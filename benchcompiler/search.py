"""Emit the SEARCHING AGENT from the spec and its canonical; freeze its stock; hand the
stock to workers.

The pipeline: the user writes the spec; the system draws the canonical; the SEARCH finds
candidates at scale; workers do instance construction through the stages; the system
verifies. This module is the search step's plumbing, and nothing in it searches: the agent
plans its own search, writes its own scripts and its own check, and the system only reads
the stock (agentcore.tools.search_tools).

    bench search-agent <spec> --canonical <json> --out agents/<family>_search --target N
        writes the agent: agent.yaml, instances/instances.jsonl (the brief: the spec's data
        paragraph verbatim, the admissibility clauses, the constraints, the identity the
        canonical's search block declares, the target and the spread), instances/schema.py.
        Nothing is drawn by a model; the brief is the spec's own words.
    bench stock-freeze <run dirs...> --out artifacts/stock/<family>
        merges the runs' exported stocks on the identity, drops malformed lines, and writes
        candidates.jsonl (append-only across freezes) and manifest.json.
    bench handover <stock/candidates.jsonl> --worker agents/<family>_worker --workers N ...
        fills each worker's `candidates` from the stock (whole repositories to one worker,
        so a per-repository cap holds) and launches the construct runs.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD = os.path.abspath(os.path.join(HERE, ".."))
if SCAFFOLD not in sys.path:
    sys.path.insert(0, SCAFFOLD)

DEFAULT_IMAGE = "python:3.12-bookworm"  # git, curl, python3, pip; the agent installs the rest

SCHEMA_PY = '''"""Instance schema for a searching agent: what it is told about the benchmark and the search."""

from __future__ import annotations

from pydantic import BaseModel


class SearchTask(BaseModel):
    benchmark_name: str
    task_description: str
    candidate_definition: str
    criteria: str = ""
    identity_fields: list[str]
    identity_normalize: dict[str, str] = {}
    data_source: str
    admissibility: list[dict] = []
    constraints: str = ""
    spread: str = ""
    sample_size: int = 100
    known_from: list[str] = []    # stock files or run dirs whose candidates do not count again
    territory: str = ""           # this attempt's part of the search space, one sentence
    sibling_territories: list[str] = []   # the parts the parallel attempts cover
'''


def _canon(canonical: dict) -> dict:
    return canonical.get("canonical_spec") if isinstance(canonical.get("canonical_spec"), dict) else canonical


def _text(v) -> str:
    if isinstance(v, list):
        return "\n".join(f"- {x}" for x in v)
    return str(v or "")


def identity_from_canonical(canonical: dict) -> tuple[list[str], dict, list[dict]]:
    from benchcompiler.worker import identity_from_canonical as _worker_identity

    return _worker_identity(canonical)


def spread_text(caps: list[dict], key: list[str]) -> str:
    parts = []
    for cap in caps or []:
        keys = [str(k) for k in (cap.get("keys") or [])]
        limit = cap.get("limit")
        if not keys or limit is None:
            continue
        what = str(cap.get("dimension") or "spread").strip()
        # the dimension is a label ("per_repo", "repo"); render it as the noun after "per"
        what = what[4:] if what.lower().startswith(("per_", "per ")) else what
        what = what.replace("_", " ").strip() or "spread"
        parts.append(f"at most {limit} per {what} ({', '.join(keys)})")
    return "; ".join(parts) if parts else "spread across repositories and organisations rather than several from one"


def search_instance(
    raw_spec: dict, canonical: dict, *, benchmark_name: str, target: int, known_from: list[str] | None = None
) -> dict:
    c = _canon(canonical)
    key, normalize, caps = identity_from_canonical(canonical)
    # Prose falls through to what the formalizer derived from it. A raw `admissibility`
    # written as a sentence is a supported shape (spec.admissibility_shape), and the model
    # turns it into clauses in the canonical. Raw clauses win only when they are a list;
    # selecting on truthiness would hand the searching agent no floor at all.
    raw_adm = raw_spec.get("admissibility")
    adm = [
        x
        for x in ((raw_adm if isinstance(raw_adm, list) else None) or c.get("admissibility") or [])
        if isinstance(x, dict)
    ]
    return {
        "benchmark_name": benchmark_name,
        "task_description": _text(raw_spec.get("task") or c.get("task")).strip(),
        "candidate_definition": _text(c.get("candidate_definition") or raw_spec.get("instance")).strip(),
        "identity_fields": key,
        "identity_normalize": normalize,
        # The spec's own data paragraph, verbatim: where candidates come from, what is not a source.
        "data_source": _text(raw_spec.get("data") or c.get("data_source")).strip(),
        "admissibility": [
            {
                "name": str(a.get("name") or ""),
                "observe": str(a.get("observe") or ""),
                "require": str(a.get("require") or ""),
            }
            for a in adm
        ],
        "constraints": _text(raw_spec.get("constraints") or c.get("constraints")).strip(),
        # The spec's criteria, verbatim: the clauses a candidate must satisfy, stated precisely
        # enough to look for and to check. Without them the agent has only the one-sentence
        # definition and the data paragraph, and can reduce a criterion such as "runnable
        # command-line program" to a crude textual match.
        "criteria": _text(raw_spec.get("criteria") or c.get("criteria")).strip(),
        "spread": spread_text(caps, key),
        "sample_size": int(target),
        "known_from": [os.path.abspath(p) for p in (known_from or [])],
    }


def render_agent_yaml(
    name: str,
    description: str,
    *,
    model: str,
    image: str = DEFAULT_IMAGE,
    budget_usd: float = 2.0,
    wallclock_s: int = 6 * 3600,
) -> str:
    head = {
        "name": name,
        "description": description,
        "loop": "standard",
        "action_protocol": "json",
        "models": {"main": {"name": model, "temperature": 0.3, "kwargs": {"max_tokens": 32000}}},
        # One container for the whole run: the agent's scripts, clones and result files live
        # there. The check runs in a FRESH container from the same image (search_tools).
        "env": {
            "type": "docker",
            "image": image,
            "workdir": "/workspace",
            "timeout": 900,
            "container_lifetime": "48h",
            "resources": {"mem_limit": "4g"},
        },
        "tools": [
            "write_to_file",
            "str_replace",
            "read_file",
            "linux_terminal",
            "launch_background",
            "poll_process",
            "stock_add",
            "stock_remove",
            "stock_verify",
        ],
        "tool_modules": ["agentcore.tools.search_tools"],
        "prompts": {"system": "../_shared/search_system.j2", "cycle": "../_shared/search_cycle.j2"},
        "prerender": "agentcore.tools.search_tools:prerender",
        # Progress is the stock growing or a verify having run; a long query script polled
        # for a few cycles is not a stall. There is no stage to force, so only the nudge fires.
        "stall": {"watch": ["search.stock", "search.verifies"], "nudge_after": 8},
    }
    tail = {
        "instance_schema": "instances.schema:SearchTask",
        "tool_output": {"strategy": "heuristic", "truncation": "head_tail"},
        "oracle": {"kind": "sentinel", "grade": "agentcore.tools.search_tools:finish_gate"},
        "budgets": {
            "meters": [
                {"kind": "cost_usd", "limit": budget_usd, "charge_on": "cycle"},
                {"kind": "wallclock", "limit": wallclock_s, "on_exhaust": "finish_cycle"},
            ]
        },
        "attempts": {"enabled": False},
        "repetition": {"exempt_tools": ["linux_terminal", "poll_process", "stock_verify", "read_file"]},
    }
    comment = (
        "# A SEARCHING AGENT finds candidates for a benchmark and nothing else: it plans its own\n"
        "# search, writes scripts that query in bulk, vets the results, and appends what it accepts\n"
        "# to stock/candidates.jsonl. It ends the run by writing stock/check.sh (every candidate\n"
        "# exists) and calling stock_verify; the system runs that script in a fresh container and\n"
        "# counts. The run ends when the check passes at the target, or when the budget is spent.\n"
        "#\n"
        "# GENERATED by `bench search-agent` from the spec and its canonical; nothing here is drawn\n"
        "# by a model. The brief is instances/instances.jsonl. Regenerate rather than edit.\n\n"
    )
    return comment + yaml.safe_dump({**head, **tail}, sort_keys=False, width=100, default_flow_style=False)


def emit_search_agent(
    spec_path: str,
    canonical_path: str,
    out_dir: str,
    *,
    target: int,
    model: str,
    name: str | None = None,
    known_from: list[str] | None = None,
    budget_usd: float = 2.0,
    wallclock_s: int = 6 * 3600,
    image: str = DEFAULT_IMAGE,
    log=print,
) -> dict:
    with open(spec_path, encoding="utf-8") as f:
        raw_spec = yaml.safe_load(f) or {}
    with open(canonical_path, encoding="utf-8") as f:
        canonical = json.load(f)
    benchmark_name = str(raw_spec.get("name") or os.path.splitext(os.path.basename(spec_path))[0])
    name = name or f"{benchmark_name}_search"
    inst = search_instance(raw_spec, canonical, benchmark_name=benchmark_name, target=target, known_from=known_from)
    os.makedirs(os.path.join(out_dir, "instances"), exist_ok=True)
    description = inst["task_description"].replace("\n", " ")
    with open(os.path.join(out_dir, "agent.yaml"), "w", encoding="utf-8") as f:
        f.write(
            render_agent_yaml(
                name, description, model=model, image=image, budget_usd=budget_usd, wallclock_s=wallclock_s
            )
        )
    with open(os.path.join(out_dir, "instances", "schema.py"), "w", encoding="utf-8") as f:
        f.write(SCHEMA_PY)
    with open(os.path.join(out_dir, "instances", "instances.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(inst) + "\n")
    with open(spec_path, "rb") as f:
        spec_sha = hashlib.sha256(f.read()).hexdigest()
    record = {
        "spec": os.path.abspath(spec_path),
        "spec_sha256": spec_sha,
        "canonical": os.path.abspath(canonical_path),
        "identity": inst["identity_fields"],
        "normalize": inst["identity_normalize"],
        "target": target,
        "model": model,
        "ts": time.time(),
    }
    with open(os.path.join(out_dir, "search.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    log(
        f"searching agent written to {out_dir}: identity {inst['identity_fields']}, target {target}, "
        f"{len(inst['admissibility'])} admissibility rules"
    )
    return {"out": out_dir, "identity": inst["identity_fields"], "name": name, "target": target}


def attempt_instance(
    agent_dir: str,
    *,
    known_from: list[str],
    target: int | None = None,
    territory: str = "",
    siblings: list[str] | None = None,
) -> dict:
    """One attempt's instance: the agent's brief with its known-set (sibling attempts' stocks
    and the frozen stock) filled in, optionally its own target, and its territory."""
    with open(os.path.join(agent_dir, "instances", "instances.jsonl"), encoding="utf-8") as fh:
        inst = json.loads(fh.readline())
    inst["known_from"] = [os.path.abspath(p) for p in known_from]
    if target is not None:
        inst["sample_size"] = int(target)
    inst["territory"] = str(territory or "").strip()
    inst["sibling_territories"] = [str(s).strip() for s in (siblings or []) if str(s).strip()]
    return inst


# ── territories: how parallel attempts split the search space ───────────────────────────

TERRITORY_PROMPT = """Parallel searches for one benchmark must cover DIFFERENT ground. Below is the brief every
searching agent receives. Split the search space into {n} territories, one per agent, so that
the territories are disjoint, together cover the space the brief describes, and each is
likely to hold a comparable number of candidates.

Choose the dimension that fits THIS benchmark: bands of repository popularity (stars) when
the brief admits one language; languages when it admits several; domains or kinds of
program (with the keywords that find them) when popularity bands would put most candidates
in one territory; or a combination. State each territory as ONE sentence an agent can act
on directly: the exact bands, languages or keywords, nothing vague.

THE BRIEF
Task: {task}
What one candidate is: {candidate}
Where candidates come from: {data}
Rules a repository must satisfy: {rules}
Constraints: {constraints}

Answer with one JSON object and nothing else:
{{"dimension": "<what the split is by, a few words>",
  "territories": ["<sentence for agent 1>", "<sentence for agent 2>", ...]}}
with exactly {n} territories."""


def check_territories(answer, n: int) -> str:
    """'' when the answer is n distinct, actionable sentences; else what is wrong."""
    if not isinstance(answer, dict):
        return "the answer is not a JSON object"
    ts = answer.get("territories")
    if not isinstance(ts, list) or len(ts) != n:
        return f"territories must be a list of exactly {n} sentences, got {ts!r}"
    if any(not isinstance(t, str) or len(t.split()) < 4 for t in ts):
        return "every territory must be a sentence of at least four words"
    if len({t.strip().lower() for t in ts}) != n:
        return "territories must be distinct"
    return ""


def draw_territories(agent_dir: str, n: int, model: str, *, log=print, retries: int = 2) -> dict:
    """Ask the model once for n disjoint territories from the agent's own brief. Returns
    {"dimension", "territories", "model"}; raises when no answer passes the check."""
    from benchcompiler.llm import call_llm_json

    with open(os.path.join(agent_dir, "instances", "instances.jsonl"), encoding="utf-8") as fh:
        inst = json.loads(fh.readline())
    rules = "; ".join(f"{a['name']}: {a['require']}" for a in inst.get("admissibility") or [])
    prompt = TERRITORY_PROMPT.format(
        n=n,
        task=inst["task_description"],
        candidate=inst["candidate_definition"],
        data=inst["data_source"],
        rules=rules or "(none)",
        constraints=inst.get("constraints") or "(none)",
    )
    last = ""
    for attempt in range(retries + 1):
        ask = prompt if not last else prompt + f"\n\nYour previous answer was refused: {last}. Answer again."
        answer = call_llm_json(model, ask, site="search.territories")
        last = check_territories(answer, n)
        if not last:
            log(f"territories ({answer.get('dimension')}): " + " | ".join(answer["territories"]))
            return {
                "dimension": str(answer.get("dimension") or ""),
                "territories": list(answer["territories"]),
                "model": model,
            }
        log(f"territory draw {attempt + 1} refused: {last}")
    raise RuntimeError(f"no territory answer passed the check after {retries + 1} draws: {last}")


# ── the frozen stock ───────────────────────────────────────────────────────────────────


def _identity_of_agent(agent_dir: str) -> tuple[list[str], dict]:
    with open(os.path.join(agent_dir, "instances", "instances.jsonl"), encoding="utf-8") as fh:
        inst = json.loads(fh.readline())
    return [str(k) for k in inst["identity_fields"]], dict(inst.get("identity_normalize") or {})


def _stock_files(run_or_file: str) -> list[str]:
    """The agent's LAST WORD on its own stock: the live file, or its newest snapshot if the
    live file holds nothing.

    stock_verify keeps a copy of the stock beside each verdict. The freeze must not union the
    live file with every copy: a candidate the agent DELETED because it failed its own
    existence check would still be in an older copy and come back.

    The copies still rescue the case of an agent saving an empty file over its own stock after
    a failed check. They are read only when the live stock has nothing in them to override.
    """
    if os.path.isfile(run_or_file):
        return [run_or_file]

    def _has_a_line(p: str) -> bool:
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                return any(ln.strip() for ln in fh)
        except OSError:
            return False

    live = ""
    for rel in ("stock/candidates.jsonl", "candidates.jsonl"):
        p = os.path.join(run_or_file, rel)
        if os.path.isfile(p):
            live = p
            break
    if live and _has_a_line(live):
        return [live]
    # nothing live: fall back to the NEWEST snapshot, not the union of all of them
    for d in (os.path.join(run_or_file, "stock"), run_or_file):
        if os.path.isdir(d):
            snaps = sorted(
                (
                    os.path.join(d, f)
                    for f in os.listdir(d)
                    if f.startswith("verify_") and f.endswith(".candidates.jsonl")
                ),
                reverse=True,
            )
            for s in snaps:
                if _has_a_line(s):
                    return [s]
            break
    return [live] if live else []


def freeze_stock(runs: list[str], out_dir: str, *, agent_dir: str, log=print) -> dict:
    """Merge the runs' exported stocks into <out_dir>/candidates.jsonl on the identity.
    Append-only: candidates already frozen stay first and keep their line. Malformed lines
    and duplicates are dropped and counted; manifest.json records where each candidate came
    from and the counts per run."""
    from agentcore.tools.search_tools import parse_stock

    fields, normalize = _identity_of_agent(agent_dir)
    os.makedirs(out_dir, exist_ok=True)
    frozen_path = os.path.join(out_dir, "candidates.jsonl")
    merged: dict = {}
    origin: dict = {}
    per_run: dict = {}
    if os.path.isfile(frozen_path):
        with open(frozen_path, encoding="utf-8") as fh:
            st = parse_stock(fh.read(), fields, normalize)
        merged.update(st.entries)
        # Each frozen candidate keeps the run it came from across freezes: the searching
        # agent's known-set reads that origin to not count its own finds against itself.
        previous = {}
        try:
            with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
                previous = json.load(fh).get("origin") or {}
        except (OSError, ValueError):
            previous = {}
        for k in st.entries:
            origin[k] = previous.get(k, "frozen")
    for run in runs:
        files = _stock_files(run)
        if not files:
            per_run[run] = {"error": "no stock file"}
            log(f"{run}: no stock file")
            continue
        texts = []
        for p in files:
            with open(p, encoding="utf-8", errors="replace") as fh:
                texts.append(fh.read())
        st = parse_stock("\n".join(texts), fields, normalize)
        new = 0
        for k, e in st.entries.items():
            if k in merged:
                continue
            merged[k] = e
            origin[k] = os.path.abspath(run)
            new += 1
        per_run[run] = {
            "files": len(files),
            "lines": st.lines,
            "distinct": st.distinct,
            "duplicates": st.duplicates,
            "malformed": len(st.defects),
            "new_to_stock": new,
        }
        log(
            f"{run}: {len(files)} file(s), {st.lines} lines, {st.distinct} distinct, {new} new to the stock, "
            f"{st.duplicates} duplicates, {len(st.defects)} malformed"
        )
    with open(frozen_path, "w", encoding="utf-8") as fh:
        for k, e in merged.items():
            fh.write(json.dumps(e, sort_keys=True) + "\n")
    manifest = {
        "identity": fields,
        "normalize": normalize,
        "count": len(merged),
        "agent_dir": os.path.abspath(agent_dir),
        "runs": per_run,
        "origin": {k: v for k, v in origin.items()},
        "frozen_at": time.time(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    log(f"stock frozen: {len(merged)} candidates in {frozen_path}")
    return manifest


# ── the hand-over to workers ───────────────────────────────────────────────────────────


def read_size_order(path: str) -> dict:
    """A CSV whose first column names a repository (owner/name) and whose second column is
    its size: repository (lower case) -> size. For `partition_stock(order=...)`."""
    import csv

    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            try:
                out[row[0].strip().lower()] = float(row[1])
            except ValueError:
                continue  # the header
    return out


def _repo_of(entry: dict, key: list[str]) -> str:
    u = str(entry.get(key[0]) or "").strip().lower().rstrip("/").removesuffix(".git")
    return u.split("github.com/")[-1]


def partition_stock(
    entries: list[dict], key: list[str], n_workers: int, shuffle_seed: int | None = None, order: dict | None = None
) -> list[list[dict]]:
    """Whole groups on the first identity field (the repository) go to one worker, so a
    per-repository cap holds per worker; groups are dealt to the emptiest worker.
    With `shuffle_seed`, the candidates are shuffled (that seed fixes the order) and dealt
    round-robin, one candidate to exactly one worker (dealt by repository, a worker can meet
    many commits of one project in a row and stall on one project's trap)."""
    if order is not None or shuffle_seed is not None:
        import random

        seq = list(entries)
        random.Random(shuffle_seed if shuffle_seed is not None else 1).shuffle(seq)
        if order is not None:
            # Smallest first: sorted by the repository's size, ties keep the
            # seeded random order, a repository of unknown size goes last; dealt one per worker
            # in turn, so every worker starts on the cheapest builds.
            # An entry for one candidate (owner/name@commit) ranks it before its repository's
            # size (commits that name a rule first).
            def rank(e):
                r = _repo_of(e, key)
                sha = str(e.get(key[1]) or "").strip().lower() if len(key) > 1 else ""
                return order.get(f"{r}@{sha}", order.get(r, float("inf")))

            seq.sort(key=rank)
        n = max(1, n_workers)
        return [p for p in (seq[i::n] for i in range(n)) if p]
    groups: dict = {}
    for e in entries:
        groups.setdefault(str(e.get(key[0]) or "").strip().lower().rstrip("/"), []).append(e)
    parts: list[list[dict]] = [[] for _ in range(max(1, n_workers))]
    for g in sorted(groups.values(), key=len, reverse=True):
        parts[min(range(len(parts)), key=lambda i: len(parts[i]))].extend(g)
    return [p for p in parts if p]


def _worker_identity(worker_dir: str) -> tuple[list[str], dict]:
    with open(os.path.join(worker_dir, "agent.yaml"), encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    key = [str(k) for k in ((spec.get("worklist") or {}).get("key") or [])]
    normalize = {str(k): str(v) for k, v in ((spec.get("worklist") or {}).get("normalize") or {}).items()}
    if not key:
        raise SystemExit(f"{worker_dir}/agent.yaml declares no worklist key; it is not a worker")
    return key, normalize


def _identity_key(ident, key: list[str], normalize: dict) -> str | None:
    """The normalized key of an identity given as a dict of the worker's fields or as the
    `field1|field2` string the harness prints (drop notices, worklist lines)."""
    from agentcore.tools.search_tools import key_of

    if isinstance(ident, dict):
        return key_of(ident, key, normalize)
    parts = str(ident).strip().split("|")
    if len(parts) != len(key):
        return None
    return key_of(dict(zip(key, parts)), key, normalize)


def finished_candidates(paths: list[str], worker_dir: str) -> dict:
    """What earlier epochs finished, as `field1|field2` identities: delivered = a recipe
    directory holding report.json and bench_last/instance.json; dropped = the harness's
    "X dropped: ..." notice in the trajectory. NOT the drop row's own `candidate` field,
    which names the candidate handed next."""
    key, normalize = _worker_identity(worker_dir)
    delivered, dropped = {}, {}
    notice = re.compile(r"^(\S+) dropped:")
    for p in paths:
        runs = (
            [p]
            if os.path.isfile(os.path.join(p, "trajectory.jsonl"))
            else sorted(os.path.dirname(t) for t in glob.glob(os.path.join(p, "*", "trajectory.jsonl")))
        )
        for run in runs:
            for rep in glob.glob(os.path.join(run, "recipes", "*", "report.json")):
                inst = os.path.join(os.path.dirname(rep), "bench_last", "instance.json")
                try:
                    with open(inst, encoding="utf-8") as fh:
                        j = json.load(fh)
                except (OSError, ValueError):
                    continue
                k = _identity_key(j, key, normalize)
                if k:
                    delivered[k] = "|".join(str(j.get(f) or "") for f in key)
            try:
                with open(os.path.join(run, "trajectory.jsonl"), encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if '"on_message"' not in line or "dropped:" not in line:
                            continue
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        m = notice.match(str(r.get("preview") or ""))
                        if m:
                            k = _identity_key(m.group(1), key, normalize)
                            if k:
                                dropped[k] = m.group(1)
            except OSError:
                pass
    return {
        "delivered": sorted(delivered.values()),
        "dropped": sorted(v for k, v in dropped.items() if k not in delivered),
    }


def read_exclusions(path: str) -> list:
    """Identities to exclude, one per line (`field1|field2`) or as JSON lines of the worker's fields."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                    continue
                except ValueError:
                    pass
            out.append(line)
    return out


def handover(
    stock_path: str,
    worker_dir: str,
    out_dir: str,
    *,
    workers: int,
    target: int,
    launch: bool = False,
    python: str | None = None,
    sample: int = 0,
    seed: int = 1,
    exclude: list | None = None,
    shuffle: bool = False,
    size_order: str | None = None,
    log=print,
) -> dict:
    """Fill each worker's `candidates` from the stock (a random sample of `sample` when given)
    and (optionally) launch the runs. `exclude`: identities (dicts or `f1|f2` strings) that
    never reach a worker -- what earlier epochs delivered or dropped."""
    from agentcore.tools.search_tools import parse_stock

    key, normalize = _worker_identity(worker_dir)
    with open(stock_path, encoding="utf-8", errors="replace") as fh:
        st = parse_stock(fh.read(), key, normalize)
    if st.defects:
        log(
            f"{len(st.defects)} stock line(s) lack the worker's identity {key} and are skipped: "
            + "; ".join(st.defects[:3])
        )
    excluded_keys = {k for k in (_identity_key(x, key, normalize) for x in (exclude or [])) if k}
    n_excluded = sum(1 for k in st.entries if k in excluded_keys)
    if excluded_keys:
        log(f"excluding {n_excluded} candidate(s) already finished ({len(excluded_keys)} identities given)")
    entries = [e for k, e in st.entries.items() if k not in excluded_keys]
    if not entries:
        raise SystemExit(f"{stock_path}: no candidate carries the worker's identity fields {key}")
    if sample and sample < len(entries):
        import random

        entries = random.Random(seed).sample(entries, sample)
        log(f"random sample of {sample} of {st.distinct} candidates (seed {seed})")
    with open(os.path.join(worker_dir, "instances", "instances.jsonl"), encoding="utf-8") as fh:
        base = json.loads(fh.readline())
    # A worker receives every field the search recorded. The identity is what it registers;
    # the rest are the search's notes (a commit, a file, a method it found; sometimes a
    # placeholder it invented), and the worker brief says so: leads to check, not facts.
    # Keeping only the identity would make a worker rediscover what the search already named.
    order = read_size_order(size_order) if size_order else None
    parts = partition_stock(entries, key, workers, shuffle_seed=seed if (shuffle or order) else None, order=order)
    if order is not None:
        log(
            f"smallest repository first ({size_order}, {len(order)} sizes), ties by seed {seed}, dealt one per worker in turn"
        )
    elif shuffle:
        log(f"shuffled with seed {seed}, dealt one candidate per worker in turn")
    os.makedirs(out_dir, exist_ok=True)
    share = [target // len(parts) + (1 if i < target % len(parts) else 0) for i in range(len(parts))]
    launches = []
    for i, part in enumerate(parts):
        inst = dict(base)
        inst["candidates"] = part
        inst["sample_size"] = share[i]
        spec_path = os.path.join(out_dir, f"worker_{i}.instances.jsonl")
        with open(spec_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(inst) + "\n")
        run_dir = os.path.join(out_dir, f"worker_{i}")
        py = python or os.environ.get("BENCH_WORKER_PYTHON") or sys.executable
        cmd = [
            py,
            os.path.join(HERE, "bench_cli.py"),
            "construct",
            "--config",
            os.path.join(worker_dir, "agent.yaml"),
            "--instance",
            spec_path,
            "--out",
            run_dir,
        ]
        launches.append((run_dir, cmd))
        log(f"worker {i}: {len(part)} candidates, target {share[i]}")
    manifest = {
        "stock": os.path.abspath(stock_path),
        "worker": os.path.abspath(worker_dir),
        "workers": len(parts),
        "total_target": target,
        "candidates": len(entries),
        "excluded": n_excluded,
        "runs": [rd for rd, _ in launches],
        "commands": [" ".join(c) for _, c in launches],
        "ts": time.time(),
    }
    if launch:
        pids = []
        for run_dir, cmd in launches:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "construct.log"), "w") as logf:
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
            with open(os.path.join(run_dir, "launcher.pid"), "w") as fh:
                fh.write(str(proc.pid))
            pids.append(proc.pid)
            log(f"launched {run_dir}: pid {proc.pid}")
        manifest["pids"] = pids
    with open(os.path.join(out_dir, "handover.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    if not launch:
        log("ready to launch (re-run with --launch, or run these yourself):")
        for _, cmd in launches:
            log("  " + " ".join(cmd))
    return manifest
