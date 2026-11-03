"""The searching agent's tools: the stock it writes, the known-set it reads, and the check
the system runs before the run ends.

The pipeline: the user writes the spec; the system draws the canonical; the SEARCH finds
candidates at scale; workers do instance construction; the system verifies. This module is
the harness side of the search, and it is deliberately small. The agent plans its own
search, writes its own scripts, vets its own results and appends to its own stock file. The
system only

  1. hands the agent the known-set: candidates earlier runs and parallel attempts found
     (known/known.jsonl, re-synced before every prompt);
  2. reads the stock and counts distinct candidates on the identity the canonical declares;
  3. runs the agent's own check script over the stock in a fresh container;
  4. ends the run when the script passes and the count reaches the target, or when the
     budget is spent.

No harness code issues a query, screens a repository or extracts a candidate: the system
cannot know where a family's candidates are or how to look.

Files in the agent's workspace:
  stock/candidates.jsonl   one JSON object per line: the identity fields, spelled as the
                           canonical names them, plus whatever evidence the agent recorded
  stock/check.sh           the agent's own existence check over the stock; exit 0 = all exist
  known/known.jsonl        the known-set, written by the system, read-only for the agent
The stock is exported to <out_dir>/stock/ every cycle, so sibling agents and the freeze
(benchcompiler.search.freeze_stock) read it while the run is live.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

from agentcore.agent.worklist import KEY_NORMALIZERS, Unnormalizable, _norm_key_part
from agentcore.tools.context import ToolContext
from agentcore.tools.registry import tool
from agentcore.tools.result import ToolResult

STOCK = "stock/candidates.jsonl"
CHECK = "stock/check.sh"
KNOWN = "known/known.jsonl"
CHECK_TIMEOUT = 1800  # seconds the agent's check script may run
# Run the check by its own shebang when it has one (chmod first: files written by the agent
# are not executable), else under sh.
_CHECK_CMD = (
    "chmod +x /stock/check.sh; if head -c 2 /stock/check.sh | grep -q '#!'; "
    "then /stock/check.sh; else sh /stock/check.sh; fi"
)
_LAST = "_search_last_verify"
_VERIFY_N = "_search_verify_n"
_KNOWN_HASH = "_search_known_hash"
_EXPORT_HASH = "_search_export_hash"
_SHRINK_N = "_search_shrink_n"


# ── identity ────────────────────────────────────────────────────────────────────────────


def identity(ctx) -> tuple[list[str], dict]:
    inst = getattr(ctx, "instance", None) or {}
    fields = [str(k) for k in (inst.get("identity_fields") or [])]
    normalize = {str(k): str(v) for k, v in (inst.get("identity_normalize") or {}).items()}
    return fields, normalize


def key_of(entry: dict, fields: list[str], normalize: dict) -> str | None:
    """The candidate's identity, normalized the way the worklist normalizes it (case, a
    trailing slash and .git cannot change which thing is meant). None when a field is
    missing or empty."""
    parts = []
    for k in fields:
        v = entry.get(k)
        if v in (None, ""):
            return None
        name = normalize.get(k)
        fn = KEY_NORMALIZERS.get(name) if name else None
        try:
            v = fn(k, str(v)) if fn else str(v)
        except Unnormalizable:
            return None
        parts.append(_norm_key_part(v, keep_case=(name == "abs_path")))
    return "|".join(parts)


@dataclass
class Stock:
    entries: dict = field(default_factory=dict)  # key -> the first entry with that key
    lines: int = 0
    duplicates: int = 0
    defects: list = field(default_factory=list)  # "line N: <what is wrong>"

    @property
    def distinct(self) -> int:
        return len(self.entries)


def parse_stock(text: str | None, fields: list[str], normalize: dict) -> Stock:
    st = Stock()
    if not text:
        return st
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        st.lines += 1
        try:
            entry = json.loads(raw)
        except ValueError as e:
            st.defects.append(f"line {n}: not a JSON object ({str(e)[:60]})")
            continue
        if not isinstance(entry, dict):
            st.defects.append(f"line {n}: not a JSON object")
            continue
        missing = [k for k in fields if entry.get(k) in (None, "")]
        if missing:
            st.defects.append(f"line {n}: missing {', '.join(missing)}")
            continue
        k = key_of(entry, fields, normalize)
        if k is None:
            st.defects.append(f"line {n}: identity fields cannot be normalized")
            continue
        if k in st.entries:
            st.duplicates += 1
            continue
        st.entries[k] = entry
    return st


# ── the workspace ───────────────────────────────────────────────────────────────────────


def _host_workspace(env) -> str | None:
    ws = getattr(env, "workspace", None)
    if ws and os.path.isdir(str(ws)):
        return str(ws)
    ws = getattr(env, "_host_workspace", None)
    return str(ws) if ws else None


def read_workspace_file(ctx, relpath: str) -> str | None:
    """A file of the agent's workspace, wherever the workspace is: inside its container, or
    on the host for a local run. None when the file does not exist."""
    env = getattr(ctx, "env", None)
    container = getattr(env, "_container", None)
    if container is not None:
        from agentcore.environments import docker_helpers as dh

        base = getattr(getattr(env, "spec", None), "workdir", None) or "/workspace"
        path = relpath if relpath.startswith("/") else f"{base}/{relpath}"
        code, out = dh.exec_in_container(container, f"cat {shlex.quote(path)}", capture_limit=0)
        return out if code == 0 else None
    ws = _host_workspace(env)
    if not ws:
        return None
    p = os.path.join(ws, relpath)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write_workspace_file(ctx, relpath: str, text: str) -> bool:
    """Write a file into the agent's workspace. A docker environment's put_file starts the
    container if it is not up yet (the first prompt renders before the first command), so the
    known-set is there before the agent's first action. Returns whether it was written."""
    env = getattr(ctx, "env", None)
    if hasattr(env, "put_file"):
        err = env.put_file(relpath, text)
        return not err
    ws = _host_workspace(env)
    if not ws:
        return False
    p = os.path.join(ws, relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return True


def _out_dir(ctx) -> str:
    sc = getattr(getattr(ctx, "agent", None), "_scratch", None) or {}
    return str(sc.get("out_dir") or getattr(getattr(ctx, "env", None), "out_dir", None) or os.getcwd())


def _keep_if_shrunk(out_dir: str, new_text, ctx) -> None:
    """Copy the exported stock aside when the agent is about to replace it with fewer lines."""
    live = os.path.join(out_dir, "candidates.jsonl")
    if not os.path.isfile(live):
        return

    def _n(text) -> int:
        return sum(1 for ln in (text or "").splitlines() if ln.strip())

    try:
        with open(live, encoding="utf-8", errors="replace") as fh:
            before = _n(fh.read())
    except OSError:
        return
    if before == 0 or _n(new_text) >= before:
        return
    n = int(ctx.scratch.get(_SHRINK_N) or 0) + 1
    ctx.scratch[_SHRINK_N] = n
    try:
        shutil.copyfile(live, os.path.join(out_dir, f"shrink_{n:03d}.candidates.jsonl"))
    except OSError:
        pass


def export_stock(ctx) -> str:
    """Copy the stock (and the check script) from the workspace to <out_dir>/stock/, so
    sibling agents and the freeze read the live stock. Returns the export directory."""
    out = os.path.join(_out_dir(ctx), "stock")
    os.makedirs(out, exist_ok=True)
    texts = {name: read_workspace_file(ctx, rel) for name, rel in (("candidates.jsonl", STOCK), ("check.sh", CHECK))}
    digest = hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()
    if ctx.scratch.get(_EXPORT_HASH) == digest:
        return out
    # A STOCK THAT SHRINKS LEAVES A COPY OF WHAT IT HELD. An agent may wipe its own stock
    # (for example `rm -f stock/candidates.jsonl && touch stock/candidates.jsonl` after its
    # first check fails). stock_verify keeps a copy beside each verdict, but an agent that
    # never called it would leave nothing to recover from.
    # This runs before every prompt, so it sees the file on every cycle. It does not change
    # what the freeze reads: `_stock_files` prefers the live stock and reaches a copy only when
    # the live file holds nothing, which is what keeps a deliberate deletion deleted.
    _keep_if_shrunk(out, texts.get("candidates.jsonl"), ctx)
    for name, text in texts.items():
        p = os.path.join(out, name)
        if text is None:
            if os.path.exists(p):
                os.remove(p)
            continue
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
    ctx.scratch[_EXPORT_HASH] = digest
    return out


# ── the known-set ───────────────────────────────────────────────────────────────────────


def known_set(ctx) -> dict:
    """key -> entry, over every path in instance.known_from: a candidates.jsonl, or a run
    or stock directory holding one. A path that does not exist yet (a sibling that has not
    exported) is simply empty."""
    inst = getattr(ctx, "instance", None) or {}
    fields, normalize = identity(ctx)
    mine = os.path.abspath(_out_dir(ctx))
    known: dict = {}
    for p in inst.get("known_from") or []:
        p = str(p)
        candidates = []
        if os.path.isdir(p):
            candidates = [os.path.join(p, "candidates.jsonl"), os.path.join(p, "stock", "candidates.jsonl")]
        else:
            candidates = [p]
        for c in candidates:
            if not os.path.isfile(c):
                continue
            # A frozen stock records where each candidate came from (manifest.json beside
            # it). A candidate THIS run found is not "already known" to it: a freeze taken
            # while the run was live would make an agent count its own candidates as known and
            # undercount what it had found.
            own = set()
            mpath = os.path.join(os.path.dirname(c), "manifest.json")
            if os.path.isfile(mpath):
                try:
                    with open(mpath, encoding="utf-8") as fh:
                        origin = json.load(fh).get("origin") or {}
                    own = {k for k, src in origin.items() if os.path.abspath(str(src)) == mine}
                except (OSError, ValueError):
                    own = set()
            with open(c, encoding="utf-8", errors="replace") as fh:
                st = parse_stock(fh.read(), fields, normalize)
            for k, e in st.entries.items():
                if k in own:
                    continue
                known.setdefault(k, e)
    return known


def _sync_known(ctx, known: dict) -> None:
    text = "".join(json.dumps(e, sort_keys=True) + "\n" for e in known.values())
    digest = hashlib.sha256(text.encode()).hexdigest()
    if ctx.scratch.get(_KNOWN_HASH) == digest:
        return
    # Written even when empty, so the file the prompt names exists. The hash is remembered
    # only when the write happened: an early version remembered it after a write that had
    # nowhere to go (the container not up yet) and the known-set never reached the agent.
    if write_workspace_file(ctx, KNOWN, text):
        ctx.scratch[_KNOWN_HASH] = digest


# ── status and the prompt hook ──────────────────────────────────────────────────────────


def _target(ctx) -> int:
    inst = getattr(ctx, "instance", None) or {}
    return int(inst.get("sample_size") or 0)


def stock_status(ctx) -> tuple[Stock, int, str]:
    """(the parsed stock, how many of its distinct candidates are new, the status text)."""
    fields, normalize = identity(ctx)
    st = parse_stock(read_workspace_file(ctx, STOCK), fields, normalize)
    known = known_set(ctx)
    already = sum(1 for k in st.entries if k in known)
    new = st.distinct - already
    target = _target(ctx)
    lines = [
        f"STOCK ({STOCK}): {new} new candidates of {target} wanted"
        + (f"; {already} already in the known-set, not counted" if already else "")
        + (f"; {st.duplicates} duplicate line(s)" if st.duplicates else "")
        + (f"; {len(st.defects)} malformed line(s)" if st.defects else "")
        + (" -- the file does not exist yet" if st.lines == 0 and not st.defects else "")
    ]
    for d in st.defects[:5]:
        lines.append(f"  malformed: {d}")
    if len(st.defects) > 5:
        lines.append(f"  ... and {len(st.defects) - 5} more malformed lines")
    if known:
        lines.append(
            f"KNOWN-SET ({KNOWN}): {len(known)} candidates found by earlier runs and by the agents "
            f"searching in parallel, refreshed before every step; they do not count again"
        )
    else:
        lines.append(f"KNOWN-SET ({KNOWN}): empty; no earlier run has found anything, nothing to skip")
    last = ctx.scratch.get(_LAST)
    if last:
        lines.append(f"LAST stock_verify: {last.get('headline')}")
    else:
        lines.append(
            f"stock/check.sh {'exists' if read_workspace_file(ctx, CHECK) is not None else 'not written yet'}; stock_verify not called yet"
        )
    return st, new, "\n".join(lines)


def prerender(ctx) -> dict:
    """Before each prompt: put the known-set in the workspace, export the stock, and render
    the stock status."""
    out: dict = {}
    try:
        _sync_known(ctx, known_set(ctx))
        export_stock(ctx)
        st, new, text = stock_status(ctx)
        out["stock_status"] = text
        # What progress means for a search, for the run's own record and its stall detector
        # (agent.yaml watches search.stock): the stock growing, and a verify having run.
        ws = getattr(ctx, "working_set", None)
        if ws is not None and hasattr(ws, "merge"):
            ws.merge(
                {
                    "search": {
                        "stock": st.distinct,
                        "new": new,
                        "malformed": len(st.defects),
                        "verifies": int(ctx.scratch.get(_VERIFY_N) or 0),
                    }
                }
            )
    except Exception as e:  # noqa: BLE001 - a prompt variable must never break a cycle
        out["stock_status"] = f"(stock status unavailable: {e})"
    return out


# ── the check, in a fresh container ─────────────────────────────────────────────────────


def _run_check(ctx, stock_dir: str) -> tuple[int, str, float]:
    """Run stock/check.sh over a copy of the stock: in a fresh container built from the
    agent's own image when the agent runs in docker, as a subprocess otherwise. Returns
    (exit code, output, seconds)."""
    env = getattr(ctx, "env", None)
    t0 = time.time()
    image = getattr(env, "_image", None) or getattr(getattr(env, "spec", None), "image", None)
    if getattr(env, "_container", None) is not None and image:
        from agentcore import recipes
        from agentcore.environments import docker_helpers as dh

        c = dh.start_container(
            image, "2h", workdir="/stock", env=dh.llm_key_passthrough_env(getattr(env.spec, "env", None) or {})
        )
        try:
            c.put_archive("/", recipes._tar_of(stock_dir, "stock"))
            dh.exec_in_container(c, dh.tokens_manifest_cmd())
            # The script's own shebang decides its interpreter (a bash script run under sh
            # died on `set -o pipefail` in the first live smoke); no shebang means sh.
            code, out = dh.exec_in_container(c, _CHECK_CMD, timeout=CHECK_TIMEOUT, workdir="/stock", capture_limit=0)
        finally:
            dh.cleanup(c, image=None)
        return code, out, time.time() - t0
    tmp = tempfile.mkdtemp(prefix="stock_check_")
    try:
        dst = os.path.join(tmp, "stock")
        shutil.copytree(stock_dir, dst)
        penv = {**os.environ, **(getattr(getattr(env, "spec", None), "env", None) or {})}
        try:
            proc = subprocess.run(
                ["sh", "-c", _CHECK_CMD.replace("/stock/", "./")],
                cwd=dst,
                env=penv,
                capture_output=True,
                text=True,
                timeout=CHECK_TIMEOUT,
            )
            out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
            return proc.returncode, out, time.time() - t0
        except subprocess.TimeoutExpired:
            return 124, f"check.sh did not finish within {CHECK_TIMEOUT} s", time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _tail(text: str, n: int = 30) -> str:
    ls = (text or "").splitlines()
    return "\n".join(ls[-n:])


@tool(
    "stock_add",
    params={
        "lines": "the candidate(s) to add: one JSON object, a list of them, or JSON lines (one object "
        "per line), each with the identity fields, found_by, evidence and judgment"
    },
    aliases={"lines": ["line", "candidate", "candidates", "content", "entry", "entries"]},
)
def stock_add(ctx: ToolContext, lines) -> ToolResult:
    """Add candidates to your stock, stock/candidates.jsonl, one line each, at the end of the
    file.

    Nothing already in the file is changed or removed. A candidate with the same identity
    fields as one already in the stock is skipped. An item that is not a JSON object, or lacks
    an identity field, is left out, and the reply says which and why: fix it and add it again.
    The reply also says how many candidates were added and how many lines the file now holds."""
    # The stock needs a safe append: an agent that meant to add a line could otherwise replace
    # the whole file with write_to_file after str_replace failed on long JSON lines. Removing
    # the lines a check rejected goes through stock_remove.
    fields, normalize = identity(ctx)
    if isinstance(lines, dict):
        items = [lines]
    elif isinstance(lines, list):
        items = list(lines)
    else:
        items = [ln for ln in str(lines or "").splitlines() if ln.strip()]
    current = read_workspace_file(ctx, STOCK) or ""
    have = set(parse_stock(current, fields, normalize).entries)
    added, already, left_out = [], 0, []
    for n, it in enumerate(items, 1):
        if isinstance(it, str):
            try:
                it = json.loads(it)
            except ValueError as e:
                left_out.append(f"item {n}: not a JSON object ({str(e)[:50]})")
                continue
        if not isinstance(it, dict):
            left_out.append(f"item {n}: not a JSON object")
            continue
        missing = [k for k in fields if it.get(k) in (None, "")]
        if missing:
            left_out.append(f"item {n}: missing {', '.join(missing)}")
            continue
        k = key_of(it, fields, normalize)
        if k is None:
            left_out.append(f"item {n}: identity fields cannot be normalized")
            continue
        if k in have:
            already += 1
            continue
        have.add(k)
        added.append(json.dumps(it, ensure_ascii=False))
    if added:
        text = current + ("" if not current or current.endswith("\n") else "\n") + "\n".join(added) + "\n"
        if not write_workspace_file(ctx, STOCK, text):
            return ToolResult.fail(f"could not write {STOCK}; nothing was added")
    total = sum(1 for ln in (read_workspace_file(ctx, STOCK) or "").splitlines() if ln.strip())
    msg = f"added {len(added)} candidate(s) to {STOCK}; it now has {total} line(s)"
    if already:
        msg += f"; {already} already in the stock (same {', '.join(fields)}), not added again"
    if left_out:
        msg += (
            "; left out, fix and add again: "
            + "; ".join(left_out[:5])
            + (f" (+{len(left_out) - 5} more)" if len(left_out) > 5 else "")
        )
    return ToolResult(
        display_text=msg, structured={"added": len(added), "already": already, "left_out": left_out, "lines": total}
    )


REMOVALS = "stock/removals.jsonl"


@tool(
    "stock_remove",
    params={
        "candidates": "the candidate(s) to take out: one JSON object or a list, each carrying the identity "
        "fields (other fields are ignored); a malformed line, which has no identity, is named by "
        "its line number in the stock, as stock_verify reports it",
        "reason": "what you checked by hand for these candidates and what it showed, quoting the words or "
        "lines of the source that show each one fails (the whole commit message, not its subject "
        "line; the diff; the file)",
    },
    aliases={"candidates": ["candidate", "lines", "line", "entries", "items"], "reason": ["why", "explanation"]},
)
def stock_remove(ctx: ToolContext, candidates, reason: str) -> ToolResult:
    """Take candidates out of your stock, stock/candidates.jsonl: only the lines you name.

    Every other line stays exactly as it is. Each removed line is kept with your reason in
    stock/removals.jsonl. A name that matches no line in the stock is reported and nothing is
    removed for it. The reply says how many lines were removed and how many the stock now holds."""
    # Removal otherwise means the agent rewriting its own stock, usually from inside its check
    # script, where a broken script can wipe valid lines. This is the remove step made safe, as
    # stock_add makes the add step safe.
    if not str(reason or "").strip():
        return ToolResult.fail(
            "Nothing was removed: give the reason -- what you checked by hand for these candidates and what it showed."
        )
    fields, normalize = identity(ctx)
    items = candidates if isinstance(candidates, list) else [candidates]
    current = read_workspace_file(ctx, STOCK) or ""
    raw = current.splitlines()
    drop_n, drop_k, unmatched = set(), set(), []
    for it in items:
        if isinstance(it, str):
            s = it.strip()
            if s.isdigit():
                it = int(s)
            else:
                try:
                    it = json.loads(s)
                except ValueError:
                    unmatched.append(f"{s[:60]!r} (neither a JSON object nor a line number)")
                    continue
        if isinstance(it, bool):
            unmatched.append(f"{it!r} (neither a JSON object nor a line number)")
        elif isinstance(it, int):
            if 1 <= it <= len(raw) and raw[it - 1].strip():
                drop_n.add(it)
            else:
                unmatched.append(f"line {it} (the stock has {len(raw)} lines)")
        elif isinstance(it, dict):
            k = key_of(it, fields, normalize)
            if k is None:
                why = [
                    f"missing {f}"
                    if it.get(f) in (None, "")
                    else f"{f} must be the full 40-character commit hash"
                    if normalize.get(f) == "full_sha"
                    else f"{f} cannot be read"
                    for f in fields
                    if it.get(f) in (None, "") or key_of({f: it.get(f)}, [f], normalize) is None
                ]
                unmatched.append(f"{json.dumps(it)[:80]} ({'; '.join(why) or 'identity cannot be read'})")
            else:
                drop_k.add(k)
        else:
            unmatched.append(f"{str(it)[:60]!r} (neither a JSON object nor a line number)")
    keep, removed, found_k = [], [], set()
    for n, line in enumerate(raw, 1):
        k = None
        try:
            entry = json.loads(line)
            k = key_of(entry, fields, normalize) if isinstance(entry, dict) else None
        except ValueError:
            entry = line
        if n in drop_n or (k is not None and k in drop_k):
            removed.append(entry)
            if k is not None:
                found_k.add(k)
        else:
            keep.append(line)
    unmatched += [f"{k} (not in the stock)" for k in sorted(drop_k - found_k)]
    if removed:
        text = "\n".join(keep) + ("\n" if keep else "")
        if not write_workspace_file(ctx, STOCK, text):
            return ToolResult.fail(f"could not write {STOCK}; nothing was removed")
        log = read_workspace_file(ctx, REMOVALS) or ""
        log += "".join(
            json.dumps({"ts": time.time(), "line": e, "reason": str(reason).strip()}, ensure_ascii=False) + "\n"
            for e in removed
        )
        write_workspace_file(ctx, REMOVALS, log)
    total = sum(1 for ln in keep if ln.strip())
    msg = f"removed {len(removed)} line(s) from {STOCK}; it now has {total} line(s)"
    if unmatched:
        msg += (
            "; not in the stock, nothing removed for: "
            + "; ".join(unmatched[:5])
            + (f" (+{len(unmatched) - 5} more)" if len(unmatched) > 5 else "")
        )
    return ToolResult(display_text=msg, structured={"removed": len(removed), "lines": total, "unmatched": unmatched})


@tool("stock_verify", truncation="none", idempotent_ok=True)
def stock_verify(ctx: ToolContext) -> ToolResult:
    """Count the new candidates in your stock and run your stock/check.sh.

    The check runs in a fresh container that holds only a copy of your stock/ directory, at
    /stock, which is its working directory. The run ends here when check.sh exits 0, no line in
    the stock is malformed, and the number of new candidates reaches the target. Otherwise the
    reply says what is short, and you go on. Once 90% of your budget is spent, the run also
    ends here when check.sh exits 0, with the stock as it is. Call it when you think the stock
    is complete, or earlier to see the count and what your check says."""
    fields, normalize = identity(ctx)
    target = _target(ctx)
    st, new, status = stock_status(ctx)
    stock_dir = export_stock(ctx)
    n = int(ctx.scratch.get(_VERIFY_N) or 0) + 1
    ctx.scratch[_VERIFY_N] = n
    from agentcore.agent.quota import BUDGET_SPENT_FRACTION, budget_fraction

    frac = budget_fraction(ctx)
    spent = frac >= BUDGET_SPENT_FRACTION

    if read_workspace_file(ctx, CHECK) is None:
        head = f"{status}\nCHECK: {CHECK} does not exist. Write it first: a script that walks {STOCK} and exits 0 only when every candidate exists."
        code, out, secs = 2, "", 0.0
    else:
        code, out, secs = _run_check(ctx, stock_dir)
        head = f"{status}\nCHECK: sh {CHECK} exited {code} after {secs:.0f} s (fresh container, holding only stock/)"
        if out.strip():
            head += "\n  last lines of its output:\n" + "\n".join("  | " + ln for ln in _tail(out).splitlines())

    short = []
    if code != 0:
        short.append(f"check.sh must exit 0 (it exited {code})")
    if st.defects:
        short.append(
            f"{len(st.defects)} malformed line(s) must leave the stock: stock_remove with the line number, then stock_add the corrected candidate"
        )
    if new < target:
        short.append(f"{target - new} more new candidates needed ({new} of {target})")

    done = not short
    honest_stop = bool(short) and spent and code == 0
    verdict = (
        "DONE: the stock is complete and its check passed. The run ends here."
        if done
        else f"BUDGET SPENT ({frac:.0%}): the run ends with {new} of {target}; the check passed."
        if honest_stop
        else "NOT DONE: " + "; ".join(short) + ". Keep going, then call stock_verify again."
    )
    record = {
        "n": n,
        "headline": f"#{n}: exit {code}, {new} new of {target}, {len(st.defects)} malformed"
        + (" -- DONE" if done else " -- budget spent" if honest_stop else ""),
        "exit": code,
        "new": new,
        "distinct": st.distinct,
        "target": target,
        "defects": list(st.defects),
        "duplicates": st.duplicates,
        "seconds": secs,
        "budget_fraction": frac,
        "accepted": done or honest_stop,
        "ts": time.time(),
        "output_tail": _tail(out, 60),
    }
    ctx.scratch[_LAST] = record
    try:
        with open(os.path.join(stock_dir, f"verify_{n:03d}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)
        # The stock as it was at this verify, kept beside the verdict: an agent whose check
        # failed could otherwise save an empty file over its lines. The freeze reads these copies too, so a
        # wipe after a failed check loses nothing that was ever verified.
        src = os.path.join(stock_dir, "candidates.jsonl")
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(stock_dir, f"verify_{n:03d}.candidates.jsonl"))
    except OSError:
        pass
    ws = getattr(ctx, "working_set", None)
    if ws is not None and hasattr(ws, "merge"):
        ws.merge(
            {
                "search": {
                    "stock": st.distinct,
                    "new": new,
                    "malformed": len(st.defects),
                    "verifies": n,
                    "last_verify": {k: record[k] for k in ("n", "exit", "new", "target", "accepted")},
                }
            }
        )
    text = f"{head}\n{verdict}"
    if done or honest_stop:
        return ToolResult(
            display_text=text,
            finished=True,
            structured={"success_declared": new > 0, "new": new, "target": target, "exit": code},
        )
    return ToolResult(
        display_text=text, structured={"new": new, "target": target, "exit": code, "malformed": len(st.defects)}
    )


def finish_gate(ctx):
    """A finish that did not come through stock_verify is refused unless the budget is
    spent: the run ends on a passing check over a complete stock, nothing else."""
    from agentcore.agent.quota import BUDGET_SPENT_FRACTION, budget_fraction
    from agentcore.oracle.base import OracleResult

    last = ctx.scratch.get(_LAST) or {}
    if last.get("accepted"):
        return OracleResult(verdict="ACCEPT", success=int(last.get("new") or 0) > 0, reason=str(last.get("headline")))
    frac = budget_fraction(ctx)
    if frac >= BUDGET_SPENT_FRACTION:
        try:
            _st, new, _t = stock_status(ctx)
            export_stock(ctx)
        except Exception:  # noqa: BLE001
            new = 0
        return OracleResult(
            verdict="ACCEPT", success=new > 0, reason=f"budget spent ({frac:.0%}); {new} new candidates in the stock"
        )
    msg = (
        f"The run is not finished. It ends when stock_verify reports DONE: your {CHECK} exits 0 over "
        f"{STOCK} and the stock holds {_target(ctx)} new candidates. Call stock_verify."
    )
    return OracleResult(verdict="REJECT_CONTINUE", success=False, reason=msg, message=msg)
