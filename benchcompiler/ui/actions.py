"""What bench-ui can do, when it is started with --allow-actions: launch a search campaign, freeze
a stock, launch a construction epoch, package, certify, and stop runs.

Every action is an existing command of the pipeline; the UI only fills in its arguments from a
form, shows the exact command, runs its check first, and runs it after a typed confirmation.

  check   the command's own --dry-run when it has one (search_campaign.py, construct_arms.py),
          otherwise a preview computed here (inputs exist, the output does not)
  run     allowed only after a passing check of the same arguments in the last 30 minutes, and
          with the confirmation word typed; long commands run detached in their own session

Nothing is chosen for the user: a model, a provider route or a budget has no default here; an
empty optional field is left out, so the script's own default applies.

Every check, run and stop is recorded in <runs>/ui_actions/jobs.jsonl with its log.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from benchcompiler.ui import data as D

CHECK_VALID_S = 30 * 60
CHECK_TIMEOUT_S = 300
MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RUN_LABEL = "com.benchcompiler.run"
#: Environment variables the commands read; the UI shows which are set, never their values.
ENV_NAMES = (
    "OPENROUTER_KEY",
    "GITHUB_TOKEN",
    "GH_FINEGRAINED_TOKEN",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "LITELLM_BASE_URL",
    "LITELLM_KEY",
    "BENCH_CACHE_URL",
    "AGENT_MODEL",
)


class Refused(ValueError):
    """A form or a request the UI will not turn into a command; the message says what to change."""


# ---- fields ----------------------------------------------------------------------------------


def field(name, label, kind="text", required=False, choices=None, help="", placeholder="", value=""):
    return {
        "name": name,
        "label": label,
        "kind": kind,
        "required": required,
        "choices": choices or [],
        "help": help,
        "placeholder": placeholder,
        "value": value,
    }


def _get(form: dict, f: dict):
    raw = form.get(f["name"])
    if f["kind"] == "multi":
        vals = [v for v in (raw if isinstance(raw, list) else [raw] if raw else []) if v]
        bad = [v for v in vals if v not in f["choices"]]
        if bad:
            raise Refused(f"{f['label']}: {bad[0]!r} is not one of the listed choices")
        if f["required"] and not vals:
            raise Refused(f"{f['label']}: tick at least one")
        return vals
    raw = raw[0] if isinstance(raw, list) and raw else raw or ""
    raw = str(raw).strip()
    if not raw:
        if f["required"]:
            raise Refused(f"{f['label']}: required (the UI does not choose it for you)")
        return None
    k = f["kind"]
    try:
        if k == "int":
            v = int(raw)
            if v < 0:
                raise ValueError
            return v
        if k == "float":
            v = float(raw)
            if v < 0:
                raise ValueError
            return v
    except ValueError:
        raise Refused(f"{f['label']}: {raw!r} is not a non-negative number")
    if k == "choice":
        if raw not in f["choices"]:
            raise Refused(f"{f['label']}: {raw!r} is not one of {', '.join(f['choices'])}")
        return raw
    if k == "model":
        if not MODEL.match(raw):
            raise Refused(f"{f['label']}: {raw!r} is not a model string (provider/name, no spaces)")
        return raw
    if k == "name":
        if not NAME.match(raw):
            raise Refused(f"{f['label']}: {raw!r}: use letters, digits, '.', '_' or '-' only")
        return raw
    if k == "json":
        try:
            json.loads(raw)
        except ValueError as e:
            raise Refused(f"{f['label']}: not valid JSON ({e})")
        return raw
    if k == "path":
        return raw
    return raw


def values(action: dict, form: dict) -> dict:
    return {f["name"]: _get(form, f) for f in action["fields"]}


# ---- the actions -----------------------------------------------------------------------------


def _bench(root: str) -> list[str]:
    exe = shutil.which("bench")
    return [exe] if exe else [sys.executable, "-m", "benchcompiler.bench_cli"]


def _rel(root: str, path: str) -> str:
    return os.path.relpath(path, root)


def search_action(tree, task: str) -> dict:
    epochs = [
        int(m.group(1))
        for d in glob.glob(os.path.join(tree.runs, "search", task, "epoch_*"))
        if (m := re.search(r"epoch_(\d+)$", d))
    ]
    refs = sorted(
        _rel(tree.root, p)
        for p in glob.glob(os.path.join(tree.root, "agents", f"{task}_search", "instances", "instances.jsonl"))
        + glob.glob(os.path.join(tree.root, "references", "search", f"{task}.instances.jsonl"))
    )
    fields = [
        field(
            "epoch",
            "epoch number",
            "int",
            True,
            value=str(max(epochs, default=0) + 1),
            help="runs/search/<task>/epoch_<n>; the next free number is filled in",
        ),
        field("model", "model", "model", True, placeholder="e.g. openrouter/vendor/model-a"),
        field("agents", "agents", "int", True),
        field("target", "candidates per agent (target)", "int", True),
        field("cost", "cost cap per agent ($)", "float", True),
        field("wall", "wall clock per agent (s)", "int", help="empty: the script's default (21600)"),
        field("max_search", "max concurrent search agents on this machine", "int", True),
        field("territory_mode", "territory mode", "choice", True, ["none", "request", "drawn"]),
        field("start_delay", "seconds between agent starts", "float", True),
        field("directions", "shared search directions", "choice", True, ["off", "on"]),
        field("directions_model", "directions model", "model", help="needed when directions are on"),
        field("sync_interval", "directions refresh (s)", "float", help="needed when directions are on"),
        field("lessons", "lessons loop", "choice", True, ["off", "on"]),
        field("lessons_model", "lessons model", "model", help="needed when lessons are on"),
        field("lessons_interval", "lessons interval (s)", "int", help="needed when lessons are on"),
        field(
            "reference",
            "reference request",
            "choice",
            choices=refs,
            help="needed when the spec or agent was revised since the reference (see README)",
        ),
        field("extra_body", "extra request body (JSON)", "json", placeholder='{"reasoning": {"enabled": false}}'),
        field("max_tokens", "max tokens per reply", "int"),
    ]

    def argv(v):
        a = [sys.executable, "scripts/search_campaign.py", task, str(v["epoch"])]
        for k in (
            "agents",
            "territory_mode",
            "start_delay",
            "directions",
            "directions_model",
            "sync_interval",
            "lessons",
            "lessons_model",
            "lessons_interval",
            "target",
            "reference",
            "model",
            "cost",
            "wall",
            "extra_body",
            "max_tokens",
            "max_search",
        ):
            if v.get(k) is not None:
                a += ["--" + k.replace("_", "-"), str(v[k])]
        return a

    return {
        "id": "search",
        "label": "Launch a search campaign",
        "task": task,
        "paid": True,
        "fields": fields,
        "doc": "scripts/search_campaign.py: the agents start one after another; each attempt is "
        "dry-run by search_replicate.py before it starts.",
        "argv": argv,
        "check": lambda v: argv(v) + ["--dry-run"],
        "detached": True,
        "log": lambda v: os.path.join(tree.root, "runs", "search", task, f"epoch_{v['epoch']}", "campaign.log"),
        "confirm": lambda v: f"{task} epoch_{v['epoch']}",
    }


def freeze_action(tree, task: str) -> dict:
    camps = [c["epoch"] for c in tree.search_campaigns(task)]
    fields = [
        field(
            "epochs", "search campaigns to freeze", "multi", True, camps, help="every attempt of the ticked campaigns"
        ),
        field("name", "stock name", "name", True, placeholder=f"{task}_<date>", help="written to stock/<name>/"),
    ]

    def attempts(v):
        return sorted(
            a for e in v["epochs"] for a in glob.glob(os.path.join(tree.runs, "search", task, e, "attempt_*"))
        )

    def argv(v):
        return [
            *_bench(tree.root),
            "stock-freeze",
            *attempts(v),
            "--out",
            os.path.join("stock", v["name"]),
            "--agent",
            os.path.join("agents", f"{task}_search"),
        ]

    def preview(v):
        out = os.path.join(tree.root, "stock", v["name"])
        lines = [f"output stock/{v['name']}/: {'EXISTS - choose another name' if os.path.exists(out) else 'new'}"]
        total = 0
        for a in attempts(v):
            n = D.count_lines(os.path.join(a, "stock", "candidates.jsonl"))
            total += n
            lines.append(f"  {_rel(tree.runs, a)}: {n} candidates in its stock file")
        lines.append(f"{total} candidates before deduplication on the identity")
        return (not os.path.exists(out) and total > 0), "\n".join(lines)

    return {
        "id": "freeze",
        "label": "Freeze a stock",
        "task": task,
        "paid": False,
        "fields": fields,
        "doc": "bench stock-freeze: merges the agents' stock files into one frozen stock, deduplicated "
        "on the identity fields.",
        "argv": argv,
        "preview": preview,
        "detached": False,
        "confirm": lambda v: v["name"],
    }


def construct_action(tree, task: str) -> dict:
    stocks = [s["name"] for s in tree.stocks(task)]
    fields = [
        field("stock", "frozen stock", "choice", True, stocks),
        field("epoch", "run name", "name", True, help="runs/construct/<task>/<name>, e.g. epoch_20"),
        field("target", "deliveries wanted (target)", "int", True),
        field("arm", "arm name", "name", True, placeholder="e.g. vendor"),
        field("model", "model", "model", True, placeholder="e.g. openrouter/vendor/model-a"),
        field("workers", "workers", "int", True),
        field("cost_usd", "cost cap per worker ($)", "float", True),
        field("wallclock", "wall clock per worker (s)", "int", True),
        field("temperature", "temperature", "float"),
        field("effort", "effort (Claude Code route)", "choice", choices=["low", "medium", "high"]),
        field("timeout", "seconds per reply (Claude Code route)", "int"),
        field("extra_body", "extra request body (JSON)", "json"),
        field("max_tokens", "max tokens per reply", "int"),
        field(
            "order",
            "hand-over order",
            "choice",
            choices=["by repository", "shuffle", "size order"],
            help="empty: by repository (the script's default)",
        ),
        field("seed", "shuffle seed", "int", help="with order = shuffle"),
        field("size_csv", "size CSV (repository,size)", "path", help="with order = size order"),
    ]

    def out_dir(v):
        return os.path.join(tree.root, "runs", "construct", task, v["epoch"])

    def arms(v):
        arm = {
            "name": v["arm"],
            "model": v["model"],
            "workers": v["workers"],
            "cost_usd": v["cost_usd"],
            "wallclock": v["wallclock"],
        }
        for k in ("temperature", "effort", "timeout", "max_tokens"):
            if v.get(k) is not None:
                arm[k] = v[k]
        if v.get("extra_body"):
            arm["extra_body"] = json.loads(v["extra_body"])
        return [arm]

    def argv(v):
        arms_path = os.path.join(out_dir(v), "arms.json")
        a = [
            sys.executable,
            "scripts/construct_arms.py",
            os.path.join("stock", v["stock"], "candidates.jsonl"),
            os.path.join("agents", f"{task}_worker"),
            _rel(tree.root, out_dir(v)),
            _rel(tree.root, arms_path),
            "--target",
            str(v["target"]),
        ]
        if v.get("order") == "shuffle":
            if v.get("seed") is None:
                raise Refused("shuffle seed: required with order = shuffle")
            a += ["--shuffle", str(v["seed"])]
        elif v.get("order") == "size order":
            if not v.get("size_csv"):
                raise Refused("size CSV: required with order = size order")
            a += ["--size-order", v["size_csv"]]
        return a

    def before(v):
        """Refuse a run directory that already has workers; write the arms file the command reads."""
        if glob.glob(os.path.join(out_dir(v), "worker_*", "launcher.pid")):
            raise Refused(f"runs/construct/{task}/{v['epoch']} already has launched workers; choose another run name")
        os.makedirs(out_dir(v), exist_ok=True)
        with open(os.path.join(out_dir(v), "arms.json"), "w", encoding="utf-8") as fh:
            json.dump(arms(v), fh, indent=1)

    return {
        "id": "construct",
        "label": "Launch construction",
        "task": task,
        "paid": True,
        "fields": fields,
        "doc": "scripts/construct_arms.py: hands the stock over to the workers and launches them. "
        "Its --dry-run does the hand-over and prints the commands without launching.",
        "argv": argv,
        "check": lambda v: argv(v) + ["--dry-run"],
        "before": before,
        "detached": True,
        "log": lambda v: os.path.join(out_dir(v), "construct_arms.log"),
        "confirm": lambda v: f"{task} {v['epoch']}",
    }


def package_action(tree, task: str) -> dict:
    runs = [r["key"] for r in tree.construction_runs(task)]
    fields = [
        field("runs", "construction runs", "multi", True, runs),
        field("set", "benchmark set name", "name", True, help="written to benchmarks/<set>/<task>/"),
        field("name", "benchmark name", "name", help="empty: the script's default"),
    ]

    def argv(v):
        a = [
            sys.executable,
            "scripts/package.py",
            "--task",
            task,
            "--runs",
            *[_rel(tree.root, tree.run_dir(k)) for k in v["runs"]],
            "--out",
            os.path.join("benchmarks", v["set"], task),
            "--contract",
            os.path.join("agents", f"{task}_worker", "contract.json"),
        ]
        if v.get("name"):
            a += ["--name", v["name"]]
        return a

    def preview(v):
        out = os.path.join(tree.root, "benchmarks", v["set"], task)
        lines = [
            f"output benchmarks/{v['set']}/{task}/: {'EXISTS - choose another set name' if os.path.exists(out) else 'new'}"
        ]
        n = 0
        for k in v["runs"]:
            r = tree.construction_run(k)
            n += len(r["delivered"])
            lines.append(f"  {r['label']}: {len(r['delivered'])} delivered")
        lines.append(f"{n} delivered recipes to package")
        return (not os.path.exists(out) and n > 0), "\n".join(lines)

    return {
        "id": "package",
        "label": "Package delivered instances",
        "task": task,
        "paid": False,
        "fields": fields,
        "doc": "scripts/package.py: turns the delivered recipes into a benchmark with its bench interpreter.",
        "argv": argv,
        "preview": preview,
        "detached": False,
        "confirm": lambda v: f"{v['set']}/{task}",
    }


def certify_action(tree, task: str) -> dict:
    sets = [b["set"] for b in tree.benchmark_sets() if b["task"] == task]
    fields = [
        field("set", "benchmark set", "choice", True, sets),
        field("fresh", "fresh images (rebuild without the cache)", "choice", True, ["yes", "no"]),
        field("jobs", "parallel replays", "int", help="empty: the script's default (4)"),
    ]

    def argv(v):
        a = [
            sys.executable,
            "scripts/certify_packaged.py",
            os.path.join("benchmarks", v["set"], task),
            "--contract",
            os.path.join("agents", f"{task}_worker", "contract.json"),
        ]
        if v["fresh"] == "yes":
            a.append("--fresh")
        if v.get("jobs"):
            a += ["--jobs", str(v["jobs"])]
        return a

    def preview(v):
        b = tree.benchmark(v["set"], task)
        ok = docker_ok()
        return ok and b["count"] > 0, (
            f"{b['count']} instances in benchmarks/{v['set']}/{task}\n"
            f"docker: {'reachable' if ok else 'NOT reachable (docker ps failed)'}\n"
            "no model is called; each instance is built and replayed"
        )

    return {
        "id": "certify",
        "label": "Certify a benchmark",
        "task": task,
        "paid": False,
        "fields": fields,
        "doc": "scripts/certify_packaged.py: replays every packaged instance from its files and writes "
        "a certification record beside the manifest.",
        "argv": argv,
        "preview": preview,
        "detached": True,
        "log": lambda v: os.path.join(
            tree.root, "runs", "ui_actions", f"certify_{v['set']}_{task}_{int(time.time())}.log"
        ),
        "confirm": lambda v: f"{v['set']}/{task}",
    }


BUILDERS = {
    "search": search_action,
    "freeze": freeze_action,
    "construct": construct_action,
    "package": package_action,
    "certify": certify_action,
}


def docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "ps", "-q"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def form_hash(action_id: str, task: str, v: dict) -> str:
    return hashlib.sha256(json.dumps([action_id, task, v], sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---- processes -------------------------------------------------------------------------------


def _proc_ids() -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


def proc_info(pid: int) -> dict | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            stat = fh.read().decode(errors="replace")
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return None
    rest = stat[stat.rindex(")") + 2 :].split()
    if rest[0] == "Z":
        return None
    return {"pid": pid, "pgid": int(rest[2]), "cmd": cmd}


def group_members(pgid: int) -> list[dict]:
    return [i for p in _proc_ids() if (i := proc_info(p)) and i["pgid"] == pgid]


def campaign_processes(task: str, epoch: str) -> list[dict]:
    """The search_campaign.py process of runs/search/<task>/epoch_<n> and its lessons loop, found by
    their exact arguments (never by a pattern match on command lines)."""
    n = epoch.removeprefix("epoch_")
    out = []
    for p in _proc_ids():
        i = proc_info(p)
        if not i:
            continue
        try:
            with open(f"/proc/{p}/cmdline", "rb") as fh:
                args = fh.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(a.endswith("search_campaign.py") for a in args) and task in args and n in args:
            out.append(i)
        elif any(a.endswith("lessons_loop.sh") for a in args) and any(
            a.rstrip("/").endswith(f"search/{task}/epoch_{n}") for a in args
        ):
            out.append(i)
    return out


def stop_targets(tree, kind: str, task: str, name: str, agent: str | None = None) -> dict:
    """What a stop would end: process groups (from launcher.pid files, plus a campaign's own
    process) and the run's labelled containers."""
    if kind == "construction":
        run_dir = tree.run_dir(name)
        agent_dirs = (
            [os.path.join(run_dir, D.safe(agent))] if agent else sorted(glob.glob(os.path.join(run_dir, "worker_*")))
        )
        extra = []
    elif kind == "search":
        run_dir = os.path.join(tree.runs, "search", D.safe(task), D.safe(name))
        agent_dirs = (
            [os.path.join(run_dir, D.safe(agent))] if agent else sorted(glob.glob(os.path.join(run_dir, "attempt_*")))
        )
        extra = [] if agent else campaign_processes(task, name)
    else:
        raise Refused(f"unknown run kind {kind!r}")
    groups: dict[int, dict] = {}
    for i in extra:
        groups.setdefault(i["pgid"], {"pgid": i["pgid"], "what": "campaign process", "members": []})
    for a in agent_dirs:
        try:
            pid = int(open(os.path.join(a, "launcher.pid")).read().strip())
        except (OSError, ValueError):
            continue
        i = proc_info(pid)
        if i:
            groups.setdefault(i["pgid"], {"pgid": i["pgid"], "what": os.path.basename(a), "members": []})
    for g in groups.values():
        g["members"] = group_members(g["pgid"])
    own = os.getpgid(0)
    if own in groups:
        raise Refused("the UI's own process group is among the targets; stop it from a shell instead")
    return {
        "run_dir": run_dir,
        "agents": [os.path.basename(a) for a in agent_dirs],
        "groups": [g for g in groups.values() if g["members"]],
        "containers": run_containers(agent_dirs),
    }


def run_containers(agent_dirs: list[str]) -> list[dict]:
    """Running containers whose run label is one of these agent directories."""
    want = {os.path.realpath(a) for a in agent_dirs} | set(agent_dirs)
    try:
        out = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"label={RUN_LABEL}",
                "--format",
                '{{.ID}}\t{{.Names}}\t{{.Label "' + RUN_LABEL + '"}}',
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and (parts[2] in want or os.path.realpath(parts[2]) in want):
            rows.append({"id": parts[0], "name": parts[1], "run": parts[2]})
    return rows


def stop(targets: dict, force: bool = False, wait_s: float = 20.0) -> dict:
    """SIGTERM (or SIGKILL with force) to each process group, then verify the groups are empty."""
    sig = signal.SIGKILL if force else signal.SIGTERM
    sent = []
    for g in targets["groups"]:
        try:
            os.killpg(g["pgid"], sig)
            sent.append(g["pgid"])
        except ProcessLookupError:
            pass
    deadline = time.time() + wait_s
    left = []
    while time.time() < deadline:
        left = [m for g in targets["groups"] for m in group_members(g["pgid"])]
        if not left:
            break
        time.sleep(0.5)
    return {"signal": sig.name, "groups": sent, "left": left}


def remove_containers(ids: list[str]) -> str:
    if not ids:
        return "no container to remove"
    r = subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, text=True, timeout=120)
    return (r.stdout + r.stderr).strip()


# ---- jobs ------------------------------------------------------------------------------------


class Jobs:
    """Every check, run and stop the UI started, in <runs>/ui_actions/jobs.jsonl."""

    def __init__(self, runs: str):
        self.dir = os.path.join(runs, "ui_actions")
        self.ledger = os.path.join(self.dir, "jobs.jsonl")
        self.procs: dict[str, subprocess.Popen] = {}

    def _record(self, job: dict) -> dict:
        os.makedirs(self.dir, exist_ok=True)
        with open(self.ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(job) + "\n")
        return job

    def all(self) -> list[dict]:
        """The latest record of each job, newest first, with whether its process still runs."""
        last: dict[str, dict] = {}
        for j in D.read_jsonl(self.ledger):
            last[j["id"]] = {**last.get(j["id"], {}), **j}
        for j in last.values():
            p = self.procs.get(j["id"])
            if p is not None and p.poll() is not None and j.get("exit") is None:
                j.update(self._finish(j["id"], p.returncode))
            j["running"] = bool(j.get("pid")) and j.get("exit") is None and proc_info(j["pid"]) is not None
        return sorted(last.values(), key=lambda j: j["started"], reverse=True)

    def get(self, job_id: str) -> dict | None:
        return next((j for j in self.all() if j["id"] == job_id), None)

    def _finish(self, job_id: str, code: int) -> dict:
        upd = {"id": job_id, "exit": code, "ended": time.time()}
        self._record(upd)
        return upd

    def new_id(self, kind: str) -> str:
        return (
            time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            + f"_{kind}_{os.getpid() % 1000:03d}{int(time.time() * 1000) % 1000:03d}"
        )

    def run_sync(self, kind: str, meta: dict, argv: list[str], cwd: str, timeout: int = CHECK_TIMEOUT_S) -> dict:
        jid = self.new_id(kind)
        log = os.path.join(self.dir, f"{jid}.log")
        os.makedirs(self.dir, exist_ok=True)
        t0 = time.time()
        try:
            r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
            code, out = r.returncode, (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired as e:
            code, out = -1, f"{e.stdout or ''}{e.stderr or ''}\n[the UI stopped waiting after {timeout} s]"
        except OSError as e:
            code, out = -1, f"could not start: {e}"
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(out)
        return self._record(
            {
                "id": jid,
                "kind": kind,
                **meta,
                "argv": argv,
                "cwd": cwd,
                "started": t0,
                "ended": time.time(),
                "exit": code,
                "log": log,
                "output": out[-20_000:],
            }
        )

    def run_detached(self, kind: str, meta: dict, argv: list[str], cwd: str, log: str) -> dict:
        jid = self.new_id(kind)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            p = subprocess.Popen(
                argv, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
            )
        self.procs[jid] = p
        return self._record(
            {
                "id": jid,
                "kind": kind,
                **meta,
                "argv": argv,
                "cwd": cwd,
                "started": time.time(),
                "pid": p.pid,
                "pgid": p.pid,
                "log": log,
                "exit": None,
            }
        )

    def note(self, kind: str, meta: dict, text: str, exit: int = 0) -> dict:
        jid = self.new_id(kind)
        return self._record(
            {
                "id": jid,
                "kind": kind,
                **meta,
                "started": time.time(),
                "ended": time.time(),
                "exit": exit,
                "output": text,
            }
        )

    def passed_check(self, fh: str) -> dict | None:
        """A passing check of these exact arguments in the last CHECK_VALID_S seconds."""
        now = time.time()
        for j in reversed(D.read_jsonl(self.ledger)):
            if (
                j.get("kind") == "check"
                and j.get("form_hash") == fh
                and j.get("exit") == 0
                and now - j.get("started", 0) < CHECK_VALID_S
            ):
                return j
        return None


def env_status() -> list[tuple[str, bool]]:
    return [(n, bool(os.environ.get(n))) for n in ENV_NAMES]
