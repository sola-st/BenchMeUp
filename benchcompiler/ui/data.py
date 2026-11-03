"""Read-only views of a BenchMeUp tree and its runs, for the UI.

Nothing in this module writes. Two places are read:

  the repository   specs/, agents/, stock/, benchmarks/      (what the pipeline produced)
  the runs root    runs/search/..., runs/construct/...,      (what agents are doing or did)
                   runs/experiment/<name>/<task>/...

A trajectory.jsonl can be tens of GB, so it is never read whole: the latest cycles come from its
end, and one cycle's events are found by bisecting the file on the cycle number.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time

NAME = re.compile(r"^[A-Za-z0-9_.:@+-][A-Za-z0-9_.:@+=-]*$")


def safe(part: str) -> str:
    """A path component taken from a URL: no separators, no `..`."""
    if not part or part in (".", "..") or not NAME.match(part):
        raise ValueError(f"not a valid name: {part!r}")
    return part


#: Parsed JSON files keyed by (path, mtime, size): a page reads a worker's agent_state.json (often
#: megabytes) several times, and pages are reloaded every few seconds while runs are live. The
#: entry is replaced as soon as the file changes. Callers treat the result as read-only.
_JSON_CACHE: dict[str, tuple[tuple, object]] = {}
_JSON_CACHE_MAX = 2048


def read_json(path: str, default=None):
    try:
        st = os.stat(path)
    except OSError:
        return default
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _JSON_CACHE.get(path)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, ValueError):
        return default
    if len(_JSON_CACHE) >= _JSON_CACHE_MAX:
        _JSON_CACHE.pop(next(iter(_JSON_CACHE)))
    _JSON_CACHE[path] = (stamp, value)
    return value


def read_text(path: str, limit: int = 400_000) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return None


def read_jsonl(path: str, limit: int = 100_000) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    out.append(d)
                if len(out) >= limit:
                    break
    except OSError:
        pass
    return out


def count_lines(path: str) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def tail_text(path: str, nbytes: int = 20_000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            data = fh.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    return text if size <= nbytes else text.split("\n", 1)[-1]


def mtime(path: str) -> float | None:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ---- liveness, budgets ---------------------------------------------------------------------


def alive(run_dir: str) -> bool:
    """The run's own process is running: launcher.pid names a live process whose command line
    holds this run directory (a reused pid would not)."""
    try:
        with open(os.path.join(run_dir, "launcher.pid"), encoding="utf-8") as fh:
            pid = int(fh.read().strip())
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError):
        return False
    return os.path.abspath(run_dir) in cmd or os.path.basename(run_dir.rstrip("/")) in cmd


def agent_status(run_dir: str) -> dict:
    """One agent run (a search attempt or a construction worker), from its own records."""
    st = read_json(os.path.join(run_dir, "agent_state.json"), {}) or {}
    meta = read_json(os.path.join(run_dir, "run_meta.json"), {}) or {}
    res = read_json(os.path.join(run_dir, "result.json"), {}) or {}
    b = st.get("budget") or {}
    cost, wall = b.get("cost_usd") or {}, b.get("wallclock") or {}
    traj = os.path.join(run_dir, "trajectory.jsonl")
    last = mtime(traj) or mtime(os.path.join(run_dir, "agent_state.json"))
    is_alive = alive(run_dir)
    if is_alive:
        state = "running"
    elif os.path.exists(os.path.join(run_dir, "result.json")) or st.get("finished"):
        state = "ended"
    elif st:
        state = "stopped"
    else:
        state = "not started"
    return {
        "dir": run_dir,
        "name": os.path.basename(run_dir.rstrip("/")),
        "state": state,
        "alive": is_alive,
        "model": ((meta.get("models") or {}).get("main") or {}).get("name"),
        "cycle": st.get("cycle"),
        "cost": cost.get("value"),
        "cost_limit": cost.get("limit"),
        "wall": wall.get("value"),
        "wall_limit": wall.get("limit"),
        "last_activity": last,
        "reason": res.get("reason"),
        "success": res.get("success") if res else st.get("success"),
    }


# ---- the repository: tasks, specs, rules, plans, stocks, benchmarks -------------------------


class Tree:
    def __init__(self, root: str, runs: str | None = None):
        self.root = os.path.abspath(root)
        self.runs = os.path.abspath(runs or os.path.join(self.root, "runs"))

    # -- tasks --
    def tasks(self) -> list[str]:
        return sorted(
            os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(self.root, "specs", "*", "spec.yaml"))
        )

    def spec(self, task: str) -> dict:
        import yaml

        task = safe(task)
        path = os.path.join(self.root, "specs", task, "spec.yaml")
        text = read_text(path) or ""
        try:
            parsed = yaml.safe_load(text) or {}
        except yaml.YAMLError:
            parsed = {}
        return {
            "path": path,
            "rel": os.path.relpath(path, self.root),
            "text": text,
            "fields": parsed if isinstance(parsed, dict) else {},
        }

    def rules(self, task: str) -> dict:
        """The canonical specification, and the search agent's brief made from it."""
        task = safe(task)
        canon = read_json(os.path.join(self.root, "specs", task, "canonical.json"), {}) or {}
        cs = canon.get("canonical_spec", canon) if isinstance(canon, dict) else {}
        brief = read_jsonl(os.path.join(self.root, "agents", f"{task}_search", "instances", "instances.jsonl"), 1)
        return {
            "outcome": canon.get("outcome"),
            "canonical": cs,
            "brief": brief[0] if brief else {},
            "search_agent": os.path.isdir(os.path.join(self.root, "agents", f"{task}_search")),
        }

    def plan(self, task: str) -> dict:
        """The construction plan: the worker's contract -- its stages (with the text each stage
        gives the worker), commands, versions, checks, oracle and probes.
        """
        task = safe(task)
        c = read_json(os.path.join(self.root, "agents", f"{task}_worker", "contract.json"), {}) or {}
        sections = c.get("_stage_sections") or []
        stages = []
        for i, st in enumerate(c.get("_stages") or []):
            if isinstance(st, dict):
                stages.append(
                    {
                        "n": i + 1,
                        "stage": st,
                        "text": sections[i] if i < len(sections) and isinstance(sections[i], dict) else {},
                    }
                )
        does = c.get("_does") or {}
        verbs = [
            {"name": v, "args": (spec or {}).get("args") or [], "does": does.get(v)}
            for v, spec in (c.get("verbs") or {}).items()
        ]
        drawn = c.get("_drawn") or {}
        return {
            "exists": bool(c),
            "stages": stages,
            "verbs": verbs,
            "variants": c.get("variants") or {},
            "checks": c.get("checks") or ([c["observable"]] if c.get("observable") else []),
            "oracle": c.get("oracle") or {},
            "negative_control": c.get("negative_control"),
            "solver_input": c.get("solver_input"),
            "measure": c.get("measure"),
            "predicates": c.get("predicates") or [],
            "identity": c.get("_identity_fields") or drawn.get("identity"),
            "instance_fields": c.get("_instance_fields") or [],
            "drawn": {k: v for k, v in drawn.items() if not isinstance(v, (dict, list))},
        }

    def lessons(self, agent: str) -> dict:
        agent = safe(agent)
        d = os.path.join(self.root, "agents", agent)
        return {
            "lessons": read_text(os.path.join(d, "lessons.md")),
            "pinned": read_text(os.path.join(d, "lessons_pinned.md")),
        }

    # -- frozen stocks --
    def stocks(self, task: str | None = None) -> list[dict]:
        out = []
        for d in sorted(glob.glob(os.path.join(self.root, "stock", "*"))):
            name = os.path.basename(d)
            t = next(
                (x for x in sorted(self.tasks(), key=len, reverse=True) if name == x or name.startswith(x + "_")), None
            )
            if task and t != task:
                continue
            m = read_json(os.path.join(d, "manifest.json"), {}) or {}
            out.append(
                {
                    "name": name,
                    "task": t,
                    "count": count_lines(os.path.join(d, "candidates.jsonl")),
                    "identity": m.get("identity"),
                    "frozen_at": m.get("frozen_at"),
                    "note": m.get("note"),
                }
            )
        return out

    def stock_file(self, name: str) -> str:
        return os.path.join(self.root, "stock", safe(name), "candidates.jsonl")

    # -- packaged benchmarks --
    def benchmark_sets(self) -> list[dict]:
        out = []
        for d in sorted(glob.glob(os.path.join(self.root, "benchmarks", "*", "*", "manifest.json"))):
            task_dir = os.path.dirname(d)
            m = read_json(d, {}) or {}
            out.append(
                {
                    "set": os.path.basename(os.path.dirname(task_dir)),
                    "task": os.path.basename(task_dir),
                    "count": m.get("count", len(m.get("instances") or [])),
                    "name": m.get("name"),
                }
            )
        return out

    def benchmark(self, set_: str, task: str) -> dict:
        d = os.path.join(self.root, "benchmarks", safe(set_), safe(task))
        m = read_json(os.path.join(d, "manifest.json"), {}) or {}
        acc = (read_json(os.path.join(self.root, "benchmarks", set_, "acceptance.json"), {}) or {}).get(task, {})
        sel = read_json(os.path.join(d, "selection_50.json"), {}) or {}
        counted = set(sel.get("ids") or [])
        rows = []
        for inst in m.get("instances") or []:
            iid = inst.get("id")
            a = acc.get(iid) or {}
            rows.append(
                {
                    "id": iid,
                    "fields": inst.get("fields") or {},
                    "rows": inst.get("rows") or {},
                    "harness_ok": inst.get("harness_verdict_ok"),
                    "accepted": a.get("accepted"),
                    "acceptance_run": a.get("run"),
                    "counted": iid in counted if counted else None,
                    "from_run": inst.get("from_run"),
                }
            )
        certs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*certify*.json")))
        return {
            "dir": d,
            "name": m.get("name"),
            "count": len(rows),
            "instances": rows,
            "excluded": m.get("excluded") or [],
            "selection_rule": sel.get("rule"),
            "certify_records": certs,
            "readme": read_text(os.path.join(d, "README.md")),
        }

    def instance(self, set_: str, task: str, iid: str) -> dict:
        d = os.path.join(self.root, "benchmarks", safe(set_), safe(task), "instances", safe(iid))
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        return {"dir": d, "files": files, "record": read_json(os.path.join(d, "instance.json"), {})}

    def instance_file(self, set_: str, task: str, iid: str, name: str) -> str | None:
        d = os.path.join(self.root, "benchmarks", safe(set_), safe(task), "instances", safe(iid))
        return read_text(os.path.join(d, safe(name)), 2_000_000)

    # -- search campaigns --
    def search_campaigns(self, task: str) -> list[dict]:
        task = safe(task)
        out = []
        for e in sorted(glob.glob(os.path.join(self.runs, "search", task, "epoch_*")), key=natural_key):
            if not os.path.isdir(e):
                continue
            atts = sorted(glob.glob(os.path.join(e, "attempt_*")), key=natural_key)
            if not atts:
                continue
            sts = [agent_status(a) for a in atts]
            camp = read_json(os.path.join(e, "campaign.json"), {}) or {}
            out.append(
                {
                    "task": task,
                    "epoch": os.path.basename(e),
                    "agents": len(atts),
                    "running": sum(s["alive"] for s in sts),
                    "cost": round(sum(s["cost"] or 0 for s in sts), 2),
                    "candidates": sum(count_lines(os.path.join(a, "stock", "candidates.jsonl")) for a in atts),
                    "model": next((s["model"] for s in sts if s["model"]), None),
                    "started": camp.get("started")
                    or min((mtime(os.path.join(a, "run_meta.json")) or 0) for a in atts)
                    or 0,
                    "marker": next(
                        (m[:-4] for m in ("ABORTED.txt", "STOPPED.txt") if os.path.exists(os.path.join(e, m))), None
                    ),
                    "campaign": bool(camp),
                }
            )
        return out

    def campaign(self, task: str, epoch: str) -> dict:
        e = os.path.join(self.runs, "search", safe(task), safe(epoch))
        camp = read_json(os.path.join(e, "campaign.json"), {}) or {}
        agents = []
        for a in sorted(glob.glob(os.path.join(e, "attempt_*")), key=natural_key):
            s = agent_status(a)
            s["candidates"] = count_lines(os.path.join(a, "stock", "candidates.jsonl"))
            s["shrinks"] = len(glob.glob(os.path.join(a, "stock", "shrink_*.candidates.jsonl")))
            launch = read_json(os.path.join(a, "launch.json"), {}) or {}
            s["target"] = launch.get("target")
            inst = read_jsonl(os.path.join(a, "instances.jsonl"), 1)
            s["territory"] = (inst[0].get("territory") if inst else None) or ""
            agents.append(s)
        markers = {
            m[:-4]: read_text(os.path.join(e, m), 4000)
            for m in ("ABORTED.txt", "STOPPED.txt")
            if os.path.exists(os.path.join(e, m))
        }
        return {
            "task": task,
            "epoch": epoch,
            "dir": e,
            "settings": {k: v for k, v in camp.items() if k != "events"},
            "events": camp.get("events") or [],
            "agents": agents,
            "markers": markers,
            "directions": read_text(os.path.join(e, "directions.md")),
            "log_tail": tail_text(os.path.join(e, "campaign.log"), 6000),
        }

    def attempt_dir(self, task: str, epoch: str, attempt: str) -> str:
        return os.path.join(self.runs, "search", safe(task), safe(epoch), safe(attempt))

    # -- construction runs --
    def construction_runs(self, task: str) -> list[dict]:
        """Run directories holding worker_*/ : runs/construct/<task>/<epoch> and
        runs/experiment/<name>/<task>."""
        task = safe(task)
        dirs = [d for d in glob.glob(os.path.join(self.runs, "construct", task, "*")) if os.path.isdir(d)]
        dirs += [d for d in glob.glob(os.path.join(self.runs, "experiment", "*", task)) if os.path.isdir(d)]
        out = []
        for d in sorted(dirs, key=natural_key):
            ws = sorted(glob.glob(os.path.join(d, "worker_*")), key=natural_key)
            ws = [w for w in ws if os.path.isdir(w)]
            if not ws:
                continue
            sts = [agent_status(w) for w in ws]
            deliv = drop = 0
            keys = []
            for w in ws:
                wk = worker_summary(w)
                deliv += wk["summary"]["n_delivered"]
                drop += wk["summary"]["n_dropped"]
                keys += [x["key"] for x in wk["delivered"]]
            out.append(
                {
                    "key": self.run_key(d),
                    "label": os.path.relpath(d, self.runs),
                    "workers": len(ws),
                    "running": sum(s["alive"] for s in sts),
                    "cost": round(sum(s["cost"] or 0 for s in sts), 2),
                    "delivered": deliv,
                    "dropped": drop,
                    "delivered_keys": keys,
                    "model": next((s["model"] for s in sts if s["model"]), None),
                    "started": min((mtime(os.path.join(w, "run_meta.json")) or 0) for w in ws) or 0,
                }
            )
        return out

    @staticmethod
    def run_key(d: str) -> str:
        """`construct~<task>~<epoch>` or `experiment~<name>~<task>`: one URL component."""
        parts = os.path.normpath(d).split(os.sep)
        return "~".join(parts[-3:])

    def run_dir(self, key: str) -> str:
        parts = [safe(p) for p in key.split("~")]
        if len(parts) != 3 or parts[0] not in ("construct", "experiment"):
            raise ValueError(f"not a construction run: {key!r}")
        return os.path.join(self.runs, *parts)

    def construction_run(self, key: str) -> dict:
        d = self.run_dir(key)
        workers = []
        delivered, dropped = [], []
        for w in sorted(glob.glob(os.path.join(d, "worker_*")), key=natural_key):
            if not os.path.isdir(w):
                continue
            s = agent_status(w)
            wk = worker_summary(w)
            s.update(wk["summary"])
            s["arm"] = (read_json(os.path.join(w, "arm.json"), {}) or {}).get("arm")
            workers.append(s)
            delivered += [{"worker": s["name"], **x} for x in wk["delivered"]]
            dropped += [{"worker": s["name"], **x} for x in wk["dropped"]]
        lessons = read_text(os.path.join(d, "lessons.md"))
        return {
            "key": key,
            "label": os.path.relpath(d, self.runs),
            "dir": d,
            "workers": workers,
            "delivered": delivered,
            "dropped": dropped,
            "lessons": lessons,
            "arms": read_json(os.path.join(d, "arms.json")),
            "handover": {
                k: v
                for k, v in (read_json(os.path.join(d, "handover.json"), {}) or {}).items()
                if not isinstance(v, (list, dict))
            },
        }

    def worker_dir(self, key: str, worker: str) -> str:
        return os.path.join(self.run_dir(key), safe(worker))

    def funnel(self, task: str, run_keys: list[str]) -> dict:
        """stage_funnel over the workers of these construction runs, with the plan's stage names."""
        plan = self.plan(task)
        ws = []
        for key in run_keys:
            ws += [
                w
                for w in sorted(glob.glob(os.path.join(self.run_dir(key), "worker_*")), key=natural_key)
                if os.path.isdir(w)
            ]
        f = stage_funnel(ws, len(plan["stages"]))
        for s in f["stages"]:
            p = plan["stages"][s["n"] - 1]["stage"] if s["n"] <= len(plan["stages"]) else {}
            s["name"], s["files"], s["rows"] = p.get("name"), p.get("files") or [], p.get("rows") or []
        f["workers"] = len(ws)
        return f

    # -- everything running now --
    def running(self) -> list[dict]:
        pats = [
            ("search", os.path.join(self.runs, "search", "*", "*", "attempt_*")),
            ("construct", os.path.join(self.runs, "construct", "*", "*", "worker_*")),
            ("construct", os.path.join(self.runs, "experiment", "*", "*", "worker_*")),
        ]
        out = []
        for kind, pat in pats:
            for d in glob.glob(os.path.join(pat, "launcher.pid")):
                run = os.path.dirname(d)
                if not alive(run):
                    continue
                s = agent_status(run)
                rel = os.path.relpath(run, self.runs).split(os.sep)
                s["kind"] = kind
                if kind == "search":
                    s["task"], s["epoch"], s["attempt"] = rel[1], rel[2], rel[3]
                    s["candidates"] = count_lines(os.path.join(run, "stock", "candidates.jsonl"))
                else:
                    s["key"] = self.run_key(os.path.dirname(run))
                    s["task"] = rel[1] if rel[0] == "construct" else rel[2]
                    s.update(worker_summary(run)["summary"])
                out.append(s)
        return sorted(out, key=lambda s: (s["kind"], s.get("task") or "", natural_key(s["dir"])))


def running_groups(live: list[dict]) -> list[dict]:
    """The running agents grouped by run: one entry per search campaign or construction run,
    with its live agents, their spend, and (construction) deliveries and drops so far."""
    groups: dict[str, dict] = {}
    for s in live:
        if s["kind"] == "search":
            k = f"search/{s['task']}/{s['epoch']}"
            g = groups.setdefault(
                k,
                {
                    "kind": "search",
                    "task": s["task"],
                    "label": s["epoch"],
                    "url": f"/search/{s['task']}/{s['epoch']}",
                    "agents": [],
                    "cost": 0.0,
                    "candidates": 0,
                    "model": s.get("model"),
                },
            )
            g["candidates"] += s.get("candidates") or 0
        else:
            k = s["key"]
            g = groups.setdefault(
                k,
                {
                    "kind": "construction",
                    "task": s["task"],
                    "label": "/".join(k.split("~")[1:]),
                    "url": f"/construction/{k}",
                    "agents": [],
                    "cost": 0.0,
                    "delivered": 0,
                    "dropped": 0,
                    "model": s.get("model"),
                },
            )
            g["delivered"] += s.get("n_delivered") or 0
            g["dropped"] += s.get("n_dropped") or 0
        g["agents"].append(s)
        g["cost"] += s.get("cost") or 0
    out = list(groups.values())
    for g in out:
        g["n"] = len(g["agents"])
        g["cost"] = round(g["cost"], 2)
    return out


# ---- construction workers --------------------------------------------------------------------


def worker_summary(w: str) -> dict:
    """A worker's candidates from its worklist: each one's state; delivered = in the plan's final
    state; dropped = rejected, with the worker's reason; current = the latest one still open."""
    st = read_json(os.path.join(w, "agent_state.json"), {}) or {}
    ws = st.get("working_set") or {}
    wl = ws.get("worklist") or {}
    cands = wl.get("candidates") or {}
    order = wl.get("order") or list(cands)
    terminal = set(wl.get("terminal") or [])
    reasons = {x.get("key"): x.get("reason") for x in (ws.get("rejected_candidates") or []) if isinstance(x, dict)}
    by_state: dict[str, int] = {}
    delivered, dropped, open_ = [], [], []
    for k in order:
        s = str((cands.get(k) or {}).get("state") or "?")
        by_state[s] = by_state.get(s, 0) + 1
        if s in terminal:
            delivered.append({"key": k, "state": s})
        elif s == "REJECTED":
            dropped.append({"key": k, "state": s, "reason": reasons.get(k)})
        else:
            open_.append({"key": k, "state": s})
    current = open_[-1] if open_ else None
    pool = read_jsonl(os.path.join(os.path.dirname(w), f"{os.path.basename(w)}.instances.jsonl"), 1)
    pool_n = len((pool[0].get("candidates") or [])) if pool else None
    return {
        "summary": {
            "current": current,
            "states": by_state,
            "n_delivered": len(delivered),
            "n_dropped": len(dropped),
            "pool": pool_n,
            "registered": len(order),
            "terminal": sorted(terminal),
        },
        "delivered": delivered,
        "dropped": dropped,
        "worklist": cands,
        "order": order,
    }


STAGE_STATE = re.compile(r"^STAGE_(\d+)_")


def stage_funnel(worker_dirs: list[str], n_stages: int) -> dict:
    """Where the candidates of these workers went, stage by stage. A worklist state STAGE_<k>_...
    means stage k is passed; a candidate's furthest k comes from its state and its history.
    Per stage: entered, passed, dropped there, left open there (the run ended or moved on) and
    being worked on now (open in a live worker). A candidate handed to two runs counts twice."""
    n = max(n_stages, 1)
    stages = [{"n": i + 1, "entered": 0, "passed": 0, "dropped": 0, "open": 0, "now": 0} for i in range(n)]
    tot = {"taken": 0, "delivered": 0, "dropped": 0, "open": 0, "now": 0}
    for w in worker_dirs:
        st = read_json(os.path.join(w, "agent_state.json"), {}) or {}
        wl = (st.get("working_set") or {}).get("worklist") or {}
        cands = wl.get("candidates") or {}
        terminal = set(wl.get("terminal") or [])
        live = alive(w)
        for k in wl.get("order") or list(cands):
            c = cands.get(k) or {}
            state = str(c.get("state") or "")
            passed = 0
            for s in [*(c.get("history") or []), state]:
                m = STAGE_STATE.match(str(s))
                if m:
                    passed = max(passed, int(m.group(1)))
            if state in terminal:
                passed = n
            passed = min(passed, n)
            tot["taken"] += 1
            for i in range(min(passed + 1, n)):
                stages[i]["entered"] += 1
            for i in range(passed):
                stages[i]["passed"] += 1
            at = stages[min(passed, n - 1)]
            if state in terminal:
                tot["delivered"] += 1
            elif state == "REJECTED":
                at["dropped"] += 1
                tot["dropped"] += 1
            else:
                at["open"] += 1
                tot["open"] += 1
                if live:
                    at["now"] += 1
                    tot["now"] += 1
    return {"stages": stages, **tot}


def recipe_for(key: str, names: list[str]) -> str | None:
    """The recipes/<slug> directory of a worklist key. The slug is made from the candidate's
    own fields (a clone URL keeps its .git, the key does not), so match on the key's longest
    part -- the commit hash."""
    parts = sorted((p for p in key.split("|") if p), key=len, reverse=True)
    for p in parts:
        tok = re.sub(r"[^A-Za-z0-9._-]+", "-", p).strip("-")
        hit = [n for n in names if tok and tok in n]
        if len(hit) == 1:
            return hit[0]
    return None


def worker_detail(w: str) -> dict:
    wk = worker_summary(w)
    recipes = {}
    for r in sorted(glob.glob(os.path.join(w, "recipes", "*"))):
        name = os.path.basename(r)
        checks = []
        for f in sorted(os.listdir(r), key=natural_key):
            if f.endswith(".json") and (f.startswith(("check_", "stage")) or f == "report.json"):
                rep = read_json(os.path.join(r, f), {}) or {}
                rows = rep.get("rows") or []
                checks.append(
                    {
                        "file": f,
                        "ok": rep.get("ok"),
                        "ts": rep.get("ts") or mtime(os.path.join(r, f)),
                        "elapsed": rep.get("elapsed_s"),
                        "rows": [(x.get("name"), x.get("ok")) for x in rows],
                    }
                )
        checks.sort(key=lambda c: c["ts"] or 0)
        recipes[name] = {"checks": checks, "delivered": os.path.isdir(os.path.join(r, "instance"))}
    items = []
    for k in wk["order"]:
        c = wk["worklist"].get(k) or {}
        hist = c.get("history") or []
        items.append(
            {
                "key": k,
                "state": c.get("state"),
                "fields": c.get("fields") or {},
                "history": hist[-12:],
                "recipe": recipe_for(k, list(recipes)),
                "reason": next((d["reason"] for d in wk["dropped"] if d["key"] == k), None),
            }
        )
    return {"items": items, "recipes": recipes, **wk}


def recipe_detail(w: str, slug: str) -> dict:
    r = os.path.join(w, "recipes", safe(slug))
    files_dir = (
        os.path.join(r, "instance") if os.path.isdir(os.path.join(r, "instance")) else os.path.join(r, "bench_last")
    )
    files = sorted(os.listdir(files_dir)) if os.path.isdir(files_dir) else []
    checks = []
    for f in sorted(os.listdir(r) if os.path.isdir(r) else [], key=natural_key):
        if f.endswith(".json"):
            rep = read_json(os.path.join(r, f), {}) or {}
            if "rows" in rep:
                checks.append(
                    {
                        "file": f,
                        "ok": rep.get("ok"),
                        "ts": rep.get("ts") or mtime(os.path.join(r, f)),
                        "elapsed": rep.get("elapsed_s"),
                        "rows": [dict(x) for x in rep.get("rows") or [] if isinstance(x, dict)],
                    }
                )
    checks.sort(key=lambda c: c["ts"] or 0)
    # what changed since the row was last seen: "newly failing" / "now passes"
    seen: dict[str, bool] = {}
    for c in checks:
        c["changed"] = 0
        for row in c["rows"]:
            before = seen.get(row.get("name"))
            row["change"] = (
                None
                if before is None or before == bool(row.get("ok"))
                else "now passes"
                if row.get("ok")
                else "newly failing"
            )
            c["changed"] += row["change"] is not None
            seen[row.get("name")] = bool(row.get("ok"))
    return {
        "dir": r,
        "files_dir": files_dir,
        "files": files,
        "checks": checks,
        "delivered": os.path.isdir(os.path.join(r, "instance")),
    }


def recipe_file(w: str, slug: str, name: str) -> str | None:
    r = os.path.join(w, "recipes", safe(slug))
    d = os.path.join(r, "instance") if os.path.isdir(os.path.join(r, "instance")) else os.path.join(r, "bench_last")
    return read_text(os.path.join(d, safe(name)), 2_000_000)


# ---- an agent's cycles (search attempts and workers alike) ---------------------------------


def _cycle_of(line: bytes) -> int | None:
    m = re.search(rb'"cycle":\s*(\d+)', line)
    return int(m.group(1)) if m else None


def reply_object(content: str) -> dict | None:
    """The JSON object in a model reply: bare, or inside a ```json fence or after some prose."""
    try:
        d = json.loads(content)
        return d if isinstance(d, dict) else None
    except ValueError:
        pass
    start = content.find("{")
    if start == -1:
        return None
    try:
        d, _ = json.JSONDecoder().raw_decode(content, start)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _summarize(events: list[dict]) -> list[dict]:
    """Per cycle: what the model said and did, what came back, what it cost."""
    cycles: dict[int, dict] = {}
    for ev in events:
        c = ev.get("cycle")
        if c is None:
            continue
        row = cycles.setdefault(
            c,
            {
                "cycle": c,
                "ts": ev.get("ts"),
                "thoughts": None,
                "tool": None,
                "args": None,
                "result": None,
                "cost": 0.0,
                "stage": ev.get("stage"),
                "candidate": ev.get("candidate"),
                "events": [],
            },
        )
        e = ev.get("event")
        row["events"].append(e)
        if e == "on_response":
            row["cost"] += float(ev.get("cost") or 0)
            content = ev.get("content") or ""
            row["raw"] = content[:20_000]
            d = reply_object(content)
            if d is not None:
                row["thoughts"] = d.get("thoughts")
                cmd = d.get("command")
                if isinstance(cmd, dict):
                    row["tool"], row["args"] = cmd.get("name"), cmd.get("args")
            else:
                row["thoughts"] = row["thoughts"] or content[:2000]
        elif e == "on_tool_result":
            row["tool"] = row["tool"] or ev.get("name")
            row["args"] = row["args"] if row["args"] is not None else ev.get("args")
            r = ev.get("result")
            text = (r.get("display_text") or r.get("output") or json.dumps(r)[:4000]) if isinstance(r, dict) else str(r)
            row["result"] = text
            row["ok"] = r.get("ok") if isinstance(r, dict) else None
        elif e in ("on_model_retry", "on_finish", "on_run_end", "on_run_start"):
            row.setdefault("notes", []).append(e)
    return [cycles[k] for k in sorted(cycles)]


def latest_cycles(run_dir: str, n: int = 25, max_bytes: int = 64_000_000) -> list[dict]:
    """The last `n` cycles of a run, read from the end of its trajectory."""
    path = os.path.join(run_dir, "trajectory.jsonl")
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    chunk = 2_000_000
    data = b""
    with open(path, "rb") as fh:
        pos = size
        while pos > 0 and len(data) < max_bytes:
            step = min(chunk, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
            lines = data.split(b"\n")
            seen = {c for c in (_cycle_of(l[:400]) for l in lines[1:]) if c is not None}
            if len(seen) > n or pos == 0:
                break
            chunk *= 2
    lines = data.split(b"\n")
    if pos > 0:
        lines = lines[1:]
    events = []
    for l in lines:
        if l.strip():
            try:
                events.append(json.loads(l))
            except ValueError:
                pass
    rows = _summarize(events)
    return rows[-n:][::-1]


def cycle_events(run_dir: str, cycle: int) -> list[dict]:
    """Every trajectory event of one cycle, found by bisecting the file on the cycle number."""
    path = os.path.join(run_dir, "trajectory.jsonl")
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    with open(path, "rb") as fh:

        def first_cycle_at(off: int):
            fh.seek(off)
            if off:
                fh.readline()
            while True:
                l = fh.readline()
                if not l:
                    return None, None
                c = _cycle_of(l[:400])
                if c is not None:
                    return c, fh.tell() - len(l)

        lo, hi = 0, size
        while hi - lo > 1_000_000:
            mid = (lo + hi) // 2
            c, _ = first_cycle_at(mid)
            if c is None or c >= cycle:
                hi = mid
            else:
                lo = mid
        fh.seek(lo)
        if lo:
            fh.readline()
        out = []
        for l in fh:
            c = _cycle_of(l[:400])
            if c is None:
                continue
            if c > cycle:
                break
            if c == cycle:
                try:
                    out.append(json.loads(l))
                except ValueError:
                    pass
        return out


def cycle_detail(run_dir: str, cycle: int) -> dict:
    """The prompt the model saw at this cycle (cycles/cycle_N.json) and what happened."""
    msgs = read_json(os.path.join(run_dir, "cycles", f"cycle_{int(cycle):03d}.json"))
    if msgs is None:
        msgs = read_json(os.path.join(run_dir, "cycles", f"cycle_{int(cycle)}.json"))
    events = cycle_events(run_dir, int(cycle))
    rows = _summarize(events)
    return {
        "cycle": int(cycle),
        "messages": msgs if isinstance(msgs, list) else [],
        "summary": rows[0] if rows else None,
        "events": events,
    }


def agent_detail(run_dir: str) -> dict:
    s = agent_status(run_dir)
    inst = read_jsonl(os.path.join(run_dir, "instances.jsonl"), 1)
    stock = os.path.join(run_dir, "stock", "candidates.jsonl")
    return {
        "status": s,
        "instance": inst[0] if inst else {},
        "overrides": read_text(os.path.join(run_dir, "overrides.yaml"), 20_000),
        "stock_count": count_lines(stock) if os.path.exists(stock) else None,
        "check_script": read_text(os.path.join(run_dir, "stock", "check.sh"), 200_000),
        "log_tail": tail_text(os.path.join(run_dir, "construct.log"), 12_000),
        "n_cycle_files": len(glob.glob(os.path.join(run_dir, "cycles", "cycle_*.json"))),
    }


def ago(ts: float | None) -> str:
    if not ts:
        return "-"
    d = max(0, time.time() - ts)
    if d < 90:
        return f"{int(d)} s ago"
    if d < 5400:
        return f"{int(d / 60)} min ago"
    if d < 172800:
        return f"{d / 3600:.1f} h ago"
    return f"{d / 86400:.1f} days ago"


def run_graph(run_dir: str) -> dict:
    """A construction run as a job graph: each worker with its status and, in the order it took
    them, its candidates: delivered, dropped (with the reason), being built now, or left unfinished."""
    workers = []
    for w in sorted(glob.glob(os.path.join(run_dir, "worker_*")), key=natural_key):
        if not os.path.isdir(w):
            continue
        s = agent_status(w)
        wk = worker_summary(w)
        names = sorted(os.listdir(os.path.join(w, "recipes"))) if os.path.isdir(os.path.join(w, "recipes")) else []
        terminal = set(wk["summary"]["terminal"])
        reasons = {d["key"]: d["reason"] for d in wk["dropped"]}
        current = (wk["summary"]["current"] or {}).get("key")
        items = []
        for k in wk["order"]:
            state = str((wk["worklist"].get(k) or {}).get("state") or "")
            kind = (
                "delivered"
                if state in terminal
                else "dropped"
                if state == "REJECTED"
                else "now"
                if s["alive"] and k == current
                else "open"
            )
            items.append(
                {"key": k, "state": state, "kind": kind, "recipe": recipe_for(k, names), "reason": reasons.get(k)}
            )
        s.update(wk["summary"])
        s["items"] = items
        workers.append(s)
    ho = read_json(os.path.join(run_dir, "handover.json"), {}) or {}
    return {"workers": workers, "handover": {k: v for k, v in ho.items() if not isinstance(v, (list, dict))}}


def campaign_graph(c: dict) -> list[dict]:
    """A search campaign's agents with how far each stock file got toward its target."""
    return [{**a, "pct": round(100 * a["candidates"] / a["target"]) if a.get("target") else None} for a in c["agents"]]
