"""Answers to "why": why candidates were dropped (grouped by the stage they reached and the check
that failed last), and the whole history of one candidate, from the search agent that found it
to the benchmark that packaged it.
"""

from __future__ import annotations

import glob
import json
import os
import re

from benchcompiler.ui import data as D

HEX = re.compile(r"\b[0-9a-f]{7,40}\b")


def furthest_stage(cand: dict, terminal: set, n_stages: int) -> int:
    """The stage a candidate was on: one past the furthest STAGE_<k> in its state and history."""
    passed = 0
    for s in [*(cand.get("history") or []), cand.get("state") or ""]:
        m = D.STAGE_STATE.match(str(s))
        if m:
            passed = max(passed, int(m.group(1)))
    if str(cand.get("state")) in terminal:
        return n_stages
    return min(passed + 1, max(n_stages, 1))


def drops(tree, task: str, run_keys: list[str]) -> list[dict]:
    """Every dropped candidate of these runs: the stage it was on, the rows its last replay check
    failed, and the worker's reason."""
    n = len(tree.plan(task)["stages"])
    out = []
    for key in run_keys:
        run_dir = tree.run_dir(key)
        for w in sorted(glob.glob(os.path.join(run_dir, "worker_*")), key=D.natural_key):
            if not os.path.isdir(w):
                continue
            det = D.worker_detail(w)
            terminal = set(det["summary"]["terminal"])
            model = D.agent_status(w)["model"]
            for it in det["items"]:
                if it["state"] != "REJECTED":
                    continue
                rc = det["recipes"].get(it["recipe"]) if it["recipe"] else None
                last = rc["checks"][-1] if rc and rc["checks"] else None
                failing = [name for name, ok in (last["rows"] if last else []) if ok is False]
                out.append(
                    {
                        "run_key": key,
                        "run": os.path.relpath(run_dir, tree.runs),
                        "worker": os.path.basename(w),
                        "model": model,
                        "key": it["key"],
                        "recipe": it["recipe"],
                        "stage": furthest_stage(det["worklist"].get(it["key"]) or {}, terminal, n),
                        "failing": failing,
                        "no_check": last is None,
                        "reason": it["reason"] or "",
                    }
                )
    return out


def drop_groups(rows: list[dict]) -> list[dict]:
    """Drops grouped by (stage, first failing check row), largest group first."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        row = r["failing"][0] if r["failing"] else ("no replay check" if r["no_check"] else "checks passed")
        g = groups.setdefault((r["stage"], row), {"stage": r["stage"], "row": row, "items": []})
        g["items"].append(r)
    return sorted(groups.values(), key=lambda g: (-len(g["items"]), g["stage"]))


# ---- one candidate's history ------------------------------------------------------------------


def parse_query(q: str) -> tuple[str | None, str | None]:
    """'owner/repo @ abc1234', a GitHub URL with a commit, or a bare hash -> (repo, hash)."""
    q = (q or "").strip().lower().replace("https://github.com/", "").replace("http://github.com/", "")
    shas = HEX.findall(q)
    sha = max(shas, key=len) if shas else None
    m = re.search(r"([a-z0-9_.-]+/[a-z0-9_.-]+?)(?:\.git)?(?:[\s@|/]|$)", q)
    repo = m.group(1) if m and (not sha or m.group(1) != sha) else None
    return repo, sha


def _matches(text: str, repo: str | None, sha: str | None) -> bool:
    t = text.lower()
    return (not sha or sha in t) and (not repo or repo in t)


def journey(tree, q: str, limit: int = 40) -> dict:
    repo, sha = parse_query(q)
    if not sha and not repo:
        return {"q": q, "repo": None, "sha": None, "found": [], "frozen": [], "built": [], "packaged": []}
    found, frozen, built, packaged = [], [], [], []
    for p in sorted(glob.glob(os.path.join(tree.runs, "search", "*", "*", "attempt_*", "stock", "candidates.jsonl"))):
        for line in D.read_jsonl(p):
            if _matches(json.dumps(line), repo, sha):
                parts = os.path.relpath(p, tree.runs).split(os.sep)
                found.append({"task": parts[1], "epoch": parts[2], "attempt": parts[3], "line": line, "ts": D.mtime(p)})
                break
    for s in tree.stocks():
        for line in D.read_jsonl(os.path.join(tree.root, "stock", s["name"], "candidates.jsonl")):
            if _matches(json.dumps(line), repo, sha):
                frozen.append({"stock": s["name"], "task": s["task"], "frozen_at": s["frozen_at"], "line": line})
                break
    for task in tree.tasks():
        n = len(tree.plan(task)["stages"])
        for r in tree.construction_runs(task):
            run_dir = tree.run_dir(r["key"])
            for w in sorted(glob.glob(os.path.join(run_dir, "worker_*")), key=D.natural_key):
                st = D.read_json(os.path.join(w, "agent_state.json"), {}) or {}
                wl = (st.get("working_set") or {}).get("worklist") or {}
                hits = [k for k in (wl.get("candidates") or {}) if _matches(k, repo, sha)]
                if not hits:
                    continue
                det = D.worker_detail(w)
                terminal = set(det["summary"]["terminal"])
                for it in det["items"]:
                    if it["key"] not in hits:
                        continue
                    rc = det["recipes"].get(it["recipe"]) if it["recipe"] else None
                    built.append(
                        {
                            "task": task,
                            "run_key": r["key"],
                            "run": r["label"],
                            "worker": os.path.basename(w),
                            "model": D.agent_status(w)["model"],
                            "key": it["key"],
                            "state": it["state"],
                            "outcome": "delivered"
                            if it["state"] in terminal
                            else "dropped"
                            if it["state"] == "REJECTED"
                            else "open",
                            "stage": furthest_stage(det["worklist"].get(it["key"]) or {}, terminal, n),
                            "history": it["history"],
                            "reason": it["reason"],
                            "recipe": it["recipe"],
                            "checks": rc["checks"] if rc else [],
                        }
                    )
                if len(built) >= limit:
                    break
    for b in tree.benchmark_sets():
        bench = tree.benchmark(b["set"], b["task"])
        for inst in bench["instances"]:
            if _matches(inst["id"] + json.dumps(inst["fields"]), repo, sha):
                packaged.append(
                    {
                        "set": b["set"],
                        "task": b["task"],
                        "id": inst["id"],
                        "accepted": inst["accepted"],
                        "counted": inst["counted"],
                        "from_run": inst["from_run"],
                    }
                )
    return {
        "q": q,
        "repo": repo,
        "sha": sha,
        "found": found[:limit],
        "frozen": frozen,
        "built": built,
        "packaged": packaged,
    }
