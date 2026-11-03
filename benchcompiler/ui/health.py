"""Warnings about live runs, for the strip under the header and the Health page.

Each warning says what happened, why it matters and what to do. None of them acts: stopping or
changing a run stays the user's decision (the Stop button, or a shell).

  stalled        a live agent wrote no trajectory event for STALL_MIN minutes
  log errors     a live agent's log shows a provider or crash signature in its last lines
  failing tools  most of a live agent's recent tool calls failed; across half a run's agents
                 it is reported once for the run (the same refusal everywhere is the run's
                 problem, not one agent's)
  pace           a construction run has spent half its budget and is far from its target
  disk           the runs folder's disk is nearly full (construction keeps an image per candidate)
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections import Counter

from benchcompiler.ui import data as D

STALL_MIN = 10
ERR = re.compile(
    r"(RateLimitError|budget_exceeded|Budget has been exceeded|\b40[23]\b|\b5\d\d\b Server|Traceback|"
    r"AuthenticationError|out of credits|insufficient credits|Killed|No space left on device)",
    re.I,
)
FAIL_SHARE = 0.75
FAIL_MIN_CYCLES = 8


def _money(v) -> str:
    return f"${v or 0:,.2f}"


def _url(s: dict) -> str:
    if s["kind"] == "search":
        return f"/search/{s['task']}/{s['epoch']}/{s['attempt']}"
    return f"/construction/{s['key']}/{s['name']}"


def _label(s: dict) -> str:
    if s["kind"] == "search":
        return f"{s['task']} search {s['epoch']} {s['attempt']}"
    return f"{'/'.join(s['key'].split('~')[1:])} {s['name']}"


def _log_errors(run_dir: str, now: float) -> list[str]:
    log = os.path.join(run_dir, "construct.log")
    m = D.mtime(log)
    if not m or now - m > 15 * 60:
        return []
    found = []
    for line in D.tail_text(log, 16_000).splitlines()[-80:]:
        hit = ERR.search(line)
        if hit:
            found.append(hit.group(1))
    return found


def _failing(run_dir: str) -> tuple[float, int, str]:
    cycles = [c for c in D.latest_cycles(run_dir, 12, max_bytes=3_000_000) if c.get("result") is not None]
    if not cycles:
        return 0.0, 0, ""
    bad = [c for c in cycles if c.get("ok") is False]
    first = Counter((c["result"] or "").strip().splitlines()[0][:160] for c in bad if c.get("result"))
    return len(bad) / len(cycles), len(cycles), (first.most_common(1)[0][0] if first else "")


def warnings(tree, now: float | None = None) -> list[dict]:
    now = now or time.time()
    out: list[dict] = []
    live = tree.running()
    failing_by_run: dict[str, list] = {}
    errors: dict[str, list] = {}
    for s in live:
        idle = now - (s.get("last_activity") or now)
        if idle > STALL_MIN * 60:
            out.append(
                {
                    "level": "warn",
                    "kind": "stalled",
                    "where": _label(s),
                    "url": _url(s) + "?tab=activity",
                    "what": f"no new step for {int(idle // 60)} min",
                    "why": "a live agent that writes nothing is waiting on a model call, a hung command or a dead container",
                    "todo": "open its activity: a long build is fine; a model call that never returns or a container "
                    "that is gone is not (stop it, then resume or relaunch)",
                }
            )
        for sig in set(_log_errors(s["dir"], now)):
            errors.setdefault(sig, []).append(s)
        share, n, sample = _failing(s["dir"])
        if n >= FAIL_MIN_CYCLES and share >= FAIL_SHARE:
            run = s.get("key") or f"search~{s['task']}~{s['epoch']}"
            failing_by_run.setdefault(run, []).append((s, share, sample))
        if s.get("cost_limit") and (s.get("cost") or 0) >= 0.9 * s["cost_limit"]:
            out.append(
                {
                    "level": "info",
                    "kind": "budget",
                    "where": _label(s),
                    "url": _url(s),
                    "what": f"spent {_money(s['cost'])} of {_money(s['cost_limit'])}",
                    "why": "it stops at its cost cap",
                    "todo": "nothing, unless it should run longer",
                }
            )
    for sig, agents in errors.items():
        out.append(
            {
                "level": "bad" if re.search(r"40[23]|credit|Authentication", sig, re.I) else "warn",
                "kind": "log errors",
                "where": ", ".join(_label(s) for s in agents[:3]) + (" …" if len(agents) > 3 else ""),
                "url": _url(agents[0]),
                "what": f"'{sig}' in the last lines of {len(agents)} live agent log(s)",
                "why": "provider refusals (402/403, credits, auth) stop every agent on that route; a Traceback can be "
                "one agent's crash",
                "todo": "read the log (Records tab); for a provider refusal check the key's credit or limit before "
                "anything else",
            }
        )
    for run, rows in failing_by_run.items():
        agents_alive = [s for s in live if (s.get("key") or f"search~{s['task']}~{s['epoch']}") == run]
        if len(rows) >= max(2, len(agents_alive) / 2):
            out.append(
                {
                    "level": "bad",
                    "kind": "failing tools",
                    "where": run.replace("~", "/"),
                    "url": _url(rows[0][0]),
                    "what": f"{len(rows)} of {len(agents_alive)} live agents: most recent tool calls fail; "
                    f"most common: {rows[0][2] or '-'}",
                    "why": "the same failure across a run's agents is usually the harness or the environment, "
                    "not the candidates, and it spends every agent's budget",
                    "todo": "open one agent's activity and read the failing result; check the infrastructure before "
                    "the prompts",
                }
            )
        else:
            for s, share, sample in rows:
                out.append(
                    {
                        "level": "warn",
                        "kind": "failing tools",
                        "where": _label(s),
                        "url": _url(s) + "?tab=activity",
                        "what": f"{int(share * 100)}% of its recent tool calls failed; most common: {sample or '-'}",
                        "why": "an agent that keeps failing the same call spends its budget without progress",
                        "todo": "read its activity; it may be on a candidate that cannot work",
                    }
                )
    out += _pace(tree, live)
    out += _disk(tree.runs)
    order = {"bad": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda w: order[w["level"]])


def _pace(tree, live) -> list[dict]:
    out = []
    for key in sorted({s["key"] for s in live if s["kind"] == "construct"}):
        r = tree.construction_run(key)
        target = (r["handover"] or {}).get("total_target") or (r["handover"] or {}).get("target")
        limit = sum(w.get("cost_limit") or 0 for w in r["workers"])
        spent = sum(w.get("cost") or 0 for w in r["workers"])
        if not target or not limit or spent < 0.5 * limit:
            continue
        got = len(r["delivered"])
        projected = int(got * limit / spent) if spent else 0
        if projected < 0.5 * target:
            out.append(
                {
                    "level": "warn",
                    "kind": "pace",
                    "where": r["label"],
                    "url": f"/construction/{key}",
                    "what": f"{_money(spent)} of {_money(limit)} spent, {got} of {target} delivered; at this pace "
                    f"about {projected} by the end",
                    "why": "the run will end at its budget well short of its target",
                    "todo": "look at the drop reasons (Dropped tab) before deciding to add budget or relaunch",
                }
            )
    return out


def _disk(path: str) -> list[dict]:
    try:
        u = shutil.disk_usage(path if os.path.exists(path) else os.path.dirname(path))
    except OSError:
        return []
    free = u.free / u.total
    if free >= 0.15:
        return []
    return [
        {
            "level": "bad" if free < 0.05 else "warn",
            "kind": "disk",
            "where": path,
            "url": "/health",
            "what": f"{u.free / 1e9:.0f} GB free ({free:.0%})",
            "why": "construction builds and keeps one image per candidate (hundreds of GB per campaign); a full disk "
            "fails every build at once",
            "todo": "remove regenerable images (README: Disk and Docker images); never one a running container uses",
        }
    ]


class Cached:
    """warnings(), recomputed at most every `ttl` seconds."""

    def __init__(self, tree, ttl: float = 30.0):
        self.tree, self.ttl, self.at, self.value = tree, ttl, 0.0, []

    def get(self) -> list[dict]:
        if time.time() - self.at > self.ttl:
            self.value, self.at = warnings(self.tree), time.time()
        return self.value
