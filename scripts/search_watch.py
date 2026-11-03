"""Watch search epochs and their lessons loops; print one line per event worth acting on.

    set -a; . ./.env; set +a
    WATCH_ROUTE=openrouter python3 -u scripts/search_watch.py [--once] [--every 120] <epoch_dir> ...

Events: START (initial state), STALL (no new trajectory event for > STALL_MIN), ERROR (a new error
signature in an agent's construct.log), SHRINK (stock lines went down), GROW (+N lines, batched per
poll), EXIT (agent process gone), LESSONS (a new lessons.md written / loop dead), PROXY (probe
status changed), SUMMARY (every 30 min). Every stock line ever seen is kept in
<epoch>/watch_stock_history/<attempt>.jsonl, so an overwrite cannot erase it.
"""

import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

#: Every stock line ever seen, per attempt, kept beside the epoch it belongs to (the harness also
#: writes stock/shrink_NNN.candidates.jsonl at each shrink).
HIST_NAME = "watch_stock_history"
STALL_MIN = float(os.environ.get("WATCH_STALL_MIN", 6))
ERR = re.compile(
    r"(RateLimitError|budget_exceeded|Budget has been exceeded|\b40[23]\b|\b5\d\d\b Server|Traceback|"
    r"patience (spent|exhausted)|AuthenticationError|out of credits|Killed)",
    re.I,
)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
once = "--once" in sys.argv
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 120
args = [a for a in args if not a.isdigit()]


def emit(msg):
    print(time.strftime("%H:%M:%S ", time.gmtime()) + msg, flush=True)


def alive_runs():
    out = subprocess.run(["pgrep", "-af", "bench_cli (construct|lessons)"], capture_output=True, text=True).stdout
    return out


def proxy_status():
    if os.environ.get("WATCH_ROUTE") == "openrouter":
        key = os.environ.get("OPENROUTER_KEY", "")
        req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=30).read()).get("data", {})
            rem = d.get("limit_remaining")
            # one status per $5 step, so the event fires each time $5 more is spent
            return (
                f"KEY ${rem:.2f}-left step{int(rem // 5)} (usage ${d.get('usage', 0):.2f} of ${d.get('limit')})"
                if rem is not None
                else "KEY no-limit"
            )
        except Exception as e:  # noqa: BLE001
            return f"KEY-UNREADABLE ({e})"
    return _proxy_status()


def _proxy_status():
    base, key = os.environ.get("LITELLM_BASE_URL", ""), os.environ.get("LITELLM_KEY", "")
    model = os.environ.get("WATCH_MODEL", "")
    if not model:
        return "NO-MODEL (set WATCH_MODEL)"
    body = json.dumps({"model": model, "max_tokens": 5, "messages": [{"role": "user", "content": "OK"}]}).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=60).read()
        return "UP (200)"
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")[:160]
        return f"DOWN ({e.code}: {txt})"
    except Exception as e:  # noqa: BLE001
        return f"DOWN ({e})"


def attempts():
    for ep in args:
        for a in sorted(glob.glob(os.path.join(ep, "attempt_*"))):
            yield ep, a


state = {}
last_proxy = None
last_summary = 0
lessons_seen = {}
first = True
while True:
    procs = alive_runs()
    now = time.time()
    tot = {}
    for ep, a in attempts():
        key = a
        task = ep.rstrip("/").split("/")[-2]
        tag = f"{task}/{os.path.basename(ep)}/{os.path.basename(a)}"
        st = state.setdefault(key, {"lines": None, "err": {}, "alive": None, "stalled": False})
        alive = False
        try:
            pid = int(open(os.path.join(a, "launcher.pid")).read().strip())
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            alive = os.path.abspath(a) in cmd or a in cmd
        except (OSError, ValueError):
            pass
        # stock + history
        sp = os.path.join(a, "stock", "candidates.jsonl")
        lines = [l.strip() for l in open(sp, encoding="utf-8", errors="replace")] if os.path.exists(sp) else []
        lines = [l for l in lines if l]
        os.makedirs(os.path.join(ep, HIST_NAME), exist_ok=True)
        hp = os.path.join(ep, HIST_NAME, f"{os.path.basename(a)}.jsonl")
        seen = set(open(hp).read().splitlines()) if os.path.exists(hp) else set()
        new = [l for l in lines if l not in seen]
        if new:
            with open(hp, "a") as fh:
                fh.write("\n".join(new) + "\n")
        n = len(lines)
        t = tot.setdefault(task, [0, 0, 0.0, 0])
        t[0] += n
        t[1] += len(seen | set(lines))
        try:
            t[2] += float(json.load(open(os.path.join(a, "agent_state.json")))["budget"]["cost_usd"]["value"])
        except Exception:  # noqa: BLE001
            pass
        t[3] += 1 if alive else 0
        if st["lines"] is not None and not first:
            if n < st["lines"]:
                emit(f"SHRINK {tag}: stock {st['lines']} -> {n} lines (all lines ever seen kept in {hp})")
            elif n > st["lines"]:
                emit(f"GROW {tag}: +{n - st['lines']} -> {n}")
        st["lines"] = n
        # last event age
        tp = os.path.join(a, "trajectory.jsonl")
        age = (now - os.path.getmtime(tp)) / 60 if os.path.exists(tp) else None
        if alive and age is not None and age > STALL_MIN and not st["stalled"]:
            emit(f"STALL {tag}: no trajectory event for {age:.0f} min")
            st["stalled"] = True
        elif age is not None and age <= STALL_MIN and st["stalled"]:
            emit(f"RESUMED {tag}")
            st["stalled"] = False
        # step counter and cut-off replies (a retry loop keeps the file growing: mtime is not enough)
        cyc, cut = 0, 0
        if os.path.exists(tp):
            with open(tp, "rb") as fh:
                fh.seek(max(0, os.path.getsize(tp) - 400000))
                for raw in fh.read().decode(errors="replace").splitlines()[1:]:
                    try:
                        e = json.loads(raw)
                    except ValueError:
                        continue
                    cyc = max(cyc, int(e.get("cycle") or 0))
                    if e.get("event") == "on_response" and e.get("finish_reason") == "length":
                        cut += 1
        if cyc != st.get("cyc"):
            st["cyc"], st["cyc_t"], st["stuck"] = cyc, now, False
        elif alive and now - st.get("cyc_t", now) > float(os.environ.get("WATCH_STUCK_S", 600)) and not st.get("stuck"):
            emit(f"STUCK {tag}: step {cyc} unchanged for {(now - st['cyc_t']) / 60:.0f} min")
            st["stuck"] = True
        if not first and cut > st.get("cut", cut):
            emit(f"CUTOFF {tag}: {cut - st['cut']} new reply(ies) cut off at the output limit (step {cyc})")
        st["cut"] = cut
        # errors in construct.log tail
        lp = os.path.join(a, "construct.log")
        if os.path.exists(lp):
            with open(lp, "rb") as fh:
                fh.seek(max(0, os.path.getsize(lp) - 20000))
                tail = fh.read().decode(errors="replace")
            for m in ERR.finditer(tail):
                sig = m.group(1).lower()
                if now - st["err"].get(sig, 0) > 900 and not first:
                    line = tail[max(0, m.start() - 80) : m.end() + 120].replace("\n", " ")
                    emit(f"ERROR {tag}: {sig}: ...{line}")
                st["err"][sig] = now if (now - st["err"].get(sig, 0) > 900) else st["err"][sig]
        if st["alive"] is not None and st["alive"] and not alive:
            emit(
                f"EXIT {tag}: process gone; stock {n} lines; result.json={'yes' if os.path.exists(os.path.join(a, 'result.json')) else 'no'}"
            )
        st["alive"] = alive
    # lessons
    for ep in args:
        lm = os.path.join(ep, "lessons.md")
        if os.path.exists(lm):
            head = open(lm).readline().strip()
            if lessons_seen.get(ep) != head:
                if ep in lessons_seen or first:
                    n_l = sum(1 for l in open(lm) if l.startswith("- "))
                    emit(f"LESSONS {ep}: {n_l} lessons; {head[:70]}")
                lessons_seen[ep] = head
        task = ep.rstrip("/").split("/")[-2]
        loop_alive = any(
            l.split(None, 1)[1].startswith(
                ("bash scripts/lessons_loop.sh " + ep, "/bin/bash scripts/lessons_loop.sh " + ep)
            )
            for l in subprocess.run(
                ["pgrep", "-af", "lessons_loop.sh"], capture_output=True, text=True
            ).stdout.splitlines()
            if len(l.split(None, 1)) == 2
        )
        k = ("loop", ep)
        if state.get(k) and not loop_alive:
            emit(f"LESSONS-LOOP-GONE {ep}")
        state[k] = loop_alive
    # proxy every 10 min or when anything stalls
    if first or now - state.get("proxy_t", 0) > 600:
        ps = proxy_status()
        state["proxy_t"] = now
        if (ps.split()[:3:2] != (last_proxy or "").split()[:3:2]) or first:
            emit(f"PROXY {ps}")
        last_proxy = ps
    if first or now - last_summary > 1800:
        emit(
            "SUMMARY "
            + "; ".join(f"{k}: {v[0]} lines now ({v[1]} ever seen), ${v[2]:.2f}, {v[3]} alive" for k, v in tot.items())
        )
        last_summary = now
    first = False
    if once:
        break
    time.sleep(every)
