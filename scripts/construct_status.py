"""One table for a construction epoch's workers: alive, cycles, spend, deliveries, drops.

    python scripts/construct_status.py runs/construct/boca/epoch_1

Counts are read from the run's own record, never from console text: a delivery is a recipe
directory carrying the three markers the packager requires (report.json, a submit_*.json and
instance/), a drop is a drop_candidate tool result.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

DROP = re.compile(r"^(.*?) dropped:", re.S)


def running(worker_dir) -> bool:
    # The worker's own pid first (launcher.pid, written by the launchers): a path match alone could
    # also match a later wave's worker whose path contains this one.
    pidfile = os.path.join(worker_dir, "launcher.pid")
    if os.path.exists(pidfile):
        try:
            pid = int(open(pidfile).read().strip())
            state = open(f"/proc/{pid}/status").read().split("State:", 1)[1].strip()[0]
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
            return state not in ("Z", "X") and os.path.basename(os.path.abspath(worker_dir)) in cmd
        except (OSError, ValueError, IndexError):
            return False
    name = os.path.basename(os.path.abspath(worker_dir))
    parent = os.path.basename(os.path.dirname(os.path.abspath(worker_dir)))
    try:
        out = subprocess.run(["pgrep", "-af", "bench_cli construct"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if f"/{parent}/{name}" not in line or "bash -c" in line or "construct_status" in line:
                continue
            pid = line.split(None, 1)[0]
            state = open(f"/proc/{pid}/status").read().split("State:", 1)[1].strip()[0]
            if state not in ("Z", "X"):
                return True
    except Exception:
        pass
    return False


def counts(worker_dir):
    delivered = dropped = 0
    for cand in glob.glob(os.path.join(worker_dir, "recipes", "*")):
        if not os.path.isdir(cand) or os.path.basename(cand) == "bench_last":
            continue
        if (
            os.path.exists(os.path.join(cand, "report.json"))
            and glob.glob(os.path.join(cand, "submit_*.json"))
            and os.path.isdir(os.path.join(cand, "instance"))
        ):
            delivered += 1
    traj = os.path.join(worker_dir, "trajectory.jsonl")
    opened, last_tool, last_ts, stage = set(), None, None, None
    try:
        with open(traj, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"on_tool_result"' in line:
                    e = json.loads(line)
                    last_tool, last_ts = e.get("name"), e.get("ts")
                    if e.get("name") == "drop_candidate":
                        dropped += 1
                elif '"on_cycle_start"' in line:
                    e = json.loads(line)
                    if e.get("candidate"):
                        opened.add(e["candidate"])
                    stage = e.get("stage") or stage
    except OSError:
        pass
    return delivered, dropped, len(opened), last_tool, last_ts, stage


def _reply_stats(rows):
    """(replies, median latency s, median cost per reply) from a worker's trajectory rows."""
    import statistics as st

    sent, lat, cost = None, [], []
    for r in rows:
        if r.get("event") == "on_prompt_sent":
            sent = r.get("ts")
        elif r.get("event") == "on_response" and r.get("role") == "main" and sent:
            lat.append(r["ts"] - sent)
            cost.append(r.get("cost") or 0)
            sent = None
    return len(lat), (st.median(lat) if lat else 0.0), (st.median(cost) if cost else 0.0)


def _position(rows):
    """The candidate in hand and the row its last check stopped at."""
    checks = [r for r in rows if r.get("event") == "on_tool_result" and r.get("name") == "bench_check"]
    pos = "stage 1, no check yet"
    if checks:
        d = (checks[-1].get("result") or {}).get("display_text", "")
        m = re.findall(r"^\s*(\d+) ([^\[]+)\[(\w+)\]\s+(PASS|FAIL|SEEN|ok)\s*(.*)$", d, re.M)
        fails = [x for x in m if x[3] not in ("PASS", "ok")]
        stg = re.search(r"stage (\d) of 3", d)
        pos = f"stage {stg.group(1) if stg else '?'}, " + (
            f"row {fails[0][0]} {fails[0][2]}" if fails else "all rows pass"
        )
    cand = (rows[-1].get("candidate") or "").split("github.com/")[-1] if rows else ""
    repo, _, sha = cand.partition("|")
    return (f"{repo[:26]}@{sha[:8]}" if sha else repo[:26] or "-"), pos


def _rows_of(traj):
    rows = []
    try:
        with open(traj, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def _print_table(head, rows):
    w = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    print("  ".join(str(h).ljust(w[i]) for i, h in enumerate(head)))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def main(epoch, detail=False):
    """ONE table, one row per worker (with --detail: candidate, position, latency and cost
    per reply in the same row), then the arms side by side, the totals, and the drops."""
    now = dt.datetime.now().timestamp()
    rows, drops = [], []
    for d in sorted(glob.glob(os.path.join(epoch, "worker_*"))):
        if not os.path.isdir(d):
            continue
        st_path = os.path.join(d, "agent_state.json")
        if not os.path.exists(st_path):
            continue
        s = json.load(open(st_path))
        b = s.get("budget") or {}
        cost = (b.get("cost_usd") or {}).get("value") or 0.0
        cap = (b.get("cost_usd") or {}).get("limit") or 0.0
        wall = (b.get("wallclock") or {}).get("value") or 0.0
        wcap = (b.get("wallclock") or {}).get("limit") or 0.0
        delivered, dropped, opened, tool, ts, stage = counts(d)
        age = f"{(now - ts) / 60:.0f} min" if ts else "-"
        alive = "yes" if running(d) else "no"
        row = {
            "worker": os.path.basename(d),
            "arm": arm_of(d),
            "alive": alive,
            "cycles": s.get("cycle") or 0,
            "opened": opened,
            "delivered": delivered,
            "dropped": dropped,
            "spend": f"${cost:.2f}/{cap:.0f}",
            "wall": f"{wall / 3600:.1f}/{wcap / 3600:.0f} h",
            "last tool": f"{tool or '-'} ({age})",
            "_cost": cost,
        }
        if detail:
            tr = _rows_of(os.path.join(d, "trajectory.jsonl"))
            n, lat, per = _reply_stats(tr)
            cand, pos = _position(tr)
            row.update(
                {"candidate": cand, "position": pos, "replies": n, "latency": f"{lat:.0f}s", "$/reply": f"{per:.4f}"}
            )
            for r in tr:
                if r.get("event") == "on_tool_result" and r.get("name") == "drop_candidate":
                    drops.append(
                        f"  {os.path.basename(d)} ({arm_of(d)}) cyc {r.get('cycle'):>3}: "
                        f"{str((r.get('args') or {}).get('reason', ''))[:150]}".replace("\n", " ")
                    )
        rows.append(row)
    cols = (
        ["worker", "arm", "alive", "cycles", "opened", "delivered", "dropped"]
        + (["candidate", "position", "replies", "latency", "$/reply"] if detail else [])
        + ["spend", "wall", "last tool"]
    )
    _print_table(tuple(cols), [tuple(r[c] for c in cols) for r in rows])
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    if len(arms) > 1:
        # The arms side by side: with several models on one family every status is the comparison,
        # never one tally.
        print()
        arow = []
        for a in arms:
            rs = [r for r in rows if r["arm"] == a]
            arow.append(
                (
                    a,
                    sum(r["opened"] for r in rs),
                    sum(r["delivered"] for r in rs),
                    sum(r["dropped"] for r in rs),
                    sum(r["cycles"] for r in rs),
                    f"${sum(r['_cost'] for r in rs):.2f}",
                    f"{sum(1 for r in rs if r['alive'] == 'yes')} of {len(rs)}",
                )
            )
        _print_table(("arm", "opened", "delivered", "dropped", "cycles", "spend", "alive"), arow)
    print(
        f"\ntotal: {sum(r['delivered'] for r in rows)} delivered, {sum(r['dropped'] for r in rows)} dropped, "
        f"{sum(r['opened'] for r in rows)} opened, ${sum(r['_cost'] for r in rows):.2f} spent, "
        f"{sum(1 for r in rows if r['alive'] == 'yes')} of {len(rows)} alive"
    )
    if drops:
        print("\ndrops:")
        print("\n".join(drops[-12:]))


def arm_of(worker_dir) -> str:
    """The arm name from arm.json (a multi-model epoch), else the model's last path element
    from run_meta.json, else '-'."""
    try:
        return json.load(open(os.path.join(worker_dir, "arm.json")))["arm"]
    except Exception:
        pass
    try:
        name = json.load(open(os.path.join(worker_dir, "run_meta.json")))["models"]["main"]["name"]
        return str(name).rsplit("/", 1)[-1]
    except Exception:
        return "-"


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="one table for a construction epoch's workers")
    ap.add_argument("epoch", nargs="?", default="runs/construct/boca/epoch_1")
    ap.add_argument("--detail", action="store_true", help="add per-worker position, latency, cost and the drops")
    ap.add_argument(
        "--watch", type=int, default=0, metavar="SECONDS", help="redraw every N seconds until Ctrl-C (a live table)"
    )
    a = ap.parse_args()
    import contextlib
    import io

    while True:
        # Build the whole frame first (reading nine trajectories takes seconds), then clear
        # and print it at once, so the screen never sits blank between frames.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(a.epoch, detail=a.detail)
        if a.watch:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(
                f"{a.epoch}  {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
                f"  (every {a.watch}s, Ctrl-C to stop)\n\n"
            )
        sys.stdout.write(buf.getvalue())
        sys.stdout.flush()
        if not a.watch:
            break
        try:
            time.sleep(a.watch)
        except KeyboardInterrupt:
            break
