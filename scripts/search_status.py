"""One table for a search epoch's attempts: alive, cycles, spend, candidates, last tool.

python scripts/search_status.py runs/search/sonarqube/epoch_1
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys


def alive(attempt_dir):
    """Is a `bench construct` process still running for this attempt?

    Not read from launcher.pid: a relaunch by hand writes the shell's pid there and the shell
    exits at once. The run itself is the python process whose command line names this output
    directory.
    """
    import subprocess

    # The attempt is named in the process's --instance path, which is written the same way by
    # the launcher script and by a hand relaunch; --out may be absolute in one and relative in
    # the other, so match on the directory name itself.
    name = os.path.basename(os.path.abspath(attempt_dir))
    try:
        out = subprocess.run(["pgrep", "-af", "bench_cli construct"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            # The status command itself runs in a shell whose command line quotes this directory, so match
            # only the run's own invocation.
            if f"/{name}/" not in line or "bash -c" in line or "search_status" in line:
                continue
            pid = line.split(None, 1)[0]
            state = open(f"/proc/{pid}/status").read().split("State:", 1)[1].strip()[0]
            if state not in ("Z", "X"):
                return True
    except Exception:
        pass
    return False


def last_event(traj):
    name = ts = None
    try:
        with open(traj, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"on_tool_result"' in line:
                    e = json.loads(line)
                    name, ts = e.get("name"), e.get("ts")
    except OSError:
        pass
    return name, ts


def main(epoch):
    rows = []
    for d in sorted(glob.glob(os.path.join(epoch, "attempt_*"))):
        st = os.path.join(d, "agent_state.json")
        if not os.path.exists(st):
            continue
        s = json.load(open(st))
        b = s.get("budget") or {}
        cost = (b.get("cost_usd") or {}).get("value") or 0.0
        cap = (b.get("cost_usd") or {}).get("limit") or 0.0
        wall = (b.get("wallclock") or {}).get("value") or 0.0
        wcap = (b.get("wallclock") or {}).get("limit") or 0.0
        stock = os.path.join(d, "stock", "candidates.jsonl")
        n = sum(1 for _ in open(stock, encoding="utf-8", errors="replace")) if os.path.exists(stock) else 0
        tool, ts = last_event(os.path.join(d, "trajectory.jsonl"))
        age = f"{(dt.datetime.now().timestamp() - ts) / 60:.0f} min ago" if ts else "-"
        rows.append(
            (
                os.path.basename(d),
                "yes" if alive(d) else "no",
                s.get("cycle") or 0,
                f"${cost:.2f}/{cap:.0f}",
                f"{wall / 3600:.1f}/{wcap / 3600:.0f} h",
                n,
                f"{tool or '-'} ({age})",
            )
        )
    head = ("attempt", "alive", "cycles", "spend", "wall", "candidates", "last tool")
    w = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    line = "  ".join(str(h).ljust(w[i]) for i, h in enumerate(head))
    print(line)
    print("  ".join("-" * x for x in w))
    for r in rows:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print(
        f"\ntotal: {sum(r[5] for r in rows)} candidates, "
        f"${sum(float(r[3].split('/')[0][1:]) for r in rows):.2f} spent, "
        f"{sum(1 for r in rows if r[1] == 'yes')} of {len(rows)} alive"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/search/sonarqube/epoch_1")
