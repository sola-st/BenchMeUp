"""One search campaign for a task: N searching agents, the way they divide the search space,
when they start, and what they are shown of each other -- every choice is given here.

    set -a; . ./.env; set +a
    python scripts/search_campaign.py <task> <epoch_n> --agents N --territory-mode MODE
        --start-delay S --directions on|off [--directions-model M] [--sync-interval S]
        --lessons on|off [--lessons-interval S] --target T --model M --cost USD [--wall S]
        --max-search K [--extra-body JSON] [--max-tokens K] [--known-extra FILE ...] [--dry-run]

The procedure, with every number given as an option: attempt 0 starts alone;
every --start-delay seconds the search directions of the running agents are summarized
(benchcompiler.directions) and the next attempt starts with them and the instruction to search
another part of the space; once all run, the directions are refreshed every --sync-interval
seconds until none is alive, then once more. The territory mode (request bands, drawn once, or
none) and the lessons loop are options; a typical procedure is territory none, directions
on, start delay 20 min, sync hourly, lessons off.

Each attempt is launched through scripts/search_replicate.py (the same request and preflight);
every attempt is dry-run before the first one starts. Run this detached (setsid nohup); it
writes <epoch>/campaign.json and logs to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
sys.path.insert(0, ROOT)


def _wait_until(t_end, *, now, sleep, any_alive, step=30):
    """Sleep until t_end; return early (False) when no attempt is alive any more."""
    while now() < t_end:
        if not any_alive():
            return False
        sleep(min(step, max(0.0, t_end - now())))
    return True


def schedule(
    n: int,
    start_delay: float,
    sync_interval: float | None,
    *,
    launch,
    sync,
    alive,
    now=time.time,
    sleep=time.sleep,
    log=print,
) -> list:
    """Start attempt 0; then every `start_delay` refresh the directions (when `sync` is given) and
    start the next attempt; once all run, refresh every `sync_interval` until none is alive, and
    once more at the end. Returns the event list [(seconds since start, what)]."""
    t0 = now()
    events = []

    def ev(what):
        events.append((round(now() - t0, 1), what))
        log(f"{time.strftime('%FT%TZ', time.gmtime())} {what}")

    any_alive = lambda: any(alive(k) for k in range(n))
    launch(0)
    ev("launch attempt_0")
    for k in range(1, n):
        if start_delay > 0:
            _wait_until(t0 + k * start_delay, now=now, sleep=sleep, any_alive=lambda: True)
        if sync:
            sync()
            ev(f"directions before attempt_{k}")
        launch(k)
        ev(f"launch attempt_{k}")
    if not sync:
        return events
    last = now()
    while any_alive():
        if not _wait_until(last + (sync_interval or 3600), now=now, sleep=sleep, any_alive=any_alive):
            break
        sync()
        last = now()
        ev("directions refresh")
    sync()
    ev("directions final")
    return events


def _alive(epoch: str, k: int) -> bool:
    d = os.path.join(epoch, f"attempt_{k}")
    try:
        pid = int(open(os.path.join(d, "launcher.pid")).read().strip())
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        return d in cmd
    except (OSError, ValueError):
        return False


def _aside(path: str, tag: str) -> None:
    """Keep an earlier campaign's file for the record, out of the agents' sight."""
    if os.path.exists(path):
        shutil.move(path, f"{path}.{tag}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("epoch")
    ap.add_argument("--agents", type=int, required=True)
    ap.add_argument("--territory-mode", required=True, choices=("request", "drawn", "none"))
    ap.add_argument("--start-delay", type=float, required=True, help="seconds between attempt starts")
    ap.add_argument("--directions", required=True, choices=("on", "off"))
    ap.add_argument("--directions-model", default=None)
    ap.add_argument("--sync-interval", type=float, default=None, help="seconds between direction refreshes")
    ap.add_argument("--lessons", required=True, choices=("on", "off"))
    ap.add_argument("--lessons-interval", type=int, default=None)
    ap.add_argument("--lessons-model", default=None)
    ap.add_argument("--territory-model", default=None, help="drawn mode: the model that draws them")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--reference", default=None, help="passed to search_replicate.py --reference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cost", type=float, required=True)
    ap.add_argument("--wall", type=int, default=21600)
    ap.add_argument("--extra-body", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument(
        "--max-search",
        type=int,
        required=True,
        help="the most search agents (all tasks) allowed to run at once; a launch beyond it is refused",
    )
    ap.add_argument("--known-extra", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.directions == "on" and not (a.directions_model and a.sync_interval):
        print("--directions on needs --directions-model and --sync-interval", file=sys.stderr)
        return 2
    if a.lessons == "on" and not (a.lessons_model and a.lessons_interval):
        print("--lessons on needs --lessons-model and --lessons-interval", file=sys.stderr)
        return 2
    os.chdir(ROOT)
    agent_dir = os.path.join(ROOT, "agents", f"{a.task}_search")
    epoch = os.path.join(ROOT, "runs", "search", a.task, f"epoch_{a.epoch}")
    os.makedirs(epoch, exist_ok=True)
    if a.extra_body:
        os.environ["LLM_EXTRA_BODY"] = a.extra_body  # the helper calls send it too
    terr_file = None
    if a.territory_mode == "drawn":
        terr_file = os.path.join(epoch, "territories.json")
        if not os.path.exists(terr_file):
            if not a.territory_model:
                print("--territory-mode drawn needs --territory-model", file=sys.stderr)
                return 2
            from benchcompiler.search import draw_territories

            with open(terr_file, "w", encoding="utf-8") as fh:
                json.dump(draw_territories(agent_dir, a.agents, a.territory_model), fh, indent=1)

    def replicate(k: int, dry: bool) -> int:
        cmd = [
            PY,
            "scripts/search_replicate.py",
            a.task,
            str(a.epoch),
            "--attempt",
            str(k),
            "--width",
            str(a.agents),
            "--territory-mode",
            a.territory_mode,
            "--target",
            str(a.target),
            "--model",
            a.model,
            "--cost",
            str(a.cost),
            "--wall",
            str(a.wall),
            "--max-search",
            str(a.max_search),
        ]
        if terr_file:
            cmd += ["--territories-file", terr_file]
        if a.extra_body:
            cmd += ["--extra-body", a.extra_body]
        if a.max_tokens:
            cmd += ["--max-tokens", str(a.max_tokens)]
        if a.known_extra:
            cmd += ["--known-extra", *a.known_extra]
        if a.reference:
            cmd += ["--reference", a.reference]
        if dry:
            cmd.append("--dry-run")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip()[-800:], r.stderr.strip()[-800:], sep="\n", flush=True)
        return r.returncode

    bad = [k for k in range(a.agents) if replicate(k, True) != 0]
    if bad:
        print(f"REFUSED: the dry run failed for attempt(s) {bad}; nothing launched", file=sys.stderr)
        return 2
    print(f"dry runs: {a.agents}/{a.agents} passed", flush=True)
    plan = {
        k: getattr(a, k)
        for k in (
            "task",
            "epoch",
            "agents",
            "territory_mode",
            "start_delay",
            "directions",
            "directions_model",
            "sync_interval",
            "lessons",
            "lessons_interval",
            "lessons_model",
            "target",
            "reference",
            "model",
            "cost",
            "wall",
            "max_tokens",
        )
    }
    plan["extra_body"] = json.loads(a.extra_body) if a.extra_body else None
    if a.dry_run:
        print("[dry-run] plan: " + json.dumps(plan))
        return 0
    # what the agents read from the agent dir must be this campaign's, or nothing
    _aside(os.path.join(agent_dir, "directions.md"), "earlier")
    if a.lessons == "off":
        _aside(os.path.join(agent_dir, "lessons.md"), "off")
    plan["started"] = time.time()
    with open(os.path.join(epoch, "campaign.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1)

    def launch(k):
        if replicate(k, False) != 0:
            raise SystemExit(f"attempt_{k} was refused at launch; the campaign stops here")

    sync = None
    if a.directions == "on":
        from benchcompiler.directions import write_directions

        sync = lambda: write_directions(epoch, agent_dir, a.directions_model)
    if a.lessons == "on":
        subprocess.Popen(
            ["bash", "scripts/lessons_loop.sh", epoch, agent_dir, a.lessons_model, str(a.lessons_interval)],
            stdout=open(os.path.join(epoch, "lessons_loop.log"), "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    events = schedule(
        a.agents, a.start_delay, a.sync_interval, launch=launch, sync=sync, alive=lambda k: _alive(epoch, k)
    )
    plan["events"] = events
    plan["ended"] = time.time()
    with open(os.path.join(epoch, "campaign.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
