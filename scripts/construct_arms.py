"""Launch one construction epoch as several ARMS: groups of workers that differ only in the
model (and its provider pin, temperature and cost cap). Every arm gets the same brief, the same
lessons file and its share of the same frozen stock, so the arms compare.

    python scripts/construct_arms.py <stock.jsonl> <worker agent dir> <epoch out dir> <arms.json> [--target N] [--dry-run]

arms.json:
    [{"name": "armA", "model": "openai/openrouter/<vendor>/<model-a>", "workers": 3,
      "cost_usd": 5, "wallclock": 28800, "temperature": 0.3,
      "provider": {"order": ["<provider>"], "allow_fallbacks": false}},
     {"name": "armB", "model": "openai/openrouter/<vendor>/<model-b>", "workers": 3, "cost_usd": 10,
      "temperature": 1, "provider": {"order": ["<provider>"], "allow_fallbacks": false}},
     ...]

What it does, in order: `bench handover` for the total number of workers (the stock is dealt
by repository, as always), then for each worker an overrides.yaml carrying the arm's budgets
and its models.main block (the overlay deep-merges into agent.yaml, so the pin in agent.yaml
is REPLACED by the arm's, not added to), an arm.json naming the arm, and the launch with
`--model <arm model>`. Workers are started with start_new_session, so launcher.pid holds the
real pid (an interactive `setsid ... &` forks and records the wrong one).
The environment must already carry the route: OPENAI_API_BASE / OPENAI_API_KEY for an `openai/`
model (any litellm or OpenAI-compatible endpoint); see docs/configuration.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CACHE = (os.environ.get("BENCH_CACHE_URL") or "").rstrip("/")


def host_cache_env() -> None:
    """The same as infra/maven-cache/env.sh: when BENCH_CACHE_URL is set and the host Maven
    cache answers, the workers (who inherit this environment) are told where it is and their
    image builds get MAVEN_MIRROR; otherwise, say so and launch without it."""
    import urllib.request

    if not CACHE:
        print("host Maven cache: BENCH_CACHE_URL not set; launching without it", file=sys.stderr)
        return
    try:
        urllib.request.urlopen(f"{CACHE}/health", timeout=5).read()
    except Exception:
        print(
            f"host Maven cache: NOT answering at {CACHE} (infra/maven-cache/run.sh starts it); launching without it",
            file=sys.stderr,
        )
        return
    with open(os.path.join(ROOT, "infra", "maven-cache", "machine_facts.txt"), encoding="utf-8") as fh:
        os.environ["BENCH_MACHINE_FACTS"] = fh.read().replace("@CACHE@", CACHE)
    args = json.loads(os.environ.get("BENCH_BUILD_ARGS") or "{}")
    # one argument per upstream, as machine_facts.txt tells the agents to declare them
    args.update(
        MAVEN_MIRROR=f"{CACHE}/maven2/",
        GRADLE_PLUGIN_MIRROR=f"{CACHE}/gradle-plugins/",
        GOOGLE_MAVEN_MIRROR=f"{CACHE}/google/",
    )
    os.environ["BENCH_BUILD_ARGS"] = json.dumps(args)
    print(
        "host Maven cache: up; agents are told, builds get MAVEN_MIRROR, GRADLE_PLUGIN_MIRROR, GOOGLE_MAVEN_MIRROR",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stock")
    ap.add_argument("worker_dir")
    ap.add_argument("out_dir")
    ap.add_argument("arms_json")
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="hand over and write the files, launch nothing")
    ap.add_argument(
        "--exclude-finished",
        nargs="*",
        default=None,
        metavar="EPOCH_DIR",
        help="skip what these earlier epochs delivered or dropped",
    )
    ap.add_argument(
        "--shuffle",
        type=int,
        default=None,
        metavar="SEED",
        help="shuffle the candidates with this seed and deal one per worker in turn",
    )
    ap.add_argument(
        "--size-order",
        default=None,
        metavar="CSV",
        help="hand out the smallest repositories first (CSV: repository, size)",
    )
    ap.add_argument(
        "--exclude", default=None, metavar="FILE", help="skip the identities listed in this file (f1|f2 per line)"
    )
    a = ap.parse_args()
    with open(a.arms_json, encoding="utf-8") as fh:
        arms = json.load(fh)
    total = sum(int(x["workers"]) for x in arms)
    os.chdir(ROOT)
    # The openai-compatible route (an `openai/...` model) needs its base and key; a `claude-code/...` model
    # carries its own login and needs neither.
    if any(str(x["model"]).startswith("openai/") for x in arms):
        for k in ("OPENAI_API_BASE", "OPENAI_API_KEY"):
            if not os.environ.get(k):
                print(f"{k} is not exported; an `openai/` model needs it", file=sys.stderr)
                return 2
    host_cache_env()
    cmd = [
        PY,
        "-m",
        "benchcompiler.bench_cli",
        "handover",
        a.stock,
        "--worker",
        a.worker_dir,
        "--workers",
        str(total),
        "--target",
        str(a.target),
        "--out",
        a.out_dir,
    ]
    if a.exclude_finished:
        cmd += ["--exclude-finished", *a.exclude_finished]
    if a.exclude:
        cmd += ["--exclude", a.exclude]
    if a.shuffle is not None:
        cmd += ["--shuffle", "--seed", str(a.shuffle)]
    if a.size_order:
        cmd += ["--size-order", a.size_order]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr)
        return r.returncode
    partitions = sorted(f for f in os.listdir(a.out_dir) if f.startswith("worker_") and f.endswith(".instances.jsonl"))
    if len(partitions) < total:
        print(
            f"handover made {len(partitions)} partition(s) for {total} workers; the last arms get fewer",
            file=sys.stderr,
        )
    i = 0
    launched = []
    for arm in arms:
        for _ in range(int(arm["workers"])):
            if i >= len(partitions):
                break
            wname = partitions[i].split(".")[0]
            wdir = os.path.join(a.out_dir, wname)
            os.makedirs(wdir, exist_ok=True)
            main_block: dict = {"name": arm["model"]}
            if arm.get("temperature") is not None:
                main_block["temperature"] = arm["temperature"]
            kwargs: dict = {}
            if arm.get("provider") is not None:
                kwargs["extra_body"] = {"provider": arm["provider"]}
            # Anything else the route needs in the request body, e.g. a cheaper service
            # tier: {"service_tier": "flex"}.
            if arm.get("extra_body"):
                kwargs.setdefault("extra_body", {}).update(arm["extra_body"])
            if arm.get("max_tokens"):
                kwargs["max_tokens"] = int(arm["max_tokens"])
            # The Claude Code route's own knobs: reasoning effort and the per-call timeout in
            # seconds (agentcore.models.claude_code reads both from kwargs), recorded per worker.
            for k in ("effort", "timeout"):
                if arm.get(k) is not None:
                    kwargs[k] = arm[k]
            if kwargs:
                main_block["kwargs"] = kwargs
            overrides = {
                "budgets": {
                    "meters": [
                        {"kind": "cost_usd", "limit": float(arm.get("cost_usd", 5)), "charge_on": "cycle"},
                        {"kind": "wallclock", "limit": int(arm.get("wallclock", 28800)), "on_exhaust": "finish_cycle"},
                    ]
                },
                "models": {"main": main_block},
            }
            with open(os.path.join(wdir, "overrides.yaml"), "w", encoding="utf-8") as fh:
                yaml.safe_dump(overrides, fh, sort_keys=False)
            with open(os.path.join(wdir, "arm.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "arm": arm["name"],
                        "model": arm["model"],
                        "provider": arm.get("provider"),
                        "temperature": arm.get("temperature"),
                        "cost_usd": arm.get("cost_usd", 5),
                        "effort": arm.get("effort"),
                        "timeout": arm.get("timeout"),
                    },
                    fh,
                    indent=1,
                )
            cmd = [
                PY,
                "-m",
                "benchcompiler.bench_cli",
                "construct",
                "--config",
                os.path.join(a.worker_dir, "agent.yaml"),
                "--instance",
                os.path.join(a.out_dir, partitions[i]),
                "--out",
                wdir,
                "--model",
                arm["model"],
                "--overrides",
                os.path.join(wdir, "overrides.yaml"),
            ]
            if a.dry_run:
                print(f"[dry-run] {wname} arm={arm['name']}: {' '.join(cmd)}")
            else:
                logf = open(os.path.join(wdir, "construct.log"), "w")
                proc = subprocess.Popen(
                    cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
                )
                with open(os.path.join(wdir, "launcher.pid"), "w") as fh:
                    fh.write(str(proc.pid))
                print(
                    f"{time.strftime('%FT%TZ', time.gmtime())} {wname} arm={arm['name']} model={arm['model']} pid={proc.pid}"
                )
                launched.append((wname, arm["name"], proc.pid))
            i += 1
    with open(os.path.join(a.out_dir, "arms.json"), "w", encoding="utf-8") as fh:
        json.dump({"arms": arms, "launched": launched, "ts": time.time()}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
