"""Launch ONE search agent that repeats a reference search request for a task on another
model. The request is the one the reference agent got in
references/search/<task>.instances.jsonl (its target, territory and sibling territories),
rebuilt from the current code; only the model, the caps and the known-set path differ.

    set -a; . ./.env; set +a
    python scripts/search_replicate.py <task> <epoch_n> --model M --cost USD [--wall S] [--dry-run]

The launch is refused, before anything is spent, when:
  - the attempt directory already holds a run (never written over);
  - the rebuilt request differs from the reference request in any field but known_from;
  - the known-set is not one empty file (the reference search started from an empty stock file);
  - the model's route does not answer one 5-token call, or the GitHub token has < 500 calls left;
  - this launch would make more than --max-search search agents run at once.
After the launch it waits for run_meta.json and refuses to call the launch good unless the
recorded main model is the one asked for (AGENT_MODEL, not agent.yaml, decides it).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def refuse(msg: str) -> int:
    print(f"REFUSED: {msg}", file=sys.stderr)
    return 2


def running_search_agents() -> list[str]:
    out = subprocess.run(["pgrep", "-af", "bench_cli construct"], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if "/runs/search/" in ln or " runs/search/" in ln]


def openrouter_remaining() -> float | None:
    """The direct OpenRouter key's remaining limit in USD (None = no limit or no answer)."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY") or ""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=30).read()).get("data", {})
    except Exception:  # noqa: BLE001
        return None
    return None if d.get("limit_remaining") is None else float(d["limit_remaining"])


def route_answers(model: str, extra_body: dict | None = None) -> str | None:
    """None when one tiny call through the model's route returns a choice, else the reason."""
    if model.startswith("openrouter/"):
        # The direct route: same model, any provider; the key is OPENROUTER_KEY in .env.
        base = "https://openrouter.ai/api/v1"
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY") or ""
        if not key:
            return "OPENROUTER_KEY not set (source .env)"
        name = model[len("openrouter/") :]
    elif model.startswith("openai/"):
        base, key = os.environ.get("OPENAI_API_BASE", ""), os.environ.get("OPENAI_API_KEY", "")
        if not base or not key:
            return "OPENAI_API_BASE / OPENAI_API_KEY not set (source .env; they are copied from LITELLM_*)"
        name = model[len("openai/") :]
    else:
        return f"{model}: only the OpenAI-compatible endpoint (openai/...) and direct OpenRouter (openrouter/...) routes are checked here"
    body = json.dumps(
        {
            "model": name,
            "temperature": 0.3,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Say OK"}],
            **(extra_body or {}),
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=90).read())
    except Exception as e:  # noqa: BLE001 - any failure means the route is not usable
        return f"route call failed: {e}"
    return None if d.get("choices") else f"route reply has no choices: {str(d)[:300]}"


def github_calls_left() -> int:
    tok = os.environ.get("GH_FINEGRAINED_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    req = urllib.request.Request("https://api.github.com/rate_limit", headers={"Authorization": f"Bearer {tok}"})
    try:
        return int(json.loads(urllib.request.urlopen(req, timeout=20).read())["resources"]["core"]["remaining"])
    except Exception:  # noqa: BLE001
        return -1


def territories_for(mode: str, ref: dict, attempt: int, width: int, drawn: list | None = None) -> tuple[str, list]:
    """(this attempt's territory, the other attempts' territories) for one way of dividing the
    search space, which is chosen at launch:
      request  the territories the reference request names (star bands): attempt i takes the i-th
      drawn    territories drawn once by a model at the start (benchcompiler.search.draw_territories)
      none     no territory: the agents divide the space by the search directions they are shown
               (benchcompiler.directions)
    Raises ValueError when the mode cannot give `width` attempts a territory each."""
    if mode == "none":
        return "", []
    if mode == "request":
        pool = [ref.get("territory", "")] + list(ref.get("sibling_territories", []))
        if width == 1:
            return pool[0], pool[1:]
    elif mode == "drawn":
        pool = list(drawn or [])
    else:
        raise ValueError(f"unknown territory mode {mode!r}: request, drawn or none")
    if width > len(pool) or not 0 <= attempt < width:
        raise ValueError(f"{mode}: {len(pool)} territories for --width {width} / --attempt {attempt}")
    return pool[attempt], [t for j, t in enumerate(pool[:width]) if j != attempt]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("epoch")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cost", type=float, required=True)
    ap.add_argument("--wall", type=int, default=21600)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-search", type=int, default=2)
    # Several agents per task: the reference request already
    # names four territories (its own star band and three siblings). Attempt i takes band i and
    # lists the other three; the task's request is split evenly (--target per attempt), and each
    # attempt's known-set holds its siblings' live stocks, refreshed before every step.
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument(
        "--territory-mode",
        required=True,
        choices=("request", "drawn", "none"),
        help="how the attempts divide the search space; see territories_for",
    )
    ap.add_argument(
        "--territories-file", default=None, help='drawn mode: {"territories": [...]} as draw_territories writes it'
    )
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--target", type=int, default=None, help="per attempt; default: the Opus request's")
    ap.add_argument(
        "--known-extra",
        nargs="*",
        default=[],
        help="more stock files every attempt counts as known (an earlier epoch's finds)",
    )
    # Sent with every call of the agent (models.main.kwargs.extra_body) and with the route check,
    # e.g. '{"reasoning": {"enabled": false}, "provider": {"allow_fallbacks": true}}'. Give the
    # provider block too: the agent loop replaces the whole body.
    ap.add_argument("--extra-body", default=None, help="JSON object")
    # The reply's output limit (models.main.kwargs.max_tokens; agent.yaml says 32000). A lower
    # limit bounds the time a degenerate, endless reply can take.
    ap.add_argument("--max-tokens", type=int, default=None)
    # The request every attempt is checked against: the reference request unless the task's
    # spec was revised after it; then the regenerated agent's own instances.jsonl.
    ap.add_argument(
        "--reference",
        default=None,
        help="request file to check against (default references/search/<task>.instances.jsonl)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    extra_body = json.loads(a.extra_body) if a.extra_body else None
    os.chdir(ROOT)
    from benchcompiler.search import attempt_instance

    agent = os.path.join(ROOT, "agents", f"{a.task}_search")
    ref_path = (
        os.path.abspath(a.reference)
        if a.reference
        else os.path.join(ROOT, "references", "search", f"{a.task}.instances.jsonl")
    )
    epoch = os.path.join(ROOT, "runs", "search", a.task, f"epoch_{a.epoch}")
    out = os.path.join(epoch, f"attempt_{a.attempt}")
    if os.path.exists(os.path.join(out, "trajectory.jsonl")) or os.path.exists(os.path.join(out, "launcher.pid")):
        return refuse(f"{out} already holds a run; pick a new epoch number")
    with open(ref_path, encoding="utf-8") as fh:
        ref = json.loads(fh.readline())
    bands = [ref.get("territory", "")] + list(ref.get("sibling_territories", []))
    drawn = None
    if a.territory_mode == "drawn":
        if not a.territories_file:
            return refuse("--territory-mode drawn needs --territories-file")
        with open(a.territories_file, encoding="utf-8") as fh:
            drawn = json.load(fh).get("territories") or []
    try:
        territory, siblings = territories_for(a.territory_mode, ref, a.attempt, a.width, drawn)
    except ValueError as e:
        return refuse(str(e))
    target = int(a.target if a.target is not None else ref["sample_size"])
    known = os.path.join(epoch, "known_empty.jsonl")
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(known):
        open(known, "w").close()
    if os.path.getsize(known) != 0:
        return refuse(f"{known} is not empty")
    for p in a.known_extra:
        if not os.path.isfile(p):
            return refuse(f"--known-extra {p}: no such file")
    siblings_stock = [
        os.path.join(epoch, f"attempt_{j}", "stock", "candidates.jsonl") for j in range(a.width) if j != a.attempt
    ]
    inst = attempt_instance(
        agent,
        known_from=[known, *siblings_stock, *[os.path.abspath(p) for p in a.known_extra]],
        target=target,
        territory=territory,
        siblings=siblings,
    )
    # its own name, so it can tell its entry among the search directions it is shown
    inst["agent_name"] = f"attempt_{a.attempt}"
    free = {"known_from", "territory", "sibling_territories", "sample_size", "agent_name"}
    diff = sorted(k for k in set(ref) | set(inst) if k not in free and ref.get(k) != inst.get(k))
    if diff:
        return refuse(f"the rebuilt request differs from the Opus request in {diff}")
    if (
        a.territory_mode == "request"
        and a.width == len(bands)
        and sorted([inst.get("territory", "")] + list(inst.get("sibling_territories", []))) != sorted(bands)
    ):
        return refuse("the territories are not the Opus request's four bands")
    print(
        f"request: equal to {os.path.relpath(ref_path, ROOT)} in every field but known_from, territory, "
        f"siblings, sample_size; sample_size {inst['sample_size']}; territory: {inst.get('territory', '')!r}; "
        f"known-set: {len(inst['known_from'])} file(s)"
    )
    if a.model.startswith("openai/"):
        os.environ.setdefault("OPENAI_API_BASE", os.environ.get("LITELLM_BASE_URL", ""))
        os.environ.setdefault("OPENAI_API_KEY", os.environ.get("LITELLM_KEY", ""))
    why = route_answers(a.model, extra_body)
    if why:
        return refuse(why)
    print(f"route: {a.model} answered")
    if a.model.startswith("openrouter/"):
        left_usd = openrouter_remaining()
        if left_usd is not None and left_usd < a.cost:
            return refuse(f"the OpenRouter key has ${left_usd:.2f} left, below this agent's ${a.cost} cap")
        print(f"openrouter key: {'no limit' if left_usd is None else f'${left_usd:.2f} left'}")
    left = github_calls_left()
    if left < 500:
        return refuse(f"GitHub token has {left} core calls left (needs >= 500; -1 = no answer)")
    print(f"github: {left} core calls left")
    running = running_search_agents()
    if len(running) + 1 > a.max_search:
        return refuse(
            f"{len(running)} search agent(s) already running (cap {a.max_search}): "
            + "; ".join(r[:160] for r in running)
        )
    print(f"search agents running now: {len(running)} (cap {a.max_search})")
    from construct_arms import host_cache_env

    host_cache_env()

    with open(os.path.join(out, "instances.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(inst) + "\n")
    overrides = {
        "budgets": {
            "meters": [
                {"kind": "cost_usd", "limit": a.cost, "charge_on": "cycle"},
                {"kind": "wallclock", "limit": a.wall, "on_exhaust": "finish_cycle"},
            ]
        },
        "models": {
            "main": {
                "name": a.model,
                "temperature": a.temperature,
                **(
                    {
                        "kwargs": {
                            k: v for k, v in (("extra_body", extra_body), ("max_tokens", a.max_tokens)) if v is not None
                        }
                    }
                    if (extra_body or a.max_tokens)
                    else {}
                ),
            }
        },
    }
    with open(os.path.join(out, "overrides.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(overrides, fh, sort_keys=False)
    cmd = [
        PY,
        "-m",
        "benchcompiler.bench_cli",
        "construct",
        "--config",
        os.path.join(agent, "agent.yaml"),
        "--instance",
        os.path.join(out, "instances.jsonl"),
        "--out",
        out,
        "--model",
        a.model,
        "--overrides",
        os.path.join(out, "overrides.yaml"),
    ]
    if a.dry_run:
        print("[dry-run] " + " ".join(cmd))
        return 0
    env = dict(os.environ, AGENT_MODEL=a.model)
    with open(os.path.join(out, "construct.log"), "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env
        )
    with open(os.path.join(out, "launcher.pid"), "w") as fh:
        fh.write(str(proc.pid))
    with open(os.path.join(out, "launch.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "task": a.task,
                "attempt": a.attempt,
                "width": a.width,
                "target": target,
                "model": a.model,
                "temperature": a.temperature,
                "cost_usd": a.cost,
                "wallclock": a.wall,
                "extra_body": extra_body,
                "max_tokens": a.max_tokens,
                "pid": proc.pid,
                "reference_request": os.path.relpath(ref_path, ROOT),
                "ts": time.time(),
            },
            fh,
            indent=1,
        )
    print(
        f"{time.strftime('%FT%TZ', time.gmtime())} launched {a.task} search attempt {a.attempt} pid {proc.pid} -> {out}"
    )
    meta = os.path.join(out, "run_meta.json")
    for _ in range(36):
        if proc.poll() is not None:
            return refuse(f"the run exited at once (code {proc.returncode}); see {out}/construct.log")
        if os.path.exists(meta):
            break
        time.sleep(5)
    try:
        with open(meta, encoding="utf-8") as fh:
            got = json.load(fh).get("models", {}).get("main", {})
    except (OSError, ValueError):
        return refuse(f"no readable {meta} after 3 min; the run is up (pid {proc.pid}), check it")
    # run_meta.json records name and temperature only; the extra body is checked in the
    # overrides.yaml the run was started with (the overlay deep-merges into agent.yaml).
    if got.get("name") != a.model:
        return refuse(f"run_meta.json records main model {got}, not {a.model}; stop pid {proc.pid}")
    print(f"run_meta.json: main model {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
