"""Resume, fork, and replay-takeover — re-enter a saved run and keep going.

All three are the same operation (restore a snapshot into a freshly built agent, then
run) parameterised differently:

  resume_run(run_dir)                      → continue the latest snapshot in place
  resume_run(run_dir, new_task=...)        → FORK: new goal, carrying prior memory, into a new dir
  resume_run(run_dir, at_cycle=N)          → REPLAY-TO-N then take over live, into a new dir

The snapshot (observability.state) plus run_meta.json (how to rebuild the agent) are
the only inputs; compact-context agents rebuild an identical prompt from the restored
action_log, so a resumed cycle is indistinguishable from the original.

Scope note: the filesystem WORKSPACE is restored (copied for forks so the original is
preserved). Live docker/deferred containers are NOT snapshotted — a resumed run starts a
fresh container over the restored workspace.
"""

from __future__ import annotations

import os
import shutil
import time

from agentcore.agent.loop import RunResult
from agentcore.config.loader import load_agent_spec
from agentcore.observability.state import StatePersistence
from agentcore.run.single import build_agent, read_run_meta, write_run_meta


def restore_state(agent, snap: dict) -> None:
    """Rehydrate a built agent's mutable state from a saved snapshot dict.

    Restores action_log, working_set, stage position, budget counters and the
    finished/success flags. Used by resume/fork/replay (and unit-testable on its own).
    """
    agent.action_log = [tuple(x) for x in snap.get("action_log", [])]
    # Absent from snapshots written before the counter existed: default to 0 rather than fail.
    agent.cycle = int(snap.get("agent_cycle") or 0)
    agent.working_set.merge(snap.get("working_set") or {})
    agent.budget.restore(snap.get("budget"))
    if agent.stages.staged:
        st = snap.get("stage")
        if st and st in agent.stages._by_name:
            agent.stages.goto(st)
        elif st is None:  # None == ran past the last stage
            agent.stages.done = True
    agent._finished = bool(snap.get("finished", False))
    agent._success = bool(snap.get("success", False))


def resume_run(
    run_dir: str,
    *,
    at_cycle: int | None = None,
    new_task: str | None = None,
    output_dir: str | None = None,
    overrides: dict | None = None,
) -> RunResult:
    run_dir = os.path.abspath(run_dir)
    meta = read_run_meta(run_dir)
    config_path = meta.get("config_path")
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"cannot resume {run_dir}: run_meta.json missing or its config_path is gone "
            f"({config_path!r}). Only runs created with run_meta support resume."
        )

    snap = StatePersistence.load(run_dir, at_cycle=at_cycle)
    if not snap:
        avail = StatePersistence.available_cycles(run_dir)
        where = f"cycle {at_cycle}" if at_cycle is not None else "latest"
        raise FileNotFoundError(f"no {where} snapshot in {run_dir} (available cycles: {avail})")

    is_fork = new_task is not None
    is_takeover = at_cycle is not None
    fresh_dir = bool(output_dir) or is_fork or is_takeover

    # continue-in-place appends to the original dir; fork/takeover get a new dir so the
    # original run stays intact.
    if fresh_dir:
        tag = "fork" if is_fork else (f"takeover-c{at_cycle}" if is_takeover else "resume")
        out_dir = output_dir or os.path.join(run_dir, "forks", f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}")
    else:
        out_dir = run_dir
    os.makedirs(out_dir, exist_ok=True)

    # workspace: reuse in place; copy for a fresh dir so the original isn't mutated.
    src_ws = meta.get("workspace")
    if fresh_dir:
        ws = os.path.join(out_dir, "workspace")
        if src_ws and os.path.isdir(src_ws):
            shutil.copytree(src_ws, ws, dirs_exist_ok=True)
    else:
        ws = src_ws or os.path.join(out_dir, "workspace")

    spec = load_agent_spec(config_path, overrides)
    instance = meta.get("instance") or {}
    agent_dir = meta.get("agent_dir") or os.path.dirname(config_path)

    # Same provenance rule as a fresh launch: record the RESOLVED models. A resume may run
    # under a different model or route than the original launch -- without this the meta
    # would keep describing the old configuration.
    write_run_meta(
        out_dir,
        config_path=config_path,
        agent_dir=agent_dir,
        instance=instance,
        workspace=ws,
        loop=spec.loop,
        name=spec.name,
        resumed_from=run_dir,
        models={k: {"name": m.name, "temperature": m.temperature} for k, m in (spec.models or {}).items()},
    )
    agent = build_agent(spec, agent_dir, instance=instance, workspace=ws, out_dir=out_dir)
    agent._scratch["out_dir"] = out_dir

    restore_state(agent, snap)
    # resume always intends to CONTINUE — clear the finished flag (a truly-complete staged run
    # still stops immediately via stages.done; a user-`stop`ped one resumes mid-stage).
    agent._finished = False

    if is_fork:
        # new goal, prior learnings retained as context; stages restart from the top
        agent.task = new_task
        if agent.stages.staged:
            agent.stages.index, agent.stages.done = 0, False
        agent._success = False
        agent._mem("[fork]", f"new task: {new_task}", role="user")

    agent.events.emit(
        "on_run_start", agent=spec.name, instance=instance, resumed_from=run_dir, at_cycle=at_cycle, forked=is_fork
    )
    try:
        result = agent.run()
    finally:
        # SAME TEARDOWN AS A FRESH RUN. `env.cleanup()` alone removes the environment
        # container and leaks every per-candidate sandbox a resumed run created.
        # `_teardown_all`
        # sweeps the instance sandboxes, cleans the env, and emits on_run_end, and never
        # lets a teardown failure replace the run's result.
        from agentcore.run.single import _teardown_all

        _teardown_all([agent], spec)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(prog="agent-resume", description="Resume / fork / replay-takeover a run.")
    p.add_argument("run_dir", help="a previous run output directory")
    p.add_argument("--at-cycle", type=int, default=None, help="replay to the end of cycle N, then take over")
    p.add_argument("--task", default=None, help="fork with a new task (carries prior memory)")
    p.add_argument("--output", default=None, help="output dir for the resumed run (forks default to <run>/forks/...)")
    args = p.parse_args(argv)

    res = resume_run(args.run_dir, at_cycle=args.at_cycle, new_task=args.task, output_dir=args.output)
    print(
        json.dumps(
            {"success": res.success, "reason": res.reason, "steps": res.steps, "cost": round(res.cost, 4)}, indent=2
        )
    )
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
