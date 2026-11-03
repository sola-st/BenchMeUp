"""Single-task run — the one construction path shared by run/batch/replay.

Builds models, env, tool registry, working-set, prompts, budget, stages, oracle and
observability, then runs LoopAgent or OracleLoop. The agent directory (where
agent.yaml lives) is put on sys.path so module refs like `hooks:judge`,
`ws:BuildState`, `instances.schema:Foo`, `oracle:build_check` resolve.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import sys
import threading
import time
from typing import Any


def terminate_on_signal() -> None:
    """SIGTERM ends the run the way an exception does, so every `finally` runs: the replay
    removes its container, the teardown removes the worker's machine.

    Python's default handler ends the process without unwinding, which would leave replay
    containers and idle machines behind. Installed only from the main thread (signal handlers cannot be set elsewhere)."""

    def handler(signum, frame):
        raise SystemExit(128 + signum)

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, handler)


from agentcore.agent.budget import BudgetController
from agentcore.agent.events import EventEmitter
from agentcore.agent.loop import LoopAgent, RunResult
from agentcore.agent.oracle_loop import OracleLoop
from agentcore.agent.prompts import PromptStore
from agentcore.agent.stages import StageController
from agentcore.agent.workingset import load_working_set
from agentcore.config.loader import load_agent_spec
from agentcore.config.schema import AgentSpec
from agentcore.environments.factory import get_environment
from agentcore.models.factory import build_models
from agentcore.observability.state import StatePersistence
from agentcore.observability.trajectory import TrajectoryWriter
from agentcore.oracle.base import build_oracle
from agentcore.tools import builtin as _builtin
from agentcore.tools.registry import ToolRegistry

_agent_dir_lock = threading.Lock()
_last_agent_dir: str | None = None

# Module names agent.yaml conventionally resolves relative to agent_dir (instance_schema,
# hooks:, oracle:, ws:, tools). A flow (agentcore/run/flow.py) runs multiple agent_dirs'
# run_single() in the same process via ThreadPoolExecutor -- Python caches imports by
# dotted name in sys.modules, so a later step's `instances.schema:BuildTask` would silently
# resolve to an earlier step's cached `instances/schema.py` (different file, same name)
# instead of its own, since sys.path.insert alone doesn't invalidate that cache.
_AMBIENT_MODULE_NAMES = ("instances", "hooks", "oracle", "tools", "ws", "validators", "prerender")


def _add_agent_dir_to_path(agent_dir: str) -> None:
    global _last_agent_dir
    agent_dir = os.path.abspath(agent_dir)
    with _agent_dir_lock:
        if agent_dir in sys.path:  # move to front so it wins name resolution over any
            sys.path.remove(agent_dir)  # other agent_dir sitting ahead of it from a prior run
        sys.path.insert(0, agent_dir)
        if agent_dir != _last_agent_dir:
            for name in list(sys.modules):
                if name.split(".", 1)[0] in _AMBIENT_MODULE_NAMES:
                    del sys.modules[name]
            _last_agent_dir = agent_dir


def write_run_meta(out_dir: str, **meta: Any) -> None:
    """Persist how to rebuild this run (config/agent_dir/instance/workspace/loop) so
    `agent resume` can reconstruct the agent without the original command line."""
    # An armed stall judge is provenance the same way AGENT_MODEL is: env-only config
    # that can kill a run's commands, so the run dir must record it, not the launcher's
    # shell. Recorded next to the run's own resolved models.
    from agentcore.environments.docker_helpers import stall_judge_config

    cfg = stall_judge_config()
    if cfg is not None and isinstance(meta.get("models"), dict):
        name, window = cfg
        meta["models"] = {**meta["models"], "stall_judge": {"name": name, "stall_window_s": window}}
    # THE PROCESS NAMES ITSELF. Liveness inferred from the pid printed at launch is
    # invalidated by a relaunch into the same directory: the old pid is gone, the new worker
    # is running, and the fleet view would say DEAD. The run's own pid and kernel start time (pid recycling cannot forge
    # the pair) are written here, by the process that owns them, for every launch path.
    meta["process"] = process_identity()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp, path)


def process_identity(pid: int | None = None) -> dict:
    """{pid, starttime} for a process -- the pair that survives pid recycling. `starttime`
    is /proc/<pid>/stat field 22 (clock ticks since boot); None where /proc is unreadable."""
    pid = int(pid or os.getpid())
    start = None
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
            stat = fh.read()
        # the command name is in parentheses and may contain spaces: split after the last ')'
        start = int(stat[stat.rindex(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        pass
    return {"pid": pid, "starttime": start}


def read_run_meta(run_dir: str) -> dict:
    path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_instance(ref: str | dict | None) -> dict:
    if ref is None:
        return {}
    if isinstance(ref, dict):
        return ref
    path, _, idx = ref.partition(":")
    if not os.path.exists(path):
        return {"task": ref}  # treat the string as a freeform task
    if path.endswith(".jsonl"):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows[int(idx or 0)]
    with open(path) as f:
        data = json.load(f)
    return data[int(idx)] if isinstance(data, list) and idx else data


def build_agent(
    spec: AgentSpec,
    agent_dir: str,
    *,
    instance: dict,
    workspace: str,
    out_dir: str,
    controller=None,
) -> LoopAgent:
    _add_agent_dir_to_path(agent_dir)

    # validate the instance against its schema (validate-at-load)
    if spec.instance_schema:
        mod, _, cls = spec.instance_schema.partition(":")
        schema_cls = getattr(importlib.import_module(mod), cls)
        instance = {**instance, **schema_cls.model_validate(instance).model_dump()}

    # image_template (Jinja over instance vars) → concrete image, e.g. SWE-bench's
    # image = f(instance_id). Only when no explicit image was given.
    if spec.env.image_template and not spec.env.image:
        from jinja2 import Template

        spec.env.image = Template(spec.env.image_template).render(**instance)

    models = build_models(spec.models)
    env = get_environment(spec.env, workspace=workspace, instance=instance, out_dir=out_dir)
    if hasattr(env, "provision_commands") and spec.env.commands:  # user-declared CLI commands
        env.provision_commands()

    registry = ToolRegistry()
    registry.register_module(_builtin)
    for mod_path in spec.tool_modules:  # opt-in shared toolkits (e.g. agentcore.tools.se)
        registry.register_module(importlib.import_module(mod_path))
    tools_path = os.path.join(agent_dir, "tools.py")
    if os.path.exists(tools_path):
        registry.register_module(importlib.import_module("tools"))

    working_set = load_working_set(spec.working_set_schema)
    prompts = PromptStore(agent_dir, spec.prompts)
    budget = BudgetController(spec.budgets)
    stages = StageController(spec.stages)
    oracle = build_oracle(spec.oracle)
    events = EventEmitter()

    cls = OracleLoop if spec.loop == "oracle" else LoopAgent
    agent = cls(
        spec,
        registry=registry,
        env=env,
        models=models,
        working_set=working_set,
        prompts=prompts,
        budget=budget,
        stages=stages,
        events=events,
        oracle=oracle,
        instance=instance,
        task=instance.get("task", spec.description),
        controller=controller,
    )

    # observability via events (no monkey-patch)
    traj = TrajectoryWriter(out_dir)
    traj.bind(agent)
    state = StatePersistence(out_dir)
    state.bind(agent)
    events.subscribe(traj)
    events.subscribe(state)
    if spec.env_watch.get("enabled"):  # opt-in per-cycle environment-change signal
        from agentcore.observability.env_watch import EnvWatcher

        watcher = EnvWatcher(spec.env_watch)
        watcher.bind(agent)
        events.subscribe(watcher)
    return agent


class _SkipCleanup(Exception):
    pass


def run_single(
    config_path: str,
    *,
    instance: str | dict | None = None,
    workspace: str | None = None,
    output_dir: str | None = None,
    overrides: dict | None = None,
    controller=None,
) -> RunResult:
    terminate_on_signal()
    spec = load_agent_spec(config_path, overrides)
    # FAIL AT LAUNCH, not per tool call. The docker SDK import is lazy inside the helpers, so
    # a run launched from an interpreter without the `docker` package would start cleanly
    # and then fail EVERY execution tool with "Docker SDK not available". Checked here
    # (the run entrypoint) rather than in the environment factory: tests legitimately build
    # docker-typed agents just to render prompts, and never touch the daemon.
    if getattr(spec.env, "type", "") in ("docker", "deferred"):
        try:
            import docker  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                f"this agent's env is {spec.env.type!r} but the python docker SDK is not "
                f"importable in this interpreter ({sys.executable}); install 'docker' (pip) "
                f"or launch with the interpreter that has it. Original error: {e}"
            ) from e
    agent_dir = os.path.dirname(os.path.abspath(config_path))
    inst = load_instance(instance)
    out_dir = output_dir or os.path.join("runs", spec.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    ws = workspace or os.path.join(out_dir, "workspace")

    if spec.attempts.enabled:
        return _run_with_attempts(spec, agent_dir, inst, ws, out_dir, config_path=os.path.abspath(config_path))

    # `models` is the RESOLVED config, after yaml defaults, env overrides (AGENT_MODEL) and
    # CLI overrides merged. The yaml alone can mislead a reader about which model ran, when
    # AGENT_MODEL points at a different one. The model that produced a run is
    # its most basic provenance fact; it must live in the run dir, not in the launcher's shell.
    write_run_meta(
        out_dir,
        config_path=os.path.abspath(config_path),
        agent_dir=agent_dir,
        instance=inst,
        workspace=ws,
        loop=spec.loop,
        name=spec.name,
        models={k: {"name": m.name, "temperature": m.temperature} for k, m in (spec.models or {}).items()},
    )
    agent = build_agent(spec, agent_dir, instance=inst, workspace=ws, out_dir=out_dir, controller=controller)
    agent._scratch["out_dir"] = out_dir
    agent._scratch["agent_dir"] = agent_dir
    agent.events.emit("on_run_start", agent=spec.name, instance=inst)
    try:
        result = agent.run()
    finally:
        _teardown_all([agent], spec)
    return result


def _teardown_all(agents, spec) -> int:
    """Tear down every agent's sandboxes, and never let the teardown replace the result.

    An exception raised from inside a `finally` substitutes itself for the value the run
    had just produced, and a teardown that fails leaks sandboxes onto the host.

    Returns how many agents were torn down. A failure per agent is reported and does not
    stop the rest, because a teardown that gives up halfway leaks exactly the containers it
    exists to remove.
    """
    done = 0
    for a in agents or []:
        for step in (
            lambda: _cleanup_instance_containers(a, spec),
            lambda: a.env.cleanup(),
            lambda: a.events.emit("on_run_end"),
        ):
            try:
                step()
            except Exception as e:  # noqa: BLE001 - teardown may not decide a run's outcome
                try:
                    a.events.emit(
                        "on_intervention",
                        site="run.teardown_failed",
                        kind="teardown_failed",
                        error=f"{type(e).__name__}: {e}",
                    )
                except Exception:  # noqa: BLE001
                    pass
        done += 1
    return done


def _cleanup_instance_containers(agent, spec) -> None:
    """Tear down any per-instance sandboxes (agentcore.tools.containers) the agent created.

    Separate from `env.cleanup()` because those containers are deliberately NOT owned by the
    environment -- they exist beside it, one per candidate instance. Without this a run that
    crashes or exhausts its budget mid-mining leaks every sandbox it had open onto the host.
    Imported lazily so the tool module stays fully optional.
    """
    # The blanket except below is for docker faults, but the skip decision happens before
    # it so a fault cannot fake a keep.
    if getattr(spec.env, "keep_containers", False):
        return
    try:
        from agentcore.tools.containers import cleanup_containers

        n = cleanup_containers(getattr(agent, "_scratch", {}))
        if n:
            agent.events.emit("on_intervention", site="run.container_cleanup", kind="container_cleanup", removed=n)
    except Exception:
        pass


def _run_with_attempts(
    spec: AgentSpec, agent_dir: str, inst: dict, ws: str, out_dir: str, *, config_path: str = ""
) -> RunResult:
    """Outer retry-with-learning / sampling loop (AttemptController)."""
    from agentcore.agent.attempts import AttemptController

    built: list = []

    def make_agent(carry):
        i = len(built)
        sub = os.path.join(out_dir, f"attempt-{i}")
        write_run_meta(
            sub,
            config_path=config_path,
            agent_dir=agent_dir,
            instance=inst,
            workspace=f"{ws}_a{i}",
            loop=spec.loop,
            name=spec.name,
        )
        a = build_agent(spec, agent_dir, instance=inst, workspace=f"{ws}_a{i}", out_dir=sub)
        a._scratch["out_dir"] = sub
        if carry is not None:
            a.working_set.merge({"carryover": carry.model_dump()})
        a.events.emit("on_run_start", agent=spec.name, instance=inst, attempt=i)
        built.append(a)
        return a

    ctrl = AttemptController(spec.attempts, make_agent, max_attempts=spec.budgets.max_attempts)
    try:
        return ctrl.run()
    finally:
        _teardown_all(built, spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent-run", description="Run a single agentcore task.")
    p.add_argument("--config", required=True, help="path to agent.yaml")
    p.add_argument("--instance", default=None, help="instances.jsonl:index | path.json | freeform task")
    p.add_argument("--workspace", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--model", default=None, help="override models.main.name")
    # There is no --budget flag: budgets come from agent.yaml / an overrides file, and a flag
    # that is parsed but ignored would silently do nothing.
    args = p.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.model:
        overrides["models"] = {"main": {"name": args.model}}
    res = run_single(
        args.config, instance=args.instance, workspace=args.workspace, output_dir=args.output, overrides=overrides
    )
    print(
        json.dumps(
            {"success": res.success, "reason": res.reason, "steps": res.steps, "cost": round(res.cost, 4)}, indent=2
        )
    )
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
