"""Layered configuration loader.

Precedence (low → high):
  1. core defaults        agentcore/config/defaults.yaml  (the `agent:` block)
  2. agent config         agents/<slug>/agent.yaml
  3. env var overrides    AGENT_MODEL, AGENT_ENV_IMAGE, AGENT_WORKSPACE, ...
  4. CLI overrides        passed in as a dict

Each layer is deep-merged; the result is validated ONCE into AgentSpec / RunSpec.
Core constants (truncation sizes, sentinels) are exposed separately via core_defaults().
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from agentcore.agent import derive
from agentcore.config.schema import AgentSpec, RunSpec

_DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


def _read_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def _defaults_doc() -> dict:
    return _read_yaml(_DEFAULTS_PATH)


def core_defaults() -> dict:
    """The `core:` constants block (truncation, sentinels, structurer...)."""
    return _defaults_doc().get("core", {})


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides() -> dict:
    """Map a handful of well-known env vars onto AgentSpec paths."""
    ov: dict[str, Any] = {}
    if model := os.getenv("AGENT_MODEL"):
        ov.setdefault("models", {}).setdefault("main", {})["name"] = model
    if image := os.getenv("AGENT_ENV_IMAGE"):
        ov.setdefault("env", {})["image"] = image
    if temperature := os.getenv("AGENT_MODEL_TEMPERATURE"):
        ov.setdefault("models", {}).setdefault("main", {})["temperature"] = float(temperature)

    # Swap the steps meter for a cost_usd cap (e.g. for a cheap model where step-count isn't
    # the real constraint) -- replaces budgets.meters wholesale (lists aren't deep-merged),
    # keeping a wallclock meter as a hard safety net regardless of spend rate.
    # AND THE CLOCK, which is the meter that often binds: a run can end on the wallclock
    # budget with most of the cost budget unspent, so capping cost alone would not shorten it.
    # A short validation run needs the clock, which avoids keeping a second copy of an agent
    # or editing a shipped agent.yaml for the duration of a run.
    #
    # A BAD VALUE MUST NOT END A RUN AT CYCLE 0. These are read from the environment, which is
    # where typos live; an unparseable limit falls back to the default rather than raising out
    # of `load_agent_spec` before the first cycle.
    def _num(name, default=None):
        raw = os.getenv(name)
        if raw in (None, ""):
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    cost_limit = _num("AGENT_BUDGET_COST_LIMIT")
    clock_limit = _num("AGENT_BUDGET_WALLCLOCK_LIMIT")
    if cost_limit is not None or clock_limit is not None:
        meters = []
        if cost_limit is not None:
            meters.append({"kind": "cost_usd", "limit": cost_limit, "charge_on": "cycle"})
        meters.append({"kind": "wallclock", "limit": clock_limit or 14400, "on_exhaust": "finish_cycle"})
        ov.setdefault("budgets", {})["meters"] = meters
    return ov


def load_agent_spec(config_path: str | Path, cli_overrides: dict | None = None) -> AgentSpec:
    """Load + validate an AgentSpec from a YAML path with layered overrides."""
    agent_doc = _read_yaml(config_path)
    merged = deep_merge(_defaults_doc().get("agent", {}), agent_doc)
    merged = deep_merge(merged, _env_overrides())
    merged = deep_merge(merged, cli_overrides or {})
    spec = AgentSpec.model_validate(merged)
    _refuse_silently_self_graded_states(spec, config_path)
    _refuse_unworkable_computes(spec, config_path)
    return spec


def _refuse_unworkable_computes(spec: AgentSpec, config_path) -> None:
    """A `computes:` the harness cannot honour is a startup error.

    The failure it prevents is silent: an unknown `agg`, a `from:` naming no field, or a
    two-field aggregation given one source produces a field that never appears -- and the
    agent is left facing a `requires` it cannot satisfy, having been told the harness would
    fill it. Refusing at load costs a second; the defect this whole feature answers cost 3.1
    hours in one worker.
    """
    wl = getattr(spec, "worklist", None)
    if wl is None or not getattr(wl, "enabled", False):
        return
    problems = []
    for st in wl.states or []:
        for name, decl in (getattr(st, "computes", None) or {}).items():
            bad = derive.declaration_problem(str(name), decl)
            if bad:
                problems.append(f"{st.name}: {bad}")
    if problems:
        raise ValueError(f"{config_path}: unworkable `computes:` declaration(s) -- " + "; ".join(problems))


def _refuse_silently_self_graded_states(spec: AgentSpec, config_path) -> None:
    """A worklist state with `requires` and no `verify:` is refused at load time.

    `requires` checks presence, which a model satisfies by typing `build_ok: "yes"`; `verify`
    is what checks the claim. Nothing in the type system told them apart, so forgetting
    `verify` was silent -- that one omission produced four of one day's confirmed bugs, in
    four different places. Making it a STARTUP error moves the whole bug class to before the
    run spends money.

    The escape hatch is deliberate and loud: a state may declare `self_graded: true`, which
    keeps the behaviour but puts the decision on record where a reviewer (and the criteria
    coverage report) can see it. Enforced here rather than in the pydantic model because unit
    tests legitimately build bare WorklistSpecs to test the state machine without validators;
    what must never happen silently is a LIVE collector running one.
    """
    wl = getattr(spec, "worklist", None)
    if wl is None or not getattr(wl, "enabled", False):
        return

    # `rederives:` gates a state: the harness re-executes the recorded command itself
    # (worklist._rederives) before any validator, which is what lets DESIGNED lifecycles
    # verify through declarations alone. An all-`none` rederives map does NOT count: every
    # entry exempted means nothing is re-executed.
    # ...BUT A DISCRIMINATES BLOCK IS NOT A STATE'S GATE. It proves the recorded ORACLE
    # reacts to a break -- and it proves that by comparing two strings the model wrote
    # (worklist._discriminates), executing nothing. Letting it gate the whole state would
    # let a state require several fields and be "verified" by "these two summaries differ"
    # while a required patch was never applied and a corruption command never run. It counts
    # as a CONTROL, not as the gate.
    def _gated(st):
        red = getattr(st, "rederives", None) or {}
        return any(isinstance(d, dict) for d in red.values())

    unverified = [
        st.name
        for st in (wl.states or [])
        if (st.requires or []) and not st.verify and not st.self_graded and not _gated(st)
    ]
    if unverified:
        raise ValueError(
            f"{config_path}: worklist state(s) {unverified} have `requires` but nothing that "
            f"executes -- the model would grade its own homework there. Name a validator, or "
            f"declare a `rederives:` entry with a cmd so the harness re-runs the command and "
            f"compares, or declare `self_graded: true` to put that decision on record. A "
            f"`discriminates:` block is a control over two recorded values and no longer "
            f"stands in for a gate: it compares strings the model wrote and runs nothing."
        )

    # ...and a CRITERION taken on the model's word is reported, every time, by name.
    # Refusing it would stop the world: collectors commonly self-grade at least one
    # criterion. Silence would let a benchmark's whole premise ("the test fails on
    # the parent and passes on the fix") be satisfied by an agent typing `fail` and `pass`.
    # The sharp form of this rule -- a state that RECORDS or COMPUTES evidence may not
    # self-grade -- is refused at design time (benchcompiler.instantiate).
    on_trust = [
        (st.name, list(st.enforces or []))
        for st in (wl.states or [])
        if st.self_graded and (getattr(st, "enforces", None) or [])
    ]
    if on_trust:
        named = "; ".join(f"{n} ({len(e)} criterion/criteria)" for n, e in on_trust)
        print(
            f"{config_path}: {len(on_trust)} state(s) grade a CRITERION on the builder's "
            f"word, with nothing executed to contradict it: {named}. That is on the record "
            f"here and in the criteria-coverage report; give each one a `rederives:` entry "
            f"or a validator when one can exist.",
            file=sys.stderr,
        )

    # A `derives:` with no validator names a field nothing computes. `self_graded: true` does
    # not excuse it: that flag admits the MODEL's claim is trusted, whereas a derive asserts the
    # HARNESS supplies the value -- so the escape hatch for one says nothing about the other,
    # and the consumer left holding the missing field is another agent in another process.
    underived = [st.name for st in (wl.states or []) if (getattr(st, "derives", None) or []) and not st.verify]
    if underived:
        raise ValueError(
            f"{config_path}: worklist state(s) {underived} declare `derives:` but no `verify:` "
            f"-- nothing would compute those fields. A derive is a promise to whoever consumes "
            f"the delivered record; name the validator that keeps it."
        )


def load_run_spec(
    config_path: str | Path,
    *,
    instances: str | None = None,
    workspace_root: str | None = None,
    output_root: str | None = None,
    parallelism: int = 1,
    isolation: str = "in_process",
    cli_overrides: dict | None = None,
) -> RunSpec:
    agent = load_agent_spec(config_path, cli_overrides)
    kw: dict[str, Any] = {"agent": agent, "parallelism": parallelism, "isolation": isolation}
    if instances is not None:
        kw["instances"] = instances
    if workspace_root is not None:
        kw["workspace_root"] = workspace_root
    if output_root is not None:
        kw["output_root"] = output_root
    return RunSpec.model_validate(kw)
