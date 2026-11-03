"""Generic collector gates: the template tier's nine validators.py bodies, as ONE module.

WHY (no task-specific coded gates): per-collector validators would differ only in two
constants — which field names the candidate's source, and which gate the terminal
state composes — and the copies would drift. The constants arrive as instance
declarations:

    identity_field       the lead field naming the candidate's source; the exclusion
                         roster runs against it (spec content: which field that is)
    terminal_inner_gate  the name of the gate the terminal state already declared,
                         composed before the consumer session ("" means none)

and every agent's `verify:` points here — `agentcore.agent.collector_gates:<fn>` —
with no per-agent Python anywhere. A missing declaration FAILS CLOSED, never open.
This module knows no task's vocabulary; the conformance test in
test_the_field_contract_knows_no_task pins it alongside field_contract.
"""

from __future__ import annotations

import re

from agentcore.tools.variants import VARIANTS_GATE_READS


def _inst(ctx) -> dict:
    return getattr(ctx, "instance", {}) or {}


def _run_in(ctx, sandbox: str, cmd: str, timeout: int = 600, capture_limit: int | None = None):
    """Run a check in the SAME environment the evidence was produced in. Never guess.

    One candidate, one sandbox: `linux_terminal` and each `container_exec` sandbox are
    separate filesystems. `exec_in_sandbox` rebuilds the NAMED sandbox from its own
    recipe when this process has no live handle. A silent fallback to the agent's own
    environment would let a gate read the wrong filesystem, so a named sandbox that
    cannot be rebuilt is an error, not a quiet substitution.
    """
    if sandbox:
        try:
            from agentcore.tools import containers as _containers

            return _containers.exec_in_sandbox(
                ctx, sandbox, cmd, timeout=timeout, capture_limit=capture_limit, revive_removed=True
            )
        except Exception as e:  # noqa: BLE001
            return 127, (
                f"sandbox {sandbox!r} could not be reached or rebuilt "
                f"({type(e).__name__}: {e}). Its recipe is kept -- recreate it "
                f"with container_create and redo the step there. Do not redo it "
                f"with linux_terminal; that is a different filesystem."
            )
    if ctx is None or getattr(ctx, "env", None) is None:
        return 127, "no environment available"
    r = ctx.env.execute(cmd, timeout=timeout)
    return r.returncode, (r.output or "") + (r.stderr or "")


def candidate_is_admissible(fields: dict, ctx) -> tuple[bool, str]:
    """Enforce whatever THIS spec says is excluded, before anything expensive happens.

    The field carrying the source identity is DECLARED (`identity_field`), because a
    hardwired spelling can disagree with the vocabulary the spec uses."""
    from agentcore.agent.exclusions import check_candidate

    fname = str(_inst(ctx).get("identity_field") or "").strip()
    if not fname:
        return False, (
            "HARNESS/SPEC FAULT: this task's instance declares no "
            "`identity_field`, so admissibility cannot run. Declare which "
            "lead field names the candidate's source; an undeclared identity "
            "is a missing rule, not a permissive one."
        )
    name = str(fields.get(fname) or "").strip()
    if not name:
        return False, (f"record {fname} -- the real, public source this candidate came from")
    return check_candidate(name, _inst(ctx))


def sandbox_is_real(fields: dict, ctx) -> tuple[bool, str]:
    """One candidate, one container that actually exists -- checked, not requested."""
    from agentcore.agent.sandboxes import sandbox_is_exclusive

    return sandbox_is_exclusive(fields, ctx)


def oracle_holds(fields: dict, ctx) -> tuple[bool, str]:
    """FAILS CLOSED until the benchmark declares or implements its real oracle.

    The two shape checks are genuinely generic preconditions (they catch a model
    recording "the test failed" instead of the output that shows it); the final refusal
    is deliberate — a benchmark whose oracle was never checked is worse than none."""
    evidence = str(fields.get("oracle_evidence") or "").strip()
    if len(evidence) < 80:
        return False, (
            "record oracle_evidence: the actual output you observed when you "
            "exercised the oracle -- the command and what it printed, not a "
            "description of it."
        )
    if not re.search(r"[\n\r]|exit|Tests run|PASS|FAIL|error|\d{3,}", evidence, re.IGNORECASE):
        return False, (
            "oracle_evidence reads like a summary rather than real output. Paste what the command actually printed."
        )
    return False, (
        "this benchmark's oracle check is NOT DECLARED. State it in the spec so the "
        "harness re-reads this evidence and decides for itself -- parse the report, "
        "diff the bytes, compare the timings, whatever admissibility means here. Until "
        "then no candidate can be delivered, and that is deliberate: a benchmark whose "
        "oracle was never checked is worse than no benchmark."
    )


def instance_is_complete(fields: dict, ctx) -> tuple[bool, str]:
    """Enough for a third party to rebuild the environment and reproduce the result."""
    recipe = str(fields.get("env_recipe") or "").strip()
    if not recipe:
        return False, (
            "record env_recipe: the base image and the exact tool versions "
            "this was validated in -- a real image:tag plus the versions that "
            "matter."
        )
    if not re.search(r"[A-Za-z][A-Za-z0-9._/-]*[:=][A-Za-z0-9._-]+", recipe):
        return False, (
            "env_recipe must name a real image and version, not a description. A prose phrase is not reproducible."
        )
    return True, "complete and reproducible"


def variants_are_materialized(fields: dict, ctx) -> tuple[bool, str]:
    """The generic materialisation gate, over the shared sandbox plumbing."""
    from agentcore.tools.variants import variants_are_materialized as _gate

    return _gate(fields, ctx, lambda sandbox, cmd, timeout: _run_in(ctx, sandbox, cmd, timeout))


variants_are_materialized.reads = VARIANTS_GATE_READS  # a re-record of these re-runs the gate


def consumer_session_runs(fields: dict, ctx) -> tuple[bool, str]:
    """Rehearse the CONSUMER's own commands in this candidate's sandbox before
    delivery, skipping the rebuild/check steps the variants gate re-runs already."""
    from agentcore.tools.session import consumer_session_runs as _gate

    return _gate(fields, ctx, lambda sandbox, cmd, timeout: _run_in(ctx, sandbox, cmd, timeout))


def admissible_and_materialized(fields: dict, ctx) -> tuple[bool, str]:
    """Both generic gates on ONE state -- admissibility first, because it is cheap and
    refuses an excluded project before the variants gate pays for rebuilds."""
    ok, detail = candidate_is_admissible(fields, ctx)
    if not ok:
        return ok, detail
    return variants_are_materialized(fields, ctx)


admissible_and_materialized.reads = VARIANTS_GATE_READS


def materialized_and_session(fields: dict, ctx) -> tuple[bool, str]:
    """Both obligations on the state that carries variants AND ends the lifecycle;
    neither may quietly replace the other."""
    ok, detail = variants_are_materialized(fields, ctx)
    if not ok:
        return ok, detail
    return consumer_session_runs(fields, ctx)


materialized_and_session.reads = VARIANTS_GATE_READS


def terminal_gate(fields: dict, ctx) -> tuple[bool, str]:
    """Whatever this lifecycle already checked at delivery, plus the consumer session.

    COMPOSES rather than pattern-matches; the inner gate's name is DECLARED
    (`terminal_inner_gate`), resolved against this module, and an unknown name is a
    loud harness error rather than a skipped check."""
    inner_name = str(_inst(ctx).get("terminal_inner_gate") or "").strip()
    if inner_name:
        inner = globals().get(inner_name)
        if inner is None or not callable(inner):
            return False, (
                f"HARNESS ERROR: the declared terminal_inner_gate "
                f"{inner_name!r} is not a gate this module defines, so "
                f"nothing can be checked until the declaration is fixed."
            )
        ok, detail = inner(fields, ctx)
        if not ok:
            return ok, detail
    return consumer_session_runs(fields, ctx)
