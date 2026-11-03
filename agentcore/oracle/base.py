"""Oracle result types + the resolver that builds an Oracle from an OracleSpec.

An Oracle decides success/progress. Three conceptual slots (see OracleSpec in
config.schema):
  - proxy:      run after each execute, in-env, returns a structured OracleResult.
  - submission: extract a prediction at submit time into an aggregate file.
  - grade:      post-run judge (in-process) or external-join spec.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal

from pydantic import BaseModel

JudgeVerdict = Literal["ACCEPT", "REJECT_CONTINUE", "FAIL"]


class OracleResult(BaseModel):
    success: bool = False
    done: bool = False  # stop the loop now (verify-then-finish)
    score: float | None = None  # metric value (coverage %, mutation score)
    sub_signals: dict[str, Any] = {}  # vector: built / coverage / crash_buckets / ...
    reason: str = ""
    # llm_judge tri-state; "" means not a judge
    verdict: JudgeVerdict | Literal[""] = ""
    message: str = ""  # for REJECT_CONTINUE: what to inject back


def _resolve(path: str):
    """Resolve 'module:attr' or a builtin oracle name to a callable/Oracle."""
    if ":" in path:
        mod, _, attr = path.partition(":")
        return getattr(importlib.import_module(mod), attr)
    # builtin oracle keys
    from agentcore.oracle import deterministic, llm_judge, submission  # noqa

    builtins = {
        "deterministic": deterministic.build,
        "llm_judge": llm_judge.build,
        "submission": submission.build,
    }
    if path in builtins:
        return builtins[path]
    raise ValueError(f"Unknown oracle reference: {path!r}")


def build_oracle(spec) -> "Oracle | None":
    """Build a proxy Oracle from an OracleSpec (or None if kind=none/sentinel)."""
    from agentcore.config.schema import OracleSpec  # local import avoids cycle

    if not isinstance(spec, OracleSpec):
        return None
    # An explicit proxy is honored regardless of kind (so kind='sentinel' + proxy works).
    if spec.proxy:
        fn = _resolve(spec.proxy)
        # hook(ctx)->OracleResult  OR  build(spec)->Oracle
        return fn(spec) if getattr(fn, "_is_oracle_builder", False) else _HookOracle(fn, spec)
    if spec.kind in ("none", "sentinel"):
        return None
    # kind without explicit proxy: dispatch to builtin by kind name
    return _resolve(spec.kind)(spec)


class _HookOracle:
    """Wraps a plain `def hook(ctx) -> OracleResult` into the Oracle protocol."""

    def __init__(self, fn, spec):
        self._fn = fn
        self._spec = spec

    def check(self, ctx) -> OracleResult:
        res = self._fn(ctx)
        return res if isinstance(res, OracleResult) else OracleResult(**(res or {}))


if False:  # pragma: no cover - typing only
    from agentcore.protocols import Oracle  # noqa
