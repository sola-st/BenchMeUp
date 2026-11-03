"""Deterministic proxy oracle.

Runs a command in the environment and grades the result with cheap, in-process
checks: return code, artifact presence (glob), and an optional numeric metric
extracted via regex compared against a threshold. Covers build / test / coverage
style checks where success is decidable without an LLM.

OracleSpec.target keys honored (all optional):
  command          : str    — command to run in ctx.env (else uses last exec output? no — required for a real check)
  cwd              : str     — working dir for the command
  timeout          : int     — per-command timeout (s)
  expect_returncode: int     — required return code (default 0)
  artifact_glob    : str     — glob (run in-env) that must match >=1 file
  metric_regex     : str     — regex with one capture group -> float metric
  threshold        : float   — metric must be >= threshold (or compare via `comparator`)
  comparator       : str     — one of ">=","<=",">","<","==" (default ">=")
  search_stderr    : bool    — also scan stderr for metric_regex (default True)
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from agentcore.oracle.base import OracleResult

_COMPARATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


class DeterministicOracle:
    """Run a command in-env and grade it deterministically."""

    def __init__(self, spec):
        self.spec = spec
        self.target: dict[str, Any] = dict(getattr(spec, "target", {}) or {})

    def check(self, ctx) -> OracleResult:
        t = self.target
        command = t.get("command")
        if not command:
            return OracleResult(success=False, done=False, reason="deterministic oracle: no 'command' in target")

        cwd = t.get("cwd", "") or ""
        timeout = t.get("timeout")
        exec_res = ctx.env.execute(command, cwd=cwd, timeout=timeout)

        returncode = int(getattr(exec_res, "returncode", 0) or 0)
        output = getattr(exec_res, "output", "") or ""
        stderr = getattr(exec_res, "stderr", "") or ""
        timed_out = bool(getattr(exec_res, "timed_out", False))

        sub_signals: dict[str, Any] = {
            "returncode": returncode,
            "timed_out": timed_out,
        }
        reasons: list[str] = []
        ok = True

        # 1) return code
        expect_rc = int(t.get("expect_returncode", 0))
        rc_ok = (returncode == expect_rc) and not timed_out
        sub_signals["returncode_ok"] = rc_ok
        if not rc_ok:
            ok = False
            reasons.append(f"returncode {returncode} != expected {expect_rc}" + (" (timed out)" if timed_out else ""))

        # 2) artifact glob (evaluated in-env)
        artifact_glob = t.get("artifact_glob")
        if artifact_glob:
            matched = self._glob_in_env(ctx, artifact_glob, cwd, timeout)
            sub_signals["artifact_matches"] = matched
            art_ok = matched > 0
            sub_signals["artifact_ok"] = art_ok
            if not art_ok:
                ok = False
                reasons.append(f"no files matched artifact_glob {artifact_glob!r}")

        # 3) numeric metric via regex
        score: float | None = None
        metric_regex = t.get("metric_regex")
        if metric_regex:
            haystack = output
            if t.get("search_stderr", True):
                haystack = output + "\n" + stderr
            score = self._extract_metric(metric_regex, haystack)
            sub_signals["metric"] = score
            threshold = t.get("threshold")
            if threshold is not None:
                comparator = t.get("comparator", ">=")
                cmp_fn = _COMPARATORS.get(comparator, _COMPARATORS[">="])
                if score is None:
                    ok = False
                    reasons.append(f"metric_regex {metric_regex!r} did not match output")
                else:
                    metric_ok = cmp_fn(score, float(threshold))
                    sub_signals["metric_ok"] = metric_ok
                    if not metric_ok:
                        ok = False
                        reasons.append(f"metric {score} not {comparator} threshold {threshold}")

        reason = "all deterministic checks passed" if ok else "; ".join(reasons)
        return OracleResult(
            success=ok,
            done=ok,
            score=score,
            sub_signals=sub_signals,
            reason=reason,
        )

    # ── helpers ────────────────────────────────────────────────────────────────
    def _glob_in_env(self, ctx, pattern: str, cwd: str, timeout) -> int:
        """Count files matching a shell glob, evaluated inside the environment."""
        # `ls -d <glob>` prints matches (one per line); empty / error => 0.
        quoted = pattern  # globs must stay unquoted to expand in the shell
        cmd = f"sh -c {shlex.quote(f'ls -d -- {quoted} 2>/dev/null | wc -l')}"
        try:
            res = ctx.env.execute(cmd, cwd=cwd, timeout=timeout)
            out = (getattr(res, "output", "") or "").strip().splitlines()
            for line in reversed(out):
                line = line.strip()
                if line.isdigit():
                    return int(line)
        except Exception:
            pass
        return 0

    @staticmethod
    def _extract_metric(pattern: str, text: str) -> float | None:
        try:
            rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error:
            return None
        last: float | None = None
        for m in rx.finditer(text or ""):
            grp = m.group(1) if m.groups() else m.group(0)
            try:
                last = float(grp)
            except (TypeError, ValueError):
                continue
        return last


def build(spec) -> DeterministicOracle:
    """Build a DeterministicOracle from an OracleSpec."""
    return DeterministicOracle(spec)


build._is_oracle_builder = True  # type: ignore[attr-defined]
