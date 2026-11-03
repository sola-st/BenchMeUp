"""Submission oracle — extract a per-instance prediction into an aggregate file.

At submit time this runs `spec.submission.command` inside the environment, captures
its stdout, and merges that into a benchmark-shaped aggregate JSON file
(`spec.submission.aggregate_file`) keyed by `spec.submission.key` (or the instance
id). The aggregate is the join surface a downstream harness (SWE-bench preds.json,
SARIF, etc.) consumes; an external grader scores it later.

This oracle never decides success itself (done=False, success=True meaning
"submitted ok"); grading is deferred. The aggregate file is written on the host so
it accumulates across instances; the captured stdout is also stashed in
ctx.working_set / ctx.scratch for downstream tools.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentcore.oracle.base import OracleResult


class SubmissionOracle:
    """Run a submit command in-env, capture stdout, update an aggregate file."""

    def __init__(self, spec):
        self.spec = spec
        self.submission = getattr(spec, "submission", None)

    def check(self, ctx) -> OracleResult:
        sub = self.submission
        if sub is None:
            return OracleResult(
                success=False,
                done=False,
                reason="submission oracle: spec.submission is not set",
            )

        key = self._resolve_key(ctx)
        captured = ""
        returncode = 0
        if getattr(sub, "command", None):
            res = ctx.env.execute(sub.command, cwd=ctx.scratch.get("cwd", "") if ctx.scratch else "")
            captured = getattr(res, "output", "") or ""
            stderr = getattr(res, "stderr", "") or ""
            returncode = int(getattr(res, "returncode", 0) or 0)
            if not captured.strip() and stderr.strip():
                captured = stderr
        else:
            # no command: fall back to a stashed prediction if a tool wrote one
            captured = self._fallback_prediction(ctx)

        value = self._coerce(captured)

        aggregate_path = self._resolve_aggregate_path(ctx, sub)
        written_to = None
        if aggregate_path:
            written_to = self._update_aggregate(aggregate_path, key, value)

        # stash for downstream/grade hooks
        if ctx.scratch is not None:
            ctx.scratch.setdefault("submissions", {})[key] = value

        sub_signals: dict[str, Any] = {
            "submitted": {key: value} if not isinstance(value, (dict, list)) else value,
            "submission_key": key,
            "returncode": returncode,
        }
        if written_to:
            sub_signals["aggregate_file"] = written_to

        return OracleResult(
            success=True,
            done=False,
            reason=f"submitted prediction for {key!r}" + (f" -> {written_to}" if written_to else ""),
            sub_signals=sub_signals,
        )

    # ── helpers ────────────────────────────────────────────────────────────────
    def _resolve_key(self, ctx) -> str:
        sub = self.submission
        if sub is not None and getattr(sub, "key", None):
            return str(sub.key)
        inst = getattr(ctx, "instance", {}) or {}
        for k in ("instance_id", "id", "name", "task_id"):
            if inst.get(k):
                return str(inst[k])
        return "default"

    def _resolve_aggregate_path(self, ctx, sub) -> str | None:
        path = getattr(sub, "aggregate_file", None)
        if not path:
            return None
        if os.path.isabs(path):
            return path
        # resolve relative paths against a host out_dir if the run provided one
        base = None
        if ctx.scratch:
            base = ctx.scratch.get("out_dir") or ctx.scratch.get("output_dir")
        if base:
            return os.path.join(str(base), path)
        return os.path.abspath(path)

    @staticmethod
    def _coerce(text: str) -> Any:
        """Parse captured stdout as JSON if possible, else keep stripped text."""
        s = (text or "").strip()
        if not s:
            return ""
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            return s

    def _fallback_prediction(self, ctx) -> str:
        ws = getattr(ctx, "working_set", None)
        if ws is not None:
            for attr in ("prediction", "patch", "submission", "answer"):
                val = getattr(ws, attr, None)
                if val:
                    return val if isinstance(val, str) else json.dumps(val)
        if ctx.scratch:
            for attr in ("prediction", "patch", "submission", "answer"):
                if ctx.scratch.get(attr):
                    v = ctx.scratch[attr]
                    return v if isinstance(v, str) else json.dumps(v)
        return ""

    @staticmethod
    def _update_aggregate(path: str, key: str, value: Any) -> str:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data: dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    data = loaded
            except (ValueError, OSError):
                data = {}
        data[key] = value
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, path)
        return path


def build(spec) -> SubmissionOracle:
    """Build a SubmissionOracle from an OracleSpec."""
    return SubmissionOracle(spec)


build._is_oracle_builder = True  # type: ignore[attr-defined]
