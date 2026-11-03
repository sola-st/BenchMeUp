"""LLM-judge oracle — tri-state verify-then-finish gate.

Used as `oracle.grade` (an in-process judge hook) or as a proxy. When the agent
declares completion, the judge is queried with the recent run evidence (history +
working set + the expected-output guidance from spec.target) and must answer with
a final verdict that maps to one of three states:

  ACCEPT          : success — stop the loop.
  REJECT_CONTINUE : not done yet — inject `.message` back and keep going.
  FAIL            : unrecoverable — stop, not a success.

Verdict parsing is robust (adapted from analysis_agent.agent._parse_final_verdict):
it scans bottom-up for a "FINAL VERDICT" / "VERDICT" marker and reads the first
verdict token after it, tolerant to punctuation/case/underscores/dashes.

This module exposes:
  build(spec)      -> LLMJudgeOracle              (Oracle with .check(ctx))
  judge(ctx)       -> OracleResult                (plain hook usable as oracle.grade)
"""

from __future__ import annotations

import sys
from typing import Any

from agentcore.oracle.base import JudgeVerdict, OracleResult

# Tokens we map to each tri-state verdict.
_ACCEPT_TOKENS = {"accept", "yes", "pass", "success", "succeeded", "approve", "approved"}
# The fallback scan takes the last line that IS a verdict, so a justification ending "No
# regressions were found." is not read as FAIL. A bare "no" on its own line still decides;
# see parse_verdict.
_FAIL_TOKENS = {"fail", "failed", "abort", "reject", "rejected", "no"}
_CONTINUE_TOKENS = {"continue", "reject_continue", "rejectcontinue", "retry", "incomplete"}

_MARKERS = ("finalverdict", "verdict", "decision")

_DEFAULT_SYSTEM = (
    "You are a strict validator. You decide whether the agent has genuinely "
    "accomplished the task, using only the supplied evidence. Be skeptical of "
    "unsupported claims of success. Respond with a short justification, then a "
    "single final line of the exact form:\n"
    "FINAL VERDICT: ACCEPT   (task is genuinely complete)\n"
    "FINAL VERDICT: REJECT_CONTINUE   (not done; more work could finish it)\n"
    "FINAL VERDICT: FAIL   (task cannot be completed / is unrecoverable)\n"
    "When unsure, prefer REJECT_CONTINUE."
)


def _pick_model(ctx) -> Any | None:
    models = getattr(ctx, "models", {}) or {}
    for key in ("validator", "judge", "main"):
        if key in models:
            return models[key]
    # last resort: any model
    return next(iter(models.values()), None)


def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    while s and s[0] in "\"'`:=-":
        s = s[1:].strip()
    return s


def _tokenize(tail: str) -> list[str]:
    tokens: list[str] = []
    cur: list[str] = []
    for ch in tail:
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def parse_verdict(text: str) -> JudgeVerdict | None:
    """Map validator free-text to ACCEPT / REJECT_CONTINUE / FAIL (or None).

    Scans bottom-up for a verdict marker line; the first recognized verdict token
    after the marker wins. Falls back to a whole-document token scan if no marker
    line is present.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("—", "-").replace("–", "-")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    def classify(tokens: list[str]) -> JudgeVerdict | None:
        for t in tokens:
            low = t.lower()
            if low in _CONTINUE_TOKENS:
                return "REJECT_CONTINUE"
            if low in _ACCEPT_TOKENS:
                return "ACCEPT"
            if low in _FAIL_TOKENS:
                # "reject" alone (without continue) reads as FAIL here
                return "FAIL"
        return None

    # 1) marker-anchored scan, bottom-up
    for ln in reversed(lines):
        low = ln.lower()
        compact = low.replace(" ", "").replace("_", "").replace("-", "")
        if not any(m in compact for m in _MARKERS):
            continue
        marker_pos = max(low.rfind("verdict"), low.rfind("decision"))
        tail = low[marker_pos:] if marker_pos != -1 else low
        tail = _strip_wrapping_quotes(tail)
        # drop the marker word itself
        for m in ("verdict", "decision", "final"):
            tail = tail.replace(m, " ")
        v = classify(_tokenize(tail))
        if v is not None:
            return v

    # 2) fallback: a line that IS a verdict, not one that merely contains a token. Taking
    # any containing line would make "No regressions were found." a FAIL.
    for ln in reversed(lines):
        toks = _tokenize(_strip_wrapping_quotes(ln.lower().strip()))
        if len(toks) != 1:
            continue
        v = classify(toks)
        if v is not None:
            return v
    return None


def _build_evidence(ctx, spec) -> str:
    parts: list[str] = []

    target = dict(getattr(spec, "target", {}) or {}) if spec is not None else {}
    expected = (
        target.get("expected_output")
        or target.get("expected")
        or target.get("expected_output_example")
        or target.get("guidance")
    )
    if expected:
        parts.append(
            "Expected output guidance / example (the real output may differ; be flexible):\n"
            "<<<EXPECTED_OUTPUT_EXAMPLE>>>\n"
            f"{expected}\n"
            "<<<END_EXPECTED_OUTPUT_EXAMPLE>>>"
        )

    # task / instance summary
    agent = getattr(ctx, "agent", None)
    task = getattr(agent, "task", "") if agent is not None else ""
    if task:
        parts.append(f"Task:\n{task}")

    instance = getattr(ctx, "instance", {}) or {}
    if instance:
        keys = ("id", "instance_id", "name", "project", "repo")
        summary = {k: instance[k] for k in keys if k in instance}
        if summary:
            parts.append("Instance: " + ", ".join(f"{k}={v}" for k, v in summary.items()))

    # recent action log (command -> summary)
    history = getattr(agent, "action_log", None) if agent is not None else None
    if history:
        recent = history[-12:]
        rendered = "\n".join(f"- {cmd}\n  -> {summ}" for cmd, summ in recent)
        parts.append("Recent executed actions (summary):\n" + rendered)

    # working set snapshot
    ws = getattr(ctx, "working_set", None)
    if ws is not None:
        try:
            snap = ws.snapshot()
            if snap:
                import json

                parts.append("Working set (current state):\n" + json.dumps(snap, indent=2, default=str)[:4000])
        except Exception as e:  # noqa: BLE001
            # A JUDGE MUST KNOW WHAT IT COULD NOT SEE. Swallowing a snapshot failure silently
            # would remove the run's entire state from the prompt, and a judge told nothing
            # about the state may ACCEPT on the strength of not having seen it.
            parts.append(
                f"Working set (current state): UNAVAILABLE -- it could not be read "
                f"({type(e).__name__}: {e}). Judge accordingly: you are missing the "
                f"run's own state, so do not treat its absence as evidence of "
                f"anything."
            )

    return "\n\n".join(p for p in parts if p).strip()


def _judge_with_spec(ctx, spec) -> OracleResult:
    model = _pick_model(ctx)
    if model is None:
        # no judge model available -> cannot accept; ask to continue (fail-open to loop)
        return OracleResult(
            success=False,
            done=False,
            verdict="REJECT_CONTINUE",
            message="No validator/judge/main model available to grade completion.",
            reason="no judge model",
        )

    evidence = _build_evidence(ctx, spec)
    user_msg = (
        "Evidence from the run:\n"
        "<<<EVIDENCE>>>\n"
        f"{evidence or '(no evidence captured)'}\n"
        "<<<END_EVIDENCE>>>\n\n"
        "Question: Is there clear evidence that the task was genuinely and "
        "correctly accomplished? Decide ACCEPT, REJECT_CONTINUE, or FAIL and end "
        "with a single 'FINAL VERDICT: <one of ACCEPT|REJECT_CONTINUE|FAIL>' line."
    )
    messages = [
        {"role": "system", "content": _DEFAULT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = model.query(messages)
    except Exception as e:  # model/network failure -> keep looping rather than crash
        return OracleResult(
            success=False,
            done=False,
            verdict="REJECT_CONTINUE",
            message=f"Validator call failed ({e}); continue and try to finish the task.",
            reason=f"judge query error: {e}",
        )

    # charge cost if a budget controller is reachable
    budget = getattr(ctx, "budget", None)
    if budget is not None and hasattr(budget, "note_cost"):
        try:
            budget.note_cost(getattr(resp, "cost", 0.0))
        except Exception as e:  # noqa: BLE001
            # The judge's own calls fire most at the END of a run, so silently losing them
            # would under-count exactly the spend nearest the cap.
            print(
                f"judge cost not recorded on the meter ({type(e).__name__}: {e}) -- the "
                f"run's spend is understated by this call",
                file=sys.stderr,
            )

    raw = (getattr(resp, "content", "") or "").strip()
    verdict = parse_verdict(raw)

    if verdict == "ACCEPT":
        return OracleResult(
            success=True,
            done=True,
            verdict="ACCEPT",
            reason="validator accepted completion",
            message="",
            sub_signals={"judge_raw": raw[:2000]},
        )
    if verdict == "FAIL":
        return OracleResult(
            success=False,
            done=True,
            verdict="FAIL",
            reason="validator judged the task unrecoverable",
            message="",
            sub_signals={"judge_raw": raw[:2000]},
        )
    # REJECT_CONTINUE or inconclusive -> fail-closed to "continue"
    msg = (
        "SUCCESS VALIDATION FAILED: the evidence does not clearly show the task "
        "is complete. Keep working and gather/produce the missing evidence before "
        "declaring completion again.\nValidator notes:\n" + raw
    )[:2000]
    return OracleResult(
        success=False,
        done=False,
        verdict="REJECT_CONTINUE",
        reason="validator requested continuation" + ("" if verdict == "REJECT_CONTINUE" else " (inconclusive verdict)"),
        message=msg,
        sub_signals={"judge_raw": raw[:2000]},
    )


class LLMJudgeOracle:
    """Tri-state LLM judge. Oracle protocol: .check(ctx) -> OracleResult."""

    def __init__(self, spec=None):
        self.spec = spec

    def check(self, ctx) -> OracleResult:
        return _judge_with_spec(ctx, self.spec)

    # callable form so it can also be used directly as an oracle.grade hook
    def __call__(self, ctx) -> OracleResult:
        return self.check(ctx)


def judge(ctx) -> OracleResult:
    """Plain judge hook usable as `oracle.grade` (no OracleSpec target).

    Reads spec target via ctx.agent.spec.oracle if available, else judges from
    history/working-set evidence alone.
    """
    spec = None
    agent = getattr(ctx, "agent", None)
    if agent is not None:
        spec = getattr(getattr(agent, "spec", None), "oracle", None)
    return _judge_with_spec(ctx, spec)


def build(spec) -> LLMJudgeOracle:
    """Build an LLMJudgeOracle from an OracleSpec."""
    return LLMJudgeOracle(spec)


build._is_oracle_builder = True  # type: ignore[attr-defined]
