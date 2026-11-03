"""AttemptController + Carryover — opt-in cross-attempt orchestration.

Two mutually-exclusive modes (DESIGN.md section E):

  * **Sampling (pass@k):** when ``spec.sampling`` is set, run ``n`` *independent*
    attempts (each ``make_agent(None).run()`` — no carryover) and pick one via
    ``select`` (first_success | best_metric | majority | judge).
  * **Retry-with-learning:** otherwise loop up to ``max_attempts``, threading a
    typed :class:`Carryover` between attempts. Return the first successful
    :class:`RunResult`; on failure fold the result into the carryover via
    ``summarize_attempt`` (default: append a one-line lesson) and try again.

``Carryover`` is the *inter-attempt* channel (retry-learning text + artifact
refs). It is deliberately distinct from the ``WorkingSet`` (intra-run typed
accumulator) and ``StatePersistence`` (crash snapshot).

Container policy (``spec.container_policy``: keep|reset|llm) is **not** enforced
here: resetting / reprovisioning the environment between attempts is the
responsibility of the ``make_agent`` closure, which owns environment
construction. This controller only decides *whether* to carry learning forward
and *which* attempt to return.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

from pydantic import BaseModel

from agentcore.agent.loop import LoopAgent, RunResult
from agentcore.config.schema import AttemptsSpec

# ── carryover ─────────────────────────────────────────────────────────────────


class Carryover(BaseModel):
    """Typed learning threaded across retry attempts.

    ``effective_prefix`` / ``effective_commands`` capture a proven recipe (build
    bring-up); ``artifacts`` references checkpoint / directory handles (a fuzzing
    corpus, a working diff) produced by a prior attempt and resolvable by the
    ``make_agent`` closure.
    """

    lessons: list[str] = []
    completed_stages: list[str] = []
    effective_commands: dict[str, list[str]] = {}  # stage -> commands
    effective_prefix: list[str] = []  # single-loop ordered recipe
    artifacts: dict[str, str] = {}  # name -> checkpoint/dir handle


# ── default summarizer ─────────────────────────────────────────────────────────


def _default_summarize_attempt(res: RunResult, carry: Carryover) -> Carryover:
    """Fold a failed attempt into the carryover by appending a terse lesson.

    Returns a *new* Carryover (does not mutate the input) so callers can keep the
    prior object around for diagnostics.
    """
    lesson = f"Attempt failed: {res.reason or 'no reason given'} (after {res.steps} steps)."
    return carry.model_copy(update={"lessons": [*carry.lessons, lesson]})


# ── score extraction (best_metric) ─────────────────────────────────────────────


def _metric(res: RunResult) -> float:
    """Numeric score for ``best_metric`` selection.

    Looks for a ``score`` key in the working_set first, then any numeric
    working_set value; falls back to success-as-1.0. Non-numeric / missing -> 0.
    """
    ws = res.working_set or {}
    if isinstance(ws, dict):
        if isinstance(ws.get("score"), (int, float)):
            return float(ws["score"])
        for v in ws.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return 1.0 if res.success else 0.0


# ── controller ──────────────────────────────────────────────────────────────────


class AttemptController:
    """Outer retry-with-learning OR independent sampling (pass@k).

    Args:
        spec: the :class:`AttemptsSpec` (drives sampling vs retry).
        make_agent: ``Callable[[Carryover | None], LoopAgent]``. Given a
            ``Carryover`` (retry mode) or ``None`` (sampling / first attempt), it
            constructs a fresh agent. The closure owns environment construction
            and is therefore responsible for honouring ``spec.container_policy``
            (keep / reset / llm) and ``spec.verify_in_clean_env``.
        summarize_attempt: optional hook ``(RunResult, Carryover) -> Carryover``
            folding a failed attempt into learning. Defaults to appending a
            single lesson line.
        max_attempts: retry-mode cap (ignored in sampling mode).
        judge: optional ``Callable[[list[RunResult]], RunResult]`` used by the
            ``judge`` selection strategy. If absent, ``judge`` falls back to
            ``first_success``.
    """

    def __init__(
        self,
        spec: AttemptsSpec,
        make_agent: Callable[[Carryover | None], LoopAgent],
        summarize_attempt: Callable[[RunResult, Carryover], Carryover] | None = None,
        max_attempts: int = 1,
        judge: Callable[[list[RunResult]], RunResult] | None = None,
    ):
        self.spec = spec
        self.make_agent = make_agent
        self.summarize_attempt = summarize_attempt or _default_summarize_attempt
        self.max_attempts = max(1, int(max_attempts))
        self.judge = judge

    # ── public ────────────────────────────────────────────────────────────────
    def run(self) -> RunResult:
        if self.spec.sampling is not None:
            return self._run_sampling(self.spec.sampling.n, self.spec.sampling.select)
        return self._run_retry()

    # ── one attempt ─────────────────────────────────────────────────────────────
    def _one_attempt(self, carry: Carryover | None) -> RunResult:
        agent = self.make_agent(carry)
        return agent.run()

    # ── retry-with-learning ─────────────────────────────────────────────────────
    def _run_retry(self) -> RunResult:
        carry = Carryover()
        last: RunResult | None = None
        for _ in range(self.max_attempts):
            res = self._one_attempt(carry)
            last = res
            if res.success:
                return res
            carry = self.summarize_attempt(res, carry)
        # exhausted: return the most recent attempt (carries the final reason)
        return last if last is not None else RunResult(success=False, reason="no attempts run")

    # ── sampling (pass@k) ────────────────────────────────────────────────────────
    def _run_sampling(self, n: int, select: str) -> RunResult:
        n = max(1, int(n))
        if select == "first_success":
            # short-circuit: stop at the first success to save budget
            last: RunResult | None = None
            for _ in range(n):
                res = self._one_attempt(None)
                last = res
                if res.success:
                    return res
            return last if last is not None else RunResult(success=False, reason="no samples")
        samples = [self._one_attempt(None) for _ in range(n)]
        return self.select(samples, select)

    # ── selection strategies ─────────────────────────────────────────────────────
    def select(self, samples: list[RunResult], how: str) -> RunResult:
        if not samples:
            return RunResult(success=False, reason="no samples")
        if how == "first_success":
            return next((s for s in samples if s.success), samples[0])
        if how == "best_metric":
            return max(samples, key=_metric)
        if how == "majority":
            return self._majority(samples)
        if how == "judge":
            if self.judge is not None:
                return self.judge(samples)
            # no judge provided: degrade to first_success
            return next((s for s in samples if s.success), samples[0])
        # unknown strategy: be conservative, prefer a success
        return next((s for s in samples if s.success), samples[0])

    @staticmethod
    def _majority(samples: list[RunResult]) -> RunResult:
        """Pick the modal submitted answer; tie-break toward successes.

        Votes over a stable signature of ``submitted`` (falling back to
        ``success``). Returns a representative RunResult for the winning bucket.
        """

        def sig(s: RunResult) -> str:
            if s.submitted is not None:
                try:
                    import json

                    return json.dumps(s.submitted, sort_keys=True, default=str)
                except Exception:
                    return repr(s.submitted)
            return f"success={s.success}"

        counts = Counter(sig(s) for s in samples)
        winner, _ = counts.most_common(1)[0]
        bucket = [s for s in samples if sig(s) == winner]
        # within the winning bucket, prefer a successful representative
        return next((s for s in bucket if s.success), bucket[0])
