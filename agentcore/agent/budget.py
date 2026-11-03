"""BudgetController — meters (steps / wallclock / cost / cpu / disk / test_runs).

Generalizes the legacy step/cost limits. Meters charge on configurable events,
and on exhaustion either abort, finish the current cycle, or fail-with-lesson.
`thresholds` couple budget consumption to control actions (e.g. force a state).
"""

from __future__ import annotations

import time
from typing import Callable

from agentcore.config.schema import BudgetSpec, Meter


class _MeterState:
    def __init__(self, spec: Meter, clock: Callable[[], float]):
        self.spec = spec
        self._clock = clock
        self._spent = 0.0
        self._start = clock()
        self.probe: Callable[[], float] | None = None  # for cpu_seconds/disk_bytes

    def charge(self, amount: float = 1.0) -> None:
        self._spent += amount

    def value(self) -> float:
        if self.spec.kind == "wallclock":
            return self._clock() - self._start
        if self.probe is not None:
            return self.probe()
        return self._spent

    def fraction(self) -> float:
        return self.value() / self.spec.limit if self.spec.limit else 0.0

    def over(self) -> bool:
        return self.value() >= self.spec.limit


class BudgetController:
    def __init__(self, spec: BudgetSpec, clock: Callable[[], float] = time.monotonic):
        self.spec = spec
        self._clock = clock
        self.meters = [_MeterState(m, clock) for m in spec.meters]
        self._fired_thresholds: set[int] = set()

    # ── charging ─────────────────────────────────────────────────────────────
    def on_event(self, event: str) -> None:
        """event in {'cycle','tool_call','stage'} — charge steps/test_runs meters."""
        for ms in self.meters:
            if ms.spec.kind in ("steps", "test_runs") and ms.spec.charge_on == event:
                ms.charge(1)

    def note_cost(self, amount: float) -> None:
        for ms in self.meters:
            if ms.spec.kind == "cost_usd":
                ms.charge(amount)

    def charge(self, kind: str, amount: float = 1.0) -> None:
        for ms in self.meters:
            if ms.spec.kind == kind:
                ms.charge(amount)

    def set_probe(self, kind: str, probe: Callable[[], float]) -> None:
        for ms in self.meters:
            if ms.spec.kind == kind:
                ms.probe = probe

    # ── evaluation ────────────────────────────────────────────────────────────
    def should_stop(self) -> tuple[bool, bool, str]:
        """Returns (stop, terminal, reason). terminal=True => raise BudgetExhausted."""
        for ms in self.meters:
            if ms.over():
                terminal = ms.spec.on_exhaust in ("abort", "fail_with_lesson")
                return True, terminal, f"{ms.spec.kind} budget reached ({ms.value():.0f}/{ms.spec.limit:.0f})"
        return False, False, ""

    def exhausted(self) -> bool:
        return self.should_stop()[0]

    def thresholds_triggered(self) -> list[dict]:
        """Return threshold actions newly crossed (by max meter fraction)."""
        frac = max((ms.fraction() for ms in self.meters), default=0.0)
        fired = []
        for i, th in enumerate(self.spec.thresholds):
            if i not in self._fired_thresholds and frac >= float(th.get("at", 1.0)):
                self._fired_thresholds.add(i)
                fired.append(th)
        return fired

    def status(self) -> dict:
        return {ms.spec.kind: {"value": round(ms.value(), 3), "limit": ms.spec.limit} for ms in self.meters}

    def status_text(self) -> str:
        """The same facts as `status()`, for a PROMPT rather than for `restore()`.

        Two consumers need two shapes. `status()` must stay a dict -- `restore()` rehydrates spent counters from it on
        resume/fork/replay -- so it cannot simply be made readable. Rendering that dict into a
        prompt would show Python punctuation to the model, so this gives the readable line.

        Never raises: a probe meter can fail, and the budget line must not be the thing that
        ends a cycle.
        """
        parts = []
        for ms in self.meters:
            try:
                parts.append(f"{ms.spec.kind}: {ms.value():.2f}/{ms.spec.limit:g}")
            except Exception:  # noqa: BLE001 - an unreadable meter is not a run-ending event
                parts.append(f"{ms.spec.kind}: unavailable")
        return " | ".join(parts) or "no budget meters"

    def restore(self, status: dict | None) -> None:
        """Rehydrate spent counters from a saved status() dict (resume/fork/replay).

        Counter meters (steps/cost/test_runs) get their spent value back; wallclock is
        rebased so the elapsed time resumes where it left off. Probe meters
        (cpu_seconds/disk_bytes) are re-measured live, so there is nothing to restore.
        """
        if not status:
            return
        for ms in self.meters:
            saved = status.get(ms.spec.kind)
            if not isinstance(saved, dict) or "value" not in saved:
                continue
            val = float(saved["value"])
            if ms.spec.kind == "wallclock":
                ms._start = self._clock() - val
            elif ms.probe is None:
                ms._spent = val
