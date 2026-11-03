"""Lifecycle events — observability attaches here instead of monkey-patching the loop.

The LoopAgent emits enumerated events; listeners (trajectory writer, cost meter,
custom hooks) subscribe. No monkey-patching of the cycle function.
"""

from __future__ import annotations

from typing import Callable

# Enumerated lifecycle events (documented contract):
LIFECYCLE_EVENTS = (
    "on_run_start",
    "on_attempt_start",
    "on_cycle_start",
    "on_prompt_sent",
    "on_response",
    "on_tool_result",
    "on_cycle_end",
    "on_attempt_end",
    "on_finish",
    "on_run_end",
)

# Framework-DECISION events — the harness's own choices, not the model's turns.
# These power the report's "why" layer: how an action was parsed, why a stage
# advanced, how tool output was reduced, and what an oracle/judge decided.
DECISION_EVENTS = (
    "on_parse_path",  # how plan() resolved an action (direct/structurer/control/format_error)
    "on_stage_transition",  # from_stage -> to_stage + reason (sentinel/fuzzy/side_effect/tool/control)
    "on_output_reduced",  # tool output raw->kept, with strategy + truncation policy
    "on_oracle_check",  # oracle/judge verdict (success/done/score/reason/source)
    "on_budget_threshold",  # budgets.thresholds fired (at/action/target/message)
    "on_model_retry",  # transient provider failure; waiting before retrying (see litellm_model)
    "on_stall",  # no measurable progress for N cycles (nudge, then force_state)
    "on_slow_candidate",  # one candidate held the slot slow_nudge cycles (nudge) / slow_defer (deferred)
)

# Human-in-the-loop: a person (or supervising program) steered a live run.
INTERVENTION_EVENTS = (
    "on_intervention",  # kind in {inject, edit, stop} + detail (what the human did)
    "on_env_change",  # the execution environment changed between cycles (fs/packages/system)
)

# Typed records for the debugging-grade log: what the
# harness said to the model, what a gate decided, and whether the model did what was asked.
RECORD_EVENTS = (
    "on_message",  # a harness-authored text entered the model's context (site, placement)
    "on_refusal",  # a gate's verdict on a candidate (origin, served, strike, escalation)
    "on_closure",  # the outcome of a message's expectation (complied/ignored/...)
    "on_decision",  # a harness decision with its inputs, rule and consequence
)

EVENTS = LIFECYCLE_EVENTS + DECISION_EVENTS + INTERVENTION_EVENTS + RECORD_EVENTS


class EventEmitter:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable]] = {}

    def on(self, event: str, fn: Callable) -> None:
        self._subs.setdefault(event, []).append(fn)

    def subscribe(self, listener: object) -> None:
        """Register every `on_*` method on `listener` that matches an event name."""
        for event in EVENTS:
            fn = getattr(listener, event, None)
            if callable(fn):
                self.on(event, fn)

    def emit(self, event: str, /, **payload) -> None:
        for fn in self._subs.get(event, ()):
            try:
                fn(**payload)
            except Exception:  # observers must never break the run
                import logging

                logging.getLogger("agentcore.events").debug("listener for %s failed", event, exc_info=True)
