"""The site registry: every place the harness speaks to the model or decides about a candidate.

A site is a stable id (`loop.stall.nudge`) for one emitter in the code. Every message,
refusal, decision and closure record in the trajectory carries one, so a finding reads
"site X, compliance 3%" and X names the function -- the code association that survives
line numbers and wording.

Each site declares:
  where     "module:qualname" of the emitter; a test imports it.
  rule      what fires it, in one sentence.
  expects   what the harness expects the model to do next (an Expect, mechanical), or NONE.
            The loop evaluates it at the following cycles and writes a `closure` record.
  recurring True for a status notice that may repeat unchanged; a repeat must not sit in
            the echoed-turn slot (an invariant the registry test checks).
  classes   the defect classes this site exists to prevent, for regression by name.

Nothing here talks to a model. Predicates read the cycle's action and the worklist store.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ── expectations ─────────────────────────────────────────────────────────────

COMPLIED, IGNORED, CONTRADICTED, REPEATED = "complied", "ignored", "contradicted", "repeated"


def action_fingerprint(action: dict | None) -> str:
    """One string per distinct action: tool name plus its arguments, bounded."""
    if not action:
        return ""
    try:
        args = json.dumps(action.get("args") or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = str(action.get("args"))
    return f"{action.get('name', '')}:{args[:500]}"


@dataclass(frozen=True)
class Expect:
    """A mechanical predicate over what the model did after a message.

    `check(opened, now)` returns COMPLIED or CONTRADICTED when the cycle decides the
    expectation, else None (still open). `opened` is the snapshot taken when the message was
    shown; `now` describes the cycle being evaluated:
      opened: {"candidate": key, "state_pos": int, "last_action_fp": str}
      now:    {"action": {"name", "args"} | None, "parsed": bool, "candidate": key,
               "state_pos": int}
    """

    kind: str
    within: int = 2
    arg: Any = None

    def check(self, opened: dict, now: dict) -> str | None:
        action = now.get("action") or {}
        name = str(action.get("name") or "")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        k = self.kind
        if k == "none":
            return None
        if k == "calls_tool":
            return COMPLIED if name == self.arg else None
        if k in ("records_field", "records_one_of"):
            wanted = {self.arg} if k == "records_field" else set(self.arg or ())
            if name in ("worklist_record", "worklist_add", "record"):
                recorded = set()
                if isinstance(args.get("fields"), dict):
                    recorded |= set(args["fields"])
                for cand in args.get("candidates") or []:
                    if isinstance(cand, dict):
                        recorded |= set(cand)
                if args.get("key"):
                    recorded.add(str(args["key"]))
                if recorded & wanted:
                    return COMPLIED
            return None
        if k == "candidate_changes":
            return COMPLIED if now.get("candidate") != opened.get("candidate") else None
        if k == "state_advances":
            return COMPLIED if int(now.get("state_pos") or 0) > int(opened.get("state_pos") or 0) else None
        if k == "action_differs_from_last":
            if not now.get("action"):
                return None
            fp = action_fingerprint(now.get("action"))
            return COMPLIED if fp != opened.get("last_action_fp") else None
        if k == "no_action_matching":
            if now.get("action") and re.search(str(self.arg), action_fingerprint(now["action"])):
                return CONTRADICTED
            return None
        if k == "action_parses":
            return COMPLIED if now.get("parsed") else None
        raise ValueError(f"unknown expectation kind {k!r}")

    def describe(self) -> str:
        return (
            f"{self.kind}({self.arg!r}, within={self.within})"
            if self.arg is not None
            else f"{self.kind}(within={self.within})"
        )


NONE = Expect("none", 0)


def records_field(name: str, within: int = 2) -> Expect:
    return Expect("records_field", within, str(name))


def records_one_of(names, within: int = 2) -> Expect:
    return Expect("records_one_of", within, tuple(sorted(str(n) for n in names)))


def calls_tool(name: str, within: int = 2) -> Expect:
    return Expect("calls_tool", within, str(name))


def candidate_changes(within: int = 2) -> Expect:
    return Expect("candidate_changes", within)


def state_advances(within: int = 3) -> Expect:
    return Expect("state_advances", within)


def action_differs_from_last(within: int = 2) -> Expect:
    return Expect("action_differs_from_last", within)


def no_action_matching(pattern: str, within: int = 2) -> Expect:
    return Expect("no_action_matching", within, str(pattern))


def action_parses(within: int = 1) -> Expect:
    return Expect("action_parses", within)


def settle(opened: dict, history: list[dict]) -> str:
    """The outcome of an expectation whose window closed without a decision.

    `history` is the window's cycles, each a `now` dict. Every action identical to the one
    before the message is REPEATED (the model did the same thing again); anything else, or
    nothing at all, is IGNORED.
    """
    fps = [action_fingerprint(h.get("action")) for h in history if h.get("action")]
    if fps and all(fp == opened.get("last_action_fp") for fp in fps):
        return REPEATED
    return IGNORED


# ── the registry ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Site:
    where: str
    rule: str
    expects: Expect = NONE
    recurring: bool = False
    classes: tuple[str, ...] = field(default_factory=tuple)


_LOOP = "agentcore.agent.loop:LoopAgent."
_WL = "agentcore.agent.worklist:Worklist."

SITES: dict[str, Site] = {
    # ── messages the loop writes into the model's context ──
    "loop.stage.finished": Site(_LOOP + "_advance_stage", "a stage completed and the next one begins"),
    "loop.publish.banner": Site(
        _LOOP + "_publish_worklist",
        "the delivered sample was written to publish_to",
        recurring=True,
        classes=("blinded-echo-slot",),
    ),
    "loop.publish.empty": Site(
        _LOOP + "_publish_worklist",
        "the run stands in a publish stage with nothing delivered",
        recurring=True,
        classes=("blinded-echo-slot",),
    ),
    "loop.budget.deferred": Site(
        _LOOP + "_apply_budget_thresholds",
        "a budget threshold fired but the exit gate or the strand veto held it",
        classes=("dead-cycles",),
    ),
    "loop.budget.forced": Site(
        _LOOP + "_apply_budget_thresholds",
        "a budget threshold moved the run to its target stage",
        classes=("dead-cycles",),
    ),
    "loop.harness_fault.halt": Site(
        _LOOP + "_halt_on_harness_fault",
        "a systemic, harness-authored fault ended the run",
        classes=("run-past-own-verdict",),
    ),
    "loop.stall.vetoed": Site(
        _LOOP + "_check_stall",
        "no progress for force_after cycles, but the exit guard refused the move",
        expects=action_differs_from_last(within=2),
        classes=("dead-cycles",),
    ),
    "loop.stall.forced": Site(
        _LOOP + "_check_stall", "no progress for force_after cycles; the stage was forced", classes=("dead-cycles",)
    ),
    "loop.stall.nudge": Site(
        _LOOP + "_check_stall",
        "no measurable progress for nudge_after cycles",
        expects=action_differs_from_last(within=2),
        recurring=True,
        classes=("dead-cycles", "blinded-echo-slot"),
    ),
    "loop.action.rejected": Site(
        _LOOP + "_inject_noop",
        "a repeated or no-op action was refused without running",
        expects=action_differs_from_last(within=1),
        classes=("refused-resolvable",),
    ),
    "loop.injected": Site(_LOOP + "inject", "a person injected an instruction"),
    "loop.controller.error": Site(_LOOP + "_before_cycle", "the cycle controller raised"),
    "loop.finish.refused": Site(
        _LOOP + "_handle_control",
        "a command-less finish was refused by the finish oracle",
        expects=action_differs_from_last(within=1),
        classes=("dead-cycles",),
    ),
    "loop.stage_exit.refused": Site(
        _LOOP + "_handle_control",
        "a stage_done signal was refused by the exit guard",
        expects=action_differs_from_last(within=1),
        classes=("dead-cycles",),
    ),
    "loop.tool.nudge": Site(
        _LOOP + "run",
        "a tool call was refused before running (NonTerminating)",
        expects=action_differs_from_last(within=1),
        classes=("refused-resolvable",),
    ),
    "loop.parse.format_error": Site(
        _LOOP + "run", "the model's reply did not parse as an action", expects=action_parses(within=1)
    ),
    "loop.error": Site(
        _LOOP + "run", "an unexpected exception in the cycle, recorded not fatal", classes=("harness-fault-misread",)
    ),
    # Every on_intervention emit names its site, so the decision records rank.
    "loop.worklist.render_failed": Site(
        _LOOP + "_before_cycle", "the worklist section could not render; the cycle ran without it"
    ),
    "loop.refusal_storm": Site(_LOOP + "run", "refusal_streak_limit consecutive refused cycles ended the run"),
    "loop.edited": Site(_LOOP + "edit_last_memory", "a person rewrote the last observation"),
    "loop.stopped": Site(_LOOP + "request_stop", "a person stopped the run"),
    "run.teardown_failed": Site(
        "agentcore.run.single:run_single", "a teardown step raised; the run's outcome is unchanged"
    ),
    "run.container_cleanup": Site("agentcore.run.single:run_single", "leftover sandboxes removed at run end"),
    "loop.unsited": Site(_LOOP + "_mem", "a message recorded without a site (a defect to fix)"),
    "tool.result": Site(_LOOP + "summarize", "what a tool returned, reduced for the model"),
    # ── the worklist's slow-candidate watchdog ──
    "worklist.slow.nudge": Site(
        _WL + "tick",
        "one candidate held the slot slow_nudge cycles",
        expects=action_differs_from_last(within=3),
        recurring=True,
        classes=("dead-cycles", "blinded-echo-slot"),
    ),
    "worklist.slow.deferred": Site(
        _WL + "tick", "one candidate held the slot slow_defer cycles; deferred", classes=("dead-cycles",)
    ),
    "worklist.empty.nudge": Site(
        _WL + "_tick_empty",
        "no candidate registered for slow_nudge cycles",
        expects=calls_tool("worklist_add", within=3),
        recurring=True,
        classes=("dead-cycles",),
    ),
    # ── gate verdicts (refusal records; the message the model sees is the tool result) ──
    "worklist.gate.refuse": Site(
        _WL + "_validated", "a gate refused fresh evidence", classes=("refused-resolvable", "frozen-verdict")
    ),
    "worklist.gate.replay": Site(_WL + "_validated", "a cached refusal was served again", classes=("frozen-verdict",)),
    "worklist.gate.harness_fault": Site(
        _WL + "_validated", "the harness's own failure at a gate", classes=("harness-fault-misread",)
    ),
    "worklist.systemic": Site(
        _WL + "_emit_systemic",
        "one refusal shape reached systemic_at unrelated candidates",
        classes=("run-past-own-verdict",),
    ),
}


def site(site_id: str) -> Site:
    """The registered site, or KeyError: an emitter must be registered before it writes."""
    return SITES[site_id]
