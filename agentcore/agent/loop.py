"""LoopAgent — the plan → execute → summarize cycle.

Default action protocol: JSON tool-calling {thoughts, command:{name,args}} with a
structurer fallback (bash_block available via action_protocol). Memory is compact:
each cycle rebuilds [system, user] from a (command, summary) history rather than an
unbounded message list. Stages, budget meters, repetition policy, lifecycle events
and an optional judge oracle are all wired here; OracleLoop (oracle_loop.py)
overrides only execute/should_finish for verify-then-finish tasks.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agentcore.agent.budget import BudgetController
from agentcore.agent.events import EventEmitter
from agentcore.agent.exceptions import (
    BudgetExhausted,
    FinishWithResult,
    FormatError,
    GoalsAccomplished,
    NonTerminating,
    Terminating,
)
from agentcore.agent.prompts import PromptStore
from agentcore.agent.reduce import OutputReducer
from agentcore.agent.stages import StageController
from agentcore.agent.workingset import WorkingSet
from agentcore.config.loader import core_defaults
from agentcore.config.schema import AgentSpec
from agentcore.oracle.base import OracleResult
from agentcore.tools import parsing
from agentcore.tools.context import ToolContext
from agentcore.tools.registry import ToolRegistry
from agentcore.tools.result import ToolResult


@dataclass
class RunResult:
    success: bool = False
    reason: str = ""
    steps: int = 0
    cost: float = 0.0
    action_log: list = field(default_factory=list)  # compact (command, summary) record
    working_set: dict = field(default_factory=dict)
    submitted: dict | None = None


def _resolve_hook(path: str):
    mod, _, attr = path.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _cmd_name(action: dict) -> str:
    return (action.get("command") or {}).get("name", "") if action else ""


class LoopAgent:
    def __init__(
        self,
        spec: AgentSpec,
        *,
        registry: ToolRegistry,
        env,
        models: dict,
        working_set: WorkingSet,
        prompts: PromptStore,
        budget: BudgetController,
        stages: StageController,
        events: EventEmitter | None = None,
        oracle=None,
        instance: dict | None = None,
        task: str = "",
        controller=None,
    ):
        self.spec = spec
        self.registry = registry
        self.env = env
        self.models = models
        self.working_set = working_set
        self.prompts = prompts
        self.budget = budget
        self.stages = stages
        self.events = events or EventEmitter()
        self.oracle = oracle
        self.instance = instance or {}
        self.task = task
        self.controller = controller  # optional human-in-the-loop cycle-boundary controller

        self.core = core_defaults()
        self.ctxspec = spec.context
        self.action_log: list[tuple[str, str]] = []  # compact (command, summary) record (always kept)
        self.transcript: list[dict] = []  # growing chat history (transcript/hybrid modes)
        self._transcript_stage: str | None = None
        self.stage_summaries: list[str] = []  # summaries of completed stages (hybrid carry)
        self._last_assistant: str = ""  # the model's most recent raw reply
        # The previous exchange, echoed back as real turns in compact mode so the model can see
        # what it just did. See ContextSpec.echo_last_turn.
        self._last_turn: dict | None = None
        self._finished = False
        self._success = False
        self._reason = ""
        self._submitted: dict | None = None
        # (repetition key, the cycle it happened on) -- the cycle is what lets a repetition
        # block EXPIRE. See _is_repetition.
        self._recent_actions: list[tuple[tuple, int]] = []
        # Monotonic cycle counter, available to tools as `ctx.agent.cycle` (e.g. for stamping
        # sandbox lifecycle events).
        self.cycle: int = 0
        self._noop_streak: dict[str, int] = {}
        self._scratch: dict[str, Any] = {}  # persistent per-run bag for tools (checkpoints, out_dir, ...)

        # resolve hooks
        self.hooks = {k: _resolve_hook(v) for k, v in (spec.hooks or {}).items()}

        # configurable tool-output reducer (heuristic / llm_summary / raw, per-tool overrides)
        self.reducer = OutputReducer(
            spec.tool_output,
            models=self.models,
            prompts=self.prompts,
            registry=self.registry,
            budget=self.budget,
            events=self.events,
        )
        self._error_streak = 0  # consecutive unexpected errors (loop-resilience backstop)
        self._last_progress_fp: tuple | None = None  # stall detector: last observed progress
        self._stall_streak = 0  # cycles since anything measurable changed
        # Budget-threshold forces the stage's exit gate refused; retried every cycle until it
        # opens, because thresholds_triggered() yields each entry only once.
        self._deferred_thresholds: list[dict] = []

        # Surface the model layer's patient transient-retry waits as real events. Without this
        # a multi-minute provider stall is indistinguishable from a hung process in the log.
        for role, m in self.models.items():
            if hasattr(m, "on_retry") and getattr(m, "on_retry", None) is None:
                m.on_retry = functools.partial(self._on_model_retry, role)
            # A model that can wait out a provider limit (a provider session limit) waits
            # only within the run's remaining wallclock.
            if hasattr(m, "wait_budget_s") and getattr(m, "wait_budget_s", None) is None:
                m.wait_budget_s = self._wallclock_left_s

    def _wallclock_left_s(self) -> float | None:
        """Seconds of wallclock budget the run has left; None when it declares none."""
        try:
            for ms in self.budget.meters:
                if ms.spec.kind == "wallclock":
                    return max(0.0, float(ms.spec.limit) - float(ms.value()))
        except Exception:
            return None
        return None

    def _on_model_retry(self, role, *, attempt, wait, remaining, error):
        # Marks the cycle as provider-starved: the stall watchdog must not count a cycle the
        # WORLD refused against the agent (otherwise the watchdog would read the silence as
        # "no progress" and force-advance the stage).
        self._cycle_was_starved = True
        self.events.emit(
            "on_model_retry",
            role=role,
            attempt=attempt,
            wait=round(wait, 2),
            remaining=round(remaining, 1),
            error=f"{type(error).__name__}: {error}",
        )

    # ── loop-resilience backstop ──────────────────────────────────────────────
    @staticmethod
    def _is_transient_provider_error(exc: BaseException) -> bool:
        """Is this a provider capacity/connectivity blip, as opposed to a real defect?

        The distinction is the whole point of the backstop split below, so it is deliberately
        narrow: only the litellm exception classes that genuinely describe "the upstream is
        busy or unreachable right now" qualify. A buggy hooks.py raising KeyError, a tool that
        always throws, a model that never parses -- none of these are transient, and waiting
        does not help them. Falls back to False (the strict, fail-fast path) if litellm isn't
        importable, so a missing dependency can never accidentally grant infinite patience.
        """
        from agentcore.models.claude_code import ClaudeCodeError

        if isinstance(exc, ClaudeCodeError):
            return exc.kind in ("transient", "session_limit")
        try:
            import litellm
        except Exception:
            return False
        names = ("RateLimitError", "ServiceUnavailableError", "InternalServerError", "APIConnectionError", "Timeout")
        classes = tuple(getattr(litellm, n) for n in names if hasattr(litellm, n))
        if bool(classes) and isinstance(exc, classes):
            return True
        # a provider 429/529 can arrive as a generic APIError: same test as the
        # model layer's retry, so the loop and the model agree on what is worth waiting for
        from agentcore.models.litellm_model import is_capacity_error

        return is_capacity_error(exc)

    @staticmethod
    def _ends_run_now(exc: BaseException) -> str | None:
        """The run's end reason when this error leaves nothing to retry; None otherwise.

        A session limit reaches the loop only when the model could not wait for its reset
        within the run's time: every further cycle would fail the same way until then, so the
        run ends once, saying when the limit lifts, instead of after five instant errors."""
        from agentcore.models.claude_code import ClaudeCodeError

        if isinstance(exc, ClaudeCodeError) and exc.kind == "session_limit":
            return f"model session limit: {exc}"
        return None

    def _error_streak_limit(self, transient: bool = True) -> int:
        """How many consecutive unexpected errors to absorb before abandoning the run.

        Two regimes, because a fixed limit would conflate two very different situations:

        * A genuinely broken run (buggy hook, always-throwing tool). Stopping early is right --
          nothing improves by trying again, and a fast, clear failure is the useful outcome.
          These keep the original limit of 5.

        * A transient provider outage. Stopping early discards everything already achieved
          (e.g. an upstream capacity blip while most of the budget is unspent). For these, be stubborn while there is
          budget left to be stubborn with, and tighten as it runs out -- late in a run an error
          streak is likelier to be terminal and there is less left to protect. The model layer
          already retries patiently inside each call, so a unit here is worth minutes.
        """
        floor, ceiling = 5, 40
        if not transient:
            return floor
        try:
            used = max((ms.fraction() for ms in self.budget.meters), default=1.0)
        except Exception:
            return floor
        remaining = max(0.0, 1.0 - used)
        return max(floor, int(ceiling * remaining))

    # ── context / prompts ─────────────────────────────────────────────────────
    def ctx(self) -> ToolContext:
        # NB: reuse one persistent scratch dict so tool state (checkpoints, out_dir,
        # submission aggregate paths) survives across cycles.
        return ToolContext(
            env=self.env,
            models=self.models,
            working_set=self.working_set,
            instance=self.instance,
            stage=self.stages,
            budget=self.budget,
            agent=self,
            scratch=self._scratch,
        )

    def _template_vars(self, **extra) -> dict[str, Any]:
        allow = self.stages.tools_allow(self.spec.tools)
        rendered_log = self._render_action_log()
        v: dict[str, Any] = {
            "task": self.task,
            "instance": self.instance,
            "action_log": rendered_log,
            "history": rendered_log,  # back-compat alias for existing prompts using {{ history }}
            # What the run has WRITTEN, which the replay above no longer carries in full. See
            # `_files_written`: this is the half that makes the per-entry clip safe.
            "files_written": self._files_written(self.action_log),
            "files_written_text": self._files_written_text(self.action_log),
            "tools_text": self.registry.describe(allow),
            # THE NAMES, separately from the prose. Templates guard capability blocks on "do I
            # have this tool"; guarding on the rendered DESCRIPTIONS would fire whenever
            # another tool's help text mentions the name, instructing the model to use a
            # tool it does not have.
            "tool_names": sorted(self.registry.names(allow) if hasattr(self.registry, "names") else (allow or [])),
            "working_set": self.working_set.snapshot(),
            # THE SENTENCE, NOT THE DICT: `{{ budget }}` is handed to templates as readable text;
            # the dict is still what `restore()` reads. See `BudgetController.status_text`.
            "budget": self.budget.status_text(),
            "core": self.core,
            # WHAT EARLIER AGENTS ON THIS FAMILY LEARNED: `bench lessons` reads
            # the complete records of a family's agents with a model and writes lessons.md
            # into the agent dir. Read from disk every cycle, like the templates, so an agent
            # that is running sees a new file on its next cycle. Empty when there is none.
            "lessons": self.prompts.read_file("lessons.md", "") if hasattr(self.prompts, "read_file") else "",
            # WHERE THE OTHER SEARCHING AGENTS LOOK: `benchcompiler.directions`
            # summarizes each agent's search direction into directions.md in the agent dir;
            # read every cycle like the lessons, so a refresh reaches running agents.
            "directions": self.prompts.read_file("directions.md", "") if hasattr(self.prompts, "read_file") else "",
            # FACTS ABOUT THE HOST, not the task: e.g. where a package cache is, set by the
            # launch (BENCH_MACHINE_FACTS).
            "machine_facts": os.environ.get("BENCH_MACHINE_FACTS", "").strip(),
        }
        for src in (self.models.get("main"), self.env, self.stages):
            try:
                v.update(src.get_template_vars())
            except Exception:
                pass
        # advertise env-provisioned CLI commands in the tool listing (run via linux_terminal)
        cmds = v.get("commands") or []
        if cmds:
            v["tools_text"] += "\n\nProvisioned shell commands (run via linux_terminal):\n" + "\n".join(
                f"- {c['name']}"
                + (f": {c['description']}" if c.get("description") else "")
                + (f"  (usage: {c['usage']})" if c.get("usage") else "")
                for c in cmds
            )
        # Per-candidate situation, when a worklist is configured. Supplies both structured vars
        # and a ready-made `worklist_text` block, so a prompt template can drop in
        # {{ worklist_text }} and stop restating the whole task every cycle.
        if getattr(self.spec, "worklist", None) and self.spec.worklist.enabled:
            try:
                from agentcore.agent.worklist_render import exclusions_line, render_text, render_worklist

                wv = render_worklist(self.spec, self.working_set, target=(self.instance or {}).get("sample_size"))
                wv["worklist_exclusions"] = exclusions_line(self.instance or {})
                v.update(wv)
                v["worklist_text"] = render_text(wv, v)
            except Exception as e:  # noqa: BLE001 - a render bug must not kill the run...
                # ...but it must not be INVISIBLE either: a silent swallow would delete the whole
                # worklist section from the prompt with no trace.
                try:
                    self.events.emit(
                        "on_intervention",
                        site="loop.worklist.render_failed",
                        kind="worklist_render_failed",
                        error=f"{type(e).__name__}: {e}",
                    )
                except Exception:  # noqa: BLE001
                    pass
        # The list of PLACES to work through, when the task declares one. This is a STANDING
        # block on purpose: a pool that is only reachable from the working set and rendered
        # nowhere leaves the agent mining blind. Bounded by sources.view_cap however large the list gets.
        if getattr(self.spec, "sources", None) and self.spec.sources.enabled:
            try:
                from agentcore.agent import sources as _src
                from agentcore.tools.source_tools import _spec as _source_spec

                pool = _src.build(_source_spec(self.ctx()), self.working_set)
                v["sources_text"] = pool.view(cycle=getattr(self, "cycle", 0))
                # What the task requires of a source, from its own declaration -- so the
                # prompt and the gate can never state different thresholds.
                v["sources_admissibility"] = _src.admissibility_block(pool.spec)
                v["sources_rows"] = pool.rows()
                v["sources_batch"] = pool.current_batch()
            except Exception:
                pass
        # programmatic pre-render hook — receives the ToolContext, so it can vary the
        # prompt by ctx.stage.name() (e.g. load a different augmented doc per stage).
        if self.spec.prerender:
            try:
                v.update(_resolve_hook(self.spec.prerender)(self.ctx()) or {})
            except Exception:
                pass
        v.update(extra)
        return v

    # How much of the echoed exchange is worth carrying. A single tool result can hold thousands of characters of raw
    # file content; echoing that whole, every cycle, would be the same waste in a new place.
    ECHO_ASSISTANT_CHARS = 4000
    ECHO_OBSERVATION_CHARS = 8000

    def _echoed_turn(self) -> tuple[str, str] | None:
        """The previous cycle's (reply, observation), when it is being echoed as real turns."""
        if self.ctxspec.mode != "compact" or not getattr(self.ctxspec, "echo_last_turn", False):
            return None
        turn = getattr(self, "_last_turn", None) or {}
        a, o = str(turn.get("assistant") or "").strip(), str(turn.get("observation") or "").strip()
        return (a, o) if a and o else None

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        """Keep the head and the TAIL: a traceback's verdict is on its last line."""
        from agentcore.tools.parsing import clip

        return clip(str(text or ""), limit)

    # THE COMMAND IS NEVER CLIPPED. It is the identity of the action: clipped, an entry is
    # neither recognisable nor repeatable, and the agent is left guessing what it already did.
    # It is also cheap: commands are short in the median, and a whole window of them is small
    # next to the observations. Tiering older commands would shrink prompts only a few percent.
    #
    # THE OBSERVATION IS WHERE THE SIZE IS. One cap for both halves would barely touch a
    # command and clip every observation. head+tail is the right shape for it -- `(exit N)` is
    # in the first line and a traceback's verdict is on the last -- and how much of it is worth
    # carrying depends on how recently it happened. What you did last you are still reasoning
    # about; what you did thirty actions ago you need to RECOGNISE, not re-read. So commands
    # stay whole, the freshest observations are kept nearly complete and older ones are clipped.
    #
    ACTION_LOG_FRESH_ENTRIES = 5
    ACTION_LOG_FRESH_CHARS = 8000
    ACTION_LOG_OLDER_CHARS = 400
    #: A command is never clipped as an INSTRUCTION. The one exception is a command that
    #: carries a payload -- `write_to_file({'path': ..., 'content': '<18k of file>'})`, or a
    #: heredoc through linux_terminal -- where the bulk is output wearing a command's clothes.
    #: 6,000 sits above any ordinary command so ordinary traffic is untouched, and `_clip` keeps head+tail, so even a clipped label
    #: still shows the tool and the path. `files_written` is what makes that safe: a script's
    #: record does not depend on the log replaying its body.
    ACTION_LOG_COMMAND_CEILING = 6000

    def _render_action_log(self) -> str:
        if not self.action_log:
            return "(no actions yet)"
        n = self.ctxspec.window or 20
        entries = self.action_log[-n:]
        # The echoed exchange is about to be sent verbatim as its own two messages; leaving it
        # in the replay as well pays for it twice. THE ECHOED ENTRY, NOT THE LAST ONE: a notice recorded with
        # `touch_last_turn=False` (the publish banner, a stall re-nudge) is the last entry on
        # the cycle it fires, and the echoed one sits just before it. Dropping by position hid
        # the notice for that cycle and replayed the tool output twice, once as the echo and
        # once at the fresh tier of the log.
        if self._echoed_turn() and entries:
            drop = self._echoed_index(entries)
            if drop is not None:
                entries = entries[:drop] + entries[drop + 1 :]
        if not entries:
            return "(nothing older than your last action)"
        fresh_from = max(0, len(entries) - self.ACTION_LOG_FRESH_ENTRIES)
        out = []
        for i, (cmd, summ) in enumerate(entries):
            cap = self.ACTION_LOG_FRESH_CHARS if i >= fresh_from else self.ACTION_LOG_OLDER_CHARS
            out.append(f"- {self._clip(cmd, self.ACTION_LOG_COMMAND_CEILING)}\n  -> {self._clip(summ, cap)}")
        return "\n".join(out)

    def _echoed_index(self, entries) -> int | None:
        """Position within `entries` of the entry `_echoed_turn` is about to send, or None."""
        turn = self._last_turn or {}
        idx = turn.get("index")
        if isinstance(idx, int):
            rel = idx - (len(self.action_log) - len(entries))
            return rel if 0 <= rel < len(entries) else None
        # A turn recorded without its index (hand-built): the observation itself identifies it.
        obs = str(turn.get("observation") or "")
        for i in range(len(entries) - 1, -1, -1):
            if str(entries[i][1]) == obs:
                return i
        return None

    # A path in a tool call's args repr, whatever the parameter is called.
    _PATH_ARG = re.compile(
        r"'(?:path|file_path|filename|dest|output_path)'\s*:\s*"
        r"'([^']{1,300})'"
    )
    # A shell write: `> f`, `>> f`, or `tee f`. `2>` and friends are excluded by requiring the
    # redirection not to be preceded by a digit -- `2>/dev/null` rides on nearly every command
    # in these runs and is not an artifact.
    _SHELL_WRITE = re.compile(
        r"(?<![0-9<>])>>?\s*([^\s;|&'\"<>\\]{1,300})"
        r"|\btee\s+(?:-a\s+)?([^\s;|&'\"<>\\]{1,300})"
    )
    _NOT_A_FILE = ("/dev/", "&")
    FILES_WRITTEN_CAP = 40

    @classmethod
    def _files_written(cls, action_log) -> list[str]:
        """Paths this run has written, oldest first, from the action log itself.

        WHY THIS EXISTS: it is what makes per-entry clipping safe. A script's only record is the
        heredoc sitting in the log, and the model reads scripts back to remind itself they exist, so
        bounding the log without replacing that record would remove the very thing it was
        re-reading for. Derived from the action log rather than tracked separately, so it costs
        no new plumbing and survives resume for free (the action log is persisted).

        Heuristic on purpose, and deliberately biased toward missing a write rather than
        inventing one: this is a reminder, not a manifest. A path listed here is one the run
        asked to write, not one verified to exist -- the sandbox is frequently not this
        filesystem, so there is nothing honest to stat.
        """
        seen: dict[str, None] = {}
        for label, _ in action_log or ():
            text = str(label)
            head = text.split("(", 1)[0]
            if head.startswith("read_file") or head.startswith("worklist"):
                continue
            found: list[str] = []
            if not head.startswith("linux_terminal"):
                found += cls._PATH_ARG.findall(text)
            else:
                for a, b in cls._SHELL_WRITE.findall(text):
                    found.append(a or b)
            for p in found:
                p = p.strip().strip("'\"")
                if p and not p.startswith(cls._NOT_A_FILE):
                    # Re-listed at its MOST RECENT write, so a file rewritten late
                    # is never the one the tail cap drops.
                    seen.pop(p, None)
                    seen[p] = None
        out = list(seen)
        return out[-cls.FILES_WRITTEN_CAP :]

    @classmethod
    def _files_written_text(cls, action_log) -> str:
        """The dashboard block. Empty when nothing was written -- a heading over an empty list
        is a line of prompt that says nothing."""
        files = cls._files_written(action_log)
        if not files:
            return ""
        return (
            "== FILES YOU HAVE WRITTEN (this run asked to write these; re-use them "
            "instead of re-typing or re-reading them) ==\n" + "\n".join(f"  {p}" for p in files)
        )

    # ── prompt assembly (mode-aware) ────────────────────────────────────────────
    def _build_messages(self) -> list[dict]:
        if self.ctxspec.mode == "compact":
            return self._compact_messages()
        self._ensure_transcript()
        return self._transcript_view()

    def _compact_messages(self) -> list[dict]:
        """Rebuild [system, user] fresh each cycle from the compact action_log."""
        vars_ = self._template_vars()
        system = self.prompts.render("system", **vars_) if self.prompts.has("system") else ""
        user_name = self.stages.prompt_name(self.spec.prompts.get("cycle", "cycle"))
        user = self.prompts.render(user_name, **vars_) if user_name else self._render_action_log()
        msgs = []
        if system.strip():
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        # The previous exchange, as REAL turns, after the standing prompt. Placed last so the
        # most recent thing is closest to generation -- the same reasoning that put the
        # worklist refresher at the end of the mining prompt. See ContextSpec.echo_last_turn.
        echoed = self._echoed_turn()
        if echoed:
            reply, observation = echoed
            msgs.append({"role": "assistant", "content": self._clip(reply, self.ECHO_ASSISTANT_CHARS)})
            msgs.append(
                {
                    "role": "user",
                    "content": "OUTPUT OF YOUR LAST ACTION (above) -- this is what actually came "
                    "back, not a new instruction:\n" + self._clip(observation, self.ECHO_OBSERVATION_CHARS),
                }
            )
        return msgs

    def _ensure_transcript(self) -> None:
        """Seed the growing transcript; reset+reseed at a stage boundary (hybrid)."""
        reset = self.ctxspec.reset_on_stage
        if reset is None:
            reset = self.ctxspec.mode == "hybrid"
        cur = self.stages.name()
        if not self.transcript:
            self._seed_transcript()
        elif reset and self._transcript_stage != cur:
            self._summarize_completed_stage()  # carry a summary across the boundary
            self._seed_transcript()

    def _seed_transcript(self) -> None:
        vars_ = self._template_vars()
        msgs: list[dict] = []
        if self.prompts.has("system"):
            s = self.prompts.render("system", **vars_)
            if s.strip():
                msgs.append({"role": "system", "content": s, "pin": True})
        if self.stage_summaries:
            msgs.append(
                {
                    "role": "system",
                    "pin": True,
                    "content": "Summary of completed stages:\n\n" + "\n\n".join(self.stage_summaries),
                }
            )
        user_name = self.stages.prompt_name(self.spec.prompts.get("cycle", "cycle"))
        if user_name:
            msgs.append({"role": "user", "content": self.prompts.render(user_name, **vars_)})
        self.transcript = msgs
        self._transcript_stage = self.stages.name()

    def _transcript_view(self) -> list[dict]:
        """Apply window / max_tokens eviction; pinned messages always survive.

        Bounded two ways so a growing transcript can never exceed the context window:
        (1) drop oldest non-pinned messages until under the token budget; (2) if a single
        remaining message is still too large (e.g. a raw report), hard-truncate its content.
        """
        pinned = [dict(m) for m in self.transcript if m.get("pin")]
        rest = [dict(m) for m in self.transcript if not m.get("pin")]
        if self.ctxspec.window:
            rest = rest[-self.ctxspec.window :]
        budget = self.ctxspec.max_tokens
        if budget:
            est = lambda ms: sum(len(m.get("content", "")) for m in ms) // 4  # noqa: E731
            while len(rest) > 1 and est(pinned + rest) > budget:
                rest.pop(0)
            if est(pinned + rest) > budget:  # a single oversized message remains -> truncate contents
                per_chars = max(2000, (budget * 4) // max(1, len(rest)))
                for m in rest:
                    if len(m.get("content", "")) > per_chars:
                        m["content"] = parsing.truncate(
                            m["content"],
                            "head_tail",
                            head=per_chars // 2,
                            middle=0,
                            tail=per_chars // 2,
                            threshold=per_chars,
                        )
        return [{"role": m["role"], "content": m["content"]} for m in (pinned + rest)]

    def _summarize_completed_stage(self) -> None:
        stage = self._transcript_stage or "stage"
        convo = [m for m in self.transcript if m.get("role") in ("assistant", "user") and not m.get("pin")]
        text = "\n".join(f"{m['role']}: {m.get('content', '')[:1000]}" for m in convo)[-6000:]
        summary = ""
        if self.ctxspec.summarize_on_stage_exit and text:
            model = self.models.get("summarizer") or self.models.get("main")
            try:
                r = model.query(
                    [
                        {
                            "role": "user",
                            "content": "Summarize concisely what was accomplished and learned in this "
                            "stage (facts the next stage needs):\n\n" + text,
                        }
                    ]
                )
                self.budget.note_cost(getattr(r, "cost", 0.0))
                summary = (getattr(r, "content", "") or "").strip()[:2000]
            except Exception:
                summary = ""
        if not summary:  # heuristic fallback (also used when summarize_on_stage_exit is off)
            summary = "; ".join(lbl for lbl, _ in self.action_log[-8:])
        self.stage_summaries.append(f"[{stage}] {summary}")

    # ── memory recording (compact action_log + transcript messages) ─────────────
    def _mem(
        self,
        label: str,
        summary: str,
        role: str = "user",
        touch_last_turn: bool = True,
        site: str | None = None,
        expects=None,
    ) -> None:
        """Record an observation/note: always into the action_log; also into the
        transcript (as a message) when in a transcript/hybrid mode.

        `touch_last_turn=False` records without claiming the echoed-turn slot. A RECURRING
        STATUS NOTICE IS NOT WHAT CAME BACK FROM YOUR LAST ACTION: `_publish_worklist` runs
        last at every cycle end, so in a publish stage its banner would overwrite the tool
        result and `_echoed_turn` would show the model the harness talking to itself under the
        heading "OUTPUT OF YOUR LAST ACTION (above) -- this is what actually came back".
        """
        self.action_log.append((label, summary))
        # Whatever came back most recently, paired with the reply that caused it, for the
        # echoed exchange. Loop annotations ([stall], [nudge]) count too: they are also "what
        # came back", and the model should see one as the immediate consequence of its reply
        # rather than as a line in a log.
        if touch_last_turn:
            self._last_turn = {
                "assistant": getattr(self, "_last_assistant", "") or "",
                "observation": summary,
                # WHICH entry this is, so the replay drops exactly the echoed
                # one and not whichever entry is last: see `_echoed_index`.
                "index": len(self.action_log) - 1,
            }
        if self.ctxspec.mode != "compact" and self.transcript:
            self.transcript.append({"role": role, "content": summary})
        self._note_message(label, summary, touch_last_turn, site, expects)

    # ── the typed log: messages, expectations, closures ─────────────────────────
    # Every text the harness puts in front of the model
    # is a `message` record with a registered site and where it landed; a site that expects
    # something of the model opens an expectation the following cycles close mechanically.

    def _note_message(self, label: str, text: str, echoed: bool, site: str | None, expects) -> None:
        from agentcore.observability import sites as _sites

        events = getattr(self, "events", None)
        if events is None:
            return
        if not site:
            site = "tool.result" if not str(label).startswith("[") else "loop.unsited"
        registered = _sites.SITES.get(site)
        if expects is None and registered is not None:
            expects = registered.expects
        n = getattr(self, "_msg_seq", 0) + 1
        self._msg_seq = n
        mid = f"m{getattr(self, 'cycle', 0)}.{n}"
        try:
            events.emit(
                "on_message",
                site=site,
                label=str(label)[:200],
                text=text,
                placement="echo" if echoed else "log",
                mid=mid,
                expects=expects.describe() if expects and expects.kind != "none" else None,
                author="tool" if site == "tool.result" else "harness",
            )
        except Exception:  # noqa: BLE001 - the log must never break the run
            pass
        if expects is not None and expects.kind != "none":
            opened = {
                "mid": mid,
                "site": site,
                "expects": expects,
                "cycle": getattr(self, "cycle", 0),
                "history": [],
                **self._closure_snapshot(),
            }
            self.__dict__.setdefault("_open_expectations", []).append(opened)

    def _closure_snapshot(self) -> dict:
        """What an expectation compares against: the candidate, its position, the action
        the model took before the message."""
        key, pos = self._candidate_position()
        return {"candidate": key, "state_pos": pos, "last_action_fp": getattr(self, "_last_action_fp", "")}

    def _candidate_position(self) -> tuple[str, int]:
        key = ""
        try:
            key = self._current_candidate_key()
        except Exception:  # noqa: BLE001
            return "", 0
        if not key:
            return "", 0
        try:
            from agentcore.agent.worklist import WORKLIST_KEY

            store = getattr(self.working_set, WORKLIST_KEY, None) or {}
            state = (store.get("candidates") or {}).get(key, {}).get("state")
            names = [s.name for s in self.spec.worklist.states]
            return key, names.index(state) if state in names else -1
        except Exception:  # noqa: BLE001
            return key, 0

    def _close_expectations(self) -> None:
        """At cycle end: decide every open expectation the model has had a chance to answer."""
        from agentcore.observability import sites as _sites

        open_ = getattr(self, "_open_expectations", None)
        if not open_:
            return
        cycle = getattr(self, "cycle", 0)
        key, pos = self._candidate_position()
        now = {
            "action": getattr(self, "_cycle_action", None),
            "parsed": bool(getattr(self, "_cycle_parsed", False)),
            "candidate": key,
            "state_pos": pos,
        }
        keep = []
        for exp in open_:
            if exp["cycle"] >= cycle:  # opened this cycle: the model answers next cycle
                keep.append(exp)
                continue
            exp["history"].append(now)
            outcome = exp["expects"].check(exp, now)
            if outcome is None and len(exp["history"]) < exp["expects"].within:
                keep.append(exp)
                continue
            if outcome is None:
                outcome = _sites.settle(exp, exp["history"])
            try:
                self.events.emit(
                    "on_closure",
                    site=exp["site"],
                    expectation=exp["mid"],
                    outcome=outcome,
                    cycles=len(exp["history"]),
                    expects=exp["expects"].describe(),
                    action=_sites.action_fingerprint(now.get("action"))[:300],
                )
            except Exception:  # noqa: BLE001
                pass
        self._open_expectations = keep

    def _record_assistant(self) -> None:
        if self.ctxspec.mode != "compact" and self.transcript:
            self.transcript.append({"role": "assistant", "content": self._last_assistant or ""})

    def _structurer_rotation(self):
        return self.spec.models["structurer"].rotation if "structurer" in self.spec.models else None

    def _structure(self, freeform: str) -> dict | None:
        allow = self.stages.tools_allow(self.spec.tools)
        extra = self.prompts.render("structurer", **self._template_vars()) if self.prompts.has("structurer") else ""
        return parsing.structure_action(
            self.models,
            freeform,
            self.registry.specs(allow),  # active tools -> dynamic listing/completion/examples
            rotation=self._structurer_rotation(),
            extra_instructions=extra,
            cost_sink=self.budget.note_cost,  # account structurer spend on the budget meter
        )

    # ── plan ───────────────────────────────────────────────────────────────────
    def plan(self) -> dict:
        msgs = self._build_messages()
        self.events.emit("on_prompt_sent", messages=msgs, stage=self.stages.name())

        if self.spec.action_protocol == "freeform":
            # SPLIT request: main model answers free-form, then the structurer converts it.
            resp = self.models["main"].query(msgs)
            self.budget.note_cost(getattr(resp, "cost", 0.0))
            content = getattr(resp, "content", "") or ""
            self._last_assistant = content
            self.events.emit(
                "on_response",
                content=content,
                attempt=0,
                cost=getattr(resp, "cost", 0.0),
                role="main",
                finish_reason=getattr(resp, "finish_reason", ""),
                source=getattr(resp, "source", "content"),
                provider=getattr(resp, "provider", None),
                generation_id=getattr(resp, "generation_id", None),
            )
            sig = self._control_signal(content)  # free-form may declare stage-done / finish
            if sig:
                self._emit_parse("control", attempt=0, signal=sig)
                return {"control": sig}
            action = self._structure(content)
            if action:
                self.events.emit(
                    "on_response", content="[structurer] " + str(action.get("command")), attempt=0, role="structurer"
                )
                self._emit_parse("freeform_structured", attempt=0, tool=_cmd_name(action))
                return action
            self._emit_parse("format_error", attempt=0)
            raise FormatError("structurer could not extract a command from the free-form response")

        last_err = None
        cut_off = False
        max_retries = int(self.core.get("loop", {}).get("max_format_retries", 5))
        for attempt in range(max_retries):
            if attempt == 0:
                ask = msgs
            elif cut_off:
                ask = msgs + self._cutoff_nudge(self._last_assistant)
            else:
                ask = msgs + self._format_nudge(last_err, self._last_assistant)
            resp = self.models["main"].query(ask)
            self.budget.note_cost(getattr(resp, "cost", 0.0))
            content = getattr(resp, "content", "") or ""
            self._last_assistant = content
            self.events.emit(
                "on_response",
                content=content,
                attempt=attempt,
                finish_reason=getattr(resp, "finish_reason", ""),
                source=getattr(resp, "source", "content"),
                cost=getattr(resp, "cost", 0.0),
                role="main",
                provider=getattr(resp, "provider", None),
                generation_id=getattr(resp, "generation_id", None),
            )
            # A REPLY CUT OFF AT THE OUTPUT LIMIT IS NEVER RUN. Its command is
            # incomplete by definition, so it is neither parsed (a repaired brace would run half
            # a script) nor handed to the structurer, and the retry does not carry it back.
            cut_off = getattr(resp, "finish_reason", "") == "length"
            if cut_off:
                self._emit_parse("cut_off", attempt=attempt)
                last_err = "Your reply was cut off at the output limit."
                continue
            action = self._parse(content)
            if action:
                self._emit_parse("direct", attempt=attempt, tool=_cmd_name(action))
                return action
            # A control-only message (finish / stage-completion sentinel) carries no
            # JSON command — recognize it instead of format-retrying into a FormatError.
            sig = self._control_signal(content)
            if sig:
                self._emit_parse("control", attempt=attempt, signal=sig)
                return {"control": sig}
            last_err = "Could not parse a tool call from your message."
        if cut_off:
            self._emit_parse("format_error", attempt=max_retries)
            raise FormatError(f"{max_retries} replies in a row were cut off at the output limit")
        # structurer fallback (json/bash protocols, after direct parsing failed)
        action = self._structure(self._last_assistant)
        if action:
            self._emit_parse("structurer", attempt=max_retries, tool=_cmd_name(action))
            return action
        sig = self._control_signal(self._last_assistant)
        if sig:
            self._emit_parse("control", attempt=max_retries, signal=sig)
            return {"control": sig}
        self._emit_parse("format_error", attempt=max_retries)
        raise FormatError(last_err or "No parseable action.")

    def _emit_parse(self, path: str, *, attempt: int, tool: str = "", signal: str = "") -> None:
        self.events.emit(
            "on_parse_path", path=path, protocol=self.spec.action_protocol, attempt=attempt, tool=tool, signal=signal
        )

    def _control_signal(self, text: str) -> str | None:
        """Detect a command-less control message: finish or stage-completion."""
        if self._sentinel_finish(text):
            return "finish"
        if self.stages.staged and self.stages.check_completion(text, self.ctx()):
            return "stage_done"
        return None

    def _parse(self, content: str) -> dict | None:
        if self.spec.action_protocol == "bash_block":
            return parsing.extract_bash_action(content)
        return parsing.extract_json_action(content)

    def _cutoff_nudge(self, reply: str) -> list[dict]:
        """The retry after a reply cut off at the output limit: what happened, not the reply.

        THE CUT TEXT DOES NOT RIDE THE RETRY. A reply cut off at the limit can be a model
        repeating one word until the limit; sending it back as its own turn invites the same
        loop, so the retry carries only a description of what happened."""
        words = re.findall(r"[A-Za-z_][\w.-]*", reply[-6000:])
        top = Counter(words).most_common(1)
        looping = bool(top) and len(words) >= 50 and top[0][1] / len(words) > 0.3
        what = (
            f'It had started repeating itself ("{top[0][0]}" over and over).'
            if looping
            else f"It was {len(reply):,} characters long."
        )
        return [
            {
                "role": "user",
                "content": "Your last reply was cut off at the output limit before its command was complete, "
                f"so nothing ran. {what} Send the same step again in smaller pieces: write a long "
                "script to a file in parts (write_to_file, then str_replace), and keep any literal "
                "list short or put it in a file first. Reply with ONE JSON object: "
                '{"thoughts":"...","command":{"name":"<tool>","args":{...}}}',
            }
        ]

    def _format_nudge(self, err: str | None, reply: str = "") -> list[dict]:
        """The retry after an unparseable reply: the model's own reply as its own turn, then
        the ask for the action THAT reply described.

        THE REPLY RIDES THE RETRY. A retry that carried only an error line would make the
        model plan afresh, and the cheapest valid JSON would win over the action it had just
        described in prose. Only the latest reply is sent, as the model's own
        turn, and the nudge says what the retry is for and what it is not."""
        turn = [{"role": "assistant", "content": reply}] if reply else []
        return turn + [
            {
                "role": "user",
                "content": f"ERROR: {err}\n"
                "Your reply above is prose, not an action. Express the action it described "
                "as ONE strict JSON object: "
                '{"thoughts":"...","command":{"name":"<tool>","args":{...}}}\n'
                "Nothing about the candidate or the task changed; a format error is not a "
                "reason to drop the candidate or to start over.",
            }
        ]

    # ── execute ─────────────────────────────────────────────────────────────────
    def execute(self, action: dict) -> ToolResult:
        cmd = action.get("command", {})
        name, args = cmd.get("name", ""), cmd.get("args", {}) or {}

        # NOTE: no finish-sentinel text check here. execute() only runs once `action`
        # already carries a real parsed command (plan() returns {"control": "finish"}
        # instead, before execute() is ever called, for genuine command-less finish
        # replies -- see _control_signal). Re-checking self._last_assistant here would
        # substring-match the finish phrase against a response that legitimately
        # contains a real tool call, e.g. if a written file's content happens to
        # contain that phrase -- silently discarding the tool call instead of running
        # it. Confirmed exploitable; do not reintroduce.

        if self._is_repetition(name, args):
            raise NonTerminating("Repetition detected: choose a different tool or approach.")

        allow = self.stages.tools_allow(self.spec.tools)
        if allow and name not in allow:
            raise NonTerminating(
                f"Tool '{name}' is not available right now"
                + (f" (stage '{self.stages.name()}')" if self.stages.staged else "")
                + f". Available tools: {', '.join(allow)}."
            )

        # Stage tool guard: is THIS call admissible NOW, given run state? (The allow
        # list above is static membership; this is dynamic.) See StageSpec.tool_guard.
        guard_ref = self.stages.tool_guard() if self.stages.staged else None
        if guard_ref:
            veto = _resolve_hook(guard_ref)(self.ctx(), name, args)
            if veto:
                raise NonTerminating(str(veto))

        args = self.registry.resolve_args(name, args, fuzzy=True) if self.registry.has(name) else args
        result = self.registry.call(name, args, self.ctx())
        # Recorded with the candidate it was for and the cycle it happened on -- see
        # _repetition_key and the decay in _is_repetition. Note this runs only after a
        # SUCCESSFUL call, so a refused attempt never becomes the thing a later attempt is
        # measured against.
        self._recent_actions.append((self._repetition_key(name, args), self.cycle))
        self.events.emit("on_tool_result", name=name, args=args, result=result, stage=self.stages.name())

        # stage transition driven by the tool
        if result.transition_to:
            self._goto_stage(result.transition_to, "tool_transition")
        return result

    # ── stage transitions (emit on_stage_transition with the reason) ──────────────
    def _advance_stage(self, reason: str) -> None:
        frm = self.stages.name()
        self.stages.advance()
        to = self.stages.name()
        self._on_stage_changed()
        # Logged HERE rather than at the call sites. There are two callers: the sentinel path
        # (_handle_control) and the side_effect path (checked after a tool call); writing the
        # marker here means a stage completed via side_effect also leaves a trace in the
        # model's context.
        #
        # And the label was written after advancing, so `[stage_done:calibrate]` meant "brainstorm
        # just finished". It named the stage ENTERED while saying "done", i.e. the opposite of
        # what it meant, in the model's context on every subsequent turn.
        self._mem(
            f"[FINISHED STAGE: {frm} -- now in: {to or 'end'}]",
            f"stage complete ({reason})",
            role="user",
            site="loop.stage.finished",
        )
        self.events.emit("on_stage_transition", from_stage=frm, to_stage=to, reason=reason, kind="advance")

    def _goto_stage(self, name: str, reason: str) -> None:
        frm = self.stages.name()
        self.stages.goto(name)
        self._on_stage_changed()
        self.events.emit("on_stage_transition", from_stage=frm, to_stage=self.stages.name(), reason=reason, kind="goto")

    def _on_stage_changed(self) -> None:
        """A new stage is a fresh start for stall accounting: the previous stage's frozen
        fingerprint says nothing about whether the new one is making progress."""
        if self.spec.stall.reset_on_stage_change:
            self._last_progress_fp, self._stall_streak = None, 0
        # Entering a publish stage announces once, even if the sample is what it already was:
        # the announcement is addressed to a model that has just been handed a new stage.
        self._last_published_sig = None
        self._publish_worklist()

    def _publish_worklist(self, sync_only: bool = False) -> None:
        """Hand the harness-verified candidates to the deliverable.

        The model does not get to retype them. Everything the worklist enforces -- dedup, real
        SHAs, a witnessed fail-to-pass transition, a reproducible environment -- applies to
        `Worklist.delivered()`; publishing is what makes those guarantees reach `final_sample`
        instead of protecting a structure nobody reads.

        Two callers: entering a publish_on stage (announces to the model), and every cycle
        end with sync_only=True (a finished instance cashes out the MOMENT it finishes,
        so every reader of final_sample sees it before wrap-up). The continuous path is
        silent bookkeeping: same list, no message, so an interrupted run's deliverable is
        already on disk in the next snapshot.
        """
        w = getattr(self.spec, "worklist", None)
        if not (w and w.enabled and w.publish_to):
            return
        announcing = self.stages.name() in (w.publish_on or [])
        if not announcing and not sync_only:
            return
        try:
            from agentcore.agent.worklist import WORKLIST_KEY, Worklist

            store = getattr(self.working_set, WORKLIST_KEY, None)
            if not isinstance(store, dict):
                return
            wl = Worklist(w, store)
            delivered = wl.delivered()
            setattr(self.working_set, w.publish_to, delivered)
            # The denominator is published mechanically too: a rejected list reconstructed from
            # memory would be the retyping risk final_sample's publish exists to prevent. The
            # worklist holds the authoritative bookings; the model never retypes them.
            rejected = [
                {"key": k, "state": c.get("state"), "reason": str(c.get("fields", {}).get("_reason") or "")}
                for k, c in (store.get("candidates") or {}).items()
                if c.get("state") in ("REJECTED", "BLOCKED")
            ]
            setattr(self.working_set, "rejected_candidates", rejected)
            if not announcing:
                return
            # ANNOUNCE ON CHANGE, NOT ON STAGE MEMBERSHIP. `sync_only` runs at every cycle end,
            # and `announcing` stays true while the run stands in a publish stage, so without
            # this check the banner would re-fire every cycle. The stage-entry announcement and
            # the cash-out-immediately guarantee are unaffected; only the repeats go.
            # `_on_stage_changed` clears this so re-entering announces.
            # Membership is the signature: which keys are delivered, in order.
            from agentcore.agent.diversity import success_from_spec

            success = success_from_spec(w)
            signature = tuple(
                k for k in (store.get("order") or []) if (store["candidates"].get(k) or {}).get("state") == success
            )
            if signature == getattr(self, "_last_published_sig", None):
                return
            self._last_published_sig = signature
            # The zero case gets its own sentence: "0 verified candidate(s)... Do not rewrite
            # that list -- only these passed every check" can be read as the model's tool
            # outputs being REPLACED. There is no list to protect when the list is empty, and
            # a recurring banner must say it is not a reaction to whatever the model just did.
            if delivered:
                self._mem(
                    "[published]",
                    f"{len(delivered)} verified candidate(s) written to "
                    f"{w.publish_to} by the harness. Do not rewrite that list -- "
                    f"only these passed every check.",
                    role="user",
                    touch_last_turn=False,
                    site="loop.publish.banner",
                )
            else:
                self._mem(
                    "[published]",
                    f"{w.publish_to} is empty: no candidate has passed every "
                    f"check yet. This is a recurring harness status notice, "
                    f"not a response to your last action; your tools work "
                    f"normally.",
                    role="user",
                    touch_last_turn=False,
                    site="loop.publish.empty",
                )
            self.events.emit(
                "on_intervention",
                site="loop.publish.banner",
                kind="worklist_publish",
                detail=f"{w.publish_to}={len(delivered)}",
            )
        except Exception:
            pass

    def _quota_would_strand(self, target: str) -> str:
        """Why forcing into `target` now would strand a run that still owes instances, or "".

        The veto below asks the CURRENT stage's exit gate. It says nothing about whether the
        TARGET stage can do the work the finish gate will go on demanding. A collector can force `wrap_up` at a budget fraction below the one at which the quota
        gate opens, and a stage exit that is a bare side_effect vetoes nothing; the run would
        then sit in a stage with no `worklist_add`, refused for lacking candidates and unable to
        create any, waiting for the budget to climb until the gate opens.

        Deferring is safe here in exactly the way the other deferral requires: the quota gate's
        opening condition IS budget-linked (target met, or BUDGET_SPENT_FRACTION), so the
        threshold re-fires rather than being cancelled -- the case that docstring calls
        guaranteed. Scoped to a target that would LOSE tools -- against the current stage and
        every stage the jump skips -- so a mandatory later stage which is equally equipped is
        never starved. A normal wrap_up is a small share of a run, so the 10% still unspent
        at 0.90 is ample.
        """
        # The tools a force leaves behind are the CURRENT stage's and every stage it jumps
        # OVER . Comparing the target with the current stage only
        # would let a run in an early stage whose tools are all also granted by wrap_up "lose
        # nothing" and be forced past intermediate stages with no way back. Spec arithmetic
        # first: it is free, and it decides whether the quota (a full working-set snapshot)
        # needs asking at all. An unknown target is left to `_goto_stage`, which reports it.
        if not all(hasattr(self.stages, m) for m in ("has", "tools_allow", "between")):
            return ""  # no stage list to compare against (an unstaged agent, a stub)
        if not self.stages.has(target):
            return ""
        allow = list(getattr(getattr(self, "spec", None), "tools", None) or [])

        def granted(name):
            return set(self.stages.tools_allow(allow, name))

        path = [self.stages.name(), *self.stages.between(target)]
        there = granted(target)
        lost = sorted(set().union(*(granted(n) for n in path)) - there)
        if not lost:
            return ""  # equally equipped: moving there strands nothing
        try:
            from agentcore.agent.quota import quota_not_met

            short = quota_not_met(self.ctx())
        except Exception:  # noqa: BLE001 - a broken gate must never trap a run in a stage
            return ""
        if not short:
            return ""
        holders = [repr(n) for n in path if granted(n) & set(lost)]
        return (
            f"it still owes instances and {target!r} does not grant {', '.join(lost)}, "
            f"which {', '.join(holders)} still {'does' if len(holders) == 1 else 'do'}. "
            f"{short}"
        )

    def _apply_budget_thresholds(self) -> None:
        """budgets.thresholds: [{at, action, target, message}] -- couples budget
        consumption to a control action. Currently supports action: force_state,
        which forces a stage transition regardless of whether the model thinks it's
        done, so a mandatory later stage (e.g. critique) can't be starved of its whole
        budget by an earlier one that runs long. Each threshold entry only ever fires once
        (BudgetController.thresholds_triggered() tracks that), and a bad/unreachable
        target is reported as a nudge rather than crashing the run.

        A FORCE MAY NOT SKIP A GATE THAT REFUSES. Skipping
        `_stage_exit_veto` on this path would let `force_state: wrap_up` fire at 0.85 while the quota gate
        opens at 0.90, leaving cycles in a stage with no `worklist_add` and no
        `worklist_record`, unable to register, advance or finish. A vetoed threshold is DEFERRED rather than dropped: `thresholds_triggered()` yields
        each entry exactly once, so discarding one would silently delete the backstop. It re-fires
        on the first cycle the guard allows -- guaranteed for the quota gate, whose own opening
        condition is the meter that armed the threshold continuing to climb.

        THAT LAST SENTENCE IS THE WHOLE PRECONDITION, and it is only true of a named `guard`.
        A bare `side_effect` has no budget-linked opening condition, so deferring on one
        cancels the backstop instead of postponing it. So this path asks `_stage_exit_veto(side_effect=False)`; the model-declared
        exit and the stall watchdog still honour both.
        """
        if not self.stages.staged:
            return
        # Carried as (threshold, already_announced) so the announce flag never touches the
        # threshold dict, which belongs to the spec and is shared.
        pending = self._deferred_thresholds + [
            (th, False) for th in self.budget.thresholds_triggered() if th.get("action") == "force_state"
        ]
        self._deferred_thresholds = []
        for th, announced in pending:
            target = th.get("target")
            if not target or target == self.stages.name():
                continue
            try:
                # GUARD ONLY. A bare `side_effect` has no budget-linked opening condition, so
                # deferring on it does not postpone this threshold, it cancels it. See `_stage_exit_veto`.
                veto = self._stage_exit_veto(side_effect=False)
            except Exception:  # noqa: BLE001 - a broken guard must never trap a run in a stage
                veto = ""
            if not veto:
                veto = self._quota_would_strand(target)
            if veto:
                self._deferred_thresholds.append((th, True))
                if not announced:
                    self._mem(
                        "[budget_threshold]",
                        f"Budget threshold reached ({th.get('at')!r}), but this stage's "
                        f"exit gate refuses to move on to {target!r}: {veto} The "
                        f"transition will happen as soon as that gate opens -- keep "
                        f"working here or in the stages before {target!r}: they still "
                        f"hold the tools for it.",
                        role="user",
                        site="loop.budget.deferred",
                    )
                    self.events.emit(
                        "on_budget_threshold", at=th.get("at"), target=target, action="deferred_by_guard", message=veto
                    )
                continue
            message = th.get("message") or (
                f"Budget threshold reached ({th.get('at')!r}) -- forced transition to "
                f"stage {target!r} so it isn't starved of the remaining budget."
            )
            try:
                self._goto_stage(target, "budget_threshold")
            except ValueError as e:
                message = f"budget threshold fired with an invalid target: {e}"
            self._mem("[budget_threshold]", message, role="user", site="loop.budget.forced")
            self.events.emit("on_budget_threshold", at=th.get("at"), target=target, message=message)

    # ── the harness convicting itself ─────────────────────────────────────────
    HALT_ON_HARNESS_FAULT_ENV = "BENCH_HALT_ON_SYSTEMIC_HARNESS_FAULT"

    def _halt_on_harness_fault(self) -> None:
        """End the run when the worklist has recorded a SYSTEMIC, harness-authored fault.

        Two independent signals have to coincide before this fires: the refusal was the
        harness's own fault (`Worklist._harness_fault` flagged its shape -- the gate said it
        was broken), and `Worklist._note_refusal_fingerprint` found that shape on SYSTEMIC_AT
        unrelated candidates, pooled across the fan-out. Neither alone ends anything: a lone
        harness fault can be a transient, and a systemic shape without the flag is three
        candidates failing a correct gate the same way -- announced, never acted on.

        This is not the agent finishing short. The target stays binding on the agent; this is
        the harness refusing to keep charging for a gate it has itself declared broken. The fix
        is a harness change and a fresh run (resumed state carries exactly what a fix cannot
        help), and every cycle spent first is waste. Set BENCH_HALT_ON_SYSTEMIC_HARNESS_FAULT=0 to fall back to announcing only.
        """
        from os import environ

        if environ.get(self.HALT_ON_HARNESS_FAULT_ENV, "1") == "0":
            return
        from agentcore.agent.exceptions import HarnessFault
        from agentcore.agent.worklist import WORKLIST_KEY

        store = getattr(self.working_set, WORKLIST_KEY, None)
        # POP, not read: the flag has done its job once the run ends on it. Left in the store
        # it would ride into the snapshot, and a resume after a fix would halt again at the end
        # of its first cycle on the old verdict.
        fault = store.pop("_harness_fault", None) if isinstance(store, dict) else None
        if not fault:
            return
        msg = (
            f"harness fault at {fault.get('state')}: the harness marked this refusal as its "
            f"own error and then found it on {fault.get('candidates')} unrelated candidates "
            f"({fault.get('refusals')} refusals) -- {str(fault.get('shape'))[:160]!r}. "
            f"No candidate can pass a gate the harness says is broken; stopping so the gate "
            f"is fixed and the run relaunched."
        )
        self._mem("[harness_fault]", msg, role="user", site="loop.harness_fault.halt")
        self.events.emit(
            "on_intervention",
            site="loop.harness_fault.halt",
            kind="harness_fault_halt",
            detail=msg,
            state=fault.get("state"),
        )
        raise HarnessFault(msg)

    # ── stall detection (no observable progress) ──────────────────────────────
    @staticmethod
    def _dig(obj: Any, path: str) -> Any:
        """Follow a dotted path into nested dicts/objects; None if any hop is missing."""
        cur = obj
        for part in path.split("."):
            if cur is None:
                return None
            cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        return cur

    @staticmethod
    def _progress_value(v: Any) -> Any:
        """Reduce a watched value to what 'progress' means for it.

        Sizes for containers, so appending a candidate registers while merely rewriting one
        does not; the value itself for scalars, so a flag flipping (e.g. calibration passing)
        registers. Falls back to a truncated repr for anything else -- bounded so a huge blob
        can't make the fingerprint expensive to compute every cycle.
        """
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, (list, tuple, set, dict, str)):
            return len(v)
        return repr(v)[:120]

    def _progress_fingerprint(self) -> tuple:
        """What the run has actually achieved, as a comparable tuple.

        Includes the side-effect count so a cycle that really ran a command or wrote a file is
        never mistaken for a stall, even when the watched fields legitimately don't move yet
        (a long build, a slow clone). Without that, honest slow work would trip the detector.
        """
        # ONE snapshot per cycle: this is a full pydantic dump of the entire working set --
        # source pool, every candidate with its fields and history, final_sample -- so its
        # cost grows with everything the run has found. _worklist_progress reads the same dict,
        # so it is handed this one.
        ws = self.working_set.snapshot()
        watched = tuple(self._progress_value(self._dig(ws, p)) for p in self.spec.stall.watch)
        return (self.stages.name(), self._n_effectful_actions(), self._worklist_progress(ws)) + watched

    @staticmethod
    def _worklist_advances(ws) -> dict:
        """How many candidates have LEFT their entry state, by the state they reached.

        The worklist is the run's deliverable ledger. Every worklist tool sits in
        `bookkeeping_tools` (calling one proves nothing), and `watch` is per-task config that
        may be empty, so without this a candidate reaching DELIVERED, the single most
        meaningful thing the run does, would count for nothing to the stall detector.

        Measured against the ENTRY state (`history[0]`, written once by `Worklist.add` and
        never mutated) rather than against history length, because history also grows on
        refusals: a refusal storm must still read as frozen, and it does -- a refusal
        leaves the state where it was.

        Registrations are deliberately not progress either. Bulk-adding thirty rows advances
        nobody.

        ONLY THE SUCCESS TERMINAL COUNTS. Counting any other state would make the detector
        unescalatable: a run grinding the same query while blocking a candidate every few cycles
        would change the tuple each time, reset the streak, and never reach `force_after`.
        Terminating a candidate is real work and it is honestly recorded, but a run can block
        candidates forever without producing a single instance, so it is not progress toward the
        deliverable. Honest slow work is still safe: `_n_effectful_actions()` sits in the same
        fingerprint, so a genuinely new command is never a frozen cycle.

        The success state is `terminal[0]` as the SPEC declared it -- the same element
        `Worklist.delivered()` reads -- never the literal "DELIVERED". A benchmark that names its
        terminal something else would package zero instances if code here guessed that
        string. With no terminal declared there is no way to tell a delivery from a dead end, so
        nothing counts rather than everything.
        """
        from agentcore.agent.worklist import WORKLIST_KEY

        store = (ws or {}).get(WORKLIST_KEY)
        if not isinstance(store, dict):
            return {}
        terminal = store.get("terminal") or []
        success = terminal[0] if terminal else None
        if success is None:
            return {}
        out: dict[str, int] = {}
        for cand in (store.get("candidates") or {}).values():
            if not isinstance(cand, dict):
                continue
            hist = cand.get("history") or []
            entry = hist[0] if hist else None
            state = cand.get("state")
            if state == success and state != entry:
                out[state] = out.get(state, 0) + 1
        return out

    def _worklist_progress(self, ws) -> tuple:
        """`ws` is the caller's snapshot: _progress_fingerprint already has one, and taking a
        second here doubled a full pydantic dump of the whole working set every cycle."""
        return tuple(sorted(self._worklist_advances(ws).items()))

    @staticmethod
    def _effect_head(label: str) -> str:
        """Normalize an action label to the part that says WHAT is being touched.

        For a shell command that is everything before the first pipe/;/&& -- the data
        source -- because the post-processing tail is where a looping model varies:
        A model can run the SAME search many times with only the formatting tail varying.
        Byte-identity misses that loop entirely; the head does not.

        LEADING `cd`/`export`/`env` SEGMENTS ARE SKIPPED FIRST. Splitting on `&&` first would reduce
        `cd /root/clone && git cat-file -t <sha> && curl .../issues/N` to `cd /root/clone` -- a
        segment that touches nothing and is identical for every command run against that clone,
        so distinct commands would collapse to a few heads and a productive run would be
        nudged as frozen.
        """
        head, sep_paren, rest = label.partition("(")
        # Peel prefixes that only set up the environment for what follows. The segment still
        # carries the args repr (`{'command': 'cd /tmp`), so the keyword is read from the text
        # after the last quote rather than from the start of the segment.
        while True:
            first, sep, tail = rest.partition("&&")
            if not sep:
                break
            word = re.split(r"['\"]", first)[-1].strip().split(" ", 1)[0]
            if word not in ("cd", "export", "env", "set", "source", "."):
                break
            rest = tail
        rest = rest.strip()
        for sep in ("|", ";", "&&"):
            rest = rest.split(sep, 1)[0]
        # Trailing punctuation belongs to the label, not to the command: without this a command
        # run bare and the same command piped into `head` differ only by the `'})` the repr left
        # behind, and count as two actions.
        rest = rest.rstrip("\"' )}]")
        return " ".join(f"{head}{sep_paren}{rest}".split())[:120]

    def _n_effectful_actions(self) -> int:
        """How many DISTINCT effectful things the run has done outside the working set.

        Counting every action would defeat the detector twice over: `record` appends to
        the action log like anything else, so key-spam would look busy; and a re-run
        of the same command is not new work, so counting raw shell calls would let a run
        re-fetch one search indefinitely while the fingerprint moved every cycle and the
        detector never fired. Distinct normalized
        heads (see _effect_head) count; repeats and formatting variations do not. A
        genuinely slow build being POLLED with the same command therefore accrues
        streak until a watched field moves -- that costs a diagnostic nudge, which is
        cheap, and the force path still respects stage-exit guards.

        Entries the LOOP writes about itself -- `[stall]`, `[error]`, `[nudge]`,
        `[budget_threshold]` -- are excluded too, and that exclusion is load-bearing
        rather than cosmetic: the stall nudge appends `[stall]` to this very log, so
        counting it would make the nudge look like progress and reset the streak it
        just measured.
        Agent actions are labelled `name(args)`; loop annotations are bracketed.
        """
        book = set(self.spec.stall.bookkeeping_tools or ())
        heads = set()
        for label, _ in self.action_log:
            label = str(label)
            if label.startswith("["):
                continue  # loop's own annotation, not an agent action
            if label.split("(", 1)[0].strip() in book:
                continue  # bookkeeping; only watched paths vouch for it
            # Class reference, not self: several test stubs borrow this method alone.
            heads.add(LoopAgent._effect_head(label))
        return len(heads)

    def _check_stall(self) -> None:
        """Escalate when nothing observable has changed for several cycles.

        Graded on purpose: explain first (the model may simply not realize which field is
        blocking it), force the stage only if explaining didn't work, and never abort -- an
        agent that is stuck in one stage may still do useful work in the next one, and budget
        meters already bound the total damage.
        """
        st = self.spec.stall
        if not st.enabled:
            return
        fp = self._progress_fingerprint()
        if fp != self._last_progress_fp:
            # Clear the starvation excuse HERE too, not only on the no-progress path below:
            # a single provider retry followed by real work would otherwise leave the flag set
            # and excuse a later, unrelated frozen cycle.
            # The flag means "THIS cycle was starved"; without this it meant "some cycle since
            # the last excused stall was".
            self._last_progress_fp, self._stall_streak = fp, 0
            self._cycle_was_starved = False
            return
        if getattr(self, "_cycle_was_starved", False):
            self._cycle_was_starved = False
            return  # the provider stalled, not the agent; do not count it
        self._stall_streak += 1

        if st.force_after and self._stall_streak >= st.force_after:
            # Forcing needs somewhere to go. An unstaged agent, or one already in its final
            # stage, has nowhere -- keep nudging there rather than logging an action that
            # silently does nothing. Budget meters remain the backstop in that case.
            nxt = self.stages.peek_next() if self.stages.staged else None
            if nxt:
                # THE INVARIANT LIVES HERE TOO: a stage whose exit guard refuses may not be
                # force-exited by the stall watchdog. Provider starvation can make
                # many cycles pass with no progress; without the veto the watchdog would shove
                # the run into wrap_up -- a stage with no mining tools -- and a quota guard wired
                # only to the SENTINEL path would never get a say. A guarded
                # veto here keeps the run in the stage that can still act; the budget
                # threshold force remains the deliberate, guard-exempt backstop.
                # AND THE STRAND VETO. A mining stage guarded only by its sentinel would otherwise
                # still be walked through by a stall force into a stage that cannot mine. The loop holds the invariant; the config does not have to.
                veto = self._stage_exit_veto() or self._quota_would_strand(nxt)
                if veto:
                    self._stall_streak = 0
                    self._mem(
                        "[stall]",
                        f"No progress for {st.force_after} cycles, but the "
                        f"stage's exit guard refuses to move on: {veto}",
                        role="user",
                        site="loop.stall.vetoed",
                    )
                    self.events.emit(
                        "on_stall",
                        action="vetoed_by_guard",
                        target=nxt,
                        stage=self.stages.name(),
                        cycles=st.force_after,
                        frozen=self._frozen_description(),
                    )
                    return
                self._stall_streak = 0
                msg = (
                    f"No progress for {st.force_after} cycles -- forcing the transition to "
                    f"{nxt!r}. Whatever you have now is what this stage produced."
                )
                # The stage is read BEFORE the transition: what stalled is the stage being
                # left, and where it went is already `target`.
                left = self.stages.name()
                self._goto_stage(nxt, "stall")
                self._mem("[stall]", msg, role="user", site="loop.stall.forced")
                self.events.emit(
                    "on_stall",
                    action="force_state",
                    target=nxt,
                    stage=left,
                    cycles=st.force_after,
                    frozen=self._frozen_description(),
                )
                return
            # fall through: re-nudge periodically instead of going silent. A REPEAT does not
            # claim the echoed-turn slot: the first nudge is the consequence of the model's
            # reply and belongs there; the tenth identical one is a status line, and letting
            # it overwrite the real tool output is the same blinding the publish banner did.
            if (self._stall_streak - st.nudge_after) % max(st.force_after, 1) == 0:
                frozen = self._frozen_description()
                self._mem(
                    "[stall]",
                    f"Still no measurable progress after {self._stall_streak} cycles: {frozen}.",
                    role="user",
                    touch_last_turn=False,
                    site="loop.stall.nudge",
                )
                self.events.emit(
                    "on_stall", action="nudge", stage=self.stages.name(), cycles=self._stall_streak, frozen=frozen
                )
            return

        if st.nudge_after and self._stall_streak == st.nudge_after:
            frozen = self._frozen_description()
            msg = f"You have used {st.nudge_after} cycles without changing anything measurable: {frozen}."
            # The recording advice is for an agent that HAS a record tool; to one without
            # (the worker, the searching agent) it names a thing it was never told about.
            try:
                has_record = "record" in set(self.registry.names(self.stages.tools_allow(self.spec.tools)))
            except Exception:  # noqa: BLE001
                has_record = True
            if has_record:
                msg += (
                    " Recording under a NEW key name does not count as progress and "
                    "does not satisfy any gate -- if a gate rejected a field, re-record THAT "
                    "field, under its original name, in the shape the gate asked for. If you are "
                    "finished with this stage, say so and move on."
                )
            self._mem("[stall]", msg, role="user", site="loop.stall.nudge")
            self.events.emit("on_stall", action="nudge", stage=self.stages.name(), cycles=st.nudge_after, frozen=frozen)

    def _frozen_description(self) -> str:
        """Name the actual stuck values, so the nudge is diagnostic rather than vague."""
        ws = self.working_set.snapshot()
        parts = [f"stage={self.stages.name()}"]
        # When a worklist exists it is what the run is FOR, so name it first -- advice about
        # recording under new key names is useless to an agent whose problem is that no
        # candidate has advanced.
        from agentcore.agent.worklist import WORKLIST_KEY

        adv = self._worklist_advances(ws)
        if isinstance((ws or {}).get(WORKLIST_KEY), dict):
            n = len((ws[WORKLIST_KEY].get("candidates") or {}))
            parts.append(
                "candidates advanced="
                + (", ".join(f"{k}:{v}" for k, v in sorted(adv.items())) or "none")
                + f" (of {n} registered)"
            )
        for p in self.spec.stall.watch:
            val = self._dig(ws, p)
            parts.append(
                f"{p}="
                + (
                    "absent"
                    if val is None
                    else (f"{len(val)} item(s)" if isinstance(val, (list, dict, tuple, set)) else str(val)[:2000])
                )
            )
        return ", ".join(parts)

    def _repetition_key(self, name: str, args: dict) -> tuple:
        """(tool, args, WHICH CANDIDATE it was for).

        The same call against a different candidate is a different action, and a byte
        comparison alone cannot see that: after recording {clone_path: ..., issue_kind: ...} for
        one candidate, the next candidate legitimately needs the same two values, and refusing
        them as a repetition leaves no other approach.

        Per-candidate looping is still caught, and better than a byte comparison ever did it:
        `_strike` counts refusals at one state and defers then blocks, and `tick_current` counts
        cycles spent on one candidate. Both name the candidate and say what to do instead.
        """
        return (name, str(args), self._current_candidate_key(), self._sandbox_generation(args))

    def _sandbox_generation(self, args: dict) -> int:
        """How many times the sandbox this call names has been created.

        The second thing a byte comparison cannot see: if the model removes a sandbox and then
        recreates it, the create call is byte-identical to the earlier one although the
        container was DELETED in between; the world moved and the key did not. Generation makes an action against a rebuilt sandbox a different
        action from the same call against the one it replaced, which also covers container_exec:
        re-running a build in a fresh sandbox is new work, not a repeat.
        """
        name = args.get("name") or args.get("sandbox") if isinstance(args, dict) else None
        if not name:
            return 0
        try:
            from agentcore.tools import containers

            return containers.generation(self.working_set, str(name))
        except Exception:  # noqa: BLE001 - repetition accounting must never end a run
            return 0

    def _current_candidate_key(self) -> str:
        """Which candidate the worklist is serving, or "" when there is no worklist."""
        if not (getattr(self.spec, "worklist", None) and self.spec.worklist.enabled):
            return ""
        try:
            from agentcore.agent.worklist import WORKLIST_KEY, Worklist

            store = getattr(self.working_set, WORKLIST_KEY, None)
            if not isinstance(store, dict):
                return ""
            cur = Worklist(self.spec.worklist, store).current()
            return cur[0] if cur else ""
        except Exception:  # noqa: BLE001 - repetition accounting must never end a run
            return ""

    def _is_repetition(self, name: str, args: dict) -> bool:
        rep = self.spec.repetition
        if not rep.enabled or name in rep.exempt_tools:
            return False
        if self.registry.has(name) and self.registry.meta(name).get("idempotent_ok"):
            return False
        key = self._repetition_key(name, args)
        decay = int(getattr(rep, "decay_cycles", 20) or 0)
        floor = (self.cycle - decay) if decay > 0 else None
        seen = sum(1 for k, at in self._recent_actions[-rep.window :] if k == key and (floor is None or at > floor))
        return seen >= max(int(getattr(rep, "threshold", 5) or 5), 1)

    # ── summarize ────────────────────────────────────────────────────────────────
    def summarize(self, action: dict, result: ToolResult) -> None:
        cmd = action.get("command", {})
        name, args = cmd.get("name", ""), cmd.get("args", {})
        summary = self.reducer.reduce(name, result)
        obs_role = "tool" if self.ctxspec.observation_role == "tool" else "user"
        from agentcore.observability.sites import action_fingerprint

        self._cycle_action = {"name": name, "args": args}
        # A gate that refused inside this tool left the site and the expectation for the
        # message the model is about to see (worklist._note_refusal); the result carries them.
        shown = (getattr(self, "_scratch", None) or {}).pop("_shown", None) or {}
        self._mem(
            f"{name}({args})",
            summary,
            role=obs_role,
            site=shown.get("site") or "tool.result",
            expects=shown.get("expects"),
        )
        self._last_action_fp = action_fingerprint(self._cycle_action)
        # let stages auto-advance on state that only makes sense to (re-)check after a
        # real tool call ran (side_effect). Sentinel/fuzzy are deliberately EXCLUDED
        # here: this call site runs after `action` already carried a real, successfully
        # parsed command, so self._last_assistant is known to contain that command's
        # JSON/args -- a naive substring search for a sentinel/fuzzy token over that
        # full text can false-positive-match text embedded in the command's own
        # arguments (e.g. a written file's content). Genuine command-less
        # sentinel/fuzzy completions are already caught correctly in plan(), via
        # _control_signal, before a command is ever parsed/executed.
        if self.stages.staged:
            reason = self.stages.match_completion(self._last_assistant, self.ctx(), kinds={"side_effect"})
            if reason:
                self._advance_stage(reason)
                self._noop_streak.clear() if self.spec.repetition.reset_on_stage_change else None

    # ── finish detection ─────────────────────────────────────────────────────────
    def _sentinel_finish(self, text: str) -> bool:
        token = self.core.get("sentinels", {}).get("finish", "goals accomplished, end the task")
        return parsing._fix_escaped_newlines(text or "").lower().find(token.lower()) >= 0  # type: ignore

    def should_finish(self) -> bool:
        if self._finished:
            return True
        stop, terminal, reason = self.budget.should_stop()
        if stop:
            self._reason = reason
            if terminal:
                raise BudgetExhausted(reason)
            # A CLEAN FINAL CHECK IS NOT LOST TO THE CLOCK: a check that cleared every row may
            # run while the wall clock runs out, before the reply that would deliver it. One more cycle, once per
            # run, while that check is still waiting (bench_check sets it, bench_deliver and
            # drop_candidate clear it).
            scratch = getattr(self, "_scratch", None) or {}
            if scratch.get("deliverable") and not scratch.get("_delivery_grace_used"):
                scratch["_delivery_grace_used"] = True
                log = getattr(self, "_log", None)
                if callable(log):
                    log(
                        f"{reason}; one more cycle to deliver the clean final check of "
                        f"{scratch['deliverable'].get('slug')}"
                    )
                return False
            return True  # finish_cycle: graceful stop
        # staged agent: finished when the stage controller is done
        if self.stages.staged and self.stages.done:
            self._finished, self._success, self._reason = True, True, "all stages complete"
            return True
        return False

    # ── run ─────────────────────────────────────────────────────────────────────
    def _on_finish_request(self, result: ToolResult) -> bool:
        """A tool/sentinel asked to finish. Optionally gate via a judge oracle.

        Returns True if the loop should actually stop.
        """
        judge = self.hooks.get("validate_success") or self.hooks.get("judge")
        grade_ref = self.spec.oracle.grade
        if grade_ref and not judge:
            judge = _resolve_hook(grade_ref)
        if judge is None:
            self._success = result.structured.get("success_declared", True) if result.structured else True
            self._submitted = result.structured
            return True
        verdict = judge(self.ctx())
        # judge may return OracleResult, (verdict,msg), or bool
        if isinstance(verdict, OracleResult):
            accepted = verdict.verdict != "REJECT_CONTINUE"
            self._emit_oracle_check(verdict, source="judge", accepted=accepted)
            if not accepted:
                self._refuse_finish(result, verdict.message or verdict.reason)
                return False
            self._success = verdict.success
            return True
        if isinstance(verdict, tuple):
            v, msg = verdict
            accepted = v != "REJECT_CONTINUE"
            self._emit_oracle_check(
                OracleResult(success=(v == "ACCEPT"), reason=msg or ""), source="judge", accepted=accepted
            )
            if not accepted:
                self._refuse_finish(result, msg)
                return False
            self._success = v == "ACCEPT"
            return True
        self._success = bool(verdict)
        self._emit_oracle_check(OracleResult(success=self._success), source="judge", accepted=True)
        return True

    def _emit_oracle_check(self, res: OracleResult, *, source: str, accepted: bool) -> None:
        self.events.emit(
            "on_oracle_check",
            source=source,
            accepted=accepted,
            success=res.success,
            done=res.done,
            score=res.score,
            reason=res.reason,
            sub_signals=res.sub_signals,
        )

    def _refuse_finish(self, result: ToolResult, reason: str) -> None:
        """Put the oracle's refusal INTO the refused call's own result, untruncated.

        The result is summarized after this returns, so rewriting it is what makes the reason
        the model's observation for the exact call it refused. Otherwise the result
        would hold the model's own words and the reason would go to a truncated side-channel
        log line, with the actionable part (written last) cut, and the model would retry blind.
        """
        reason = (reason or "the finish oracle refused; no reason was given").strip()
        result.finished = False
        result.error = reason
        result.display_text = f"NOT ACCEPTED -- the run is not finished. {reason}"
        result.truncation = "none"
        # Same streak accounting as any rejected no-op, but WITHOUT the extra memory line: the
        # reason is already the observation, and a duplicate entry is prompt noise.
        self._noop_streak["validation"] = self._noop_streak.get("validation", 0) + 1

    def _inject_noop(self, cls: str, message: str) -> None:
        self._noop_streak[cls] = self._noop_streak.get(cls, 0) + 1
        self._mem(f"[{cls}_rejected]", message[:2000], role="user", site="loop.action.rejected")

    # ── human-in-the-loop primitives (driven by an optional CycleController) ──────
    def inject(self, text: str, role: str = "user") -> None:
        """Add a human instruction the agent sees next cycle (compact: enters history)."""
        text = (text or "").strip()
        if not text:
            return
        self._mem("[injected]", text, role=role, site="loop.injected")
        self.events.emit("on_intervention", site="loop.injected", kind="inject", detail=text[:2000])

    def edit_last_memory(self, new_text: str) -> None:
        """Rewrite the most recent observation before it is re-rendered into the prompt."""
        if not self.action_log:
            return
        label, _old = self.action_log[-1]
        self.action_log[-1] = (label, new_text)
        # The echoed slot holds a copy. Left alone, the OLD text would be echoed and the new
        # text would be the entry the replay de-duplicated away, so the edit would never reach
        # the model.
        turn = self._last_turn or {}
        if turn.get("index") == len(self.action_log) - 1 or turn.get("observation") == _old:
            turn["observation"] = new_text
        if self.ctxspec.mode != "compact" and self.transcript:
            for m in reversed(self.transcript):
                if m.get("role") in ("user", "tool"):
                    m["content"] = new_text
                    break
        self.events.emit("on_intervention", site="loop.edited", kind="edit", detail=f"{label}: {str(new_text)[:2000]}")

    def request_stop(self, reason: str = "stopped by user") -> None:
        """Finish the run now (graceful); leaves a clean snapshot for `agent resume`."""
        self._finished = True
        self._reason = reason
        self.events.emit("on_intervention", site="loop.stopped", kind="stop", detail=reason)

    def _stage_exit_veto(self, side_effect: bool = True) -> str:
        """A stage's `completion.guard` may refuse a model-declared exit, with a reason.

        `guard` names an entry in the spec's `hooks:` map; the callable gets the ToolContext
        and returns "" to allow the exit or a reason to refuse it. Fail-open on any guard
        error: a broken guard must degrade to the old behaviour, never trap a run in a stage.

        `side_effect=False` asks for the GUARD-ONLY verdict, and only the budget-threshold path
        asks for it -- see the block below and `_apply_budget_thresholds`.
        """
        c = self.stages.current() if self.stages.staged else None
        comp = (getattr(c, "completion", None) or {}) if c else {}

        # A side_effect completion IS a mechanical exit gate: the stage ends when the
        # predicate holds, and not before. The stall watchdog honours it too, not only
        # `guard`: forced out of a stage whose predicate never held, a run would land in the
        # next stage with nothing to work from. Same invariant as the guard veto: forcing may
        # not skip a gate that refuses. Budget-threshold forces are vetoed by a named guard only
        # (deferred, not dropped); see below.
        #
        # AND THE BUDGET PATH IS EXEMPT AGAIN, FOR THIS GATE ONLY. A deferral is only a
        # deferral if the gate can open. A named `guard` wired to the budget meter that armed
        # the threshold self-opens; a bare `side_effect` is an expression over working-set
        # state with no budget link at all, so a worker that can never satisfy it could never
        # reach the stage that publishes, and would run out of wallclock first.
        #
        # The other two forced-transition paths keep honouring it, because neither is armed
        # by a budget meter: the model-declared exit (saying the magic words is exactly what a
        # side_effect gate exists to refuse) and the stall watchdog.
        if side_effect and "side_effect" in comp:
            try:
                if not self.stages._eval_side_effect(comp["side_effect"], self.ctx()):
                    return (
                        f"stage {c.name!r} completes only when its side_effect gate "
                        f"holds, and it does not yet: {comp['side_effect']}"
                    )
            except Exception:  # noqa: BLE001 - a broken predicate must not trap the run
                pass

        guard_name = comp.get("guard")
        if not guard_name:
            return ""
        fn = self.hooks.get(guard_name)
        if fn is None:
            return ""
        try:
            return str(fn(self.ctx()) or "")
        except Exception:  # noqa: BLE001
            return ""

    def _tick_worklist_cycle(self) -> None:
        """Charge this cycle to the current candidate; surface the harness's nudge if it has one.

        THE one place that knows a cycle has begun, which is why the accounting lives here rather
        than in render (called more than once per cycle) or in the tools (a cycle that calls no
        worklist tool is exactly the kind being counted). A worklist fault must never end a run,
        so everything here is best-effort.
        """
        if not (getattr(self.spec, "worklist", None) and self.spec.worklist.enabled):
            return
        try:
            from agentcore.agent.worklist import WORKLIST_KEY, Worklist

            store = getattr(self.working_set, WORKLIST_KEY, None)
            if not isinstance(store, dict):
                return
            wl = Worklist(self.spec.worklist, store, ctx=self.ctx(), ws=self.working_set)
            msg, repeat, site = wl.tick()
            if msg:
                # A REPEAT DOES NOT CLAIM THE ECHOED-TURN SLOT: a re-nudge would be shown under
                # "OUTPUT OF YOUR LAST ACTION" in place of the tool result. The worklist says which nudges are
                # repeats; the first one and a deferral keep the slot.
                self._mem("[worklist]", msg, role="user", touch_last_turn=not repeat, site=site)
                self.events.emit("on_slow_candidate", message=msg)
        except Exception:  # noqa: BLE001
            pass

    def _before_cycle(self) -> None:
        """Cycle-boundary HITL seam. No-op unless a controller is attached; a controller
        fault must never crash the run."""
        if self.controller is None:
            return
        try:
            self.controller.before_cycle(self)
        except (KeyboardInterrupt, EOFError):
            self.request_stop("interrupted by user")
        except Exception as e:  # noqa: BLE001 - a controller bug is recorded, not fatal
            self._mem("[controller_error]", f"{type(e).__name__}: {e}", role="user", site="loop.controller.error")

    def _handle_control(self, signal: str) -> None:
        """Handle a command-less control message (finish / stage completion)."""
        self._cycle_action = {"name": f"control:{signal}", "args": {}}
        if signal == "finish":
            res = ToolResult(finished=True, structured={"success_declared": True})
            if self._on_finish_request(res):
                self._finished = True
            elif res.display_text:
                # No tool call to carry the refusal here (a sentinel finish is command-less),
                # so the reason goes in as its own untruncated memory line instead.
                self._mem("[finish_refused]", res.display_text, role="user", site="loop.finish.refused")
        elif signal == "stage_done":
            veto = self._stage_exit_veto()
            if veto:
                # Refuse the exit HERE, where the model still has this stage's tools. The
                # finish oracle fires in wrap_up -- a stage with no worklist or container
                # tools, so "you have 3 unfinished candidates" would be a demand the model has
                # no means left to satisfy.
                self._mem("[stage_exit_refused]", veto, role="user", site="loop.stage_exit.refused")
            else:
                self._advance_stage("stage_done_control")
                if self.spec.repetition.reset_on_stage_change:
                    self._noop_streak.clear()

    def run(self) -> RunResult:
        self.events.emit("on_attempt_start", agent=self)
        try:
            while not self.should_finish():
                self._before_cycle()  # human-in-the-loop seam (inject/edit/stop)
                if self._finished:  # a controller asked to stop
                    break
                self.cycle += 1
                self._cycle_action, self._cycle_parsed = None, False
                self.budget.on_event("cycle")
                self._apply_budget_thresholds()
                self._tick_worklist_cycle()
                self.events.emit("on_cycle_start", stage=self.stages.name(), step=len(self.action_log))
                try:
                    action = self.plan()
                    self._cycle_parsed = True
                    self._record_assistant()  # transcript modes: record the model's turn
                    if "control" in action:
                        self._handle_control(action["control"])
                    else:
                        self.budget.on_event("tool_call")
                        result = self.execute(action)
                        # The finish oracle runs BEFORE the result is summarized, because on a
                        # refusal it rewrites the result to carry the reason. Summarizing first would make the
                        # observation for a refused goals_accomplished the model's own words
                        # echoed back, with the reason on a truncated side channel.
                        if result.finished and self._on_finish_request(result):
                            self._finished = True
                        self.summarize(action, result)
                    self._error_streak = 0
                    self._refusal_streak = 0
                except NonTerminating as e:
                    self._mem("[nudge]", str(e), role="user", site="loop.tool.nudge")
                    # Refusal-storm breaker. A model re-emitting one refused action
                    # costs a FULL LLM call per cycle while changing nothing.
                    # Dead cycles are free for the agent but not the budget: a long
                    # unbroken refusal run is unambiguous pathology (healthy runs
                    # intersperse refusals with dispatched work, which resets this),
                    # so end the run honestly and let the judge see what exists.
                    self._refusal_streak = getattr(self, "_refusal_streak", 0) + 1
                    limit = int(getattr(self.spec.stall, "refusal_streak_limit", 60) or 0)
                    if limit and self._refusal_streak >= limit:
                        self.events.emit(
                            "on_intervention",
                            site="loop.refusal_storm",
                            kind="refusal_storm",
                            detail=f"{self._refusal_streak} consecutive refused cycles",
                        )
                        raise BudgetExhausted(
                            f"refusal storm: {self._refusal_streak} consecutive refused "
                            f"cycles -- every one a paid LLM call that changed nothing"
                        ) from e
                except FormatError as e:
                    self._mem("[format_error]", str(e), role="user", site="loop.parse.format_error")
                except Terminating:
                    raise  # GoalsAccomplished/BudgetExhausted -> handled below
                except Exception as e:  # noqa: BLE001 - unexpected (model/API/tool); don't crash the run
                    ends = self._ends_run_now(e)
                    if ends:
                        self._reason = ends
                        break
                    self._error_streak += 1
                    self._mem("[error]", f"{type(e).__name__}: {e}", role="user", site="loop.error")
                    if self._error_streak >= self._error_streak_limit(self._is_transient_provider_error(e)):
                        self._reason = f"aborted after {self._error_streak} consecutive errors: {type(e).__name__}: {e}"
                        break
                # after the cycle's effects have landed, so the fingerprint reflects this cycle
                self._close_expectations()
                self._check_stall()
                self._publish_worklist(sync_only=True)
                self.events.emit("on_cycle_end", stage=self.stages.name())
                # Last, so the halting cycle is cashed out and snapshotted like any other.
                self._halt_on_harness_fault()
        except Terminating as e:  # graceful: finalize a result instead of crashing
            if isinstance(e, (GoalsAccomplished, FinishWithResult)):
                self._finished = self._success = True
                self._reason = self._reason or str(e) or "finished"
                if isinstance(e, FinishWithResult):
                    self._submitted = e.result
            else:  # BudgetExhausted / LimitsExceeded
                self._reason = str(e)
        result = RunResult(
            success=self._success,
            reason=self._reason or ("finished" if self._success else "stopped"),
            steps=len(self.action_log),
            cost=sum(m.cost() for m in self.models.values()),
            action_log=list(self.action_log),
            working_set=self.working_set.snapshot(),
            submitted=self._submitted,
        )
        self.events.emit("on_finish", result=result)
        return result
