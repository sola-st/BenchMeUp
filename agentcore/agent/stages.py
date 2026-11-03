"""StageController over a StageGraph.

Linear-with-sentinel is the default shape (fits most tasks). Back-edges
(transitions), per-stage tools, and several completion kinds are opt-in config.
A tool may also drive a transition via ToolResult.transition_to.
"""

from __future__ import annotations

from typing import Any

from agentcore.config.schema import StageSpec


def _normalize(s: str) -> str:
    return " ".join((s or "").lower().replace("-", "_").split())


class StageController:
    def __init__(self, stages: list[StageSpec]):
        self.stages = list(stages)
        self._by_name = {s.name: s for s in self.stages}
        self.index = 0
        self.done = False

    # ── queries ───────────────────────────────────────────────────────────────
    @property
    def staged(self) -> bool:
        return bool(self.stages)

    def current(self) -> StageSpec | None:
        if not self.stages or self.done:
            return None
        return self.stages[self.index]

    def name(self) -> str | None:
        c = self.current()
        return c.name if c else None

    def tools_allow(self, global_allow: list[str], name: str | None = None) -> list[str]:
        """The tools the current stage grants -- or a NAMED stage's, so a caller can ask what
        a stage it is not standing in would be able to do. `tools: None` inherits the global
        allowlist; `tools: []` grants nothing. One rule, however the stage is chosen."""
        c = self.current() if name is None else self._by_name.get(name)
        if c and c.tools is not None:
            return c.tools
        return global_allow

    def has(self, name: str) -> bool:
        return name in self._by_name

    def tool_guard(self) -> str | None:
        c = self.current()
        return c.tool_guard if c else None

    def prompt_name(self, default: str | None) -> str | None:
        c = self.current()
        if c and c.prompt:
            return c.prompt
        return default

    def between(self, target: str) -> list[str]:
        """Names of the stages a jump from the current stage to `target` would skip, in order,
        exclusive of both ends. Empty for an adjacent, backward or unknown target. Lets a
        caller ask what a force jumps OVER, not only where it lands."""
        dst = self._by_name.get(target)
        if dst is None or self.done:
            return []
        return [s.name for s in self.stages[self.index + 1 : self.stages.index(dst)]]

    def peek_next(self) -> str | None:
        """Name of the linear next stage, or None if this is the last one.

        Lets a caller decide whether advancing is even possible before doing it (the stall
        detector must not "force" a transition out of the final stage, where there is nowhere
        to go and forcing would silently do nothing).
        """
        if self.done or self.index + 1 >= len(self.stages):
            return None
        return self.stages[self.index + 1].name

    # ── transitions ────────────────────────────────────────────────────────────
    def goto(self, name: str) -> None:
        if name not in self._by_name:
            raise ValueError(f"Unknown stage {name!r}")
        self.index = self.stages.index(self._by_name[name])
        self.done = False

    def advance(self) -> None:
        """Move to the linear next stage; mark done after the last stage."""
        if self.index + 1 < len(self.stages):
            self.index += 1
        else:
            self.done = True

    def match_completion(self, assistant_text: str, ctx: Any, *, kinds: set[str] | None = None) -> str | None:
        """Which completion predicate (if any) fired for the current stage?

        Returns the predicate kind ("sentinel" | "fuzzy" | "side_effect") so callers
        can record WHY a stage advanced, or None if the stage has not completed.

        `kinds`, if given, restricts which predicate types are considered (default:
        all). "sentinel" and "fuzzy" do a substring/normalized-containment search over
        the FULL assistant text -- which, when called after a real tool call has
        already been parsed from that same text, can false-positive-match a tool
        call's own arguments (e.g. a written file whose content happens to contain the
        sentinel string). Both are meant, by contract, to fire only on a genuinely
        command-less reply (see check_completion's caller, _control_signal, which is
        only reached when parsing found no command) -- callers that run this check
        AFTER a command was already executed (e.g. loop.py's summarize()) must pass
        kinds excluding "sentinel"/"fuzzy" and keep only state-based kinds like
        "side_effect", which legitimately need to be (re-)checked post-execution.
        """
        c = self.current()
        if c is None:
            return None
        comp = c.completion or {}
        text = assistant_text or ""
        if "sentinel" in comp and (kinds is None or "sentinel" in kinds):
            token = comp["sentinel"].replace("{name}", c.name)
            if token.lower() in text.lower():
                return "sentinel"
        if "fuzzy" in comp and (kinds is None or "fuzzy" in kinds):
            if _normalize(comp["fuzzy"]) in _normalize(text):
                return "fuzzy"
        if "side_effect" in comp and (kinds is None or "side_effect" in kinds):
            if self._eval_side_effect(comp["side_effect"], ctx):
                return "side_effect"
        # tool_signal completions are driven by ToolResult.transition_to, handled by the loop
        return None

    def check_completion(self, assistant_text: str, ctx: Any) -> bool:
        """Did the current stage signal completion via the model's text?"""
        return self.match_completion(assistant_text, ctx) is not None

    def _eval_side_effect(self, expr: str, ctx: Any) -> bool:
        """Evaluate an author-trusted predicate over a small namespace.

        The namespace exposes the env/working_set/instance objects AND the env's
        published template vars (e.g. `container_present`, `phase`) so predicates can
        reference environment state without reaching into private attributes.
        """
        env = getattr(ctx, "env", None)
        agent = getattr(ctx, "agent", None)
        ns: dict[str, Any] = {
            "env": env,
            "ws": getattr(ctx, "working_set", None),
            "instance": getattr(ctx, "instance", {}),
            # The RESOLVED spec, so a gate can read its own config (e.g.
            # collector.required_before_scale) instead of re-reading agent.yaml from disk --
            # which silently ignores runtime overrides and duplicates config loading.
            "spec": getattr(agent, "spec", None),
            # The agent's RESOLVED hooks (spec.hooks), so a gate can call a real function.
            # The eval namespace has no __builtins__, so an expression cannot import anything --
            # `__import__('hooks').f(ws)` raises NameError and, because the eval is wrapped in a
            # bare except, silently evaluates to False. A gate that can never pass is worse than
            # no gate: the run stalls with no diagnostic anywhere.
            "hooks": getattr(agent, "hooks", {}) or {},
        }
        try:
            if env is not None:
                ns.update(env.get_template_vars())
        except Exception:
            pass
        try:
            return bool(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307 - author config, sandboxed ns
        except Exception:
            return False

    def get_template_vars(self) -> dict[str, Any]:
        # `stage_sentinel` lets a SHARED cycle prompt name the exit token of whichever stage is
        # current. Without it a shared prompt cannot satisfy the stage-exit contract (the token a
        # sentinel stage requires must appear in that stage's prompt, or the exit is unreachable
        # and the model burns its budget guessing synonyms).
        c = self.current()
        return {
            "stage": self.name(),
            "stage_index": self.index,
            "n_stages": len(self.stages),
            "stage_sentinel": (c.completion or {}).get("sentinel", "") if c else "",
        }
