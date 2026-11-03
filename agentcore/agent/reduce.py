"""OutputReducer — configurable compression of tool output into agent memory.

Unifies the two strategies the real agents used:
  - heuristic   : truncation + failure-region extraction (analysis_agent)
  - llm_summary : a model condenses the output each cycle (ExecutionAgent)
  - raw         : keep the output as-is (e.g. a file the agent explicitly read)

Selected per agent via the `tool_output:` block, with optional per-tool overrides.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentcore.config.schema import ToolOutputSpec
from agentcore.tools import parsing
from agentcore.tools.result import ToolResult

_DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following tool output for an engineering agent's memory. "
    "Preserve errors, failures, key results/metrics and any findings; drop noise. "
    "Be concise."
)


@dataclass
class _Eff:
    strategy: str
    truncation: str
    threshold: int
    head: int
    middle: int
    tail: int
    n_lines: int
    summary_model: str
    summary_prompt: str | None
    max_summary_chars: int


class OutputReducer:
    def __init__(self, spec: ToolOutputSpec, *, models: dict, prompts=None, registry=None, budget=None, events=None):
        self.spec = spec
        self.models = models or {}
        self.prompts = prompts
        self.registry = registry
        self.budget = budget  # so the per-cycle llm_summary cost lands on the budget meter
        self.events = events  # emits on_output_reduced so the report can show the "why"

    # ── config resolution ──────────────────────────────────────────────────────
    def _effective(self, tool_name: str, result: ToolResult) -> _Eff:
        s = self.spec
        ov = (s.per_tool or {}).get(tool_name, {}) or {}

        def pick(key, default):
            return ov[key] if key in ov else getattr(s, key, default)

        strategy = pick("strategy", s.strategy)

        # truncation precedence: explicit on the ToolResult > per-tool > agent spec
        #                        > the tool's @tool-declared policy > head_tail
        if result.truncation and result.truncation != "inherit":
            truncation = result.truncation
        elif "truncation" in ov and ov["truncation"]:
            truncation = ov["truncation"]
        elif s.truncation:
            truncation = s.truncation
        elif self.registry is not None and self.registry.has(tool_name):
            truncation = self.registry.meta(tool_name).get("truncation", "head_tail")
        else:
            truncation = "head_tail"

        return _Eff(
            strategy=strategy,
            truncation=truncation,
            threshold=pick("threshold_chars", s.threshold_chars),
            head=pick("head_chars", s.head_chars),
            middle=pick("middle_chars", s.middle_chars),
            tail=pick("tail_chars", s.tail_chars),
            n_lines=pick("n_lines", s.n_lines),
            summary_model=pick("summary_model", s.summary_model),
            summary_prompt=pick("summary_prompt", s.summary_prompt),
            max_summary_chars=pick("max_summary_chars", s.max_summary_chars),
        )

    # ── reduction ──────────────────────────────────────────────────────────────
    def reduce(self, tool_name: str, result: ToolResult) -> str:
        e = self._effective(tool_name, result)
        text = result.display_text or ""
        out, path = self._compute(e, text, result)
        if self.events is not None:
            self.events.emit(
                "on_output_reduced",
                tool=tool_name,
                strategy=e.strategy,
                truncation=e.truncation,
                path=path,
                raw_len=len(text),
                reduced_len=len(out),
            )
        return out

    def _compute(self, e: _Eff, text: str, result: ToolResult) -> tuple[str, str]:
        """Return (reduced_text, path) where path names the branch taken (for the report)."""
        # a structured digest provided by the tool short-circuits text reduction
        if result.structured is not None and e.strategy != "llm_summary" and result.truncation == "none":
            return text, "structured"
        if e.strategy == "raw":
            return text, "raw"
        truncated = parsing.truncate(
            text,
            e.truncation,
            head=e.head,
            middle=e.middle,
            tail=e.tail,
            threshold=e.threshold,
            n_lines=e.n_lines,
        )
        if e.strategy == "llm_summary":
            return self._summarize(truncated, e), "llm_summary"
        return truncated, "heuristic"

    def _summarize(self, text: str, e: _Eff) -> str:
        model = self.models.get(e.summary_model) or self.models.get("knowledge") or self.models.get("main")
        if model is None:
            return text[: e.max_summary_chars]
        try:
            if e.summary_prompt and self.prompts is not None and self.prompts.has(e.summary_prompt):
                instruction = self.prompts.render(e.summary_prompt)
            else:
                instruction = _DEFAULT_SUMMARY_PROMPT
            resp = model.query([{"role": "user", "content": f"{instruction}\n\n{text}"}])
            if self.budget is not None:
                self.budget.note_cost(getattr(resp, "cost", 0.0))
            out = (getattr(resp, "content", "") or "").strip() or text
            return out[: e.max_summary_chars]
        except Exception:
            return text[: e.max_summary_chars]
