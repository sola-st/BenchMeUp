"""ToolResult — the structured return of every tool call.

A tool returns display_text (what the model sees, subject to truncation) and may
attach a structured payload (bypasses truncation), request a stage transition,
end the loop, or signal a mid-loop oracle hit.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

TruncationPolicy = Literal["inherit", "head_tail", "first_error", "latest_n_lines", "none"]


class ToolResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    display_text: str = ""
    structured: dict[str, Any] | None = None
    truncation: TruncationPolicy = "inherit"
    transition_to: str | None = None  # request a stage transition (e.g. repair)
    finished: bool = False  # request loop end (e.g. submit)
    oracle_hit: dict[str, Any] | None = None  # mid-loop success capture
    error: str | None = None

    @classmethod
    def text(cls, s: str, **kw) -> "ToolResult":
        return cls(display_text=s, **kw)

    @classmethod
    def fail(cls, msg: str, **kw) -> "ToolResult":
        return cls(display_text=msg, error=msg, **kw)
