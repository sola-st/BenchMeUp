"""WorkingSet — typed, persisted, cross-cycle blackboard.

Authors subclass this with task-specific fields (a list of Findings, a coverage
scratchpad, a list of review Comments, a build effective_prefix...). It survives
across cycles and stages, is persisted by observability.state, and is reachable
from tools via ToolContext.working_set.

Distinct from:
  - Carryover (agent.attempts): inter-attempt retry-learning.
  - StatePersistence (observability.state): the crash snapshot mechanism.
"""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel


class WorkingSet(BaseModel):
    """Base blackboard. Empty by default; subclass to add typed fields."""

    model_config = {"extra": "allow"}

    def merge(self, data: dict[str, Any] | "WorkingSet | None") -> None:
        """Shallow-merge a dict or another WorkingSet into this one in place."""
        if data is None:
            return
        items = data.items() if isinstance(data, dict) else data.model_dump().items()
        for k, v in items:
            setattr(self, k, v)

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "WorkingSet":
        return cls(**(data or {}))


def load_working_set(schema_path: str | None) -> WorkingSet:
    """Resolve 'module:ClassName' to an instance, or return a bare WorkingSet."""
    if not schema_path:
        return WorkingSet()
    mod_name, _, cls_name = schema_path.partition(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls()
