"""ToolContext — the typed handle every tool receives.

Tools mutate run state (the container, the working set, the current stage). Rather
than pretend tools are pure, we sanction this by giving them a structured context.
This is a plain dataclass (not pydantic) so it can hold live protocol objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentcore.agent.workingset import WorkingSet
    from agentcore.protocols import Environment, Model


@dataclass
class ToolContext:
    env: "Environment"
    models: dict[str, "Model"]
    working_set: "WorkingSet"
    instance: dict = field(default_factory=dict)
    stage: Any = None  # StageController (agent.stages)
    budget: Any = None  # BudgetController (agent.budget)
    agent: Any = None  # back-reference to the LoopAgent (last resort)
    scratch: dict = field(default_factory=dict)  # free per-run mutable bag for tools

    @property
    def model(self) -> "Model":
        """The primary model (convenience)."""
        return self.models["main"]
