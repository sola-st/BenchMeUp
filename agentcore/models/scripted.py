"""ScriptedModel — an offline, deterministic model for tests, smoke runs and CI.

Returns canned responses from `kwargs.script` (a list of assistant message
strings) in order; when exhausted, emits a goals_accomplished action so any loop
terminates. Register as model class "scripted".
"""

from __future__ import annotations

import json
from typing import Any

from agentcore.protocols import ModelResponse

_FINISH = json.dumps(
    {"thoughts": "done", "command": {"name": "goals_accomplished", "args": {"reason": "scripted end"}}}
)


class ScriptedModel:
    def __init__(self, name: str = "scripted", script: list[str] | None = None, **_: Any):
        self.name = name
        self.script = list(script or [])
        self._i = 0
        self.n_calls = 0

    def query(self, messages: list[dict], **kwargs) -> ModelResponse:
        self.n_calls += 1
        if self._i < len(self.script):
            content = self.script[self._i]
            self._i += 1
        else:
            content = _FINISH
        return ModelResponse(content=content, cost=0.0)

    def cost(self) -> float:
        return 0.0

    def get_template_vars(self) -> dict[str, Any]:
        return {"model_name": self.name, "model_cost": 0.0, "n_model_calls": self.n_calls}
