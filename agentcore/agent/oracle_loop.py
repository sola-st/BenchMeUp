"""OracleLoop — verify-then-finish.

The HARNESS runs the proxy oracle in-env after each execute and owns should_finish;
the model proposes candidates, it does not self-declare done. This is the common
deterministic-inner-loop shape (build bring-up, repair, coverage test-gen).
"""

from __future__ import annotations

from agentcore.agent.loop import LoopAgent
from agentcore.oracle.base import OracleResult
from agentcore.tools.result import ToolResult


class OracleLoop(LoopAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_oracle = OracleResult()

    def execute(self, action: dict) -> ToolResult:
        result = super().execute(action)
        # mid-loop oracle hit captured by a tool takes precedence
        if result.oracle_hit is not None:
            self._last_oracle = OracleResult(
                success=True, done=True, sub_signals=result.oracle_hit, reason="tool reported oracle hit"
            )
            self._emit_oracle_check(self._last_oracle, source="tool_hit", accepted=True)
        elif self.oracle is not None:
            try:
                self._last_oracle = self.oracle.check(self.ctx())
            except Exception as e:  # noqa: BLE001
                self._last_oracle = OracleResult(success=False, reason=f"oracle error: {e}")
            self.working_set.merge(self._last_oracle.sub_signals)
            self._emit_oracle_check(self._last_oracle, source="proxy", accepted=self._last_oracle.done)
        return result

    def should_finish(self) -> bool:
        if self._last_oracle.done:
            self._finished = True
            self._success = self._last_oracle.success
            self._reason = self._last_oracle.reason or "oracle satisfied"
            return True
        return super().should_finish()
