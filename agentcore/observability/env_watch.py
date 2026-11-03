"""EnvWatcher — surface environment changes back to the agent as a per-cycle signal.

Opt-in via agent.yaml::

    env_watch:
      enabled: true
      files: true        # watch the workspace top-level (default true)
      packages: false    # diff `pip freeze` each cycle (default false — it's the costly probe)

It re-fingerprints the env at each cycle end, diffs against the previous snapshot, and when
anything changed: (1) injects a concise `[env]` note so the NEXT cycle's prompt sees it, and
(2) emits an on_env_change event for the trajectory/report. A no-op when nothing changed, so a
stable env costs only the (cheap) filesystem scan.
"""

from __future__ import annotations

from typing import Any

from agentcore.context.env_fingerprint import diff_fingerprints, fingerprint


class EnvWatcher:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.files = bool(cfg.get("files", True))
        self.packages = bool(cfg.get("packages", False))
        self.max_signals = int(cfg.get("max_signals", 12))
        self._agent = None
        self._prev: dict | None = None

    def bind(self, agent) -> None:
        self._agent = agent

    def _snapshot(self) -> dict:
        return fingerprint(self._agent.ctx(), files=self.files, packages=self.packages, system=False)

    def on_run_start(self, **_: Any) -> None:
        if self._agent is not None:
            self._prev = self._snapshot()

    def on_cycle_end(self, **_: Any) -> None:
        if self._agent is None:
            return
        cur = self._snapshot()
        if self._prev is None:
            self._prev = cur
            return
        signals, changes = diff_fingerprints(self._prev, cur)
        self._prev = cur
        if not signals:
            return
        shown = signals[: self.max_signals]
        if len(signals) > self.max_signals:
            shown.append(f"…(+{len(signals) - self.max_signals} more)")
        summary = "; ".join(shown)
        self._agent._mem("[env]", summary, role="user")  # feed back into the prompt
        self._agent.events.emit("on_env_change", summary=summary, changes=changes)
