"""StatePersistence — JSON crash snapshot of agent memory + WorkingSet.

Saved after each cycle (subscribe to on_cycle_end) so a crashed/killed run can be
inspected or resumed. Distinct from the trajectory (append-only event log).

RETENTION. The snapshot holds the WHOLE agent memory and grows with the run, so writing one
per cycle and keeping them all costs bytes quadratic in cycle count. Nothing reads these but `--at-cycle`
(agentcore/cli.py), a manual fork/debug entry point, so they are decimated on write: cycle 1,
every KEEP_EVERY-th, and a sliding window of the last KEEP_LAST. agent_state.json (the latest snapshot) is
untouched -- every automated consumer reads that one.
"""

from __future__ import annotations

import json
import os
from typing import Any

#: Keep cycle 1 (forks and tests take over there), every KEEP_EVERY-th snapshot, and the last
#: KEEP_LAST. Set BENCH_STATE_KEEP_EVERY=0 to retain every cycle as before.
KEEP_EVERY = int(os.environ.get("BENCH_STATE_KEEP_EVERY") or 25)
KEEP_LAST = int(os.environ.get("BENCH_STATE_KEEP_LAST") or 10)


class StatePersistence:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, "agent_state.json")
        self.state_dir = os.path.join(out_dir, "state")
        os.makedirs(self.state_dir, exist_ok=True)
        self._agent = None
        # Resume into the same dir continues the per-cycle numbering instead of clobbering.
        # The HIGHEST retained cycle, not the file COUNT: decimation leaves gaps, and a count
        # would restart numbering below the surviving snapshots and overwrite them.
        self._cycle = self._highest_existing_snapshot()

    def _highest_existing_snapshot(self) -> int:
        cycles = self.available_cycles(self.out_dir)
        return cycles[-1] if cycles else 0

    def bind(self, agent) -> None:
        self._agent = agent

    def on_cycle_end(self, **_: Any) -> None:
        self._cycle += 1
        self.save(self._snapshot_path(self._cycle))  # per-cycle (restorable to a RETAINED N)
        self._prune(self._cycle)

    def _prune(self, cycle: int) -> None:
        """Drop the snapshot that just fell out of the recent window, unless it is a keeper."""
        if KEEP_EVERY <= 0:
            return
        old = cycle - KEEP_LAST
        if old <= 1 or old % KEEP_EVERY == 0:
            return
        try:
            os.remove(self._snapshot_path(old))
        except OSError:
            pass

    def on_finish(self, **_: Any) -> None:
        self.save()

    def _snapshot_path(self, cycle: int) -> str:
        return os.path.join(self.state_dir, f"cycle_{cycle:03d}.json")

    def save(self, also_to: str | None = None) -> None:
        if self._agent is None:
            return
        a = self._agent
        snap = {
            "cycle": self._cycle,
            # The AGENT's own counter, distinct from this observer's snapshot count. Two
            # mechanisms compute with it -- the sandbox lifecycle ledger stamps events with it,
            # and repetition entries expire by it -- so a resume that restarts it at 0 stamps
            # every later event "cycle 0" and makes the decay window meaningless.
            "agent_cycle": int(getattr(a, "cycle", 0) or 0),
            "action_log": a.action_log,
            "working_set": a.working_set.snapshot(),
            "stage": a.stages.name(),
            "budget": a.budget.status(),
            "finished": a._finished,
            "success": a._success,
        }
        self._atomic_write(self.path, snap)  # latest (crash snapshot)
        if also_to:
            self._atomic_write(also_to, snap)

    @staticmethod
    def _atomic_write(path: str, obj: dict) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)

    @staticmethod
    def load(out_dir: str, at_cycle: int | None = None) -> dict:
        """Latest snapshot, or the snapshot as of the END of cycle `at_cycle`."""
        if at_cycle is not None:
            path = os.path.join(out_dir, "state", f"cycle_{at_cycle:03d}.json")
        else:
            path = os.path.join(out_dir, "agent_state.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def available_cycles(out_dir: str) -> list[int]:
        state_dir = os.path.join(out_dir, "state")
        out: list[int] = []
        try:
            for f in os.listdir(state_dir):
                if f.startswith("cycle_") and f.endswith(".json"):
                    try:
                        out.append(int(f[len("cycle_") : -len(".json")]))
                    except ValueError:
                        continue
        except OSError:
            pass
        return sorted(out)
