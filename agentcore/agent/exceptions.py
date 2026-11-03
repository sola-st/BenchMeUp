"""Control-flow exceptions for the agent loop.

NonTerminating exceptions are caught by the loop and fed back to the model as an
observation. Terminating exceptions end the run.
"""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base for all agentcore control-flow signals."""


# ── non-terminating (handled, fed back to the model) ─────────────────────────


class NonTerminating(AgentError):
    pass


class FormatError(NonTerminating):
    """The model's output could not be parsed into an action."""


class ExecutionTimeoutError(NonTerminating):
    """A tool/command execution timed out."""


class RepetitionError(NonTerminating):
    """The model repeated a recent action; nudge it to try something else."""


# ── terminating (end the run) ────────────────────────────────────────────────


class Terminating(AgentError):
    def __init__(self, message: str = "", result: Any = None):
        super().__init__(message)
        self.result = result


class GoalsAccomplished(Terminating):
    """The agent (or an oracle) declared the task complete and successful."""


class FinishWithResult(Terminating):
    """A tool requested loop end, carrying a typed result (e.g. submit)."""


class BudgetExhausted(Terminating):
    """A budget meter hit its limit with on_exhaust=abort/fail_with_lesson."""


class HarnessFault(Terminating):
    """The harness found its OWN gate broken -- a refusal it marked HARNESS ERROR/FAULT has
    become systemic across unrelated candidates -- and stopped spending on it. Not the agent
    finishing short: nothing the agent does can pass a gate the harness says is broken, and
    every further cycle is paid for nothing. The fix is a harness change and a fresh run,
    which this hands back early."""


class LimitsExceeded(BudgetExhausted):
    """Back-compat alias used by the legacy agents."""
