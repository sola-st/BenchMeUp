"""The harness's gate thresholds, in ONE place.

Each of these is a default a spec may override (`worklist:` fields and `environment.
max_sandboxes` in config/schema.py). The engines that enforce them read the spec first and
fall back to the constant here, so a spec that says nothing behaves exactly as before, and
the schema's defaults ARE these constants -- there is no second copy to drift from.

This module imports nothing from the package so both the schema and the engines can import it.
"""

#: Refusals at one state before a candidate is deferred for others. Most candidates that
#: eventually pass a state need at most three refusals there, so deferring at 4 sets aside few
#: eventual passes. Deferral is reversible; the slot switch is the cost.
STUCK_DEFER = 4
#: Refusals at one state before a candidate is terminated. Terminal, so it needs enough
#: attempts to be confident the candidate really cannot proceed rather than that the model
#: was briefly confused.
STUCK_BLOCK = 8
#: Distinct candidates on one refusal shape before it stops looking like their fault. Two
#: can be coincidence in a small sample; three unrelated candidates failing one gate the same
#: way is a statement about the gate.
SYSTEMIC_AT = 3
#: Cycles one candidate may hold the slot before the harness says something.
SLOW_NUDGE = 40
#: ...and before it is deferred so others get a turn.
SLOW_DEFER = 100
#: The limit a `caps:` entry gets when it declares none. Stopping at 2-4 per source is typical;
#: at a target of 20 instances, 4 forces at least five.
CAP_DEFAULT_LIMIT = 4
#: How many sandboxes one run may hold live at once. Host memory is the real bound, and the
#: per-container mem_limit already carries it.
MAX_SANDBOXES = 32
