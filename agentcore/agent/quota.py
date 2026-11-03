"""The sample-size target is binding until the budget is genuinely spent.

Refusing while registered candidates are unfinished is not enough: a run can drive every
registered candidate to a terminal state and still be far from the target with most of its
budget unspent. "All terminal" does not mean "exhausted".
A mining stage that can register new candidates is never out of moves
while budget remains; it is only out of the moves it already made.

The goal is the target. Whether the budget is consumed entirely before reaching it is not a
risk this gate weighs -- the user sets the budget, and spending all of it chasing the target
is the declared price.
A run that grinds to budget exhaustion short of the target finishes honestly then,
and only then. Consequently there is NO pushback cap: the configured budget and wallclock limits
are the backstops, not an additional counter.
"""

from __future__ import annotations

# The harness's own dead states come from the one module that defines them. A task's
# WorklistSpec.terminal is the SUCCESS vocabulary (config/schema.py: "REJECTED and BLOCKED
# are always terminal in addition to these"), so the dead pair is genuinely a harness
# constant -- the pair is defined once, in diversity._DEAD_STATES.
from agentcore.agent.diversity import _DEAD_STATES, success_from_spec, terminal_from

# The gate stands down only this close to the limit -- at 90% there is no realistic
# room to clone, build and prove another candidate; refusing there would trap the run.
BUDGET_SPENT_FRACTION = 0.9


def budget_fraction(ctx) -> float:
    """How much of the run's budget is spent, as the fraction of its fullest meter.

    The one reading every gate that opens on "budget spent" uses (quota here, the searching
    agent's stock_verify). Never raises: no budget information means 0.0, so a gate cannot
    open on a missing meter."""
    try:
        b = getattr(ctx.agent, "budget", None)
        meters = getattr(b, "meters", None)
        if meters:
            # The real BudgetController: same max-fraction rule its own
            # thresholds_triggered() uses. BudgetController has no
            # b.spent/b.limit attributes, so the meters are read directly.
            return max((ms.fraction() for ms in meters), default=0.0)
        spent = float(getattr(b, "spent", 0) or 0)
        limit = float(getattr(b, "limit", 0) or 0)
        return spent / limit if limit else 0.0
    except Exception:  # noqa: BLE001 - no budget info means do not block on budget
        return 0.0


def quota_not_met(ctx) -> str:
    """Return "" when finishing is legitimate, else the refusal to send back.

    Legitimate means exactly: no target declared, target met, or budget >= 90% spent.
    """
    ws = ctx.working_set.snapshot()
    inst = getattr(ctx, "instance", {}) or {}
    target = int(inst.get("sample_size") or 0)
    if not target:
        return ""

    store = ws.get("worklist") or {}
    wl = store.get("candidates") or {}
    # The task's own success state, not the literal "DELIVERED". A benchmark declaring
    # `terminal: [ACCEPTED]` would otherwise count zero deliveries and this gate could never
    # be satisfied. `diversity.terminal_from` answers this question.
    success = terminal_from(ctx) or success_from_spec(store)
    delivered = sum(1 for c in wl.values() if c.get("state") == success)
    if delivered >= target:
        return ""

    frac = budget_fraction(ctx)
    if frac >= BUDGET_SPENT_FRACTION:
        return ""  # the user's boundary; a short sample is honest here

    finished_states = set(_DEAD_STATES) | {success}
    unfinished = [k for k, c in wl.items() if c.get("state") not in finished_states]
    if unfinished:
        return (
            f"You have delivered {delivered} of {target} and are trying to finish "
            f"with {len(unfinished)} candidate(s) still unfinished: "
            f"{', '.join(k[:40] for k in unfinished[:5])}"
            f"{'...' if len(unfinished) > 5 else ''}. Only {frac:.0%} of the budget "
            f"is spent. Drive those to {success}, {', '.join(_DEAD_STATES[:-1])} or "
            f"{_DEAD_STATES[-1]} first -- "
            f"leaving them half-done is not a smaller sample, it is an abandoned one."
        )

    return (
        f"You have delivered {delivered} of {target}. Every candidate you registered "
        f"is finished, but that is not the same as being done: only {frac:.0%} of the "
        f"budget is spent, and the target is {target}. Register NEW candidates -- a "
        f"different project, a different organisation, a different part of the history "
        f"of one that worked -- and keep going. The run ends at {target} delivered or "
        f"at the budget limit, whichever comes first."
    )
