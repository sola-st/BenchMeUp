"""Source-list tools -- opt-in via `tool_modules: ["agentcore.tools.source_tools"]`.

Two calls, mirroring the discipline of the candidate worklist one level up:

  sources_fetch   hand me the next places to work through (or re-read a range I lost)
  sources_done    I looked at these and there is nothing here, and here is why

`sources_fetch` exists so the agent stops carrying a cursor: a long list recorded early falls
out of a bounded context window and would otherwise be silently abandoned.

`sources_done` exists because the harness must never INFER that a place is empty. Handing ten
names to a sweep script is not evidence the script queried ten: it can crash after three or
rate-limit after five. A statement
from the agent is evidence and is recorded as such; it sinks the entry to the bottom of the
order and never deletes it. See `agentcore.agent.sources` for the full reasoning.
"""

from __future__ import annotations

from agentcore.agent import diversity as _diversity
from agentcore.agent import sources as _sources
from agentcore.tools.context import ToolContext
from agentcore.tools.registry import tool
from agentcore.tools.result import ToolResult


def _instance_admissibility(ctx) -> list:
    """The clauses this run's own work order states, or [] when it states none."""
    inst = getattr(ctx, "instance", None)
    if not isinstance(inst, dict):
        return []
    return [c for c in (inst.get("admissibility") or []) if isinstance(c, dict)]


def _spec(ctx):
    """The task's declared source list, or None when it declares none."""
    agent = getattr(ctx, "agent", None)
    spec = getattr(agent, "spec", None)
    ss = getattr(spec, "sources", None)
    if ss is None or not getattr(ss, "enabled", False):
        return None
    wl = getattr(spec, "worklist", None)
    return _sources.SourceSpec(
        key=getattr(ss, "key", "source_pool"),
        caps=list(getattr(wl, "caps", None) or []),
        terminal=_diversity.success_from_spec(wl),
        batch_size=int(getattr(ss, "batch_size", _sources.DEFAULT_BATCH) or _sources.DEFAULT_BATCH),
        view_cap=int(getattr(ss, "view_cap", _sources.DEFAULT_VIEW_CAP) or _sources.DEFAULT_VIEW_CAP),
        noun=getattr(ss, "noun", "source") or "source",
        verify=getattr(ss, "verify", None),
        # EVERY declared field, or the gate silently drops a check. Omitting these three is how
        # a pool concentrated in three organisations passed a spread requirement of five: the
        # reader built a SourceSpec whose spread_by was None, so the check never ran.
        requires=list(getattr(ss, "requires", None) or []),
        evidence=list(getattr(ss, "evidence", None) or []),
        spread_by=getattr(ss, "spread_by", None),
        exclusion_fields=list(getattr(ss, "exclusion_fields", None) or []),
        # THE WORK ORDER WINS, where it states clauses. An agent's `sources.admissibility` is a
        # snapshot taken when the agent was written; the INSTANCE is derived from the canonical
        # at launch. Keeping the config's copy authoritative meant a regenerated spec left the
        # model shown one procedure (the brief's ADMISSIBILITY block) and gated on another (this
        # one), with neither side wrong about itself.
        #
        # Falls back to the config when the instance declares none, so a task that keeps its
        # clauses in agent.yaml is untouched.
        admissibility=(_instance_admissibility(ctx) or list(getattr(ss, "admissibility", None) or [])),
    )


def _pool(ctx):
    spec = _spec(ctx)
    if spec is None:
        return None, None
    return spec, _sources.build(spec, ctx.working_set)


def _disabled() -> ToolResult:
    return ToolResult.fail(
        "This task declares no source list (sources.enabled is false), so "
        "there is nothing to fetch from or mark. Register candidates "
        "directly."
    )


def _row_payload(r) -> dict:
    out = dict(r.entry)
    out["source"] = r.value
    out["leads_so_far"] = r.candidates
    if r.done_reason:
        out["you_marked_done"] = r.done_reason
    return out


@tool(
    "sources_fetch",
    params={
        "count": "how many places to hand you. Leave it out for the task's batch size. The harness "
        "picks which ones -- never-used first, then the ones that have produced before -- "
        "so you never have to remember where you stopped.",
        "start": "with `end`, RE-READ that slice of the list instead of taking new work. Use this "
        "when the batch you were given has scrolled out of your history. Re-reading costs "
        "you nothing and takes nothing.",
    },
)
def sources_fetch(ctx: ToolContext, count: int = 0, start: int = -1, end: int = -1) -> ToolResult:
    """Hand me the next places to work through, or re-read part of the list."""
    spec, pool = _pool(ctx)
    if pool is None:
        return _disabled()
    if not pool.entries:
        return ToolResult.fail(
            f"Your {spec.noun} list is empty -- nothing has been recorded under "
            f"'{spec.key}' yet. Find real {spec.noun}s that meet the requirements and record "
            f"them first; there is nowhere to mine from until you do."
        )

    if start >= 0 and end > start:
        rows = pool.read_range(start, end)
        if not rows:
            return ToolResult(
                display_text=f"Entries {start}-{end} are past the end of your list ({len(pool.rows())} entries).",
                structured={"sources": [], "reread": True},
            )
        return ToolResult(
            display_text=(
                f"Entries {start}-{start + len(rows)} of your {spec.noun} list "
                f"(re-read -- nothing was taken):\n"
                + "\n".join(
                    f"  {r.index:>4}  {r.value}"
                    + (f"  ({r.candidates} lead(s))" if r.candidates else "")
                    + (f"  [you marked done: {r.done_reason}]" if r.done_reason else "")
                    for r in rows
                )
            ),
            structured={"sources": [_row_payload(r) for r in rows], "reread": True},
        )

    n = int(count) if count and int(count) > 0 else spec.batch_size
    cycle = int(getattr(getattr(ctx, "agent", None), "cycle", 0) or 0)
    rows = pool.lease(n, cycle)
    if not rows:
        return ToolResult(
            display_text=(
                f"Nothing left to hand you: every {spec.noun} on your list has hit "
                f"the per-{spec.noun} limit. To deliver more you need NEW {spec.noun}s "
                f"-- find them and record them under '{spec.key}'."
            ),
            structured={"sources": []},
        )

    revisits = [r for r in rows if r.done_reason]
    body = "\n".join(
        f"  {r.value}"
        + (f"  ({r.candidates} lead(s) so far)" if r.candidates else "")
        + (
            f"  [you marked this done at cycle {(pool.state.get(r.value) or {}).get('done', {}).get('cycle')}: "
            f"{r.done_reason}]"
            if r.done_reason
            else ""
        )
        for r in rows
    )
    head = f"Your next {len(rows)} {spec.noun}(s) to work through:"
    tail = (
        "\nWhen you are done: sources_done([...], reason) for the ones with nothing in "
        "them, then sources_fetch() for the next batch."
    )
    if revisits:
        tail = (
            f"\nNOTE: {len(revisits)} of these came back because nothing else is left. You "
            f"marked them done earlier for the reason shown -- if that reason was really "
            f"about your QUERY rather than the {spec.noun}, try a different one." + tail
        )
    return ToolResult(display_text=f"{head}\n{body}{tail}", structured={"sources": [_row_payload(r) for r in rows]})


@tool(
    "sources_done",
    params={
        "sources": "the list of source names to set aside -- pass them all in ONE call, the way you swept them.",
        "reason": 'REQUIRED: what you found, in your own words ("no merged PRs close a linked '
        'issue", "archived"). Reasons are counted across sources, and the same reason '
        "appearing many times usually means the QUERY was wrong rather than that every "
        "one of those places was empty. Nothing is deleted -- a source you mark sinks to "
        "the bottom and will come back if the list runs out.",
    },
)
def sources_done(ctx: ToolContext, sources, reason: str = "") -> ToolResult:
    """I looked at these and there is nothing here for us."""
    spec, pool = _pool(ctx)
    if pool is None:
        return _disabled()
    if not str(reason or "").strip():
        return ToolResult.fail(
            "sources_done needs a `reason`. It is counted across every source you set aside, "
            "and that count is the only way to tell 'these places are empty' apart from 'my "
            "query is wrong' -- which is the more common of the two."
        )
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list) or not sources:
        return ToolResult.fail("`sources` must be a non-empty list of source names, as shown in your list view.")

    cycle = int(getattr(getattr(ctx, "agent", None), "cycle", 0) or 0)
    marked, unknown = pool.mark_done_checked(sources, str(reason).strip(), cycle)
    left = [r for r in pool.rows() if r.bucket in (_sources.NEVER, _sources.NO_LEADS)]
    msg = f"Set aside {marked} {spec.noun}(s): {reason.strip()!r}. {len(left)} still to work."
    if unknown:
        msg += (
            f" {len(unknown)} name(s) are not in your list and were ignored: "
            f"{', '.join(unknown[:5])} -- use the names exactly as your list shows them."
        )
    reasons = pool.done_reasons()
    if reasons and reasons[0][1] >= 5:
        msg += (
            f" You have now set aside {reasons[0][1]} for the same reason "
            f"({reasons[0][0]!r}). If that many real {spec.noun}s look empty, suspect the "
            f"query before the {spec.noun}s."
        )
    return ToolResult(
        display_text=msg, structured={"marked": marked, "unknown": unknown, "reasons": reasons, "remaining": len(left)}
    )


@tool(
    "sources_add",
    params={
        "sources": "the places to add, as a list of objects carrying every field the task requires "
        "-- pass them all in ONE call. Entries already on your list are skipped (and "
        "you are told which), and entries that miss the requirements are refused with "
        "the value that fell short, so nothing is silently dropped."
    },
)
def sources_add(ctx: ToolContext, sources) -> ToolResult:
    """Add places to your list. The list GROWS -- nothing you already have is lost.

    This is the only way in, and that is deliberate. `record` overwrote the whole list, so a
    second pass at surveying could shrink it; nothing deduplicated, so a place already set
    aside could be handed straight back out; and the admissibility clauses were enforced only
    by the survey stage's exit gate, which meant anything added later -- `record` is in the
    mining stage's toolset -- skipped the vetting entirely. A model that ran out of sources
    mid-run could escape every requirement just by adding them late.

    It also turns running out into something recoverable: when everything on the list is worked
    or full, the answer is to add more, here, under the same requirements.
    """
    spec, pool = _pool(ctx)
    if pool is None:
        return _disabled()
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list) or not sources:
        return ToolResult.fail(f"`sources` must be a non-empty list of {spec.noun} objects.")

    known = {r.value: r for r in pool.rows()}
    added, dupes, refused = _sources.admit_entries(pool, sources)

    if added:
        entries = list(pool.entries) + added
        ctx.working_set.merge({spec.key: entries})
        pool.entries = entries

    parts = [f"Added {len(added)} {spec.noun}(s)."]
    if dupes:
        told = []
        for v in dupes[:5]:
            row = known.get(v)
            why = (
                f" (you set it aside at cycle "
                f"{(pool.state.get(v) or {}).get('done', {}).get('cycle')}: "
                f"{row.done_reason})"
                if row is not None and row.done_reason
                else ""
            )
            told.append(f"{v}{why}")
        parts.append(f"{len(dupes)} already on your list: " + "; ".join(told) + ("..." if len(dupes) > 5 else ""))
    if refused:
        shown = "; ".join(f"{v}: {why}" for v, why in refused[:4])
        parts.append(
            f"{len(refused)} refused -- {shown}" + (f" (+{len(refused) - 4} more)" if len(refused) > 4 else "")
        )
    left = [r for r in pool.rows() if r.bucket in (_sources.NEVER, _sources.NO_LEADS) and not r.full]
    parts.append(f"{len(pool.rows())} {spec.noun}(s) on the list, {len(left)} still unworked.")
    return ToolResult(
        display_text=" ".join(parts),
        structured={"added": len(added), "duplicates": len(dupes), "refused": len(refused), "unworked": len(left)},
    )
