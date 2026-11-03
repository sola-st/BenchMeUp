"""The pool of places to look: counters and a ranking, never an inferred verdict.

An agent that mines a corpus works in two levels. It first decides WHERE to look -- a set of
repositories, package indexes, corpus directories, whatever the spec says a source is -- and
then works candidates out of those places. `agentcore.agent.worklist` handles the second level
well: the harness presents one candidate at a time and recomputes its state from evidence, so
the model never has to carry a cursor. This module covers the first level: a recorded pool must
stay visible in the prompt, and each candidate is tied back to the place it came from, so
deliveries from sources outside the pool can be seen.

WHAT THIS MODULE WILL AND WILL NOT CLAIM. The harness can know YIELD and CAPACITY: how many
candidates and deliveries each entry produced (joined through the very function the diversity
caps use, so a lead is attributed to its source with no cooperation from the model) and whether
an entry's cap is full. Both are recomputed from the worklist on every call and stored nowhere,
so they cannot go stale or disagree with the gate.

The harness CANNOT know effort. Handing ten names to a sweep script is not evidence the script
queried ten: it can crash after three, rate-limit after five, or skip some silently. So there is no state meaning "swept
and empty", and none is inferred from "was handed out and produced nothing". That inference
would delete real sources permanently on a transient failure, and it would look like progress
while doing it.

What replaces it is a RANKING. A priority order is robust to being wrong in a way a verdict is
not: an entry that keeps yielding nothing sinks, and comes back if the list runs dry, which is
the right behaviour when the alternative is mining off-pool. The one deliberate signal is the
agent MARKING sources done with a reason -- evidence, not inference -- and even that only
sinks an entry to the bottom. The reasons are aggregated, because eighteen sources marked done
for the same reason says the query is wrong far more often than it says eighteen large projects
are empty, and neither the harness nor the model would see that from a count.

Nothing here knows what a repository is. An entry's identity is whatever the declared caps
already normalize, so a task whose sources are package names or corpus directories gets the
same machinery from the same spec fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentcore.agent import diversity

# Harness-owned bookkeeping lives beside the agent's own recorded list, never inside it: the
# pool is the model's deliberate artifact (and `_lift_pool_file` only ever grows it), so
# annotating its entries in place would blur whose statement is whose.
# NOT underscore-prefixed: pydantic stores underscore extras as private attrs and excludes
# them from model_dump(), so such a slot would not survive a snapshot and finished sources
# would be re-leased after a resume.
STATE_KEY = "source_state_harness"
_LEGACY_STATE_KEY = "_source_state"

DEFAULT_BATCH = 10
# Up to 100 named entries per view is acceptable. The layered view
# below usually names far fewer -- the current batch is the only actionable part -- but a large
# pool must not be able to push the prompt over on its own.
DEFAULT_VIEW_CAP = 100

NEVER = "never_handed_out"
NO_LEADS = "no_leads_yet"
PRODUCING = "producing"
FINISHED = "finished"
DONE = "marked_done"

# Priority for the next batch. Never-handed-out first; then entries that have already yielded,
# because a source that produced once is a better bet than one that has not; then the ones that
# came back empty; then, only when everything else is spent, the ones the agent marked done.
_RANK = {NEVER: 0, PRODUCING: 1, NO_LEADS: 2, DONE: 3}

_BUCKET_LINES = {
    FINISHED: "FINISHED -- these hit the {limit}-lead limit. Nothing more can be taken from them.",
    PRODUCING: "PRODUCING -- {leads} lead(s) so far, and they still have room.",
    DONE: "YOU MARKED DONE -- you said these have nothing for us. They come back only if the list runs out.",
    NO_LEADS: "HANDED TO YOU, NO LEADS YET -- you were given these in an earlier batch and "
    'no lead has been recorded from them. That is not the same as "empty".',
    # NOT a passive count: when every candidate held is blocked, "I am out of sources" feels
    # true locally and a number in a summary does not contradict it. The line has to say what
    # to do with them.
    NEVER: "NOT TOUCHED YET -- consider these BEFORE going out to gather more sources. "
    "sources_fetch() hands you the next of them and costs nothing; finding and "
    "vetting a new source costs several cycles.",
}
_BUCKET_ORDER = [FINISHED, PRODUCING, DONE, NO_LEADS, NEVER]


@dataclass
class SourceSpec:
    """Everything task-specific, declared by the spec rather than known here.

    `requires` and `evidence` are the whole reason this gate can be task-blind. `requires`
    fields must merely be present. `evidence` fields must be present AND differ between
    entries, because an evidence field records something OBSERVED about one particular place --
    a scoreboard, a README line, a last-changed date -- and thirty entries carrying the same
    sentence is a statement nobody went and looked. That distinction is general; which fields
    fall on which side is not, so the spec says.
    """

    key: str = "source_pool"  # working-set key holding the agent's list
    caps: list = field(default_factory=list)  # reused for identity; the first usable one wins
    terminal: str = "DELIVERED"
    batch_size: int = DEFAULT_BATCH
    view_cap: int = DEFAULT_VIEW_CAP
    noun: str = "source"  # what the spec calls a place to look
    verify: str | None = None  # hook ref: admit an entry into the pool, or refuse
    requires: list = field(default_factory=list)  # fields every entry must carry
    evidence: list = field(default_factory=list)  # fields that must ALSO differ between entries
    spread_by: str | None = None  # field whose distinct values measure spread
    exclusion_fields: list = field(default_factory=list)  # fields to match the roster against
    # Measurable clauses: {name, observe, require}. Each one adds a required field, a gate
    # check, and a line in the prompt -- from ONE declaration, so the three cannot drift.
    # See agentcore.agent.admissibility for the grammar and why `observe` is load-bearing.
    admissibility: list = field(default_factory=list)


@dataclass
class SourceRow:
    value: str
    index: int
    entry: dict
    fetched: int
    last_fetched: int | None
    candidates: int
    delivered: int
    held: int  # what the cap counts: delivered + in flight
    limit: int
    done_reason: str | None
    bucket: str

    @property
    def full(self) -> bool:
        return self.limit > 0 and self.held >= self.limit


class SourcePool:
    """A view over (the agent's recorded list, the worklist, the harness's own bookkeeping)."""

    def __init__(self, spec: SourceSpec, entries: list, worklist: dict, state: dict):
        self.spec = spec
        self.entries = [e for e in (entries or []) if isinstance(e, dict)]
        self.worklist = worklist if isinstance(worklist, dict) else {}
        self.state = state if isinstance(state, dict) else {}
        self._keys, self._urls, self._limit = self._cap()

    # ── identity ────────────────────────────────────────────────────────────
    def _cap(self):
        for cap in self.spec.caps or []:
            rc = diversity.resolve_cap(cap)
            if rc is None:
                continue
            # No limit<=0 guard here, unlike the gates: an unlimited cap still supplies the
            # identity keys, and `self._limit` is consulted as `if self._limit and ...`.
            return rc.keys, rc.urls, rc.limit
        return [], True, 0

    def identity(self, entry: dict) -> str | None:
        """The same string the caps compute, so a lead joins to its source for free."""
        if not self._keys:
            return None
        return diversity.value_of(entry or {}, self._keys, urls=self._urls)

    # ── derived counters ────────────────────────────────────────────────────
    def _yield_counts(self) -> tuple[dict, dict, dict]:
        cands, delivered = {}, {}
        for c in (self.worklist.get("candidates") or {}).values():
            if not isinstance(c, dict):
                continue
            v = self.identity(c.get("fields") or {})
            if not v:
                continue
            cands[v] = cands.get(v, 0) + 1
            if c.get("state") == self.spec.terminal:
                delivered[v] = delivered.get(v, 0) + 1
        held = (
            diversity._counts(self.worklist, self._keys, self._urls, self.spec.terminal, reserving=True)
            if self._keys
            else {}
        )
        return cands, delivered, held

    def rows(self) -> list[SourceRow]:
        cands, delivered, held = self._yield_counts()
        out, seen = [], set()
        for i, entry in enumerate(self.entries):
            v = self.identity(entry)
            if not v or v in seen:
                continue  # a pool the agent recorded twice is one source
            seen.add(v)
            st = self.state.get(v) or {}
            n_c, n_h = cands.get(v, 0), held.get(v, 0)
            done = (st.get("done") or {}).get("reason")
            if self._limit and n_h >= self._limit:
                bucket = FINISHED
            elif n_c:
                # Evidence beats a stale statement: a source that has since produced is not
                # done, whatever the model said about it earlier.
                bucket = PRODUCING
            elif done:
                bucket = DONE
            elif st.get("fetched"):
                bucket = NO_LEADS
            else:
                bucket = NEVER
            out.append(
                SourceRow(
                    value=v,
                    index=i,
                    entry=entry,
                    fetched=int(st.get("fetched") or 0),
                    last_fetched=st.get("last_fetched"),
                    candidates=n_c,
                    delivered=delivered.get(v, 0),
                    held=n_h,
                    limit=self._limit,
                    done_reason=done if bucket == DONE else None,
                    bucket=bucket,
                )
            )
        return out

    def row(self, value: str) -> SourceRow | None:
        return next((r for r in self.rows() if r.value == value), None)

    def buckets(self) -> dict[str, list[SourceRow]]:
        out: dict[str, list[SourceRow]] = {b: [] for b in _BUCKET_ORDER}
        for r in self.rows():
            out[r.bucket].append(r)
        return out

    def off_pool(self) -> list[SourceRow]:
        """Sources that produced candidates without ever being in the pool.

        Not refused by default (a poor pool would deadlock the run); made visible instead.
        """
        known = {self.identity(e) for e in self.entries}
        cands, delivered, held = self._yield_counts()
        return [
            SourceRow(
                value=v,
                index=-1,
                entry={},
                fetched=0,
                last_fetched=None,
                candidates=n,
                delivered=delivered.get(v, 0),
                held=held.get(v, 0),
                limit=self._limit,
                done_reason=None,
                bucket="off_pool",
            )
            for v, n in sorted(cands.items())
            if v not in known
        ]

    def done_reasons(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for r in self.rows():
            if r.bucket == DONE and r.done_reason:
                counts[r.done_reason] = counts.get(r.done_reason, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    # ── handing work out ────────────────────────────────────────────────────
    def next_batch(self, count: int) -> list[SourceRow]:
        """The ranking. Full entries are excluded outright; nothing else ever is."""
        ranked = sorted(
            (r for r in self.rows() if not r.full),
            key=lambda r: (
                _RANK.get(r.bucket, 9),
                r.fetched,
                r.last_fetched if r.last_fetched is not None else -1,
                r.index,
            ),
        )
        return ranked[: max(0, int(count))]

    def lease(self, count: int, cycle: int) -> list[SourceRow]:
        rows = self.next_batch(count)
        for r in rows:
            st = self.state.setdefault(r.value, {})
            st["fetched"] = int(st.get("fetched") or 0) + 1
            st["last_fetched"] = cycle
        self.state["_last_batch"] = {
            "values": [r.value for r in rows],
            "cycle": cycle,
            "first_index": rows[0].index if rows else None,
            "last_index": rows[-1].index if rows else None,
        }
        return rows

    def read_range(self, start: int, end: int) -> list[SourceRow]:
        """Re-read an exact slice, leasing nothing. The agent loses its batch out of a 40-cycle
        window routinely; recovering it must not consume work it has not done."""
        rows = self.rows()
        lo = max(0, int(start))
        hi = min(len(rows), int(end))
        return rows[lo:hi] if lo < hi else []

    def current_batch(self) -> list[SourceRow]:
        last = self.state.get("_last_batch") or {}
        wanted = list(last.get("values") or [])
        by_value = {r.value: r for r in self.rows()}
        return [by_value[v] for v in wanted if v in by_value]

    # ── the agent's own signal ──────────────────────────────────────────────
    def mark_done(self, values, reason: str, cycle: int) -> int:
        return self.mark_done_checked(values, reason, cycle)[0]

    def mark_done_checked(self, values, reason: str, cycle: int) -> tuple[int, list[str]]:
        known = {r.value for r in self.rows()}
        n, unknown = 0, []
        for v in values or []:
            v = str(v).strip()
            if v not in known:
                unknown.append(v)
                continue
            self.state.setdefault(v, {})["done"] = {"reason": reason, "cycle": cycle}
            n += 1
        return n, unknown

    # ── the standing view ───────────────────────────────────────────────────
    def view(self, cycle: int | None = None) -> str:
        noun = self.spec.noun
        if not self.entries:
            return (
                f"YOUR {noun.upper()} LIST IS EMPTY. Nothing has been added yet. Find "
                f"real {noun}s that meet the requirements above and add them with "
                f"`sources_add` -- until then there is nowhere to mine from."
            )

        rows = self.rows()
        b = self.buckets()
        budget = max(1, int(self.spec.view_cap or DEFAULT_VIEW_CAP))
        lines = [f"YOUR {noun.upper()} LIST -- {len(rows)} recorded"]
        leads = sum(r.candidates for r in b[PRODUCING])
        for name in _BUCKET_ORDER:
            if not b[name]:
                continue
            txt = _BUCKET_LINES[name].format(limit=self._limit, leads=leads)
            lines.append(f"  {len(b[name]):>3}  {txt}")

        reasons = self.done_reasons()
        if reasons:
            said = "; ".join(f"{r!r} x{n}" for r, n in reasons[:3])
            lines.append(f"       reasons you gave: {said}")

        batch = self.current_batch()
        if batch:
            last = self.state.get("_last_batch") or {}
            span = (
                f"entries {last.get('first_index')}-{last.get('last_index')}"
                if last.get("first_index") is not None
                else ""
            )
            lines.append(f"\nYOUR CURRENT BATCH ({span}, given to you at cycle {last.get('cycle')}):")
            for r in batch[:budget]:
                got = f"  ({r.candidates} lead(s) so far)" if r.candidates else ""
                lines.append(f"   {r.value}{got}")
            budget -= len(batch[:budget])
            lines.append(
                f"   -> finished with these? sources_done([...], reason) for the ones "
                f"with nothing in them, then sources_fetch({self.spec.batch_size}) "
                f"for the next batch."
            )
            nxt = [r for r in self.next_batch(3) if r not in batch]
            if nxt and budget > 0:
                lines.append(
                    "NEXT UP (a peek -- not yours until you fetch): " + ", ".join(r.value for r in nxt[:budget])
                )
                budget -= len(nxt[:budget])
        else:
            # Nothing has been handed out yet, so show the ends of the list: this is the one
            # cheap chance anyone gets to notice the pool is full of the wrong thing.
            head = rows[: max(1, budget // 2)]
            tail = rows[-max(1, budget - len(head)) :] if len(rows) > len(head) else []
            lines.append("\nLIST STARTS: " + ", ".join(r.value for r in head))
            if tail:
                lines.append("LIST ENDS:   " + ", ".join(r.value for r in tail))
            lines.append(
                f"   -> sources_fetch({self.spec.batch_size}) hands you the next batch "
                f"to work through; you never have to remember where you stopped."
            )

        off = self.off_pool()
        if off:
            n_leads = sum(r.candidates for r in off)
            lines.append(
                f"\nNOTE: {n_leads} lead(s) came from {len(off)} {noun}(s) that are "
                f"not in your list. That is allowed, but your list is the part you "
                f"vetted."
            )
        return "\n".join(lines)


def admit(entries: list, spec: SourceSpec, ctx=None) -> tuple[list, list]:
    """Run the declared entry gate over recorded entries. Returns (admitted, [(entry, why)]).

    This is where "is this actually a software project" belongs, and it is the only thing that
    keeps a bad place out for good -- everything downstream is a ranking that lets an entry
    come back. A gate that only counts entries,
    organisations and present fields can admit a list of directories and tutorial indexes.

    What admissible MEANS is spec content -- the harness supplies the hook and nothing else.

    Fails closed on a gate that cannot be imported, matching the worklist's `verify:`: a gate
    that silently vanishes is worse than no gate, because the list then looks vetted.
    """
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    if not spec.verify:
        return entries, []
    try:
        import importlib

        mod, _, attr = str(spec.verify).partition(":")
        check = getattr(importlib.import_module(mod), attr)
    except Exception as e:  # noqa: BLE001 - reported, never raised into the cycle
        why = (
            f"HARNESS ERROR: the {spec.noun} gate {spec.verify!r} could not be loaded "
            f"({type(e).__name__}: {e}). No {spec.noun} can be admitted until that is "
            f"fixed; this is a configuration fault, not something to work around."
        )
        return [], [(e_, why) for e_ in entries]

    admitted, refused = [], []
    for entry in entries:
        try:
            ok, detail = check(entry, ctx)
        except Exception as e:  # noqa: BLE001 - one bad entry must not sink the batch
            ok, detail = False, f"{type(e).__name__}: {e}"
        (admitted if ok else refused).append(entry if ok else (entry, detail or "refused"))
    return admitted, refused


def required_fields(spec: SourceSpec) -> list[str]:
    """Declared `requires`, plus every clause name. Derived so a task never lists a field
    twice and cannot list one in the gate but not the prompt."""
    from agentcore.agent import admissibility as adm

    out = list(spec.requires or [])
    for name in adm.field_names(spec.admissibility):
        if name not in out:
            out.append(name)
    return out


def admissibility_block(spec: SourceSpec) -> str:
    """What the model is told it must record about each place, and how to obtain it.

    A threshold the model is never shown is a trap: it records what it was asked for, gets
    refused for a value it did not know about, and cannot tell what the number should have
    been. Prose that is asked for but not
    enforced can be satisfied by pasting a description already at hand.
    """
    clauses = [c for c in (spec.admissibility or []) if isinstance(c, dict)]
    if not clauses:
        return ""
    noun = spec.noun or "source"
    width = max(len(str(c.get("name") or "")) for c in clauses)
    lines = [
        f"WHAT MAKES A {noun.upper()} ADMISSIBLE. Record all of these for every entry, and "
        f"LOOK EACH ONE UP -- a value you inferred from a name or a description is not an "
        f"observation, and the gate checks every one of them:"
    ]
    for c in clauses:
        lines.append(f"  {str(c.get('name') or ''):<{width}}  {c.get('require')}")
        lines.append(f"  {'':<{width}}  how: {c.get('observe')}")
    lines.append(
        f"A {noun} that misses any of these is not admissible -- leave it out rather "
        f"than recording it and hoping. The gate names the value that failed and what "
        f"it needed, so read the refusal instead of guessing."
    )
    return "\n".join(lines)


def _absent(value) -> bool:
    """Missing, versus recorded-as-false.

    `not str(v or "").strip()` reads False and 0 as absent, because `False or ""` is `""`.
    That would refuse `is_fork: false` -- a correct observation of exactly the thing the
    clause asks about -- as an unrecorded field, and the same for a legitimate `0`.
    """
    if value is None:
        return True
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return False
    return not str(value).strip()


def admit_entries(pool: "SourcePool", raw, *, now: str | None = None) -> tuple[list, list, list]:
    """Decide what may join a pool: (added, duplicate_values, [(value, why_refused)]).

    THE one place that decides. Both ways into the pool (`sources_add` and the
    pool-file lift) go through it, so they cannot disagree. The stage gate re-checks the whole
    pool, so an entry refused here would otherwise become a pool-wide block.

    Decides only. Applying the result is the caller's, because the tool records a message and
    the lift merges silently.
    """
    from agentcore.agent import admissibility as adm

    known = {r.value for r in pool.rows()}
    added, dupes, refused, seen = [], [], [], set()
    for entry in raw or []:
        if not isinstance(entry, dict):
            refused.append(("(not an object)", "each entry must be an object"))
            continue
        value = pool.identity(entry)
        if not value:
            # NAME the fields: a refusal that does not say which fields identify an entry
            # can only be answered by guessing.
            id_fields = ", ".join(pool._keys) if pool._keys else "(none declared)"
            refused.append(
                (
                    "(unidentifiable)",
                    f"missing the field(s) that identify a {pool.spec.noun}: "
                    f"record {id_fields} (e.g. not 'name' -- the identity is read "
                    f"from those exact keys)",
                )
            )
            continue
        if value in known or value in seen:
            dupes.append(value)
            continue
        missing = [f for f in required_fields(pool.spec) if _absent(entry.get(f))]
        if missing:
            refused.append((value, f"missing {missing}"))
            continue
        # THE SAME ANCHOR THE GATE USES: `gate()` passes `now=`, so a recency clause such as
        # `last_commit: within 12 months` gets one answer here and in the gate.
        problems = adm.check(pool.spec.admissibility, entry, now=now)
        if problems:
            refused.append((value, "; ".join(problems)))
            continue
        seen.add(value)
        added.append(dict(entry))
    return added, dupes, refused


def _cap_limit(spec: SourceSpec, field: str | None) -> int:
    """The declared limit governing `field`, or the first declared cap when field is None.

    A cap is matched by the FIELD NAMES it is keyed on, not by its dimension label. `spread_by`
    names a field the entries carry ("org"); a cap names a dimension ("organisation"). Matching
    the two as strings would find nothing, and a lookup that silently returns "no limit" turns
    the spread requirement off without saying so.
    """
    for cap in spec.caps or []:
        rc = diversity.resolve_cap(cap)
        if rc is None:
            continue
        # The floor follows the limit ADMISSION enforces, declared or defaulted: a cap that
        # admits 4 per source makes a 3-source pool unable to hold 12, whether or not the
        # spec wrote the 4 down (WorklistSpec fills `cap_default_limit` into a cap that
        # declares none, so a loaded spec always carries it). A floor that skipped undeclared
        # limits would let through a pool that the collector could not use.
        if field is None or field in rc.keys or field == rc.dimension:
            return max(1, rc.limit)
    return 0


def gate(pool, spec: SourceSpec, instance: dict, ctx=None, now: str | None = None) -> tuple[bool, str]:
    """Admit a source pool, or refuse it and SAY WHY. Task-blind by construction.

    Nothing here is specific to one kind of source (a `repo_url` field, an http(s) prefix, a
    language field, a named spread dimension): a pool of contest rounds or corpus directories
    can pass it as well as a pool of repositories.

    Every branch returns a sentence the model can act on, naming the field, the count, or the
    arithmetic that failed.

    What is checked is only what is true of ANY collection of places to look:
      * the declared fields are present,
      * the declared EVIDENCE fields differ between entries -- nobody looks at eight things and
        writes the same observation eight times,
      * there are enough entries, and enough spread, to hold the target under the declared caps,
      * nothing is on the spec's exclusion roster,
      * and the spec's own validator accepts each entry.
    Every noun in those checks comes from the spec.
    """
    import math

    noun = spec.noun or "source"

    # The noun is the TASK's word ("repository", "corpus", "round"), so it cannot be
    # pluralized by adding an s ("repositorys"). Counted
    # phrasings put the number first and leave the noun alone.
    def n_of(k: int) -> str:
        return f"{k} {noun}" if k == 1 else f"{k} x {noun}"

    if not isinstance(pool, list) or not pool:
        return False, (
            f"No {noun} entries recorded yet. Record a list under {spec.key!r}: each entry "
            f"needs {list(spec.requires) or 'the fields this task declares'}"
            + (
                f", including {list(spec.evidence)} which must record what you actually observed about THAT {noun}."
                if spec.evidence
                else "."
            )
        )

    entries = [e for e in pool if isinstance(e, dict)]
    if len(entries) != len(pool):
        return False, (
            f"{len(pool) - len(entries)} entry(ies) under {spec.key!r} are not "
            f"objects. Every entry must be an object with "
            f"{list(spec.requires) or 'the declared fields'}."
        )

    # 1. declared fields present -- including every clause name, derived
    for fname in required_fields(spec):
        bad = [i for i, e in enumerate(entries) if _absent(e.get(fname))]
        if bad:
            return False, (
                f"{len(bad)} of {len(entries)} {noun} entries are missing {fname!r} "
                f"(first at position {bad[0]}). Every entry needs all of "
                f"{required_fields(spec)}."
            )

    # 2. evidence must have been observed, not asserted
    for fname in spec.evidence or []:
        bad = [i for i, e in enumerate(entries) if _absent(e.get(fname))]
        if bad:
            return False, (
                f"{len(bad)} of {len(entries)} {noun} entries are missing {fname!r}. That "
                f"field records what you SAW about one {noun}; go and look."
            )
        seen: dict[str, int] = {}
        for e in entries:
            v = str(e.get(fname)).strip()
            seen[v] = seen.get(v, 0) + 1
        worst, n = max(seen.items(), key=lambda kv: kv[1])
        # A couple of genuine coincidences are fine; a majority is not an observation.
        if n > 2 and n > len(entries) / 2:
            return False, (
                f"{n} of {len(entries)} {noun} entries carry the SAME {fname!r} "
                f"({worst[:60]!r}). That field must record what you observed about "
                f"each one; identical text means it was not looked up."
            )

    # 3. enough entries, and enough spread, for the target under the declared caps
    target = int((instance or {}).get("sample_size") or 0)
    if target:
        # The per-source cap: whichever cap is keyed on a field the entries carry as their
        # identity. Falls back to the first declared cap.
        per = 0
        for fname in spec.requires or []:
            per = _cap_limit(spec, fname)
            if per:
                break
        per = per or _cap_limit(spec, None)
        if per:
            need = math.ceil(target / per)
            if len(entries) < need:
                return False, (
                    f"{len(entries)} {noun} entries cannot hold {target} instances at "
                    f"{per} per {noun}: you need at least {need}. Find more."
                )
        if spec.spread_by:
            per_group = _cap_limit(spec, spec.spread_by)
            if per_group:
                need_g = math.ceil(target / per_group)
                # Case-blind, like every other identity in this file:
                # "Apache" and "apache" are one organisation, not two toward the floor.
                groups = {str(e.get(spec.spread_by) or "").strip().lower() for e in entries} - {""}
                if len(groups) < need_g:
                    return False, (
                        f"your {noun} entries span {len(groups)} distinct "
                        f"{spec.spread_by!r} value(s); {target} instances at "
                        f"{per_group} per {spec.spread_by} needs at least "
                        f"{need_g}. Widen the pool."
                    )

    # 4. the task's own measurable thresholds, per entry
    if spec.admissibility:
        from agentcore.agent import admissibility as adm

        for e in entries:
            problems = adm.check(spec.admissibility, e, now=now)
            if problems:
                who = next((str(e.get(k)) for k in (spec.requires or []) if e.get(k)), "?")
                return False, (
                    f"{noun} {who!r} does not meet this benchmark's requirements: "
                    + "; ".join(problems)
                    + f". Every {noun} on the list has to clear all of them -- "
                    f"drop the ones that do not."
                )

    # 4. the spec's exclusion roster, over the fields the spec says carry a name
    if spec.exclusion_fields:
        try:
            from agentcore.agent.exclusions import check_candidate

            for e in entries:
                for fname in spec.exclusion_fields:
                    v = str(e.get(fname) or "").strip()
                    if not v:
                        continue
                    ok, why = check_candidate(v, instance or {})
                    if not ok:
                        return False, (
                            f"{noun} {v!r} is excluded by this benchmark: {why} Remove it from {spec.key!r}."
                        )
        except Exception:  # noqa: BLE001 - an unreachable roster must not wedge the stage
            pass

    # 5. the spec's own judgment about what belongs
    admitted, refused = admit(entries, spec, ctx=ctx)
    if refused:
        first = refused[0]
        return False, (
            f"{len(refused)} of {len(entries)} {noun} entries do not meet this "
            f"benchmark's requirements. First: {first[1]} "
            f"Drop those from {spec.key!r} and find {noun}s that do."
        )

    return True, (
        f"{len(entries)} {noun} entries recorded, all carrying "
        f"{required_fields(spec) or 'the declared fields'}"
        + (f" with distinct {list(spec.evidence)}" if spec.evidence else "")
        + (
            f", spanning "
            f"{len({str(e.get(spec.spread_by) or '').strip().lower() for e in entries} - {''})} "
            f"{spec.spread_by} value(s)"
            if spec.spread_by
            else ""
        )
        + "."
    )


def build(spec: SourceSpec, working_set) -> SourcePool:
    """Assemble a pool from a working set, creating the harness's bookkeeping slot if needed."""
    from agentcore.agent.worklist import WORKLIST_KEY

    snap = working_set.snapshot() if hasattr(working_set, "snapshot") else dict(working_set or {})
    entries = snap.get(spec.key)
    state = getattr(working_set, STATE_KEY, None)
    if not isinstance(state, dict):
        # read-both migration: an in-process working set touched by pre-rename code may
        # still hold the slot under the legacy private name
        state = getattr(working_set, _LEGACY_STATE_KEY, None)
    if not isinstance(state, dict):
        state = snap.get(STATE_KEY) if isinstance(snap.get(STATE_KEY), dict) else {}
    if getattr(working_set, STATE_KEY, None) is not state:
        if hasattr(working_set, "__setattr__"):
            try:
                setattr(working_set, STATE_KEY, state)
            except Exception:  # noqa: BLE001 - a dashboard must never break a cycle
                pass
    return SourcePool(spec, entries if isinstance(entries, list) else [], snap.get(WORKLIST_KEY) or {}, state)


def from_an_admitted_source(fields: dict, ctx) -> tuple[bool, str]:
    """(ok, why) -- is this candidate's source one the pool admitted?

    WHY THIS IS HERE AND NOT IN THREE VALIDATORS. Every lead validator's success message says
    "shapes are real and the repository is admissible", and none of them consulted this module
    The claim is cheap to earn: `SourcePool.identity` is the
    same `diversity.value_of` the caps already compute, so a lead joins to its source with no
    cooperation from the model and no field names known here. One implementation, so copies of
    the join cannot drift apart.

    ADMITS, LOUDLY, IN THREE CASES. A task that declares no pool (`sources.enabled: false`)
    gets no gate at all -- an empty gate that refuses everything is worse than none, which is
    `excluded_names`' contract too. An EMPTY pool is not a pool that refused: the survey gate
    is what makes a pool exist, and a second gate contradicting it would refuse the first lead
    of every run. And a candidate whose identity cannot be computed is admitted, because a
    refusal naming nothing is one the model cannot act on.
    """
    from agentcore.tools.source_tools import _pool  # local: source_tools imports this module

    try:
        spec, pool = _pool(ctx)
    except Exception:  # noqa: BLE001 - a gate must not die on a context shape it did not expect
        return True, ""
    if spec is None or pool is None:
        return True, ""
    known = {r.value for r in pool.rows() if r.value}
    if not known:
        return True, ""
    value = pool.identity(fields or {})
    if not value or value in known:
        return True, ""
    return False, (
        f"{value} is not in the {spec.noun} pool. The pool is where admissibility is "
        f"checked -- every clause in the brief is evaluated when a {spec.noun} is admitted, "
        f"and a candidate mined outside it has satisfied none of them. Either record "
        f"{value} with sources_add first (with the observations each clause asks for), or "
        f"take this candidate from a {spec.noun} already admitted. "
        f"Admitted so far: {', '.join(sorted(known)[:8]) or 'nothing yet'}"
        f"{' ...' if len(known) > 8 else ''}."
    )
