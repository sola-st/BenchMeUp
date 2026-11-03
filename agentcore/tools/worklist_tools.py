"""Worklist tools — opt-in via `tool_modules: ["agentcore.tools.worklist_tools"]`.

Deliberately four narrow tools, and none of them lets the model set a state. The model reports
*evidence*; `Worklist` recomputes the state from what is present. See agentcore/agent/worklist.py
for why (short version: a state whose preconditions are unmet must be unreachable, or the model
will talk past it -- as one real run did for ~90 consecutive cycles).

  worklist_add       register candidates found during discovery (duplicates refused)
  worklist_record    attach evidence to the candidate in hand -> harness advances its state
  worklist_drop      finish with a candidate: REJECTED (not admissible) or BLOCKED (we can't process it)
  worklist_status    the state histogram

`worklist_record` takes no candidate id: it always applies to the candidate the harness is
currently presenting. That removes a whole class of mistake (updating the wrong row) and one more
thing the model would otherwise have to track.
"""

from __future__ import annotations

import importlib
import re

from agentcore.agent.worklist import BLOCKED, REJECTED, WORKLIST_KEY, Worklist
from agentcore.tools.context import ToolContext
from agentcore.tools.registry import tool
from agentcore.tools.result import ToolResult


def _resolve(path: str, agent_dir: str | None):
    """Resolve "module:function", trying the agent's own directory first so a collector can
    ship its own validators next to its agent.yaml."""
    mod, _, attr = path.partition(":")
    try:
        return getattr(importlib.import_module(mod), attr)
    except Exception:
        import sys

        if agent_dir and agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        return getattr(importlib.import_module(mod), attr)


def _wl(ctx: ToolContext) -> Worklist | None:
    agent = getattr(ctx, "agent", None)
    spec = getattr(agent, "spec", None)
    wspec = getattr(spec, "worklist", None)
    if wspec is None or not wspec.enabled:
        return None
    store = getattr(ctx.working_set, WORKLIST_KEY, None)
    if not isinstance(store, dict):
        store = {}
        setattr(ctx.working_set, WORKLIST_KEY, store)
    # Resolve declared validators once per call. These are what make the harness -- rather than
    # the model -- the thing that confirms a step actually happened.
    agent_dir = (getattr(agent, "_scratch", {}) or {}).get("agent_dir")
    validators = {}
    for st in wspec.states:
        if st.verify:
            try:
                validators[st.name] = _resolve(st.verify, agent_dir)
            except Exception as e:  # noqa: BLE001
                # FAIL CLOSED. Skipping a missing validator would keep the run alive, but worse:
                # the state silently loses its gate, every candidate walks through unchecked, and
                # the run delivers unverified instances that are indistinguishable from verified
                # ones. The whole design rests on the harness confirming claims, so a gate that
                # cannot be loaded must refuse, not wave things through.
                #
                # Still not an exception: the run continues and the refusal is actionable, so a
                # typo'd `verify:` shows up as "this state cannot be entered, here's why" rather
                # than a dead process or, worse, a clean-looking bad result.
                validators[st.name] = _unloadable(st.verify, e)
    return Worklist(wspec, store, validators=validators, ctx=ctx)


def _unloadable(ref: str, err: Exception):
    """A stand-in validator for a `verify:` that could not be imported. Always refuses."""

    def _refuse(fields, ctx):
        return False, (
            f"HARNESS ERROR: this state's validator {ref!r} could not be loaded "
            f"({type(err).__name__}: {err}). No candidate can pass this state until "
            f"that is fixed -- this is a configuration fault, not something to work "
            f"around by recording different evidence. Report it and move on."
        )

    return _refuse


def _disabled() -> ToolResult:
    return ToolResult.fail("This agent has no worklist configured (worklist.enabled is false).")


def _no_active(wl: Worklist, doing: str) -> ToolResult:
    """Nothing to work on -- and WHY, since the store knows.

    A generic "No active candidate. Use worklist_add to register candidates first." would
    mislead a model that HAS registered candidates, all already finished, and send it off
    re-registering leads. An empty list and a list whose every entry
    is finished are different situations with different next moves; say which it is.
    """
    h = wl.histogram()
    if not h:
        return ToolResult.fail(
            f"No candidate to {doing}: the worklist is empty. Register candidates with worklist_add first."
        )
    where = ", ".join(f"{k} {v}" for k, v in h.items())
    return ToolResult.fail(
        f"No candidate to {doing}: every registered candidate is finished ({where}), and a "
        f"finished candidate is not worked on again. Register new candidates with worklist_add."
    )


@tool(
    "worklist_add",
    params={
        "candidates": (
            "a LIST of dicts, one per candidate. Each dict maps field name -> value; the "
            "REQUIRED names are this task's identity key plus everything its first step "
            "requires -- they are listed in the WORKLIST section of this prompt, and any "
            "refusal repeats the full list. Unknown names are ignored (you are told which); "
            "an empty string counts as missing."
        )
    },
    aliases={"candidates": ["candidate", "entries", "leads", "items"]},
)
def worklist_add(ctx: ToolContext, candidates: list[dict]) -> ToolResult:
    """Register discovered candidates. Pass a list of dicts of known fields.

    Duplicates are refused automatically -- you do not need to check first. Add candidates in
    bulk as you discover them; you will be given them back one at a time to work on.
    """
    wl = _wl(ctx)
    if wl is None:
        return _disabled()
    if isinstance(candidates, dict):
        candidates = [candidates]
    if not isinstance(candidates, list):
        return ToolResult.fail("`candidates` must be a list of objects.")
    # THE CAP, AT THE DOOR.
    #
    # The diversity cap is enforced by the first state's validator, which runs after the model
    # has already picked a candidate, cloned its repository, swept its history and registered
    # it -- so a spent source is discovered at the END of the work, one candidate per cycle.
    #
    # Nothing about such a candidate can succeed, which is exactly the test `worklist_add`
    # already applies to duplicates and incomplete entries. Refusing here, in ONE reply for the
    # whole batch, replaces dozens of one-per-cycle drops.
    #
    # The in-flight reservation comes along for free: check_caps counts delivered plus
    # in-flight, so two candidates already vetting fill a cap of two.
    from agentcore.agent import diversity

    caps = diversity.caps_from(ctx)
    full_counts: dict[str, int] = {}
    full_detail: dict[str, str] = {}

    added, dupes, bad = [], 0, []
    for c in candidates:
        if not isinstance(c, dict):
            bad.append("not an object")
            continue
        if caps:
            ok_cap, why = diversity.check_caps(c, ctx, caps=caps)
            if not ok_cap:
                value = why.split(" has already given")[0].split(" is ")[0].strip() or "?"
                full_counts[value] = full_counts.get(value, 0) + 1
                full_detail.setdefault(value, why)
                continue
        ok, (kind, detail) = wl.add(c)
        if ok:
            added.append(detail)
        elif kind == "duplicate":
            dupes += 1
        else:
            bad.append(detail)
    msg = f"Added {len(added)} new candidate(s)."
    if full_counts:
        n = sum(full_counts.values())
        # Name each spent source ONCE with its count. check_caps' own detail repeats the
        # source name and the same advice, so it is deliberately not appended -- the whole
        # point of refusing here is to replace many messages with one short one.
        named = ", ".join(f"{v} (x{k})" for v, k in list(full_counts.items())[:4])
        msg += (
            f" {n} not registered -- their source is FULL: {named}"
            + (f" (+{len(full_counts) - 4} more)" if len(full_counts) > 4 else "")
            + ". Nothing is wrong with those candidates; that source has no room left and "
            "another from it will be refused the same way. Bring candidates from a "
            "DIFFERENT source."
        )
    if dupes:
        msg += f" {dupes} already in the worklist."
    if bad:
        # Say WHAT is missing. Reporting these as duplicates left the model with no fixable
        # signal, and it re-added variants of the same candidate until the stall detector fired.
        # Whole reasons, never a character cap: the old [:200] over the JOINED string cut the
        # tail refusals mid-field-list, which is the schema-reveal problem all over again. The
        # reasons repeat (a bulk add usually fails one way), so distinct ones are few.
        uniq = list(dict.fromkeys(bad))
        shown = "; ".join(uniq[:3])
        more = f" (+{len(uniq) - 3} more distinct reasons)" if len(uniq) > 3 else ""
        msg += f" {len(bad)} could not be added: {shown}{more}"
    # A STRIKE CHARGED HERE IS SHOWN HERE. A candidate registered with a later state's inputs
    # already in the payload has that state's gate run by add() -> _recompute, and a refusal
    # there is a counted strike, so the reply must carry the refusal rather than say only
    # "Added 1 new candidate(s)". Nothing about the check or its strike changes.
    refused_on_add = [f"{k}: NOT ACCEPTED at registration -- {err}" for k in added if (err := wl.validation_error(k))]
    if refused_on_add:
        msg += " " + " ".join(refused_on_add)
    msg += f" Worklist now: {wl.histogram()}"
    return ToolResult(
        display_text=msg,
        structured={
            "added": len(added),
            "duplicates": dupes,
            "unusable": len(bad),
            "cap_full": sum(full_counts.values()),
            "full_sources": sorted(full_counts),
            "histogram": wl.histogram(),
        },
    )


CAVEAT_CHARS = 480


def advance_note(detail: str | None) -> str:
    """What to add to an advance message when the check that passed said something first.

    A gate can pass and still have observed something the agent would want to know -- the
    negative-control probe reports UNPROBED when the declared break left the checked tree
    byte-identical, and passes anyway because some task families cannot be probed there. That
    reason would otherwise go onto the RECORD and nowhere else: refusal detail is shown to the
    agent, success detail is dropped, so an instance could be delivered whose break pointed at
    a directory the agent had itself deleted.

    Silent for an ordinary pass. Most validators return routine prose or nothing, and none of it
    belongs on the turn -- every character here is charged to a prompt that already renders on
    every cycle.
    """
    from agentcore.tools.variants import UNPROBED

    text = str(detail or "")
    if UNPROBED not in text:
        return ""
    at = text.find(UNPROBED)
    shown = text[at : at + CAVEAT_CHARS].rstrip()
    if len(text) - at > CAVEAT_CHARS:
        shown += " [...]"
    # NAME THE FIELD AND THE MOVE. A caveat alone ("fix it now if it is yours to fix") names
    # neither the field nor the action, and may lead to dropping a sound candidate over a field
    # that could be re-recorded. The probe reports the field, so both can be said.
    field = re.search(r"\(([A-Za-z_][A-Za-z0-9_]*)\)", shown)
    what = (
        f" Re-record {field.group(1)} with a value that works in this sandbox and the check will be probed properly."
        if field
        else " Re-record the field it names and the check will be probed properly."
    )
    return (
        f" It advanced, but the check that passed it did not prove everything it checks: "
        f"{shown}{what} Nothing is blocking you and this is NOT a reason to drop the "
        f"candidate -- everything else it has proven still stands, and the note stays on "
        f"its record until the field is fixed."
    )


@tool(
    "worklist_record",
    params={
        "fields": (
            "a dict mapping field name -> the evidence you observed (real command output, "
            "never a summary). Only names this task's lifecycle reads are kept -- the DO "
            "THIS NOW section of this prompt lists exactly what the current step needs, and "
            "a refusal names every accepted field. Empty values count as missing."
        )
    },
    aliases={"fields": ["values", "evidence", "data", "record"]},
)
def worklist_record(ctx: ToolContext, fields: dict) -> ToolResult:
    """Attach evidence to the candidate you are currently working on.

    Pass only the fields you have just established, e.g.
    {"parent_build": "ok", "toolchain": "jdk8"}. The state advances automatically once the
    required evidence for the next step is present -- you never set the state yourself.

    Structured values are passed as real JSON values, not quoted text. A field holding
    per-input values is an object, and its commands field an object with the SAME keys:
      {"output_digests": {"small": "3f2a...64 hex...", "large": "9c1b...64 hex..."},
       "digest_cmd": {"small": "./prog inputs/small.txt", "large": "./prog inputs/large.txt"}}
    Writing the object as a quoted string ('{"small": ...}') is tolerated but never
    needed -- do not spend turns re-escaping quotes.

    RECORD OBSERVATIONS, NOT STATISTICS. Where this task's lifecycle declares a quantity as
    computed (a median, a coefficient of variation, a ratio, a rate), record the raw numbers
    you measured -- {"timings": [671, 677, 682]} -- and the harness computes it and tells you
    what it computed. Do not implement arithmetic in a container shell; the reply names the
    observations each computed field needs.
    """
    wl = _wl(ctx)
    if wl is None:
        return _disabled()
    cur = wl.current()
    if cur is None:
        return _no_active(wl, "record evidence for")
    key, before = cur[0], cur[1]["state"]
    if not isinstance(fields, dict) or not fields:
        return ToolResult.fail("`fields` must be a non-empty object of evidence to record.")
    # Field names nothing in the lifecycle reads are not kept (update() filters them), and the
    # model must be TOLD -- silently dropping a typo'd name turns one wrong keystroke into a
    # guessing game. When every name is unknown, nothing will change, so answer with the whole
    # vocabulary instead of a misleading "still needs" list.
    vocab = wl.vocabulary()
    ignored = sorted(k for k in fields if k not in vocab)
    if len(ignored) == len(fields):
        return ToolResult.fail(
            f"Nothing recorded: no step reads {ignored}. The field names this task's "
            f"lifecycle reads are: {sorted(vocab)}."
        )
    note = f" Ignored {ignored}: no step reads those names (known: {sorted(vocab)})." if ignored else ""
    before_derived = dict((wl.get(key) or {}).get("fields", {}).get("_derived") or {})
    c = wl.update(key, fields)
    # WHAT THE HARNESS COMPUTED FOR YOU, said out loud on the turn it happens. A quantity that
    # silently appears is a quantity the agent will compute again next cycle, at real cost.
    now_derived = (c or {}).get("fields", {}).get("_derived") or {}
    fresh = {
        k: v
        for k, v in now_derived.items()
        if not v.get("agent_value") and (k not in before_derived or before_derived[k].get("value") != v.get("value"))
    }
    if fresh:
        note += (
            " Computed for you: "
            + "; ".join(
                f"{k}={v.get('value')} ({v.get('agg')} of {', '.join(v.get('from') or [])})"
                for k, v in sorted(fresh.items())
            )
            + ". Record the raw observations; the harness does this arithmetic."
        )
    for problem in wl.derive_problems(key):
        note += f" {problem}"
    # The row may have been RE-KEYED by that update -- the model corrected an identity field,
    # or a validator rewrote one (sha_is_real resolves an abbreviated commit and writes the full
    # hash back). Everything below looks the candidate up by key, and a stale key silently
    # returns "nothing missing, no refusal", which sent the model the tool's own
    # "this is a harness bug" message on the turn it did exactly the right thing.
    key = next((k for k, cand in wl.store["candidates"].items() if cand is c), key)
    after = c["state"]
    if after == before:
        nxt = wl.next_state_after(before)
        need = wl.unmet(key)  # same `_absent` test the state machine uses
        refused = wl.validation_error(key)
        if refused:
            # The harness checked the evidence and did not accept it. Say exactly why -- this is
            # the difference between "you are stuck" and "here is what to fix".
            return ToolResult(
                display_text=f"NOT ACCEPTED for {key}. {refused}{note}",
                structured={"candidate": key, "state": after, "advanced": False, "refused": refused},
            )
        if not need:
            # Should be unreachable: no unmet requirement and no refusal, yet no advance. Say so
            # plainly rather than emitting "still needs: (nothing)", which reads as a harness bug
            # and gives the model nothing to act on.
            return ToolResult.fail(
                f"{key} did not advance and nothing is recorded as missing. Re-record the "
                f"evidence for {nxt.name if nxt else 'the next step'} with real values "
                f"(an empty string counts as missing), or worklist_drop it."
            )
        return ToolResult(
            display_text=(
                f"Recorded, but {key} is still {before}. To reach "
                f"{nxt.name if nxt else '(final)'} it still needs: {need}.{note}"
            ),
            structured={"candidate": key, "state": after, "advanced": False, "still_needs": need},
        )
    # MOVING BACKWARDS IS A REFUSAL, NOT AN ADVANCE. When an input of an already-passed state
    # is re-recorded, its gate re-runs and may refuse, and `_recompute` sets the candidate back
    # to the state before it. That must not be reported as an advance: the strike is charged
    # and the refusal that caused the move has to be shown.
    names = [s.name for s in wl.spec.states]
    if after in names and before in names and names.index(after) < names.index(before):
        refused = wl.validation_error(key) or "(no refusal is recorded)"
        return ToolResult(
            display_text=(
                f"NOT ACCEPTED for {key}. {refused} The value you just recorded "
                f"is an input of that check, which re-ran and refused it, so {key} "
                f"fell back from {before} to {after}. Everything else it recorded "
                f"is kept and is re-checked once that check passes.{note}"
            ),
            structured={
                "candidate": key,
                "state": after,
                "advanced": False,
                "refused": refused,
                "fell_back_from": before,
            },
        )
    verdict = (c.get("_checked") or {}).get(after)
    caveat = advance_note(verdict[1] if isinstance(verdict, list) and len(verdict) > 1 else "")
    # THE FATAL REFUSAL IS SHOWN ON THE TURN IT LANDS. The last permitted strike sets the row
    # REJECTED/BLOCKED inside the strike path, which composes "<detail> -- refused 8 times
    # now ... move on" and caches it; this branch must show that text rather than the state
    # change alone. Display only: the verdict and the count are unchanged.
    from agentcore.agent.worklist import BLOCKED, REJECTED

    why = wl.validation_error(key) if after in (REJECTED, BLOCKED) else None
    if why:
        caveat = f"{caveat} {why}"
    return ToolResult(
        display_text=f"{key}: {before} -> {after}.{note}{caveat}",
        structured={
            "candidate": key,
            "state": after,
            "advanced": True,
            "caveat": bool(caveat),
            "histogram": wl.histogram(),
        },
    )


def _review_blocked_reason(ctx: ToolContext, wl: Worklist, key: str, reason: str) -> str:
    """One non-blocking model call reviewing a refusal-free blocked booking's stated
    reason against the candidate's record. Returns advice text or ""; never raises."""
    model = getattr(ctx, "model", None)
    if model is None:
        return ""
    c = wl.get(key) or {}
    fields = {k: str(v)[:2000] for k, v in (c.get("fields") or {}).items() if not k.startswith("_")}
    history = [str(h)[:2000] for h in (c.get("history") or [])][-8:]
    prompt = (
        "A builder agent inside a benchmark-construction harness wants to book a "
        "candidate BLOCKED -- 'the pipeline cannot process it', which removes it from "
        "the sample's denominator -- with NO check having refused it. Review whether the "
        "stated reason is supported by the record, or whether it is a misdiagnosis or a "
        "method fault the agent could fix (e.g. 'no network access' when the real error "
        "is unbuilt SNAPSHOT sibling modules; 'environment too slow' when the agent's "
        "own script is the slow part). The booking will stand either way -- your verdict "
        "is recorded next to it. At most 60 words. First line exactly 'FAULT: "
        "environment' or 'FAULT: method' or 'FAULT: candidate'.\n\n"
        f"STATED REASON: {reason}\n\nHISTORY:\n"
        + "\n".join(f"- {h}" for h in history)
        + f"\n\nRECORDED FIELDS:\n{fields}"
    )
    try:
        resp = model.query([{"role": "user", "content": prompt}])
        verdict = (getattr(resp, "content", "") or "").strip()[:2000]
        # A REVIEW THAT RAN AND SAID NOTHING IS STILL A FACT. Returning "" here made a
        # dead model, a timeout and a genuinely empty reply all read as "no advisor line",
        # which the record cannot tell apart from "reviewed, no dispute". The design promise is
        # "disputed blocks become visible and countable at merge time"; a silent failure is
        # neither.
        return verdict or f"{ADVISOR_UNAVAILABLE} (empty reply)"
    except Exception as e:  # noqa: BLE001 - the booking stands either way; say WHY no verdict
        return f"{ADVISOR_UNAVAILABLE} ({type(e).__name__})"


#: How the advisor's verdict is written into a BLOCKED booking's reason, and the words a
#: reader (benchcompiler's merge manifest) needs to classify it. The writer and the reader
#: share these so a new verdict kind cannot be counted as a clean review by an old reader.
ADVISOR_TAG = "[ADVISOR: "
ADVISOR_UNAVAILABLE = "UNAVAILABLE"  # the review did not happen; say why
ADVISOR_DISPUTED = ("FAULT: method", "FAULT: candidate")  # the agent could have gone on


def advisor_verdict(reason: str) -> str:
    """The advisor text inside a booking reason, "" when the reason carries none."""
    i = str(reason or "").rfind(ADVISOR_TAG)
    if i < 0:
        return ""
    return str(reason)[i + len(ADVISOR_TAG) :].rstrip("]").strip()


@tool(
    "worklist_drop",
    params={
        "reason": "why, in one sentence quoting what you saw; it stays with the candidate and is "
        "read by whoever audits the sample",
        "blocked": "false = this candidate is not admissible (a refactor, no test, the transition "
        "does not hold) -- a real screening rejection, counted against the sample. "
        "true = WE cannot process it (toolchain missing, dependency rot, rate limit, a "
        "cap already spent) -- a pipeline fault, NOT counted against the sample. "
        "Pick by whose fault it is; if the candidate itself is wrong, it is false.",
    },
    aliases={"reason": ["why", "message", "note"]},
)
def worklist_drop(ctx: ToolContext, reason: str, blocked: bool = False) -> ToolResult:
    """Finish with the current candidate.

    Use blocked=false when the candidate is genuinely not admissible (a refactor, no test, the
    transition does not hold). Use blocked=true when WE cannot process it (toolchain missing,
    dependency rot, rate limit) -- that is information about the pipeline, not a bad candidate,
    and it is recorded as such rather than counted against the sample.
    """
    wl = _wl(ctx)
    if wl is None:
        return _disabled()
    cur = wl.current()
    if cur is None:
        return _no_active(wl, "drop")
    key = cur[0]
    state = BLOCKED if blocked else REJECTED
    # THE CLASS COMES FROM THE MECHANISM, not from the model's choice of flag. Candidate
    # faults -- a log analyser whose output carries a live timestamp, a network server, an
    # interactive curses tool -- booked blocked=true would be removed from the sample's denominator and makes "all gates passed" vacuous.
    # A validator that RAN and refused this candidate is evidence about the CANDIDATE: the
    # pipeline worked, the candidate failed it. That evidence vetoes blocked=true. A blocked
    # booking survives only when no check ever refused the candidate at its current state --
    # toolchain missing, dependency rot, rate limits: things that prevented checking at all.
    if blocked:
        c = wl.get(key) or {}
        history = [str(h) for h in (c.get("history") or [])]
        # Refusals are logged under the state being ENTERED while the candidate remains at
        # the previous one, so the veto cannot match f"{current_state}:REFUSED:". The
        # mechanical form that matches reality: a refusal for a state the candidate never SUBSEQUENTLY
        # achieved is evidence about the candidate as it stands; a refusal at a hurdle it
        # later cleared says nothing about why it is being dropped now.
        # IN ORDER. Counting every advance entry regardless of position would let a state
        # cleared ONCE excuse every later refusal at it -- a candidate that cleared BUILT,
        # re-recorded a BUILT input its gate refused and fell back could then be booked
        # blocked with that refusal open. One pass: a refusal is open until an advance to that state
        # FOLLOWS it.
        open_refusals: dict[str, int] = {}
        for h in history:
            if ":REFUSED:" in h:
                st_name, _, what = h.partition(":REFUSED:")
                # A harness fault is not a check that ran and refused this candidate.
                # Some records carry each fault as a second,
                # ":REFUSED:HARNESS FAULT at ..." line next to the fault's own; counting it
                # here would book REJECTED, into the denominator, exactly the booking the fault
                # message tells the agent not to make.
                if what.startswith("HARNESS FAULT at "):
                    continue
                open_refusals[st_name] = open_refusals.get(st_name, 0) + 1
            elif ":" not in h:  # an advance entry clears the refusals before it
                open_refusals.pop(h, None)
        systemic = wl.systemic_shapes(key) if hasattr(wl, "systemic_shapes") else {}
        if open_refusals and systemic:
            # THE ONE BLOCKED BOOKING WHERE THE REFUSALS ARE THE THING IN DISPUTE.
            # The systemic notice tells an agent to book exactly this, and the
            # override below would flip it to REJECTED and skip the advisor --
            # the opposite of what the notice promises. The
            # exemption rests on the HARNESS's own cross-candidate finding, so an agent
            # cannot reach it by asserting that a gate is wrong.
            named = ", ".join(sorted(systemic))
            reason = (
                f"{reason or '(no reason given)'} [booked BLOCKED: the harness has "
                f"itself found this refusal shape systemic at {named}, so these "
                f"refusals are what is in dispute -- reviewed rather than counted "
                f"against the candidate]"
            )
            review = _review_blocked_reason(ctx, wl, key, reason)
            if review:
                reason = f"{reason} {ADVISOR_TAG}{review}]"
        elif open_refusals:
            state = REJECTED
            named = ", ".join(f"{s} ({n}x)" for s, n in open_refusals.items())
            reason = (
                f"{reason or '(no reason given)'} [booked REJECTED, not blocked: "
                f"checks ran and refused this candidate at {named} without it ever "
                f"clearing them -- a candidate its checks refuse is a rejected "
                f"candidate, and it stays in the sample's denominator]"
            )
        else:
            # A refusal-free blocked booking rests ENTIRELY on the agent's prose -- no
            # check ran at the death point, so the veto has no evidence, and such reasons
            # are often misdiagnoses, each wrongly out of the denominator. The advisor reviews
            # the reason against the record; its verdict is APPENDED, never a state
            # change -- disputed blocks become visible and countable at merge time.
            review = _review_blocked_reason(ctx, wl, key, reason)
            if review:
                reason = f"{reason or '(no reason given)'} {ADVISOR_TAG}{review}]"
    wl.terminate(key, state, reason or "(no reason given)")
    nxt = wl.current()
    # SAY THE STATE THAT WAS BOOKED, not the flag that was asked for. This read
    # `'BLOCKED' if blocked else 'REJECTED'`, so every booking the veto above flipped
    # announced "-> BLOCKED" while REJECTED went into the record -- the agent's only
    # feedback contradicting the run's own data, and the explanation in the reason text
    # arguing with the headline right above it.
    return ToolResult(
        display_text=(f"{key} -> {state} ({reason}). Next candidate: {nxt[0] if nxt else '(none left)'}"),
        structured={"candidate": key, "state": state, "histogram": wl.histogram()},
    )


@tool(
    "worklist_defer",
    params={
        "reason": "why you are setting it aside. The candidate KEEPS all its recorded progress and "
        "comes back after the others -- prefer this to worklist_drop whenever the "
        "candidate might still be good later (a full cap, a flaky network, a build you "
        "have not worked out yet)."
    },
)
def worklist_defer(ctx: ToolContext, reason: str = "") -> ToolResult:
    """Set the current candidate aside and move to another, KEEPING its progress.

    Use this when you are stuck on this candidate but it is not clearly bad -- you will be
    offered it again after the others. Prefer it to worklist_drop, which throws the work away.
    """
    wl = _wl(ctx)
    if wl is None:
        return _disabled()
    cur = wl.current()
    if cur is None:
        return _no_active(wl, "defer")
    key = cur[0]
    wl.defer(key)
    nxt = wl.current()
    moved = bool(nxt) and nxt[0] != key
    if nxt and not moved:
        # DEFERRING THE ONLY CANDIDATE IS A NO-OP, AND MUST NOT READ AS A MOVE. `current()`
        # serves the least-deferred, furthest-along ACTIVE candidate, so with one candidate it
        # comes straight back; a reply naming the same key on both sides would read as success.
        # Say what happened, and name the two things that would change it.
        return ToolResult(
            display_text=(
                f"{key} is the ONLY candidate in hand, so deferring it changed nothing -- it "
                f"is still the current one and will be until there is something else to work "
                f"on. Its progress is kept either way. Two things move you forward from here: "
                f"worklist_add more candidates and this one goes to the back of that queue, "
                f"or -- if you believe it cannot be made to work at all -- worklist_drop it "
                f"with the reason, which is a good outcome and not a failure."
            ),
            structured={"deferred": key, "next": key, "moved": False},
        )
    return ToolResult(
        display_text=(
            f"Deferred {key}{' (' + reason + ')' if reason else ''}; its progress is "
            f"kept. Now on: {nxt[0] if nxt else '(none left)'}"
        ),
        structured={"deferred": key, "next": nxt[0] if nxt else None, "moved": moved},
    )


@tool("worklist_status", idempotent_ok=True)
def worklist_status(ctx: ToolContext) -> ToolResult:
    """Where the candidates are, and which sources still have room.

    A source's room counts the candidates you registered from it as well as the delivered
    ones. The reply names the registered candidates that can no longer be delivered because
    their source is full, whatever else is true of them."""
    # The histogram alone tells the model nothing it does not already have: it rides on the
    # `structured` payload of every add, record and drop. A source can be closed by
    # candidates the model registered itself (the reservation in `diversity.check_caps`), and
    # that count appears in no histogram. So this reports capacity the way the gate computes it.
    wl = _wl(ctx)
    if wl is None:
        return _disabled()
    h = wl.histogram()
    if not h:
        return ToolResult.text("Worklist is empty. Use worklist_add after discovery.")
    lines = [f"Worklist ({sum(h.values())} candidates):"]
    lines += [f"  {k:<20} {v}" for k, v in h.items()]

    capacity, dead = _capacity_rows(wl, ctx)
    for cap in capacity:
        full = [c for c in cap["values"] if c["full"]]
        open_ = [c for c in cap["values"] if not c["full"]]
        lines.append(f"{cap['dimension']} cap {cap['limit']}:")
        for c in full:
            extra = f" ({c['delivered']} delivered + {c['in_flight']} in flight)" if c["in_flight"] else ""
            lines.append(f"  FULL      {c['value']} {c['held']}/{cap['limit']}{extra}")
        for c in open_:
            lines.append(f"  room {cap['limit'] - c['held']:<4} {c['value']} {c['held']}/{cap['limit']}")
        if not open_:
            lines.append(f"  every {cap['dimension']} you have used is full -- bring a NEW one")
    if dead:
        lines.append(
            f"{len(dead)} registered candidate(s) CANNOT be delivered: their "
            f"{'/'.join(sorted({d['dimension'] for d in dead}))} is already full. "
            f"Working on them cannot succeed -- drop or defer them, do not verify "
            f"them one at a time:"
        )
        for d in dead[:12]:
            lines.append(f"  {d['candidate']}  ({d['value']} is full)")
        if len(dead) > 12:
            lines.append(f"  ... and {len(dead) - 12} more")

    flat = [dict(v, dimension=c["dimension"], limit=c["limit"]) for c in capacity for v in c["values"]]
    return ToolResult(
        display_text="\n".join(lines),
        structured={"histogram": h, "delivered": len(wl.delivered()), "capacity": flat, "undeliverable": dead},
    )


def _capacity_rows(wl, ctx) -> tuple[list[dict], list[dict]]:
    """(per-cap value rows, in-flight candidates that no longer fit).

    Counted through `diversity._counts`, the same function the gate uses, so this can never
    advertise room that a record call will refuse.
    """
    from agentcore.agent import diversity

    caps = getattr(wl.spec, "caps", None) or []
    success = diversity.success_from_spec(wl.spec)
    rows, dead = [], []
    for cap in caps:
        rc = diversity.resolve_cap(cap)
        if rc is None or rc.limit <= 0:
            continue
        keys, limit, urls, dim = rc
        held = diversity._counts(wl.store, keys, urls, success, reserving=True)
        done = diversity._counts(wl.store, keys, urls, success)
        values = [
            {"value": v, "held": n, "delivered": done.get(v, 0), "in_flight": n - done.get(v, 0), "full": n >= limit}
            for v, n in sorted(held.items(), key=lambda kv: -kv[1])
        ]
        if not values:
            continue
        rows.append({"dimension": dim, "limit": limit, "values": values})
        spent = {v["value"] for v in values if v["full"]}
        for key in wl.store.get("order") or []:
            cand = (wl.store.get("candidates") or {}).get(key) or {}
            if cand.get("state") in (success, *diversity._DEAD_STATES):
                continue
            v = diversity.value_of(cand.get("fields") or {}, keys, urls=urls)
            if v in spent:
                dead.append({"candidate": key, "value": v, "dimension": dim})
    return rows, dead
