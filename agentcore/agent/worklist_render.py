"""Render the current candidate as a short, situated instruction.

This is the half of the worklist design that the model actually sees, and its whole purpose is
size and specificity. A long block of standing guidance makes the model re-derive its situation
every cycle, and differently each time.

What is rendered instead is: which candidate, which step, what specifically is required to
finish the step, and how to give up on it. A few dozen words, all of them about right now.

Standing guidance does not disappear -- it moves to the step it belongs to. The "seven ways a
test can fail" warning is genuinely important at FAILS_AT_PARENT and is noise in the other 400
turns, so it lives in that state's `instruction` and is shown only there.
"""

from __future__ import annotations

from agentcore.agent.worklist import WORKLIST_KEY, Worklist
from agentcore.tools.parsing import clip

# Effectively "show everything a candidate has recorded". This is a safety valve against a
# pathological candidate, not a size control: every value is summarised to at most ~80 characters
# before it gets here, so the whole block is bounded regardless.
#
# A small cap would truncate fields the prompt depends on:
#   * keeping the OLDEST fields could drop `triggering_test`, so the step whose instruction says
#     "run the SAME test" would never name it;
#   * keeping the NEWEST fields could drop `sandbox`, and the "this candidate's machine already
#     exists" instruction is conditional on that field.
# A field the model recorded and then cannot see is worse than a longer prompt.
KNOWN_FIELD_CAP = 40


def _live_sandboxes(working_set):
    """Names of the sandboxes that exist right now, from the working set's recipe book
    (containers.py writes it and says which recipes are live). None when there is no recipe
    book at all, so a caller can tell "none live" from "unknown"."""
    from agentcore.tools import containers

    book = (
        working_set.get(containers.RECIPE_KEY)
        if isinstance(working_set, dict)
        else getattr(working_set, containers.RECIPE_KEY, None)
    )
    if not isinstance(book, dict):
        return None
    return set(containers.live_sandboxes(book))


def _known(extra: dict, wspec, idx: int, cap: int = KNOWN_FIELD_CAP) -> dict:
    """What to show under `already recorded:` -- everything, in practice.

    Order still matters for the rare truncation: the fields that say WHERE the candidate lives
    first, then whatever the previous step required, then the rest.
    """
    # ALWAYS keep the fields that say WHERE the candidate lives: the "this candidate's machine
    # already exists" instruction is conditional on `sandbox`, so truncating it would silently
    # stop that instruction rendering. These three are never optional.
    always = {k: extra[k] for k in ("sandbox", "clone_path", "source_path") if k in extra}
    pinned = {}
    if 0 < idx <= len(wspec.states):
        prev = wspec.states[idx - 1]
        pinned = {k: extra[k] for k in (prev.requires or []) if k in extra}
    room = max(cap - len(always) - len(pinned), 2)
    recent = {k: v for k, v in list(extra.items())[-room:]}
    return {**always, **pinned, **recent}


def render_worklist(spec, working_set, target=None) -> dict:
    """Template vars for the current candidate. Empty dict when the worklist is off/empty.

    `target` is the number of instances the task asks for. It belongs in the situated block --
    the one thing rendered on EVERY mining turn -- because "am I done?" is otherwise a judgement
    the model has no basis for when the prompt only says "1 delivered" and never names a target.
    """
    wspec = getattr(spec, "worklist", None)
    if wspec is None or not wspec.enabled:
        return {}
    # The registration FORMAT, from the spec itself, so the field names are known up front
    # rather than learned by being refused.
    first_req = list((wspec.states[0].requires or []) if wspec.states else [])
    reg_fields = list(dict.fromkeys([*(wspec.key or []), *first_req]))
    # THE ENTRY STATE'S OWN WORDS. This renderer shows the instruction of the state a
    # candidate is going to NEXT, and `worklist_add` puts a new candidate at states[0] the
    # instant it succeeds -- so states[0] is never any candidate's next state and its
    # `instruction:` would otherwise never render. What is in it is not decoration -- e.g.
    # `repo_url (the clone URL as printed)`, `issue_url (a full http(s) link, not '#123')` --
    # each is a rule the entry-state validator refuses for. It belongs in the refill branch,
    # next to the field names that branch derives from the same state's `requires`.
    entry_instruction = (wspec.states[0].instruction or "").strip() if wspec.states else ""
    store = getattr(working_set, WORKLIST_KEY, None)
    if not isinstance(store, dict):
        return {
            "worklist_active": False,
            "worklist_needs_refill": True,
            "worklist_target": target,
            "worklist_register_fields": reg_fields,
            "worklist_entry_instruction": entry_instruction,
            "worklist_empty": True,
            "worklist_histogram": {},
            "worklist_delivered": 0,
        }
    wl = Worklist(wspec, store)
    hist = wl.histogram()
    cur = wl.current()
    if cur is None:
        # NO CURRENT CANDIDATE. What matters to the prompt is whether there is work in hand,
        # not whether rows have ever existed; otherwise, once the first candidate went terminal,
        # neither a candidate nor a refill instruction would render.
        return {
            "worklist_active": False,
            "worklist_register_fields": reg_fields,
            "worklist_needs_refill": True,
            "worklist_entry_instruction": entry_instruction,
            "worklist_empty": not hist,
            "worklist_target": target,
            "worklist_histogram": hist,
            "worklist_delivered": len(wl.delivered()),
            "worklist_progress": progress_line(len(wl.delivered()), target, len(wl.active()), _n_rejected(wl)),
            "worklist_banked": _banked(wl),
            "worklist_sources": _source_quota(wl, wspec),
            "worklist_rejection_pattern": _rejection_pattern(wl),
        }

    key, cand = cur
    nxt = wl.next_state_after(cand["state"])
    # A state the CURRENT spec does not declare (a lifecycle edit between snapshot and
    # resume) must render, not raise: the loop wraps this render in a keep-alive except, so
    # raising here would silently delete the worklist section from the prompt. Step 0, and
    # the state value itself carries the note.
    names = [s.name for s in wspec.states]
    shown_state = cand["state"]
    if cand["state"] in names:
        idx = names.index(cand["state"]) + 1
    else:
        idx = 0
        shown_state = (
            f"{cand['state']} (NOT in the current lifecycle -- the spec was "
            f"edited since this candidate was recorded; re-record its fields "
            f"to re-key it, or worklist_drop it)"
        )
    missing = wl.unmet(key)  # `_absent` semantics, matching the state machine

    # Only identity + a few short scalars: dumping the whole record would reintroduce the bulk
    # this exists to remove.
    ident = {k: cand["fields"].get(k) for k in wspec.key}
    # SUMMARISE a long or non-scalar value; do not drop the field. Dropping it means the model
    # cannot tell whether it recorded the thing at all -- e.g. a field of three 64-char hashes
    # would vanish at the step whose instruction refers to it.
    extra = {}
    for k, v in cand["fields"].items():
        if k in ident or k.startswith("_"):
            continue
        if isinstance(v, (list, tuple, set)):
            extra[k] = f"<recorded, {len(v)} item{'' if len(v) == 1 else 's'}>"
        elif isinstance(v, dict):
            extra[k] = f"<recorded, {len(v)} field{'' if len(v) == 1 else 's'}>"
        elif isinstance(v, (str, int, float, bool)):
            t = str(v)
            extra[k] = t if len(t) <= 80 else t[:60] + f"...<{len(t)} chars>"
        else:
            extra[k] = "<recorded>"

    return {
        "worklist_active": True,
        "worklist_needs_refill": False,
        "worklist_target": target,
        "worklist_empty": False,
        "worklist_key": key,
        "worklist_identity": ident,
        # The NEWEST fields, plus whatever the previous step required, always.
        #
        # A tight cap, oldest first, would drop exactly the fields the current step is about:
        # an instruction saying "run the SAME test" is useless if `triggering_test` was
        # truncated away, and the model would invent a target.
        "worklist_known": _known(extra, wspec, idx),
        # The sandboxes that EXIST right now, so the prompt can tell "this candidate's
        # machine already exists" from "a name was recorded but no machine was ever made".
        # None when the working set carries no recipe book at all (tests, non-sandbox tasks).
        "worklist_live_sandboxes": _live_sandboxes(working_set),
        "worklist_state": shown_state,
        "worklist_step": idx,
        "worklist_total_steps": len(wspec.states),
        "worklist_next_state": nxt.name if nxt else None,
        "worklist_instruction": (nxt.instruction if nxt else "") or "",
        "worklist_needs": missing,
        # heavy, step-specific context (e.g. the admissibility criteria) surfaced only here
        "worklist_context": list(nxt.context) if nxt else [],
        "worklist_histogram": hist,
        "worklist_remaining": len(wl.active()),
        "worklist_delivered": len(wl.delivered()),
        "worklist_progress": progress_line(len(wl.delivered()), target, len(wl.active()), _n_rejected(wl)),
        "worklist_banked": _banked(wl),
        "worklist_sources": _source_quota(wl, wspec),
        "worklist_rejection_pattern": _rejection_pattern(wl),
        # WHY THIS CANDIDATE IS SITTING HERE. Without it the situated block says what to
        # record and never what was wrong with the last attempt -- see `_last_refusal`.
        "worklist_last_refusal": _last_refusal(cand, wspec),
        # A 2-3 line repeat of the operative step, rendered at the END of the prompt. The
        # action log is most of a long prompt and renders after the situated block, so the last
        # thing the model reads before generating would otherwise be the tail of a log dump.
        "worklist_now": _now_line(key, cand["state"], nxt, missing),
    }


def _source_quota(wl, wspec) -> str:
    """Per-source capacity against the cap: where mining is open, where it is spent.

    The cap refuses the N+1th instance from one source, but a refusal is discovered one wall
    at a time unless the prompt says that source is spent. This line is the map, so "find a
    DIFFERENT source" is actionable before the refusal rather than after it.

    IT MUST COUNT WHAT THE GATE COUNTS: `diversity.check_caps` counts delivered PLUS in-flight
    (the reservation), so a line counting DELIVERED only would show `2/3` while the gate is
    full. The counting happens in `diversity._counts`, once, for both.

    The in-flight figure is spelled out rather than folded into the total: `2/3 FULL` on a
    source with two deliveries reads as a harness bug, and the model cannot work out that
    dropping its own registrations reopens the source unless it can see them.
    """
    from agentcore.agent import diversity

    for cap in getattr(wspec, "caps", None) or []:
        rc = diversity.resolve_cap(cap)
        if rc is None or rc.limit <= 0:
            continue
        keys, limit, urls, _dim = rc
        success = diversity.success_from_spec(wl.spec)
        held = diversity._counts(wl.store, keys, urls, success, reserving=True)
        done = diversity._counts(wl.store, keys, urls, success)
        if not held:
            return ""
        parts = []
        for src, n in sorted(held.items(), key=lambda kv: -kv[1]):
            flight = n - done.get(src, 0)
            seg = f"{src} {n}/{limit}"
            if n >= limit:
                seg += " FULL"
            if flight:
                seg += f" ({done.get(src, 0)} delivered + {flight} in flight)"
            parts.append(seg)
        return " | ".join(parts)
    return ""


def exclusions_line(instance: dict) -> str:
    """The spec's forbidden sources, one standing line for the mining prompt.

    A roster that lives only in the source-requirements prose is read once and forgotten, and
    the most famous candidates are exactly the ones a spec tends to forbid, each costing a
    container and a clone before the gate refuses it. A standing line next to the progress bar
    is where the model actually looks.
    """
    try:
        from agentcore.agent.exclusions import excluded_names

        names = excluded_names(instance or {})
    except Exception:  # noqa: BLE001 - a parser hiccup must not break rendering
        return ""
    if not names:
        return ""
    return "NEVER mine these (the spec excludes them; the gate WILL refuse them): " + ", ".join(sorted(names)[:24])


# Served refusals are mostly well under this length. A cut keeps head AND tail, because the
# instruction is the last sentence ("...then re-record `fix_patch_path`") and a head-only cut
# would remove exactly that.
# `worklist.REFUSAL_DETAIL_CHARS` bounds what a gate may SAY at 12,000; this bounds what is
# re-rendered every cycle the candidate stays parked, which is a different and much longer bill.
LAST_REFUSAL_CHARS = 2400


def _last_refusal(cand: dict, wspec) -> dict | None:
    """The check that is holding this candidate, in the harness's own words.

    `_recompute` walks the states in order and BREAKS at the first one whose check refuses, so
    the state holding a candidate is the first falsy entry in `_checked` -- which is not always
    the state whose fields the block below is asking for. When the FIRST state refuses, the
    candidate's own `state` stays at states[0] (the default `reached`), the render shows
    states[1]'s instruction, and nothing recorded against it can move the candidate. A candidate
    can then be held by an earlier gate while being told to record fields for a later step, and
    re-recording the held gate's own input is the only move that can work.

    Returns `{"state": ..., "detail": ...}` or None -- None is the common case and must stay
    cheap, or this becomes decoration rather than a signal.
    """
    checked = cand.get("_checked") or {}
    for st in wspec.states:
        entry = checked.get(st.name)
        if entry and not entry[0]:
            detail = str(entry[1] if len(entry) > 1 else "")
            return {"state": st.name, "detail": clip(detail, LAST_REFUSAL_CHARS)}
    return None


def _now_line(key: str, state: str, nxt, missing) -> str:
    """The operative facts, short enough to be the prompt's last line and nothing more."""
    if nxt is None:
        return f"Candidate in hand: {key} at {state} (final state) -- deliver or move on."
    instr = (nxt.instruction or "").strip()
    # A staged instruction opens with its title line ("STAGE 2 of 4: build ..."); that line is
    # the summary, not the first sentence of its first paragraph.
    first = (
        instr.splitlines()[0].strip().rstrip(".")
        if instr.startswith("STAGE ")
        else instr.split(". ")[0].strip().rstrip(".")
    )
    need = f" Still needs: {', '.join(missing)}." if missing else ""
    step = f" {first}." if first else ""
    return f"Candidate in hand: {key}, at {state}; next step is {nxt.name}.{step}{need}"


def _lookup(vars: dict, name: str):
    """Resolve a context var name, allowing dotted paths.

    Task-level material lives under `instance` (e.g. `instance.criteria_list`), so a bare-name
    lookup would silently find nothing and the step would render without its criteria.
    """
    cur = vars
    for part in name.split("."):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return cur


# Headings for the material a step pulls in. The variable name is not a heading: the model has
# never seen `instance.criteria_list` and cannot tell whether it is a rule, a hint or a label.
_CONTEXT_HEADINGS = {
    "instance.criteria_list": "A CANDIDATE ONLY COUNTS IF ALL OF THIS IS TRUE (you need this for the step above):",
    "instance.quality_gates_list": "CHECKS A REVIEWER WILL APPLY TO WHAT YOU DELIVER:",
    "instance.examples": "A KNOWN-GOOD EXAMPLE, FOR COMPARISON:",
}


def context_body(val) -> str:
    """Render a step's declared context material for a READER.

    A list of strings is a bulleted list. A list of DICTS is not printed as a Python repr:
    braces, quotes and key punctuation spend words on nothing, and a dict repr that reached a
    shell would be wrong. Every key and value still appears.
    """
    if isinstance(val, (list, tuple)):
        return "\n".join(_context_item(v) for v in val)
    if isinstance(val, dict):
        return _context_item(val)
    return str(val)


def _context_item(v) -> str:
    if not isinstance(v, dict):
        return f"  - {v}"
    # the item's own name leads, when it has one; everything else follows as key: value
    label = next((str(v[k]) for k in ("name", "id", "title") if v.get(k)), "")
    rest = [f"{k}: {x}" for k, x in v.items() if k not in ("name", "id", "title")]
    head = f"  - {label}" if label else "  -"
    return head + ("\n" + "\n".join(f"      {r}" for r in rest) if rest else "")


def _context_heading(name: str) -> str:
    return _CONTEXT_HEADINGS.get(name, name.rsplit(".", 1)[-1].replace("_", " ").upper() + ":")


def render_text(vars: dict, extra: dict | None = None) -> str:
    """The plain-text block for a cycle prompt (also what the tests assert on).

    `extra` supplies the values for the step's declared `context` vars, so heavy material is
    rendered only at the step that needs it rather than in every prompt.
    """
    if not vars.get("worklist_active"):
        hist = vars.get("worklist_histogram") or {}
        done = vars.get("worklist_delivered", 0)
        tgt = vars.get("worklist_target")
        prog = f"{done} delivered" + (f" of {tgt} wanted" if tgt else "")
        stuck = hist.get("BLOCKED", 0) + hist.get("REJECTED", 0)
        fmt = ""
        reg = vars.get("worklist_register_fields") or []
        if reg:
            body = ", ".join(f'"{f}": "<{f}>"' for f in reg)
            fmt = f" Each candidate is a dict with exactly these fields: [{{{body}}}, ...]."
        # WHAT EACH OF THOSE FIELDS MUST LOOK LIKE. The state that DEFINES the fields is the one
        # state whose instruction this renderer never shows; this is the only situation it can
        # be shown in, and the right one: the model is about to register.
        entry = str(vars.get("worklist_entry_instruction") or "").strip()
        if entry:
            fmt += f"\n\nWHAT A CANDIDATE MUST BE, AND WHAT EACH FIELD MUST LOOK LIKE:\n{entry}"
        # A WORKER DOES NOT SEARCH. It is handed candidates by the search step and
        # builds them; "find real candidates" is the search step's instruction. A worker is the agent
        # that delivers through bench_deliver and records nothing.
        granted = set((extra or {}).get("tool_names") or [])
        worker = "bench_deliver" in granted and "worklist_record" not in granted
        if worker:
            if not hist:
                head = f"No candidate in hand ({prog})."
                nxt = "The system hands you the next candidate from the pool; nothing to do until then."
            elif stuck and stuck >= sum(hist.values()) - done:
                head = (
                    f"Every candidate you registered has finished: {hist} ({prog}). "
                    "BLOCKED and REJECTED are final: those candidates do not come back, and "
                    "re-adding one is refused as a duplicate."
                )
                nxt = "The system hands you the next candidate from the pool, or ends the run when it is used up."
            else:
                head = f"No candidate in hand. So far: {hist} ({prog})."
                nxt = "The system hands you the next candidate from the pool."
            if tgt and done >= tgt:
                nxt = f"You have {done} of the {tgt} instances asked for; the run ends here."
            return f"{head}\n{nxt}"
        if not hist:
            head = f"No candidates registered yet ({prog})."
            nxt = (
                "NEXT: find real candidates and register them with worklist_add. "
                "Duplicates are refused automatically, so add in bulk." + fmt
            )
        elif stuck and stuck >= sum(hist.values()) - done:
            # The prompt must say these candidates are gone for good, or the model keeps
            # circling them (hand-searching, or building a container for a candidate that does
            # not exist).
            head = (
                f"Every candidate you registered has finished: {hist} ({prog}). "
                "BLOCKED and REJECTED are FINAL -- those candidates never come back, cannot "
                "be retried, and re-adding the same one is refused as a duplicate."
            )
            nxt = (
                "NEXT: register DIFFERENT candidates with worklist_add -- a different project, "
                "or a different commit in one you have not tried. Do not re-add the ones "
                "above. Do not create a sandbox yet: sandboxes are made for a candidate that "
                "exists, at the step that asks for one." + fmt
            )
        else:
            head = f"No candidates left to work on. So far: {hist} ({prog})."
            nxt = (
                "NEXT: find more real candidates and register them with worklist_add. "
                "Duplicates are refused automatically, so add in bulk." + fmt
            )
        if tgt and done >= tgt:
            nxt = (
                f"You have {done} of the {tgt} instances asked for. Finish the stage, or keep "
                f"mining if you think more are worth it."
            )
        return f"{head}\n{nxt}"
    ident = ", ".join(f"{k}={v}" for k, v in (vars.get("worklist_identity") or {}).items())
    done, tgt = vars.get("worklist_delivered", 0), vars.get("worklist_target")
    progress = f"{done} finished" + (f" out of the {tgt} wanted" if tgt else "")
    # A WRONG IDENTITY FIELD IS CORRECTABLE: dropping a candidate (clone, build, test run)
    # to fix one string would be wasteful, though the field is part of the identity key.
    # `update()` calls `_rekey` whenever a key field is recorded, so re-recording the corrected
    # value moves the row and keeps everything it has proven; `worklist_tools` even carries a
    # comment about "the model corrected an identity field". The
    # clause rides on the identity line that already renders, because everything added here is
    # charged to every collector on every cycle.
    lines = [
        "THE CANDIDATE YOU ARE WORKING ON NOW",
        f"  it is:            {ident}",
        "                    (if one of those is wrong, re-record it -- the candidate keeps everything it has proven)",
        f"  it has passed:    {vars['worklist_step'] - 1} of {vars['worklist_total_steps']} steps"
        f"  (currently at: {vars['worklist_state']})",
        f"  overall:          {progress}; {vars['worklist_remaining']} "
        f"candidate{'' if vars['worklist_remaining'] == 1 else 's'} still in the list",
    ]
    if vars.get("worklist_known"):
        # A WORKER'S EXTRA FIELDS ARE THE SEARCH'S NOTES: under "already recorded" a worker
        # could take noted details for the instance itself instead of checking them.
        granted = set((extra or {}).get("tool_names") or [])
        label = (
            "  search notes:     (leads to start from and CHECK, not facts; the candidate is the repository) "
            if "bench_deliver" in granted and "worklist_record" not in granted
            else "  already recorded: "
        )
        lines.append(label + ", ".join(f"{k}={v}" for k, v in vars["worklist_known"].items()))
    if vars.get("worklist_next_state"):
        lines += [
            "",
            "DO THIS NOW",
            f"  {vars.get('worklist_instruction') or 'gather the fields listed below.'}",
        ]
        # THE REFUSAL THAT IS HOLDING THIS CANDIDATE, EVERY CYCLE IT HOLDS IT. Otherwise it
        # reaches the model only as a tool result inside the action log, which has a recency
        # window; once it scrolls out the agent is told what to record and not what was wrong.
        ref = vars.get("worklist_last_refusal") or {}
        if ref.get("detail"):
            held_here = ref.get("state") == vars.get("worklist_next_state")
            lines += [
                "",
                f"  THE LAST ATTEMPT AT {ref['state']} WAS REFUSED. What the harness said, verbatim:",
                "",
                f"    {ref['detail']}",
            ]
            if not held_here:
                # NAMING THE RIGHT STEP. `_recompute` breaks at the first refusing state, so
                # this candidate cannot reach the fields the envelope below asks for, though
                # the envelope's closing line promises a move to the next step if they pass.
                # Saying so points the agent at the gate that is actually closed.
                lines += [
                    "",
                    f"  {ref['state']} is EARLIER than the step below, so this candidate cannot advance",
                    "  until that check passes -- whatever you record below. Fix what the refusal names,",
                    f"  then re-record one of {ref['state']}'s own inputs (the same value is fine) to make",
                    "  the check run again.",
                ]
        # If this candidate already has a machine, SAY SO as an instruction, not as a value in a
        # list of recorded fields. A recorded
        # `sandbox=...` in the "already recorded" line is easily missed, and the model may
        # create a second sandbox or check liveness, which container_exec already handles by
        # rebuilding a dead sandbox from its recipe.
        sb = (vars.get("worklist_known") or {}).get("sandbox")
        live = vars.get("worklist_live_sandboxes")
        if sb and live is not None and sb not in live:
            # A RECORDED NAME IS NOT A MACHINE. `worklist_add` requires `sandbox` while telling
            # the model "Do not create a sandbox yet", so the field holds a name the model
            # chose; asserting the machine exists would then be wrong.
            lines += [
                "",
                f"  sandbox={sb} is RECORDED but no such sandbox exists yet. Create it under exactly that name first:",
                "",
                "  {",
                '    "thoughts": "<brief reasoning>",',
                f'    "command": {{"name": "container_create", "args": {{"name": "{sb}", "image": "...", ...}}}}',
                "  }",
                "",
                "  then run this candidate's commands there with container_exec.",
            ]
        elif sb:
            lines += [
                "",
                "  This candidate's machine already exists. Run its commands there:",
                "",
                "  {",
                '    "thoughts": "<brief reasoning>",',
                f'    "command": {{"name": "container_exec", "args": {{"name": "{sb}", "command": "..."}}}}',
                "  }",
                "",
                "  Do not create it again, and do not check whether it is alive -- if it died it",
                "  is rebuilt from its recipe automatically on your next command.",
            ]
        if vars.get("worklist_needs"):
            # The LITERAL reply, not `worklist_record(fields={...})` pseudo-code: a model that has
            # to derive the JSON envelope itself can get it wrong once payloads get long.
            # A KEY IS A NAME; THE ANNOTATION GOES BESIDE IT. `unmet()` returns strings that do
            # two jobs: as prose they carry the reason ("timings_ms (raw observations -- the
            # harness computes median_ms and cv from it; do not compute them yourself)"). As
            # JSON keys they are wrong, and this envelope is rendered as the LITERAL reply to
            # send, so a model that copies it gets `worklist_record`'s "no step reads [...]"
            # back. Split at the point of rendering; `unmet()` is unchanged, because
            # the "still needs" message and the NOW line both want the prose.
            needs = list(vars["worklist_needs"])
            fields = ",\n".join(f'          "{n.split(" (", 1)[0]}": "<what you saw>"' for n in needs)
            notes = [n for n in needs if " (" in n]
            granted = set((extra or {}).get("tool_names") or [])
            staged = "bench_deliver" in granted and "worklist_record" not in granted
            if staged:
                # A STAGED collector: the step's token is recorded by bench_deliver after a clean
                # replay, never typed, and worklist_record is not granted to it.
                lines += [
                    "",
                    "  When this stage's files are in bench/ on your machine, check them:",
                    "",
                    "  {",
                    '    "thoughts": "<brief reasoning>",',
                    '    "command": {"name": "bench_check", "args": {}}',
                    "  }",
                    "",
                    "  Fix what the report names and check again. When every check of the stage is ok:",
                    "",
                    "  {",
                    '    "thoughts": "<brief reasoning>",',
                    '    "command": {"name": "bench_deliver", "args": {}}',
                    "  }",
                    "",
                    f"  On a clean report this candidate moves to step {vars['worklist_step'] + 1} "
                    f"({vars['worklist_next_state']}) and you will be shown what to do there.",
                ]
            else:
                lines += [
                    "",
                    "  Then send exactly this, filled in with what the command actually printed:",
                    "",
                    "  {",
                    '    "thoughts": "<brief reasoning>",',
                    '    "command": {',
                    '      "name": "worklist_record",',
                    '      "args": {',
                    '        "fields": {',
                    fields,
                    "        }",
                    "      }",
                    "    }",
                    "  }",
                    "",
                    f"  If the check on those fields passes, this candidate moves to step "
                    f"{vars['worklist_step'] + 1} ({vars['worklist_next_state']}) and you will be shown",
                    "  what to do there. If it fails, you will be told what was wrong and can retry.",
                ]
            for note in notes:
                lines += ["", f"  {note}"]
    else:
        lines += ["", "DO THIS NOW", "  Nothing -- this candidate has passed every step and is finished."]
    lines += [
        "",
        "  If this candidate cannot pass this step no matter what you try, give up on it by",
        "  sending this instead, and you will be given a different one. Dropping a bad candidate",
        "  is a good outcome.",
        "",
        "  {",
        '    "thoughts": "<brief reasoning>",',
        '    "command": {"name": "worklist_drop", "args": {"reason": "<why>", "blocked": true}}',
        "  }",
    ]
    for name in vars.get("worklist_context") or []:
        val = _lookup(extra or {}, name)
        if val:
            lines.append(f"\n{_context_heading(name)}\n{context_body(val)}")
    # THE IDENTITY, LAST AS WELL AS FIRST. The block above opens the prompt and is followed by
    # the stage text, the refusal history, the envelope and the context sections -- thousands
    # of characters -- so the identity is far behind by the time the answer is composed. A
    # model can then claim the candidate data is missing; one line where it is looking is the
    # cure.
    if ident:
        lines += [
            "",
            f"(you are working on {ident} -- it is above and it is here; "
            f"a candidate is never missing from this prompt)",
        ]
    return "\n".join(lines)


def _n_rejected(wl) -> int:
    store = getattr(wl, "store", {}) or {}
    return sum(1 for c in (store.get("candidates") or {}).values() if c.get("state") in ("BLOCKED", "REJECTED"))


def _rejection_pattern(wl, floor: int = 3) -> str:
    """The reason candidates keep failing, when one reason dominates.

    Refusals arrive one candidate at a time, and a pattern (the same rejection sentence again
    and again) only exists across them. A run that keeps sampling the same kind of source keeps
    getting the same refusal without noticing.

    Naming the pattern is not naming a source: the model still chooses where to look. It just
    gets to see that its current strategy has a systematic failure rather than bad luck.

    AND WHEN THE PROSE CANNOT SEE IT, THE GATE CAN. Grouping on `_reason` works for
    harness-authored deaths, whose sentences are byte-identical, and fails for agent-authored
    `worklist_drop` prose about the SAME gate, where several spellings of one refusal count as
    unrelated causes. Grouping the deaths by the STATE whose gate last refused them catches
    those; `floor` keeps small counts silent. The harness recorded which check refused each
    candidate, so it never has to guess.

    Prose stays first: when it speaks it quotes an actual sentence, which is more actionable
    than a state name. The gate grouping only speaks where there would otherwise be silence.
    """
    import collections
    import re

    store = getattr(wl, "store", {}) or {}
    dead = [c for c in (store.get("candidates") or {}).values() if c.get("state") in ("BLOCKED", "REJECTED")]
    reasons = []
    for c in dead:
        why = str((c.get("fields") or {}).get("_reason") or "")
        if why:
            # collapse to a shape: drop identifiers so "PR #5 does not close" and "PR #70 does
            # not close" count as the same pattern.
            reasons.append(re.sub(r"[#$]?\b[\w.\-/]*\d[\w.\-/]*\b", "N", why.lower())[:70])
    if len(reasons) >= floor:
        top, n = collections.Counter(reasons).most_common(1)[0]
        if n >= floor and n >= len(reasons) / 2:
            return (
                f"{n} of your {len(reasons)} rejections share one cause: "
                f'"{top.strip()}...". That is a property of where you are looking, not '
                f"of the individual candidates -- a different kind of source will do more "
                f"than another candidate from this one."
            )

    # `getattr`, because the whole of `render_worklist` sits inside a bare `except: pass` in
    # loop.py -- an AttributeError here does not surface as an error, it silently deletes the
    # entire situated block from the prompt.
    wspec = getattr(wl, "spec", None)
    # THE DENOMINATOR BELOW IS EVERY DEATH, not every death a gate touched. Filtering the
    # gate-less ones out of the denominator as well as the numerator would make "3 of 8 at one
    # check" report itself as "3 of your 3" -- a majority manufactured by dropping the
    # disagreeing half.
    gates = [g for g in (_refusing_state(c, wspec) for c in dead) if g]
    if len(gates) < floor:
        return ""
    gate, n = collections.Counter(gates).most_common(1)[0]
    if n < floor or n < len(dead) / 2:
        return ""
    return (
        f"{n} of your {len(dead)} dead candidates were last refused by the SAME check: "
        f"{gate}. Unrelated candidates failing one check is evidence about that check, or "
        f"about what it is asking for, and not about the candidates -- read its refusal "
        f"above and fix what it names before spending another candidate on it."
    )


def _refusing_state(cand: dict, wspec) -> str:
    """Which state's gate last refused this candidate, in spec order.

    `_checked` is the verdict cache and holds the answer directly; `_strikes` is the fallback
    for a candidate whose cache was dropped by `recheck` but whose strike count survives.
    """
    checked = cand.get("_checked") or {}
    strikes = cand.get("_strikes") or {}
    for st in getattr(wspec, "states", None) or []:
        entry = checked.get(st.name)
        if entry and not entry[0]:
            return st.name
        if st.name in strikes:
            return st.name
    return ""


def _banked(wl, cap: int = 25) -> str:
    """The instances already banked, by name. Without this the model has a COUNT and no list.

    It cannot tell whether the five it delivered came from five projects or one, cannot avoid
    re-mining a commit it already has, and cannot see that all its instances come from one
    project family.
    """
    from agentcore.agent import diversity

    store = getattr(wl, "store", {}) or {}
    # The task's own success state, not the literal "DELIVERED": the terminal name is per-task
    # configuration, and a collector that renames it would render "(none yet)" forever.
    success = diversity.success_from_spec(getattr(wl, "spec", None))
    done = [k for k, c in (store.get("candidates") or {}).items() if c.get("state") == success]
    if not done:
        return "(none yet)"
    shown = [f"  {i + 1}. {k}" for i, k in enumerate(done[:cap])]
    if len(done) > cap:
        shown.append(f"  ... and {len(done) - cap} more")
    return "\n".join(shown)


def progress_line(delivered: int, target: int, in_flight: int, rejected: int) -> str:
    """A bar the model can read at a glance, plus the counts behind it.

    A target stated once, long ago, is no substitute for a running count of what has actually
    been banked.
    """
    target = max(int(target or 0), 0)
    done = max(int(delivered or 0), 0)
    if not target:
        return f"delivered {done}"
    width = 20
    filled = min(width, round(width * done / target)) if target else 0
    bar = "#" * filled + "." * (width - filled)
    return (
        f"[{bar}] {done}/{target} delivered"
        + (f"  |  {in_flight} still in flight" if in_flight else "")
        + (f"  |  {rejected} rejected" if rejected else "")
    )
