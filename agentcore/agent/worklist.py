"""Worklist — per-candidate state, owned by the harness rather than the model.

Why this exists. The collector stages tell an agent *what phase the run is in*
(`mine_at_scale`); nothing tells it *which candidate it is working on and what specifically
comes next*. Without that, every turn the model re-derives its situation from a long prompt,
and re-derives it differently, switching between projects and rarely changing any state.

The fix is not a better prompt. It is to stop asking the model to remember where it is.

THE CENTRAL RULE: **the model never names a state.** It supplies evidence fields for the
candidate it is working on; the harness recomputes that candidate's state from the evidence
present. A state whose `requires` are unmet simply cannot be reached, so steps cannot be skipped,
claimed, or talked past -- the same reason `calibrate`'s exit is a side-effect check rather than
a sentinel.

Three properties follow for free:

  * duplicates are impossible -- the worklist is keyed on the candidate's identity fields;
  * an incomplete instance cannot reach a later state -- `requires` is a precondition, not a
    post-hoc audit;
  * where a run dies becomes a histogram over states rather than "the run underperformed".

Lifecycles are declared per task in `agent.yaml`. Different tasks have different shapes (a
build, a determinism check, a timing step, a search state); hardcoding one lifecycle here would
make this a single-task tool rather than a benchmark compiler.

The worklist lives in the WorkingSet, so it is snapshotted every cycle and survives resume --
the same reason sandbox recipes live there (see agentcore/tools/containers.py).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, NamedTuple

from agentcore.agent import derive, gates
from agentcore.agent.field_contract import fields_read as _contract_fields_read

WORKLIST_KEY = "worklist"


class Nudge(NamedTuple):
    """What `Worklist.tick` says to the model, whether it is a repeat, and which registered
    site said it (agentcore/observability/sites.py)."""

    msg: str
    repeat: bool
    site: str


# Reserved states every lifecycle has, regardless of task. Terminal states hold candidates that
# are finished with -- successfully or not -- so they are never handed to the model again.
REJECTED = "REJECTED"
BLOCKED = "BLOCKED"

# How many times one candidate may be refused AT THE SAME STATE before the harness intervenes.
# The per-candidate refusal counts at a single state can be very large (one candidate can
# consume over half a run). See `_strike` and tests/test_stuck_candidates.py.
#
# The two thresholds are deliberately far apart. Deferring is cheap and reversible -- the
# candidate keeps every field it has proven and simply yields to others -- so it fires early.
# Blocking is terminal, so it needs enough attempts to be confident the candidate really cannot
# proceed rather than that the model was briefly confused. A validator refusing eight times on
# eight genuinely different pieces of evidence is not a communication problem.
#
# The numbers live in agentcore/agent/gates.py, which the spec schema reads for its defaults
# too; `_knob` prefers the spec's value. Re-exported here for existing readers.
STUCK_DEFER = gates.STUCK_DEFER
STUCK_BLOCK = gates.STUCK_BLOCK


def _refusal_shape(detail: str) -> str:
    """The shape of a refusal, not its instance-specific detail: the first PROBLEM sentence
    with paths, ids and digests stripped. Shared by the systemic-shape detector and the
    advisor's staleness key -- both need 'same refusal or a different one?'."""
    text = str(detail or "")
    marker = text.find("[1] ")
    head = (text[marker + 4 :] if marker >= 0 else text).split(". ")[0]
    shape = re.sub(r"[/\w.-]*/[/\w.-]+|\b[0-9a-f]{7,}\b|\d+", "*", head)
    # ONE GATE, ONE SHAPE. A refusal that ENUMERATES what it found -- the no-op probe lists
    # every replaced executable -- would strip to `(*)`, `(*, *, *, *, *)` and so on: several
    # fingerprints for one gate, so the cross-worker pool would split and SYSTEMIC_AT would
    # be off by that factor. A run of stripped items is one item.
    shape = re.sub(r"\*(?:\s*[,;/]\s*\*)+", "*", shape)
    return shape[:160]


# How many CYCLES one candidate may occupy at a single state before the harness says something.
# `_strike` above counts REFUSALS, so it only fires when the model submits evidence and is told
# no. A candidate can consume a whole run without ever being refused: a long build that
# fails, is rebuilt, fails again, holds the current slot for hours -- and the strike counter
# never sees it, because the candidate was never wrong, only slow. A prompt asks the model to
# notice; the harness notices instead.
#
# Deferral, never blocking. A slow candidate is not a bad one -- a real Maven build legitimately
# takes hours -- so the escalation is the reversible one: it keeps every proven field and yields
# the slot to another candidate, and comes back if nothing else pans out.
SLOW_NUDGE = gates.SLOW_NUDGE
SLOW_DEFER = gates.SLOW_DEFER


# A git hash abbreviated to 7-12 characters and the full 40 are the SAME object -- git resolves
# both -- so identity has to treat one as a prefix of the other, not merely as unequal strings.
# Bounded deliberately: `fastapi` must not swallow `fastapi-users`, and `eo` must not swallow
# `eole`, so the rule applies only to values that look like hex hashes and only above a length
# where an accidental collision is not credible.
_HASHISH = re.compile(r"^[0-9a-f]{7,40}$")


def _norm_key_part(value, *, keep_case: bool = False) -> str:
    """One key field, reduced to what identifies the artifact rather than how it was typed.

    Case, a trailing slash and a `.git` suffix cannot change WHICH thing is meant, so they are
    removed. Nothing else is: this must never merge two artifacts that differ.

    EXCEPT for a filesystem path (`keep_case`): on the case-sensitive
    filesystems the sandboxes run, /w/AES.c and /w/aes.c ARE two files, and the generic
    fold would merge them, violating this docstring's own last sentence.
    """
    text = str(value).strip().rstrip("/")
    if keep_case:
        return text
    if text.lower().endswith(".git"):
        text = text[:-4]
    return text.lower()


class Unnormalizable(ValueError):
    """A key field's value has no canonical form the declared normalizer can compute.

    Raised INSTEAD of guessing: a relative path cannot be made absolute from outside the
    sandbox, and admitting it under a guessed key could build, time and package the
    same file twice. add() turns this into a "malformed" refusal at the door, with the
    message as the shape to use."""


def _key_norm_git_url(field: str, value: str) -> str:
    # The generic treatment already collapses case, trailing slash and .git -- declaring
    # git_url documents the intent and reserves room for more (scheme, host aliases) without
    # shifting banked identities.
    return value


def _key_norm_abs_path(field: str, value: str) -> str:
    if not str(value).startswith("/"):
        raise Unnormalizable(
            f"{field} must be an ABSOLUTE path inside the sandbox (e.g. "
            f"/workspace/zstd/programs/zstdcli.c), got {value!r}. It is part of this "
            f"candidate's identity, and the same file written two ways becomes two "
            f"candidates."
        )
    return value


# The CLOSED vocabulary a spec's `normalize:` may name (see config/schema.py). The spec
# chooses which applies to which field; nothing here keys on a field's NAME, because a rule
# in code cannot be overridden by a spec and would narrow what benchmarks are expressible.
def _key_norm_full_sha(field: str, value: str) -> str:
    # An identity field that is a commit must be the full 40-hex hash: an abbreviation of the
    # same commit would otherwise register as a second candidate, and a depth-1 fetch cannot
    # resolve an abbreviation at all.
    v = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", v):
        raise Unnormalizable(
            f"{field} must be the FULL 40-character commit hash, got {value!r}. An abbreviated "
            f"hash is the same commit spelled differently and cannot be fetched by itself; "
            f"record what `git rev-parse <ref>` prints in a clone."
        )
    return v


KEY_NORMALIZERS = {
    "git_url": _key_norm_git_url,
    "abs_path": _key_norm_abs_path,
    "full_sha": _key_norm_full_sha,
    "case_fold": lambda field, value: value,  # the generic lowercasing already applies
    "none": lambda field, value: value,
}


def _load_known(kf) -> dict[str, str]:
    """{normalised value on `kf.keys` -> the path that registered it}, read from earlier
    runs (agent_state.json -> working_set.worklist.candidates[*].fields) and pools
    (instances.json -> [*].fields). A missing or unreadable path contributes nothing: a
    run that never ran registered nothing, and the door must not refuse to open for it."""
    if not kf:
        return {}
    keys = list(getattr(kf, "keys", None) or [])
    paths = list(getattr(kf, "paths", None) or [])
    if not keys or not paths:
        return {}
    from agentcore.agent.diversity import value_of

    known: dict[str, str] = {}
    for path in paths:
        for fields, ending in _registered_fields(path):
            value = value_of(fields, keys, urls=True)
            if value and value not in known:
                known[value] = path
                # HOW IT ENDED travels with the path: the door says not only "already
                # registered" but the ending, and a sampler that reads "REJECTED: 70 ms flat
                # across inputs" can stop proposing that kind of program.
                _KNOWN_ENDINGS[(id(known), value)] = ending
    return known


def _known_stamp(kf) -> tuple:
    """(path, mtime) of every known_from state file: changed means re-read."""
    out = []
    for path in list(getattr(kf, "paths", None) or []):
        for name in ("agent_state.json", "instances.json"):
            p = os.path.join(path, name)
            try:
                out.append((p, os.stat(p).st_mtime_ns))
            except OSError:
                out.append((p, None))
    return tuple(out)


_KNOWN_ENDINGS: dict = {}


def _ending_of(c: dict) -> str:
    """How a registered candidate ended, for the door's refusal: its state, and for a
    rejected or blocked one the reason recorded with it (the last history line that starts
    with that state, or the last refusal shown), cut to a sentence."""
    state = str(c.get("state") or "?")
    hist = [str(h) for h in (c.get("history") or [])]
    reason = ""
    for h in reversed(hist):
        if h.startswith(f"{state}:"):
            reason = h.split(":", 1)[1].strip()
            break
    if not reason and state not in ("DELIVERED", "DONE"):
        for h in reversed(hist):
            if ":REFUSED:" in h:
                reason = h.split(":REFUSED:", 1)[1].strip()
                break
    reason = " ".join(reason.split())[:240]
    return f"{state}: {reason}" if reason else state


def _registered_fields(path: str) -> list[tuple[dict, str]]:
    """Every registration's (fields, how it ended) under `path`, whichever of the two shapes
    it is. A pool entry ended 'in pool'."""
    out: list[tuple[dict, str]] = []
    try:
        state = os.path.join(path, "agent_state.json")
        if os.path.isfile(state):
            with open(state) as fh:
                d = json.load(fh)
            ws = d.get("working_set") if isinstance(d, dict) else None
            wl = (ws or {}).get("worklist") if isinstance(ws, dict) else None
            for c in ((wl or {}).get("candidates") or {}).values():
                if isinstance(c, dict) and isinstance(c.get("fields"), dict):
                    out.append((c["fields"], _ending_of(c)))
        pool = os.path.join(path, "instances.json")
        if os.path.isfile(pool):
            with open(pool) as fh:
                d = json.load(fh)
            insts = d if isinstance(d, list) else (d.get("instances") if isinstance(d, dict) else [])
            for inst in insts or []:
                if isinstance(inst, dict) and isinstance(inst.get("fields"), dict):
                    out.append((inst["fields"], "in pool"))
    except (OSError, ValueError):
        return out
    return out


def _same_artifact(a: str, b: str) -> bool:
    """Two normalized keys naming one artifact.

    Equal, or equal in every field except one where a hash-shaped value is a prefix of the
    other -- an abbreviated hash and its full form
    are two spellings of the same commit.
    """
    if a == b:
        return True
    pa, pb = a.split("|"), b.split("|")
    if len(pa) != len(pb):
        return False
    differing = 0
    for x, y in zip(pa, pb):
        if x == y:
            continue
        differing += 1
        if differing > 1:
            return False
        if not (_HASHISH.match(x) and _HASHISH.match(y)):
            return False
        if not (x.startswith(y) or y.startswith(x)):
            return False
    return differing == 1


#: How much of what a command PRINTED goes into the refusal about it. Short outputs are
#: carried whole; very long ones are clipped, since the extra size buys little.
REFUSAL_OUTPUT_CHARS = 8000
#: The whole refusal. Must exceed REFUSAL_OUTPUT_CHARS by the gate's own explanation, which
#: runs 865-977 characters on the three real refusals generated to measure it -- a cap below
#: that would discard exactly what the wider output was for.
REFUSAL_DETAIL_CHARS = 12000


class Worklist:
    """A candidate table with harness-computed states.

    Backed by a plain dict living in the WorkingSet (JSON-serializable, snapshot-safe). This
    class is a view over that dict rather than an owner of state, so restoring a snapshot
    restores the worklist with no extra machinery.
    """

    # Set by _exec_rederive when the exec path had to rebuild the candidate's sandbox
    # from its recipe; refusal messages carry it (honest candidates would otherwise be
    # rejected on digests hashed in silently-fresh workspaces).
    _last_exec_note = ""

    def _revival_suffix(self) -> str:
        if not self._last_exec_note:
            return ""
        return (
            f" NOTE: {self._last_exec_note} -- a rebuilt sandbox holds only what its "
            f"recipe's setup recreates. If your clone/builds were done with "
            f"container_exec rather than the sandbox's setup, this check ran against "
            f"a fresh workspace: extend the setup (container_create replays it on "
            f"every rebuild) so the recorded state survives, then re-verify."
        )

    def __init__(self, spec, store: dict, *, validators: dict | None = None, ctx=None, ws=None):
        self.spec = spec
        self.store = store
        # The WorkingSet itself, when the caller has one -- needed to derive calibration state.
        self.ws = ws if ws is not None else getattr(ctx, "working_set", None)
        self.store.setdefault("candidates", {})
        self.store.setdefault("order", [])
        # The task's own success state, written INTO the state so the record is self-describing.
        # agent_state.json carries no spec, so every consumer of a finished run -- the packager
        # above all -- had to guess, and two of them guessed the literal "DELIVERED". A
        # benchmark that names its terminal anything else then packaged zero instances out of a
        # successful run, and its quota gate counted zero deliveries forever.
        if self.spec.terminal:
            self.store["terminal"] = list(self.spec.terminal)
        # Resolved {state_name: callable}. Absent (the default) means presence-only checking,
        # which is all the pure-data tests need; a live run passes real validators so the
        # harness verifies claims instead of trusting them.
        self.validators = validators or {}
        self.ctx = ctx
        # WHAT THIS CAMPAIGN ALREADY REGISTERED: {normalised value on the
        # declared dimension -> the earlier run or pool that registered it}. Empty unless
        # the spec (an `--overrides` overlay at launch) declares `known_from`.
        self.known: dict[str, str] = _load_known(getattr(self.spec, "known_from", None))

    def _known_refusal(self, fields: dict) -> str | None:
        """The door's refusal when this candidate's value on the campaign's counting
        dimension was registered by an earlier run; None when it is new or nothing is
        declared. A second commit of a counted program is worth zero, so the refusal names
        the run that has it and asks for a different program, not a different commit."""
        kf = getattr(self.spec, "known_from", None)
        if not kf:
            return None
        # LIVE: the declared paths are re-read when their state files have
        # changed since the last look, so concurrent attempts exclude each other (a set read
        # once at launch, while sibling directories were empty, would let them all register
        # the same program).
        stamp = _known_stamp(kf)
        if stamp != getattr(self, "_known_stamp", None):
            self.known = _load_known(kf)
            self._known_stamp = stamp
        if not self.known:
            return None
        from agentcore.agent.diversity import value_of

        value = value_of(fields, list(kf.keys or []), urls=True)
        if not value:
            return None
        where = self.known.get(value)
        if where is None:
            return None
        ending = _KNOWN_ENDINGS.get((id(self.known), value), "")
        ended = f" and ended {ending}" if ending else ""
        hint = (
            " If it died on its workload, bring a program of a different KIND, not a neighbour of this one."
            if ending.startswith(("REJECTED", "BLOCKED"))
            else ""
        )
        return (
            f"{' / '.join(kf.keys)} = {value!r} was already registered by "
            f"{os.path.basename(where.rstrip('/')) or where} ({where}){ended}: this campaign "
            f"counts a program once, whatever the commit. Bring a DIFFERENT program.{hint}"
        )

    # ── identity ─────────────────────────────────────────────────────────────
    def key_of(self, fields: dict) -> str | None:
        """Identity of a candidate, from the task-declared key fields.

        Returns None if any key field is missing -- an unidentifiable candidate cannot be
        deduplicated, so it is refused at the door rather than added as a near-duplicate.

        The KEY is normalized; the recorded fields are not. Identity is not the literal text the
        model typed: the same commit registered under "Org/Repo" and "org/repo", or as an
        abbreviated and a full hash, must be one row. The diversity CAPS already normalize
        (`value_of` lowercases and strips `.git`), and identity is the half that decides whether
        two rows are one row.

        What is normalized here is only what cannot change which artifact is meant: case, a
        trailing slash, a `.git` suffix. The stored `fields` keep the model's own spelling,
        because the clone URL it verified is the one a worker should use.
        """
        parts = []
        declared = getattr(self.spec, "normalize", None) or {}
        for k in self.spec.key:
            v = fields.get(k)
            if v in (None, ""):
                return None
            name = declared.get(k)
            if name is not None:
                fn = KEY_NORMALIZERS.get(name)
                if fn is None:
                    # A typo'd normalizer must be loud: silently skipping it would let a
                    # declaration do nothing.
                    raise ValueError(
                        f"worklist normalize: {k}: {name!r} is not a known normalizer "
                        f"({', '.join(sorted(KEY_NORMALIZERS))}). Fix the spec."
                    )
                v = fn(k, str(v))
            parts.append(_norm_key_part(v, keep_case=(name == "abs_path")))
        return "|".join(parts)

    # ── mutation ─────────────────────────────────────────────────────────────
    def add(self, fields: dict) -> tuple[bool, tuple[str, str]]:
        """Add a candidate. Returns (added, (kind, detail)).

        `kind` is one of "added" | "duplicate" | "incomplete". It is a typed value rather than
        prose because the caller has to tell those cases apart to report them, and matching on
        the message text breaks when a new refusal is added (an incomplete candidate could be
        reported as a DUPLICATE, whose only rational response is to try a different SHA).
        """
        try:
            key = self.key_of(fields)
        except Unnormalizable as e:
            # Refused AT THE DOOR, like an incomplete candidate: admitting it under a guessed
            # key could register one file twice, and the tool layer reports this kind's
            # detail verbatim so the model gets the shape to use.
            return False, ("malformed", str(e))
        if key is not None:
            for existing in self.store["candidates"]:
                if _same_artifact(key, existing):
                    return False, ("duplicate", existing)
        known = self._known_refusal(fields)
        if known:
            return False, ("known", known)
        # A candidate that cannot even satisfy the FIRST state is refused at the door, and the
        # refusal says what is missing -- ALL of it, in one refusal. Identity keys and the first
        # state's requires are checked together, so a candidate short on both does not learn
        # its schema one round trip at a time.
        required = list(self.spec.key) + [r for r in (self.spec.states[0].requires or []) if r not in self.spec.key]
        # ABSENT and BLANK are different mistakes and want different fixes: a field sent as
        # `merge_commit: ""` is not "missing". Likewise, when an unrecognised key was supplied
        # and dropped (a typo for a schema field), the refusal says so, since that is what
        # identifies the typo.
        absent = [r for r in required if r not in fields]
        blank = [r for r in required if r in fields and _absent(fields.get(r))]
        if absent or blank:
            parts = []
            if absent:
                parts.append(f"field(s) {absent} not supplied")
            if blank:
                parts.append(
                    f"field(s) {blank} supplied but BLANK (you sent the key with an "
                    f"empty value -- fill it in, do not re-send it empty)"
                )
            vocab = self.vocabulary()
            unknown = [k for k in fields if k not in vocab]
            if unknown:
                parts.append(
                    f"key(s) {sorted(unknown)} are not part of a candidate and were "
                    f"ignored -- if one of them was meant to be a required field, it "
                    f"is under the wrong name"
                )
            return False, ("incomplete", "; ".join(parts) + f" -- registering a candidate needs all of {required}")
        # An identity that collides with an existing candidate's CURRENT fields is a duplicate
        # even if no stored key matches -- see _rekey().
        for other, c in self.store["candidates"].items():
            try:
                k2 = self.key_of(c["fields"])
            except Unnormalizable:
                k2 = None  # admitted before the normalizer was declared; judge it as-is
            if other != key and k2 is not None and _same_artifact(k2, key):
                return False, ("duplicate", other)
        self.store["candidates"][key] = {
            "fields": dict(fields),
            "state": self.spec.states[0].name,
            "history": [self.spec.states[0].name],
        }
        self.store["order"].append(key)
        self._recompute(key)
        return True, ("added", key)

    def vocabulary(self) -> set[str]:
        """Every field name the lifecycle actually reads.

        Identity keys, each state's requires/requires_true, and the caps' keys -- caps read
        fields (language, upstream_url...) that no state requires, and a vocabulary without
        them would silently disable the cap by filtering its input out.
        """
        from agentcore.agent import diversity

        vocab = set(self.spec.key or [])
        for st in self.spec.states:
            vocab |= set(st.requires or []) | set(st.requires_true or [])
            # The negative control reads three fields by name, and they need not appear in any
            # state's requires -- a field may belong to an earlier state, and a task could
            # name fields belonging to none. Omitted here, `update()` would filter out the very
            # evidence `_discriminates` then demands.
            vocab |= {f for f in (getattr(st, "discriminates", None) or {}).values() if f}
            # The re-derivation control's cmd fields, for the same reason: `largest_run_cmd`
            # need not appear in any requires, and filtering it out of update() would deadlock
            # the exact evidence _rederives demands. Normalizer fields likewise (step F).
            vocab |= {
                str(d.get("cmd"))
                for d in (getattr(st, "rederives", None) or {}).values()
                if isinstance(d, dict) and d.get("cmd")
            }
            vocab |= {
                str(d.get("normalize"))
                for d in (getattr(st, "rederives", None) or {}).values()
                if isinstance(d, dict) and d.get("normalize")
            }
            # The RAW OBSERVATIONS a computed field is computed FROM. `timings` need not appear
            # in any requires -- what a later state requires is `median_runtime` -- and without
            # this the one field the agent is asked to record would be filtered out of update()
            # while the field it must never compute is demanded: the same deadlock the
            # discrimination control's fields already taught, wearing different clothes.
            for name, decl in (getattr(st, "computes", None) or {}).items():
                vocab.add(str(name))
                vocab |= set(derive.sources_of(decl))
            # THE GENERAL CASE. Each block above admits one family of fields a gate reads
            # but no `requires` names. The instance's contract steps are the declared, complete
            # list of what a gate reads (`field:` clauses, `{placeholders}` in `run:`/`require:`),
            # so every one of them is recordable by construction.
            vocab |= self._contract_fields(st)
            vocab |= self._validator_reads(st)
        for cap in getattr(self.spec, "caps", None) or []:
            rc = diversity.resolve_cap(cap)
            if rc is not None:
                vocab |= set(rc.keys)
        return vocab

    def update(self, key: str, fields: dict) -> dict | None:
        """Merge evidence into a candidate and recompute its state.

        Only names in the vocabulary merge, and only non-empty values. Both filters exist
        because a self-report belonging to no step, from a REFUSED record, would otherwise sit
        permanently in a candidate's "already recorded" line, and an empty string would
        overwrite verified evidence and regress the candidate's state. `False` is kept:
        recording is_bug_fix=false is a load-bearing value meaning "reject me".
        """
        c = self.store["candidates"].get(key)
        if c is None:
            return None
        vocab = self.vocabulary()
        merged = {k: v for k, v in fields.items() if k in vocab and not _absent(v)}
        changed = set(merged)
        c["fields"].update(merged)
        # THE HARNESS DOES THE ARITHMETIC (see agentcore/agent/derive.py). Here, before the
        # state machine reads anything: a computed field is a legitimate `requires`, so it has
        # to be present by the time `missing` is computed. The agent's own value is never
        # overwritten -- `apply` returns updates only for fields it did not record -- so this
        # cannot silently change evidence a validator already accepted.
        changed |= self._derive(c)
        # Invalidate ONLY the verdicts whose own inputs changed. Dropping all of them meant that
        # recording the fix-side report re-ran the parent-side validator, which re-read a report
        # path the checkout had since overwritten -- so proving one half destroyed the other and
        # the candidate oscillated indefinitely. Verified evidence is a fact about a moment; it
        # does not expire because unrelated work happened afterwards.
        checked = c.get("_checked") or {}
        for st in self.spec.states:
            # A state's verdict depends on its `requires` AND on whatever its negative control
            # compares -- which is often a field from an EARLIER state. Keying invalidation on
            # `requires` alone would leave a candidate refused for "corrupted_digest is
            # identical to rebuild_digest" refused after fixing rebuild_digest, with a message
            # that had become false; and a passing candidate could have its clean value edited
            # to match the perturbed one and keep the verdict.
            if st.name in checked and (changed & self._verdict_inputs(st)):
                checked.pop(st.name, None)
        if any(k in self.spec.key for k in fields):
            key = self._rekey(key) or key
        self._recompute(key)
        # A VALIDATOR may itself have rewritten an identity field: sha_is_real resolves an
        # abbreviated commit against the real clone and writes the full hash back, because
        # abbreviated commit against the real clone and writes the full hash back, because
        # refusing abbreviations would be a large share of refusals. That happens during
        # _recompute, i.e. after the re-key above, so without this second pass the row stays
        # filed under `commons-io|e806e6b` while its own fields say the full hash -- and every
        # identity-keyed guarantee here (dedup, publication, resume) is computed from the key.
        key = self._rekey(key) or key
        return self.store["candidates"][key]

    # ── derived quantities (the harness's arithmetic, never the model's) ─────
    def computes(self) -> dict[str, dict]:
        """Every `computes:` declaration in the lifecycle, keyed by the field it produces.

        Merged across states rather than applied per state: the observations may be recorded at
        one step and the summary required at a later one, and a value that exists only while the
        candidate sits in one state is a value no consumer can rely on.
        """
        out: dict[str, dict] = {}
        for st in self.spec.states:
            for name, decl in (getattr(st, "computes", None) or {}).items():
                out[str(name)] = decl
        return out

    def _derive(self, c: dict) -> set[str]:
        """Compute the declared quantities from what is recorded. Returns the names written.

        Everything interesting is in agentcore/agent/derive.py; this is the part that touches
        the candidate. Provenance is kept on the record (`_derived`) because a number whose
        origin is not recorded is exactly what this project keeps paying for: the value, the
        aggregation, the fields it came from, and -- when the agent recorded its own -- both
        values and the disagreement.
        """
        decls = self.computes()
        if not decls:
            return set()
        fields = c["fields"]
        before = dict(fields.get("_derived") or {})
        # TO A FIXPOINT. A quantity computed FROM computed quantities -- a separation ratio
        # over two medians -- would otherwise be derived only on the NEXT recording, stalling
        # a candidate one state short of delivery. Each pass derives what the previous one
        # made derivable; the count of declarations bounds the passes.
        updates, provenance, problems = derive.apply(decls, fields)
        working = {**fields, **updates}
        for _pass in range(len(decls)):
            # only the declarations still pending, whose sources the earlier passes made
            # present -- never re-applying a settled one, which would read its own result
            # as an agent-recorded value
            pending = {
                name: decl
                for name, decl in decls.items()
                if name not in updates and name not in fields and all(s in working for s in derive.sources_of(decl))
            }
            if not pending:
                break
            u, p, _probs = derive.apply(pending, working)
            if not u:
                break
            updates.update(u)
            provenance.update(p)
            working.update(u)
        fields.update(updates)
        if provenance:
            fields.setdefault("_derived", {}).update(provenance)
        # Problems belong to THIS recording: they are replaced, not accumulated, so a corrected
        # source stops being complained about.
        if problems:
            fields["_derive_problems"] = problems
        else:
            fields.pop("_derive_problems", None)
        # History records a derivation ONCE, when its value or its verdict changes. update() runs
        # on every recording, and a candidate that takes twelve of them would otherwise carry
        # twelve identical lines -- history is read to reconstruct what happened, and a repeated
        # line is noise in the one place noise is expensive.
        fresh = {
            k: v
            for k, v in updates.items()
            if json.dumps(before.get(k, {}).get("value"), default=str) != json.dumps(v, default=str)
        }
        if fresh:
            c["history"].append(
                "derived:" + ", ".join(f"{k}={json.dumps(v, default=str)[:60]}" for k, v in sorted(fresh.items()))
            )
        for name, entry in sorted(provenance.items()):
            m = entry.get("mismatch")
            if m and before.get(name, {}).get("mismatch") != m:
                c["history"].append(f"derive-mismatch:{name}:{m[:120]}")
        return set(updates)

    def derive_problems(self, key: str) -> list[str]:
        """What the last recording could not compute, for the tool to say out loud."""
        c = self.store["candidates"].get(key)
        return list((c or {}).get("fields", {}).get("_derive_problems") or [])

    def _rekey(self, key: str) -> str | None:
        """Re-point a row whose identity fields changed.

        sha_is_real instructs the model to resolve an abbreviated hash; the row must then move
        to the new key, or re-adding the corrected candidate would create a second row for the
        same commit.
        """
        c = self.store["candidates"][key]
        try:
            new = self.key_of(c["fields"])
        except Unnormalizable:
            return key  # a validator wrote a non-canonical value; keep the admitted key
        if not new or new == key:
            return key
        if new in self.store["candidates"]:
            return key  # would collide; keep the original row
        self.store["candidates"][new] = self.store["candidates"].pop(key)
        self.store["order"] = [new if k == key else k for k in self.store["order"]]
        c["history"].append(f"rekey:{key}->{new}")
        return new

    def terminate(self, key: str, state: str, reason: str) -> dict | None:
        """Move a candidate to a terminal state with a recorded reason.

        BLOCKED is deliberately first-class and not a failure of the agent: a candidate the
        pipeline cannot process is data about the pipeline. Recording why is what turns "the run
        underperformed" into "9 candidates blocked on toolchain".
        """
        c = self.store["candidates"].get(key)
        if c is None:
            return None
        c["state"] = state
        c["fields"]["_reason"] = reason
        c["history"].append(f"{state}:{reason[:2000]}")
        return c

    # ── state computation (the harness's job, never the model's) ─────────────
    def _recompute(self, key: str) -> None:
        c = self.store["candidates"][key]
        if c["state"] in (REJECTED, BLOCKED):
            return  # terminal: never revived by later evidence
        reached = self.spec.states[0].name
        for st in self.spec.states:
            # `_is_false`, not plain truthiness: the string "false" is the spelling models
            # actually send, and it would read as true. See _FALSE_WORDS.
            refuted = [
                r for r in (st.requires_true or []) if not _absent(c["fields"].get(r)) and _is_false(c["fields"].get(r))
            ]
            if refuted:
                # Quote the value the model actually wrote, so "recorded build_ok='false'"
                # reads back as the thing that was said rather than a normalised paraphrase.
                said = ", ".join(f"{r}={c['fields'].get(r)!r}" for r in refuted)
                self.terminate(key, REJECTED, f"recorded {said}")
                return
            missing = [r for r in (st.requires or []) if _absent(c["fields"].get(r))]
            if missing:
                break
            if not self._validated(c, st):
                # _validated may itself have made the candidate terminal (STUCK_BLOCK). Without
                # this check the loop below would immediately overwrite BLOCKED with whichever
                # state the evidence reaches, and the candidate would keep being served -- the
                # exact loop the strike counter exists to end.
                if c["state"] in (REJECTED, BLOCKED):
                    return
                break  # evidence present but the harness could not confirm it
            reached = st.name
        if reached != c["state"]:
            c["state"] = reached
            c["history"].append(reached)
            self._maybe_derive_calibration(key, reached)
            self._maybe_snapshot(key, reached)
            self._maybe_fold_history(key, reached)

    def _maybe_snapshot(self, key: str, state: str) -> None:
        """Commit the container the moment a candidate is fully validated.

        The moment matters. Sandboxes are torn down at the end of a run, and the delivered record
        then cites `/workspace/.../TEST-Foo.xml` in a container that no longer exists.
        Committing here captures the environment while it is
        still the one the evidence was produced in, including whatever the agent installed ad hoc
        that no recipe records.

        Failure is deliberately silent-but-recorded: an instance that cannot be snapshotted is
        still a valid instance, and refusing it over a docker hiccup would be an over-refusal.
        """
        terminal = (self.spec.terminal or [None])[0]
        if state != terminal or self.ws is None:
            return
        c = self.get(key) or {}
        sandbox = str((c.get("fields") or {}).get("sandbox") or "").strip()
        if not sandbox:
            return
        try:
            from agentcore.tools import provenance

            if provenance.snapshot_for(self.ws, key):
                return  # this candidate already has one
            from agentcore.tools.containers import SCRATCH_KEY

            env = getattr(self.ctx, "env", None)
            live = (getattr(self.ctx, "scratch", {}) or {}).get(SCRATCH_KEY, {})
            entry = live.get(sandbox) if isinstance(live, dict) else None
            image_id = ""
            if entry is not None and hasattr(entry.get("container"), "commit"):
                # TAG it. An untagged commit shows up as <none> and is indistinguishable from
                # build junk, so `docker image prune` silently deletes the provenance artifacts.
                #
                # Commit through docker_helpers, NOT container.commit(): a built Java repo's
                # writable layer takes minutes to tar and docker-py's default client gives up at
                # 60 seconds.
                safe = re.sub(r"[^A-Za-z0-9_.-]", "-", key)[:96].strip("-.").lower() or "candidate"
                from agentcore.environments import docker_helpers as _dh

                img = _dh.commit_container(entry["container"], repository="benchcompiler-snapshot", tag=safe)
                image_id = str(getattr(img, "id", "") or getattr(img, "short_id", ""))
            elif env is not None and hasattr(env, "checkpoint"):
                image_id = str(env.checkpoint() or "")
            if not image_id:
                # No exception, no image: the sandbox is gone from this process and the env
                # could not stand in. Say so -- an unexplained empty `image` in a shipped
                # artifact would go undiagnosed.
                provenance.record_snapshot_failure(
                    self.ws,
                    key,
                    f"no image committed for sandbox {sandbox!r}: "
                    f"{'no live container in this process' if entry is None else 'commit returned no id'}"
                    f"{'' if env is not None else ' and no environment to fall back on'}",
                    sandbox=sandbox,
                )
            if image_id:
                provenance.record_snapshot(
                    self.ws,
                    key,
                    image_id,
                    note=f"committed when {key} reached {state}",
                    sandbox=sandbox,
                    trace_mark=len(provenance.trace_for(self.ws, sandbox)),
                )
                # Refresh the record's provenance block. The DELIVERED validator built it while
                # running, which is BEFORE this transition fires -- so without a refresh the block
                # names no snapshot and its Dockerfile header reads "(no snapshot recorded)" even
                # though the commit has just succeeded. The delivered instance has to name the
                # authoritative artifact; that is the entire point of taking it.
                fields = c.get("fields") or {}
                if isinstance(fields.get("provenance"), dict):
                    recipes = getattr(self.ws, "sandbox_recipes", None) or {}
                    fields["provenance"] = provenance.provenance_block(
                        self.ws, sandbox, recipes.get(sandbox) or {}, candidate=key
                    )
        except Exception as e:  # noqa: BLE001 - never fail a good candidate over provenance
            # record_snapshot(..., "") would store NOTHING for an empty image id, so the
            # failure must be recorded explicitly.
            try:
                from agentcore.tools import provenance as _prov

                _prov.record_snapshot_failure(self.ws, key, f"commit failed: {type(e).__name__}: {e}", sandbox=sandbox)
            except Exception:  # noqa: BLE001
                pass

    def _maybe_fold_history(self, key: str, state: str) -> None:
        """On delivery, collapse the old action log into one steps-and-lessons summary.

        The action log dominates the prompt, and most of it is the full outputs of work on
        candidates that are already finished. Once a candidate is delivered, the detailed
        sequence has done its job -- what the NEXT candidate needs is which steps proved out
        (from this candidate's verified evidence) and which refusals to avoid repeating (from
        its history), not the pip logs. The most recent entries survive verbatim so in-flight
        context is not lost; per-candidate state lives in the situated block either way.

        Failure here must never cost a delivered candidate, so everything is guarded.
        """
        keep = getattr(self.spec, "fold_history_keep", 8) or 0
        terminal = (self.spec.terminal or [None])[0]
        if keep <= 0 or state != terminal:
            return
        agent = getattr(self.ctx, "agent", None) if self.ctx is not None else None
        log = getattr(agent, "action_log", None)
        if not isinstance(log, list) or len(log) <= keep + 1:
            return
        try:
            c = self.get(key) or {}
            lines = [f"{state}. Steps that proved it:"]
            evidence = c.get("evidence") or {}
            for st in self.spec.states:
                ev = evidence.get(st.name)
                if ev:
                    lines.append(f"  {st.name}: {str(ev.get('detail') or '')[:2000]}")
            # One lesson per state where a refusal happened -- the wording of the refusal is
            # the lesson ("record source_path as a plain absolute path" fired ten times in one
            # run because nothing carried it forward).
            lessons, seen = [], set()
            for h in c.get("history") or []:
                if ":REFUSED:" in str(h):
                    st_name = str(h).split(":REFUSED:")[0]
                    if st_name not in seen:
                        seen.add(st_name)
                        lessons.append(f"  {str(h)[:2000]}")
            if lessons:
                lines.append("Refusals hit on the way (do not repeat these):")
                lines.extend(lessons[:4])
            lines.append("(older log entries folded; the delivered record keeps the details)")
            folded = [(f"[delivered {key}]", "\n".join(lines))]
            log[:] = folded + log[-keep:]
        except Exception:  # noqa: BLE001 - folding is a prompt optimisation, never a failure
            pass

    def _maybe_derive_calibration(self, key: str, state: str) -> None:
        """First candidate to complete every verified step proves the pipeline works.

        Derived rather than reported: if the model wrote calibration_result itself, the
        "mechanical" calibrate gate would be satisfiable with record(key=..., {"passed": true}).
        """
        target = getattr(self.spec, "derive_calibration_to", "")
        terminal = (self.spec.terminal or [None])[0]
        if not target or state != terminal or self.ws is None:
            return
        if getattr(self.ws, target, None):
            return  # already proven; keep the first one
        setattr(
            self.ws,
            target,
            {"passed": True, "candidate": key, "note": "derived by the harness from a completed candidate"},
        )

    def _discriminates(self, c: dict, st) -> tuple[bool, str]:
        """The declared negative control: the oracle must react to a deliberate break.

        Generic on purpose -- see WorklistStateSpec.discriminates. The task names which field
        holds the clean value, which holds the value after the break, and which describes the
        break; the harness knows nothing else about any of them, and compares them as strings.
        """
        d = getattr(st, "discriminates", None) or {}
        clean_f, perturbed_f = d.get("clean"), d.get("perturbed")
        if not (clean_f and perturbed_f):
            return True, ""
        fields = c["fields"]
        # A describe value with whitespace is PROSE -- the description itself, not a field
        # name. Demanding a field by a sentence-shaped name would stall candidates whose
        # clean and perturbed values were already recorded. Resolved, not refused: the design
        # already said what was broken.
        describe_f = d.get("describe")
        if describe_f and " " in str(describe_f).strip():
            describe_f = None
        missing = [f for f in (clean_f, perturbed_f, describe_f) if f and _absent(fields.get(f))]
        if missing:
            return False, (
                f"prove this candidate's oracle can tell right from wrong before it "
                f"counts: break something that MUST change the result, re-measure, "
                f"and record {', '.join(missing)}."
            )
        perturbed = str(fields.get(perturbed_f)).strip().lower()
        raw = fields.get(clean_f)
        clean = [str(v).strip().lower() for v in (raw if isinstance(raw, list) else [raw])]
        if perturbed in clean:
            return False, (
                f"{perturbed_f} is identical to {clean_f}, so this oracle cannot "
                f"detect a wrong answer -- every result would 'match', including a "
                f"broken one. Either the break did not affect what is measured, or "
                f"what is measured is not really being captured. Fix that or drop "
                f"this candidate; a check that never fails proves nothing."
            )
        return True, f"the oracle reacted to a deliberate break ({clean_f} -> {perturbed_f})"

    @staticmethod
    def _ours(code, out) -> str:
        """The refusal text when the HARNESS could not run something, else "".

        Leads with HARNESS ERROR so `_harness_fault` recognises it (no strike) even though
        the caller would otherwise wrap it in "<field> exited N ... " -- which reads as an
        accusation and buries the prefix mid-sentence.
        """
        text = str(out or "")
        try:
            from agentcore.tools.containers import SANDBOX_UNREACHABLE

            unreachable = int(code) == SANDBOX_UNREACHABLE
        except Exception:  # noqa: BLE001
            unreachable = False
        # ...and by what it SAYS, so any path that has not been updated to the sentinel is
        # still caught. These phrases are the harness's own words about its own plumbing; a
        # candidate's build or test cannot print them.
        OURS_PHRASES = (
            "no usable recipe",
            "could not reach sandbox",
            "could not reach this candidate's sandbox",
            "No sandbox named",
            "no environment is available",
            "no environment available",
            "could not be (re)started",
        )
        if unreachable or text.startswith("HARNESS ERROR") or any(ph.lower() in text.lower() for ph in OURS_PHRASES):
            body = text[len("HARNESS ERROR:") :].strip() if text.startswith("HARNESS ERROR") else text.strip()
            return (
                f"HARNESS ERROR: {body or 'the harness could not execute this'} "
                f"Nothing about this candidate's evidence was tested, so no attempt is "
                f"charged against it."
            )
        return ""

    def _lane_hold(self, c: dict, label: str):
        """The host-wide measurement lane for ONE timed re-derivation (observe mode): the
        recorded wall-clock is taken with no other timed execution on the host, and the
        candidate's sandbox pinned to the lane's cores while it runs. Digest mode hashes
        output and times nothing, so it never queues here -- holding the lane for every
        digest would serialize a fleet's builds behind each other again."""
        from agentcore import measurement_lane

        return measurement_lane.hold(label, container=self._lane_container(c))

    def _lane_container(self, c: dict):
        """The docker container behind this candidate's sandbox, for pinning; None when
        there is none or it cannot be resolved (pinning is a courtesy, never a refusal)."""
        sandbox = str((c.get("fields") or {}).get("sandbox") or "").strip()
        if not sandbox or self.ctx is None:
            return None
        try:
            from agentcore.tools import containers as _containers

            entry = (_containers._live(self.ctx) or {}).get(sandbox) or {}
            cont = entry.get("container")
            ident = getattr(cont, "name", None) or getattr(cont, "id", None)
            return str(ident) if ident else None
        except Exception:  # noqa: BLE001 - no live entry, no client: no pin
            return None

    def _exec_rederive(self, c: dict, cmd: str, timeout: int = 900) -> tuple[int, str, float]:
        """Run a re-derivation command in the candidate's OWN sandbox: (exit, output, wall_ms).

        The same single path container_exec and the validators' _run_in use (_ensure_live
        rebuilds the named sandbox from its recipe), because evidence read in a different
        environment than it was produced in is a known source of refusals. Split out as a
        method so tests stub the execution, not the policy."""
        import time as _time

        from agentcore.tools.containers import SANDBOX_UNREACHABLE, UNREACHABLE_PREFIX

        sandbox = str(c["fields"].get("sandbox") or "").strip()
        t0 = _time.monotonic()
        self._last_exec_note = ""
        try:
            if sandbox:
                from agentcore.tools import containers as _containers

                # exec_in_sandbox, not a bare exec: it runs at the sandbox RECIPE's workdir,
                # the same cwd the agent's own container_exec uses. Otherwise a digest control
                # would execute relative-path commands from the image's default cwd and hash
                # error output no honest record could match.
                sink: list = []
                code, out = _containers.exec_in_sandbox(
                    self.ctx, sandbox, cmd, timeout=timeout, revive_removed=True, note_sink=sink
                )
                self._last_exec_note = " ".join(sink)
            elif self.ctx is not None and hasattr(self.ctx, "env"):
                r = self.ctx.env.execute(cmd, timeout=timeout)
                code, out = r.returncode, (r.output or "") + (r.stderr or "")
            else:
                return (
                    SANDBOX_UNREACHABLE,
                    f"{UNREACHABLE_PREFIX}no environment is available to run this in; nothing was executed.",
                    0.0,
                )
        except Exception as e:  # noqa: BLE001 - a broken sandbox must refuse, not kill the run
            # NOT 127. Failing to reach the container is ours; 127 is the shell saying a
            # command is missing, which is the candidate's.
            return (
                SANDBOX_UNREACHABLE,
                f"{UNREACHABLE_PREFIX}the harness could not reach this candidate's "
                f"sandbox ({type(e).__name__}: {e}). Nothing was executed.",
                0.0,
            )
        return code, str(out or ""), (_time.monotonic() - t0) * 1000.0

    def _rederive_one(self, c: dict, fname: str, decl, fields: dict, observed: dict) -> str | None:
        """Run ONE field's re-derivation. Returns the refusal message or None.
        Split out so _rederives can run EVERY field and report all failures in
        one refusal, rather than one problem per cycle."""
        # A COMPUTED quantity is re-derived by RE-COMPUTING it, over the observations the record
        # already carries. That covers exactly the case the exec path cannot: a statistic over
        # many runs has no single reproducing command, which is why `none` had to exist for it,
        # and `none` checked nothing at all. This does not REPLACE an execution the design also
        # asked for -- if the declaration names a command, the re-run happens too, below. The
        # recomputation proves the number summarises its own sample; only re-running proves the
        # sample is of the right program.
        computed = self.computes().get(fname)
        if computed is not None:
            _, prov, problems = derive.apply({fname: computed}, fields)
            entry = prov.get(fname) or {}
            if entry.get("mismatch"):
                return (
                    f"{fname} does not match its own observations: {entry['mismatch']} "
                    f"({entry.get('agg')} of {', '.join(entry.get('from') or [])}). The "
                    f"harness recomputes this field from what you recorded, so either the "
                    f"observations are not the ones this number summarises, or the number "
                    f"is wrong. Drop {fname} and let the harness compute it, or record the "
                    f"observations it really came from."
                )
            if problems:
                return "; ".join(problems)
            if not entry:
                # The field is under the control, and the observations it summarises are not
                # here -- so nothing checked it: a number whose provenance is the agent's word.
                srcs = ", ".join(derive.sources_of(computed))
                return (
                    f"record {srcs}: the raw observations {fname} summarises. The harness "
                    f"computes {fname} from them itself; without them the recorded value "
                    f"is unchecked."
                )
            # Consistent with its own sample. If the design ALSO named a reproducing command,
            # fall through and run it -- that is a different question (is the sample of the
            # right program?) and losing it here would be a quiet downgrade of an existing gate.
        if decl is None or (isinstance(decl, str) and decl.strip().lower() == "none"):
            return None  # explicit exemption (YAML spells it "none")
        if not isinstance(decl, dict):
            return f"rederives entry for {fname!r} is {decl!r}; declare {{cmd: <field>, mode: digest|observe}} or none."
        cmd_field = str(decl.get("cmd") or "")
        mode = str(decl.get("mode") or "digest").strip().lower()
        want_exit, exit_decl_err = self._expected_exit(decl)
        if exit_decl_err:
            return f"{fname}'s re-derivation declaration cannot be read: {exit_decl_err}"
        cmd = fields.get(cmd_field)
        if _absent(cmd):
            return (
                f"record {cmd_field}: the exact command that produced {fname}, "
                f"complete enough to re-run. The harness re-executes it rather "
                f"than believing the recorded value."
            )
        # RENDER IT FIRST, THROUGH THE SAME SEAM THE VARIANTS GATE USES. Executing the recorded
        # string verbatim would hand a command holding a `{placeholder}` to the shell with the
        # braces still in it (e.g. `$(cat {input_path})` would `cat` a file of that literal
        # name, fail, and hash a no-input run), and the control would then demand the recorded
        # digest equal THAT.
        #
        # `render_cmd` already answers this correctly and says so in words the agent can act
        # on -- "references ['input_path'], which this candidate has not recorded". This
        # control is the third consumer of that rule and should use it too.
        # Shell braces are safe: the placeholder pattern requires a bare identifier, so
        # `awk '{print $1}'` is untouched.
        # THE INPUT IT CLAIMS TO RUN ON. `on_input:` names the field holding the
        # path this command must actually use. Declared, never inferred: pairing `*_run_cmd`
        # with `*_input_path` by suffix would invent a naming convention in the harness and
        # bind every future benchmark to it.
        #
        # Without it, a recorded input path could sit beside a run command that never passes it,
        # attributing the digest and runtime to an input nothing passed to anything.
        #
        # Absent the declaration this does nothing.
        on_input = str(decl.get("on_input") or "").strip()
        if on_input:
            want_path = fields.get(on_input)
            if _absent(want_path):
                return (
                    f"this state declares that {cmd_field} runs on {on_input}, and "
                    f"{on_input} has not been recorded. Record it, or correct the "
                    f"`on_input:` in the lifecycle -- the harness cannot check a command "
                    f"against a path that is not there."
                )
            want_path = str(want_path)
            texts = (
                [str(v) for v in cmd.values()]
                if isinstance(cmd, dict)
                else [str(v) for v in cmd]
                if isinstance(cmd, (list, tuple))
                else [str(cmd)]
            )
            if not any(want_path in t or "{" + on_input + "}" in t for t in texts):
                return (
                    f"{cmd_field} never mentions {on_input} ({want_path}), but this state "
                    f"records {fname} as a measurement ON that input. As written the "
                    f"command runs on whatever it defaults to, so {fname} is not about "
                    f"{want_path} at all. Put the path in the command -- literally, or as "
                    f"{{{on_input}}}, which the harness substitutes before re-running."
                )

        # KEEP THE SHAPE. `cmd` is a string, a LIST of commands, or a DICT keyed by input --
        # `test_pairwise_digests` and `test_observe_rederive_keyed` cover all three, and
        # rendering `str(cmd)` would turn a dict into its repr.
        # Render each command in place; the container is untouched.
        from agentcore.tools.variants import render_cmd, with_machinery_input

        # `{input_path}` IS RESOLVABLE. The design's measure block declares it as a
        # placeholder the machinery fills, a contract step may DEMAND it in variant_run_cmds,
        # and packaging binds it to the recorded input -- so this control must not refuse it as
        # "not a recorded field". Same binding as packaging, from the same function; a record
        # with no input at all is still refused by name.
        fields = with_machinery_input(fields)

        def _render(one):
            out, why = render_cmd(str(one), fields)
            return out, why

        if isinstance(cmd, dict):
            rendered_map, bad = {}, None
            for k, v in cmd.items():
                r, why = _render(v)
                if r is None:
                    bad = f"[{k}] {why}"
                    break
                rendered_map[k] = r
            rendered, why = (None, bad) if bad else (rendered_map, "")
        elif isinstance(cmd, (list, tuple)):
            rendered_list, bad = [], None
            for i, v in enumerate(cmd):
                r, why = _render(v)
                if r is None:
                    bad = f"[{i}] {why}"
                    break
                rendered_list.append(r)
            rendered, why = (None, bad) if bad else (rendered_list, "")
        else:
            rendered, why = _render(cmd)
        if rendered is None:
            return (
                f"{cmd_field} cannot be re-run as recorded: {why} The harness re-executes "
                f"this command to check {fname}, so a placeholder it cannot resolve would "
                f"be passed to the shell literally and {fname} would be checked against "
                f"the output of a command that did not do what you meant."
            )
        cmd = rendered
        if mode == "digest":
            # THE OUTPUT NORMALIZER. A program can be deterministic in every way a benchmark
            # cares about and still fail byte-equality (e.g. the same counts under a fresh
            # "Generated on <time>" header). The BUILDER judges the diff and writes a small
            # script that reduces the output to one canonical form; containment is absolute:
            # the HARNESS pipes through it here (the agent never reports its result),
            # the script is a recorded field that ships with the instance, the same
            # script covers every later comparison, and the discrimination control is
            # the swallow-guard -- clean and corrupted digests both flow through it,
            # so a script that eats the answer makes them equal and the control
            # refuses. Both raw and normalized digests are recorded, ALWAYS, with line
            # counts -- how much the normalizer changes is never invisible.
            norm_field = str(decl.get("normalize") or "")
            norm_path = str(fields.get(norm_field) or "").strip() if norm_field else ""
            if norm_field and norm_path and not norm_path.startswith("/"):
                return (
                    f"{norm_field} must be the ABSOLUTE path of the "
                    f"normalizer script inside the sandbox; got {norm_path!r}."
                )
            # Where the two comparable runs leave their hashed bytes. Named per field so two
            # rederive entries in one state cannot overwrite each other's evidence, and under
            # /tmp because the candidate's own tree is what is being measured.
            _slug = re.sub(r"[^A-Za-z0-9_]", "_", str(fname))[:40]
            _KEEP_A = f"/tmp/.rederive_{_slug}.a"
            _KEEP_B = f"/tmp/.rederive_{_slug}.b"

            def _one_digest(cmd_text: str, keep_at: str = ""):
                """Run ONE command and hash its (optionally normalized) output.
                Returns (refusal_message_or_None, got, extra).

                `keep_at` names a path the HASHED bytes are left in instead of deleted, so
                that two runs which disagree can be diffed without executing the command a
                third time. That matters here more than it looks: the commands this diff
                helps most are the ones that rebuild, where a third run is a third build.
                """
                if norm_path:
                    wrapped = (
                        # `__c` IS THE COMMAND'S STATUS, captured before anything else runs.
                        # Without it this wrapper ends on `rm` and reports 0 however the command
                        # went, so the `code != 0` check below could never fire. The
                        # normalizer's own `|| exit 97` contract is a different failure and
                        # still wins.
                        f'__o=$(mktemp); ( {cmd_text} ) > "$__o" 2>&1; __c=$?; '
                        f"echo raw=$(sha256sum < \"$__o\" | awk '{{print $1}}')"
                        f',$(wc -l < "$__o"); '
                        # Direct exec, NOT `bash <path>`: bash interprets the file
                        # as bash whatever its shebang says, so Python normalizers could
                        # never run under it. chmod first; the kernel then honors
                        # the shebang.
                        f"chmod +x {norm_path} 2>/dev/null; "
                        f'{norm_path} < "$__o" > "$__o.n" || exit 97; '
                        f"echo norm=$(sha256sum < \"$__o.n\" | awk '{{print $1}}')"
                        # The NORMALIZED bytes are the ones that were hashed, so those are
                        # the ones worth diffing: a raw diff of a goaccess header the
                        # normalizer already removes would point at the wrong thing.
                        f',$(wc -l < "$__o.n"); '
                        + (f'mv "$__o.n" {keep_at} 2>/dev/null; ' if keep_at else "")
                        + 'rm -f "$__o" "$__o.n"; '
                        "exit $__c"
                    )
                    code, out, wall_ms = self._exec_rederive(c, wrapped)
                    if code == 97:
                        # The script's own stderr is in the exec output (after the
                        # raw= line) -- without it, three agents across two tasks
                        # iterated blind on normalizers that worked under
                        # `python3 x.py` but died on direct exec.
                        err_tail = "\n".join(l for l in out.strip().splitlines() if not l.startswith("raw="))[-300:]
                        return (
                            (
                                f"the normalizer {norm_path} exited nonzero on the "
                                f"real output; a normalizer must read stdin and "
                                f"write the canonical form to stdout, exit 0. The "
                                f"script is executed DIRECTLY, so it needs a "
                                f"shebang line (e.g. #!/usr/bin/env python3 or "
                                f"#!/bin/sh) as its first line."
                                + (f" Its error output: {err_tail}" if err_tail else "")
                                + self._revival_suffix()
                            ),
                            "",
                            {},
                        )
                    parsed = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
                    raw_d, _, raw_n = str(parsed.get("raw", "")).partition(",")
                    got, _, norm_n = str(parsed.get("norm", "")).partition(",")
                    got = got.strip().lower()
                    extra = {
                        "raw_sha256": raw_d.strip().lower(),
                        "raw_lines": raw_n.strip(),
                        "normalized_lines": norm_n.strip(),
                        "normalizer": norm_path,
                    }
                else:
                    # NOT a pipeline: `( cmd ) | sha256sum | awk` exits with awk's status,
                    # which is 0 whatever the command did. Same defect as the normalized
                    # wrapper above and the same fix -- hash from a file and exit with `__c`.
                    wrapped = (
                        f'__o=$(mktemp); ( {cmd_text} ) > "$__o" 2>&1; __c=$?; '
                        f"sha256sum < \"$__o\" | awk '{{print $1}}'; "
                        + (f'mv "$__o" {keep_at} 2>/dev/null; ' if keep_at else "")
                        + 'rm -f "$__o"; exit $__c'
                    )
                    code, out, wall_ms = self._exec_rederive(c, wrapped)
                    got = out.strip().splitlines()[-1].strip().lower() if out.strip() else ""
                    extra = {}
                # THE SAME DECLARATION THE OBSERVE PATH READS. `rederives` has an
                # `expect_exit` because a state can require its command to FAIL, and it is
                # wired into `observe`; this line must honour it too, or a state such as
                # `enforces: ['fails on the pre-fix commit']` with `mode: digest` would have
                # every valid candidate refused for exiting non-zero.
                self._exit_cmd_text = str(cmd_text)
                bad_exit = self._exit_mismatch(cmd_field, fname, want_exit, code, lambda: "")
                if bad_exit:
                    return (bad_exit + self._revival_suffix()), "", {}
                # An expected exit status does not excuse unhashable output: reproducing the
                # recorded digest is what this mode is for.
                if bad_exit is None and not re.match(r"^[0-9a-f]{64}$", got):
                    return (
                        (
                            f"re-running {cmd_field} did not produce a readable "
                            f"digest (exit {code}): "
                            f"{out.strip()[:REFUSAL_OUTPUT_CHARS] or 'no output'}. The command must "
                            f"run in this candidate's sandbox as recorded."
                            + self._exit_127_hint(code)
                            + self._revival_suffix()
                        ),
                        "",
                        {},
                    )
                if bad_exit == "" and (code != 0 or not re.match(r"^[0-9a-f]{64}$", got)):
                    return (
                        (
                            f"re-running {cmd_field} did not produce a readable "
                            f"digest (exit {code}): "
                            f"{out.strip()[:REFUSAL_OUTPUT_CHARS] or 'no output'}. The command must "
                            f"run in this candidate's sandbox as recorded."
                            + self._exit_127_hint(code)
                            + self._revival_suffix()
                        ),
                        "",
                        {},
                    )
                from agentcore.agent.digests import is_empty_digest

                if is_empty_digest(got):
                    return (
                        (
                            f"{cmd_field} produced EMPTY output, and a digest of "
                            f"nothing proves nothing -- every wrong answer would "
                            f"'match' it. The same happens when the command ends in "
                            f"'> /dev/null'. Make the "
                            f"command emit the program's real results."
                        ),
                        "",
                        {},
                    )
                return None, got, extra

            def _diagnose(cmd_text: str, got: str):
                """Re-run once and report (stable, second_value).

                A mismatch has two causes that one message would conflate: the record
                disagrees with a REPRODUCIBLE command (one of the two is wrong, and
                only the agent knows which -- a command that runs the wrong binary
                looks identical from here, so the harness must NOT adopt), or the
                output itself varies between runs, in which case no record can ever
                pass and the answer is a normalizer."""
                err2, got2, _ = _one_digest(cmd_text, keep_at=_KEEP_B)
                if err2 or not got2:
                    return True, got  # cannot tell; treat as stable
                return (got2 == got), got2

            def _what_differs() -> str:
                """The first lines by which the two kept outputs disagree, or "".

                This refusal names a defect and a fix but can still leave the agent guessing,
                because it can only offer a list of things that COULD vary -- and for builds
                that run before they execute (e.g. `make -j$(nproc)`), output interleaves
                differently every time. Two lines of build log say that; "timestamps, PIDs,
                ordering" does not.

                Best-effort by construction. A diff that cannot run leaves a refusal that
                was already worth sending exactly as it was.
                """
                try:
                    code, out, _ = self._exec_rederive(
                        c, f"diff {_KEEP_A} {_KEEP_B} | head -14; rm -f {_KEEP_A} {_KEEP_B}", timeout=60
                    )
                except Exception:  # noqa: BLE001 - never let the extra step lose the refusal
                    return ""
                # `diff` exits 1 when the files differ, which is the case we are in; only a
                # tool that is missing or a file that is not there gets a higher status.
                if code not in (0, 1):
                    return ""
                text = "\n".join(l for l in str(out or "").splitlines() if l.strip())
                return text[:600]

            def _disagreement(cmd_text: str, got: str, rec0: str, key: str = ""):
                """The record differs from what the harness just measured: adopt, or refuse.

                ADOPT (returns None) when the command is reproducible. Refusing a mismatch
                asks the agent to retype a number the harness has just computed and already
                stores, and it can ping-pong when a command rebuilds (`make -j$(nproc)`) and the
                digest hashes a parallel build log, so the two-run check calls it stable while
                the digests keep changing.

                Given up knowingly: a
                command pointing at the WRONG BUILD looks identical from here, and adopting
                certifies it. That case does not escape -- an `optimized_run_cmd` running the
                default binary lands at ~1.0x and the separation control refuses it -- it is
                merely caught at packaging rather than at record time.

                REFUSE only when the OUTPUT ITSELF varies: no measured digest can anchor a
                reference for a stream that is different every time, and that message names
                both the defect and the fix, which is the bar a refusal has to clear.
                """
                where = f" for input {key!r}" if key else ""
                # NEVER ADOPT A VALUE MEASURED IN A SANDBOX THAT HAD TO BE REBUILT. If the
                # sandbox died mid-validation, `_ensure_live` rebuilds it from a recipe holding the
                # installs but not the agent's clone and builds, and the re-derivation would hash
                # a fresh workspace's ERROR output -- perfectly stable, and wrong. Adopting
                # there would write that error's digest into the shipped instance as its
                # reference, silently, which is strictly worse than a refusal.
                # The disagreement is evidence about the ENVIRONMENT, not about the record.
                if self._last_exec_note:
                    return (
                        f"{cmd_field}{where} re-ran in a sandbox that had to be REBUILT "
                        f"and produced {got}, which disagrees with the recorded {rec0}. "
                        f"The harness adopts what it measures, but not from an "
                        f"environment that is not the one you built in: a rebuilt sandbox "
                        f"holds only what its recipe's setup installs -- not a clone you "
                        f"made by hand, not a build you ran in the shell. Put those steps "
                        f"into the recipe's setup so a rebuild reproduces your workspace, "
                        f"then re-record." + self._revival_suffix()
                    )
                stable, second = _diagnose(cmd_text, got)
                if stable:
                    return None
                through = f" (through the recorded normalizer {norm_path})" if norm_path else ""
                differs = _what_differs()
                shown = f" THE LINES THAT DIFFER between the two runs:\n{differs}\n" if differs else ""
                return (
                    f"{cmd_field}{where} is NOT REPRODUCIBLE{through}: two runs "
                    f"of your own command produced {got} and then {second}. No "
                    f"recorded digest can ever pass this gate while the output "
                    f"varies -- this is not a bookkeeping error."
                    + shown
                    # NO GUESS ABOUT WHAT THE VARYING BYTES ARE. Offering a fixed
                    # list ("timestamps, PIDs, paths, ordering, addresses") or a hint about
                    # build logs would write an ecosystem assumption into an engine.
                    # The diff says what varies; this says what to do about anything.
                    + " Record an output normalizer that removes exactly what "
                    "changes, or measure something that does not vary." + self._revival_suffix()
                )

            recorded = fields.get(fname)

            # A string that parses as a JSON object IS the dict it encodes
            # ('{"minimal.txt": "aae7..."}'); the scalar path would mismatch it.
            # Resolve, never refuse.
            def _maybe_json(v):
                if isinstance(v, str) and v.strip()[:1] in "{[":
                    try:
                        return json.loads(v)
                    except (ValueError, TypeError):
                        return v
                return v

            cmd = _maybe_json(cmd)
            recorded = _maybe_json(recorded)
            # Pairwise mode: a dict of commands verifies a dict of digests key by
            # key. Accepting any matching value would verify only whichever value the single
            # command happened to reproduce and ship the rest unchecked -- mixed shapes are
            # refused instead, with both valid forms named. Lists remain the
            # stability-repetition notation.
            if isinstance(cmd, dict) or isinstance(recorded, dict):
                if not (isinstance(cmd, dict) and isinstance(recorded, dict)):
                    return (
                        f"{fname} and {cmd_field} must have MATCHING shapes: either "
                        f"one command whose whole output stream yields one digest "
                        f"(scalar {cmd_field}, scalar {fname}), or matching dicts "
                        f"keyed by input ({cmd_field}[k] re-derives {fname}[k], "
                        f"each pair verified separately). A per-input digest dict "
                        f"cannot be verified by a single command -- the harness "
                        f"hashes that command's concatenated stream." + self._revival_suffix()
                    )
                if set(cmd) != set(recorded):
                    only_c = sorted(set(map(str, cmd)) - set(map(str, recorded)))
                    only_r = sorted(set(map(str, recorded)) - set(map(str, cmd)))
                    return (
                        f"{cmd_field} and {fname} must share the same input keys; "
                        f"only in {cmd_field}: {only_c or '-'}, only in {fname}: "
                        f"{only_r or '-'}." + self._revival_suffix()
                    )
                if not cmd:
                    # {} == {} passes every check above while executing nothing -- the
                    # unable-to-FAIL shape inside the control built against it.
                    return (
                        f"{cmd_field} and {fname} are EMPTY maps: an empty command map "
                        f"re-derives nothing, so this control would pass without executing "
                        f"anything. Record at least one input's command and its digest "
                        f"(matching keys), or record the scalar form (one command, one "
                        f"digest)." + self._revival_suffix()
                    )
                per: dict[str, dict] = {}
                adopted: dict[str, str] = {}
                for k in sorted(map(str, cmd)):
                    err, got, extra = _one_digest(str(cmd[k]), keep_at=_KEEP_A)
                    if err:
                        return f"input {k!r}: {err}"
                    rec_k = str(recorded[k]).strip().lower()
                    drift: dict[str, str] = {}
                    if got != rec_k:
                        refusal = _disagreement(str(cmd[k]), got, rec_k, key=k)
                        if refusal:
                            return refusal
                        drift = {"recorded_by_agent": rec_k}
                        adopted[k] = got
                    per[k] = {"exit": 0, "output_sha256": got, **drift, **extra}
                if adopted:
                    # The FIELD, not just the evidence: this value ships with the instance
                    # and every later control compares against it, so leaving the agent's
                    # copy in place would hand packaging a digest nothing measured.
                    fields[fname] = {**{str(kk): str(vv) for kk, vv in recorded.items()}, **adopted}
                observed[fname] = {"mode": "digest", "per_input": per}
            else:
                err, got, extra = _one_digest(str(cmd), keep_at=_KEEP_A)
                if err:
                    return err
                rec = [str(v).strip().lower() for v in (recorded if isinstance(recorded, list) else [recorded])]
                drift: dict[str, str] = {}
                if got not in rec:
                    refusal = _disagreement(str(cmd), got, str(rec[0]))
                    if refusal:
                        return refusal
                    drift = {"recorded_by_agent": str(rec[0])}
                    # The FIELD, not just the evidence -- see the pairwise path.
                    fields[fname] = got
                observed[fname] = {"mode": "digest", "exit": 0, "output_sha256": got, **drift, **extra}
        else:  # observe
            cmds = cmd
            if isinstance(cmds, str) and cmds.strip()[:1] in "{[":
                try:
                    cmds = json.loads(cmds)
                except ValueError:
                    cmds = cmd
            if isinstance(cmds, (dict, list, tuple)):
                # STRUCTURED commands run element by element: a dict's PYTHON REPR must
                # not reach the shell as `( {'huge.txt': 'zstd ...'} ) > /dev/null` (exit 127,
                # blamed on the candidate for the harness's own parse), and the same holds for a
                # LIST (one command per repetition). Any non-scalar runs per element, and the
                # refusal names which one.
                items = sorted(cmds.items()) if isinstance(cmds, dict) else list(enumerate(cmds))
                if not items:
                    # An empty structure skips the loop, so its loop variable must not
                    # be read afterwards.
                    return (
                        f"{cmd_field} is an empty "
                        f"{'map' if isinstance(cmds, dict) else 'list'}: it runs nothing, "
                        f"so {fname} cannot be re-derived from it. Record the command(s) "
                        f"that produce {fname}, one per element." + self._revival_suffix()
                    )
                walls: dict = {}
                for k2, one in items:
                    # A timed execution: queued on the host-wide measurement lane so
                    # the wall-clock it records is not a busy fleet's (measurement_lane).
                    with self._lane_hold(c, f"rederive {fname}[{k2!r}]"):
                        code, out, wall_ms = self._exec_rederive(c, f"( {str(one)} ) > /dev/null 2>&1")
                    self._exit_cmd_text = str(one)
                    verdict = self._exit_mismatch(
                        f"{cmd_field}[{k2!r}]",
                        fname,
                        want_exit,
                        code,
                        lambda o=one, ou=out: self._observe_diagnosis(c, o, ou) or "",
                    )
                    if verdict is not None and verdict != "":
                        return verdict + self._revival_suffix()
                    if verdict == "":
                        ours = self._ours(code, out)
                        if ours:
                            return ours
                        return (
                            f"{cmd_field}[{k2!r}] exited {code} when the harness ran "
                            f"it, so {fname} cannot be re-derived: "
                            + (self._observe_diagnosis(c, one, out) or "no output")
                            + " "
                            + self._exit_127_hint(code)
                            + self._revival_suffix()
                        )
                    walls[str(k2)] = round(wall_ms, 3)
                observed[fname] = {
                    "mode": "observe",
                    "exit": code,
                    "expect_exit": want_exit,
                    "harness_wall_ms": walls,
                    "recorded": fields.get(fname),
                }
                return None
            with self._lane_hold(c, f"rederive {fname}"):
                code, out, wall_ms = self._exec_rederive(c, f"( {cmds} ) > /dev/null 2>&1")
            self._exit_cmd_text = str(cmds)
            verdict = self._exit_mismatch(
                cmd_field, fname, want_exit, code, lambda: self._observe_diagnosis(c, cmds, out) or ""
            )
            if verdict is not None and verdict != "":
                return verdict + self._revival_suffix()
            if verdict == "":
                ours = self._ours(code, out)
                if ours:
                    return ours
                return (
                    f"{cmd_field} exited {code} when the harness ran it, so "
                    f"{fname} cannot be re-derived: "
                    f"{self._observe_diagnosis(c, cmds, out) or 'no output'}"
                    + self._exit_127_hint(code)
                    + self._revival_suffix()
                )
            observed[fname] = {
                "mode": "observe",
                "exit": code,
                "expect_exit": want_exit,
                "harness_wall_ms": round(wall_ms, 3),
                "recorded": fields.get(fname),
            }
        return None

    @staticmethod
    def _expected_exit(decl: dict) -> tuple[str, str]:
        """Read a `rederives` entry's `expect_exit`. Returns (kind, error).

        kind is "zero", "nonzero", or an exact code as a string. Absent means "zero", which is
        what every spec written before this existed meant. Notation is resolved, not refused --
        a designer writing 1, "1", "non-zero" or "NONZERO" plainly means the same thing -- but a
        declaration that cannot be read at all is an error rather than a silent fallback,
        because falling back to the default is precisely what changes which candidates survive.
        """
        if "expect_exit" not in decl or decl.get("expect_exit") is None:
            return "zero", ""
        raw = decl["expect_exit"]
        if isinstance(raw, bool):
            return "", (
                f"expect_exit: {raw!r} is a boolean and there is no obvious exit status "
                f"it names. Write 'zero', 'nonzero', or the exact code."
            )
        text = str(raw).strip().lower().replace("-", "").replace("_", "")
        if text in ("zero", "0", "ok", "success"):
            return "zero", ""
        if text in ("nonzero", "fail", "failure", "any"):
            return "nonzero", ""
        if text.isdigit():
            return ("zero" if text == "0" else text), ""
        return "", (
            f"expect_exit: {raw!r} cannot be read as an exit status. Write 'zero' (the "
            f"default), 'nonzero' for a command the state requires to fail, or the exact "
            f"code as a number."
        )

    # Statements that report success no matter what ran before them. A shell list exits with
    # its LAST command's status, so any of these in final position replaces the status of the
    # work with its own. `echo` is the usual case (`; echo "EXIT_CODE=$?"`).
    _STATUS_NEUTRAL = ("echo", "true", ":", "printf")

    @classmethod
    def _swallowed_status_note(cls, cmd) -> str:
        """Name a trailing statement that took over the command's exit status, or "".

        Only `;` and `&` split a list in a way that discards the earlier status unconditionally.
        `&&` and `||` are deliberate and their last element runs only on a condition, so they
        are not accused here; neither is a pipeline, whose last element IS the work.
        """
        text = str(cmd or "").strip().rstrip(";").strip()
        if not text:
            return ""
        cut = max(text.rfind(";"), text.rfind("&"))
        if cut < 0 or (cut and text[cut - 1] in "&|"):  # && or || -- conditional, not a swap
            return ""
        tail = text[cut + 1 :].strip()
        head = (tail.split() or [""])[0].strip("(")
        if head not in cls._STATUS_NEUTRAL:
            return ""
        return (
            f" Note the last statement of the command, {tail!r}: a shell list exits with the "
            f"status of its LAST command, so this one reports success however the work "
            f"before it turned out, and the exit status the harness sees is its, not the "
            f"test's. Record the command without that trailing statement."
        )

    _exit_cmd_text = ""

    def _exit_mismatch(self, cmd_field: str, fname: str, want: str, code: int, diagnose) -> str | None:
        """Refuse when the observed exit status is not the one the state declared it wants.

        The `nonzero` case is the load-bearing one: a state that requires its command to fail is
        making a claim ABOUT THE CANDIDATE, and a command that succeeds refutes it. Saying only
        "expected nonzero, got 0" would leave the agent to work out what that implies, so the
        refusal says what the success means.

        `diagnose` is a callable, not a string: it re-runs the command un-silenced, and a
        matching exit status must not pay for that. Calling it eagerly would break the guarantee that
        a passing `observe` runs exactly once, which is its own test.
        """
        if want == "zero":
            return None if code == 0 else ""  # "" -> caller's existing refusal text
        if want == "nonzero":
            if code != 0:
                return None
            note = self._swallowed_status_note(self._exit_cmd_text)
            head = (
                f"{cmd_field} exited 0, but this state requires it to FAIL "
                f"(expect_exit: nonzero), so {fname} does not show what it must."
            )
            guess = (
                f"A command that succeeds here means the behaviour this candidate is "
                f"supposed to demonstrate did not occur -- check that {cmd_field} really "
                f"selects the triggering case and that you ran it against the right "
                f"revision."
            )
            # ORDER IS THE MESSAGE. With the mechanism appended AFTER this hypothesis, a model
            # tends to go to `git status && git log` and inspect a repository that has nothing
            # wrong with it. A trailing status-neutral statement is a CERTAINTY read off the
            # command; the wrong-revision story is a GUESS. When both are available the certainty
            # leads and the guess becomes the fallback -- and when there is no mechanism to name,
            # the guess is all there is and leads exactly as before.
            if note:
                return (
                    head + note + f" If it still exits 0 with that gone, then the test really did pass "
                    f"here: {guess[0].lower() + guess[1:]}" + f" {diagnose()}".rstrip()
                )
            return head + " " + guess + f" {diagnose()}".rstrip()
        if str(code) == want:
            return None
        return (
            f"{cmd_field} exited {code}, but this state declares expect_exit: {want}."
            + self._swallowed_status_note(self._exit_cmd_text)
            + f" {diagnose()}".rstrip()
        )

    def _observe_diagnosis(self, c: dict, one: str, out: str) -> str:
        """What a silenced `observe` command actually printed, fetched only on failure.

        `observe` runs `( cmd ) > /dev/null 2>&1` because what it measures is the exit code and
        the harness's own wall-clock, and capturing a full Maven log would both cost and distort
        that timing. Right for the measurement, wrong for the diagnosis: on failure `out` is
        empty and the refusal would read "no output" as though the command had been silent.

        Without the output, a bare "exited 1 ... cannot be re-derived: no output" carries almost
        no information when `exit 1` is the EXPECTED outcome (e.g. a buggy revision): the
        agent cannot tell a correctly-failing test from a broken selector, a missing
        dependency or a build that never compiled.

        The re-run is paid only on a path that is already a dead end, and never on success.
        """
        if str(out or "").strip():
            return str(out).strip()[:REFUSAL_OUTPUT_CHARS]
        try:
            _code, again, _wall = self._exec_rederive(c, f"( {one} ) 2>&1")
        except Exception:  # noqa: BLE001 - a diagnosis must never replace the refusal
            return ""
        from agentcore.tools.variants import informative_tail

        return informative_tail(again, limit=REFUSAL_OUTPUT_CHARS) if str(again or "").strip() else ""

    @staticmethod
    def _exit_127_hint(code) -> str:
        """Exit 127 has one overwhelmingly likely cause; NAME it.

        A command that exited 127 usually ran on the wrong machine (e.g. python3 exists on the
        agent's own machine and not in the sandbox), so every piece of evidence was produced
        in the wrong place; a refusal that says only "exited 127 ... no output" leaves the
        cause unnamed."""
        if code == 126:
            # THE SAME WRONG-DIRECTORY CAUSE, ONE CODE OVER: `./jq '.' input.json` recorded
            # from inside the build directory; from the recipe's workdir `./jq` is the SOURCE
            # DIRECTORY of that name, and executing a directory is exit 126. A refusal saying
            # only "exit 126" would not name that.
            return (
                " Exit 126 means the shell found something at that path but could not "
                "execute it -- when the harness re-runs the command from the sandbox "
                "recipe's workdir, a RELATIVE path like ./prog usually names a directory "
                "or a non-executable file there, not the binary you ran from inside the "
                "build tree. Record the command with an absolute path to the binary, or "
                "put the `cd` inside the command itself."
            )
        if code != 127:
            # SANDBOX_UNREACHABLE is deliberately a different code: telling an agent to
            # install a tool into a container that no longer exists is the wrong
            # instruction.
            return ""
        # THE COMMON CAUSE FIRST. Exit-127 refusals are mostly a RELATIVE path
        # (`./src/iperf3 --version`) recorded from inside a `cd`: the harness re-runs the
        # command from the sandbox recipe's workdir, where `./src` does not resolve. Opening
        # with "you ran it on the main machine" would send the model looking for a different
        # filesystem instead of an absolute path.
        return (
            " Exit 127 means the command (or a tool it calls) is NOT FOUND in this "
            "candidate's SANDBOX when the harness re-runs it from the sandbox recipe's "
            "workdir. The usual cause is a RELATIVE path (./src/tool) that only resolved "
            "from the directory you happened to be in: record the command with an "
            "absolute path, or put the `cd` inside the command itself. If the tool is "
            "genuinely missing there, you ran it on the main machine -- a different "
            "filesystem the harness never reads: install the tool in the sandbox (or "
            "rewrite the command with tools the sandbox has -- its manifest was printed "
            "at container_create), redo the work THERE, and re-record."
        )

    def _bounds(self, c: dict, st) -> tuple[bool, str]:
        """A computed quantity outside its DECLARED bound refuses the state.

        `computes: {speedup: {from: [default_ms, optimized_ms], agg: ratio, at_least: 1.05}}`
        -- the number is the spec's, projected by the designer from the contract; the harness
        applies it without knowing what it means. Such a bound is decidable from the record
        before delivery, rather than being found at packaging.
        The refusal names the quantity, the value and the bound, and says the lever is the
        workload: the builder still has the sandbox and can try another kind."""
        for name, decl in (getattr(st, "computes", None) or {}).items():
            if not isinstance(decl, dict) or not derive.bounds_of(decl):
                continue
            if name not in c["fields"]:
                continue  # not computed yet: presence is the requires' business
            why = derive.bound_problem(name, decl, c["fields"].get(name))
            if why:
                srcs = ", ".join(derive.sources_of(decl))
                return False, (
                    f"{why} (computed from {srcs}). This is a verdict on the "
                    f"WORKLOAD and the measurement, not on the program: change "
                    f"what is measured -- a different input kind, more of the "
                    f"program's own computation per byte, less I/O or startup "
                    f"-- re-record {srcs}, and the harness recomputes {name}."
                )
        return True, ""

    def _rederives(self, c: dict, st) -> tuple[bool, str]:
        """The declared re-derivation control: recorded execution results are re-executed.

        Generic on purpose -- see WorklistStateSpec.rederives. The task names, per field, the
        FIELD holding the reproducing command and a mode; the harness knows nothing else.
        `digest` re-runs and requires sha256 equality with the recorded value (the exact
        recipe the deterministic state uses: both streams, hashed in-container, so
        recorded digests compare). `observe` requires the command to RUN and records the
        harness's own wall-clock next to the agent's number -- measured, not judged, the
        certifier's stance. An explicit `none` is an on-record exemption; the audit refuses
        absence.
        """
        decls = getattr(st, "rederives", None) or {}
        fields = c["fields"]
        observed: dict[str, dict] = {}
        problems: list[str] = []
        for fname, decl in decls.items():
            msg = self._rederive_one(c, fname, decl, fields, observed)
            if msg is not None:
                m = str(msg)
                problems.append(m if fname in m else f"[{fname}] {m}")
        if problems:
            if len(problems) == 1:
                return False, problems[0]
            head = f"{len(problems)} problems, each independently checked -- fix ALL of them before re-recording: "
            return False, head + " ".join(f"[{i + 1}] {pr}" for i, pr in enumerate(problems))

        if observed:
            # A LIST, like every other writer of this field. The variants gate does
            # `fields.setdefault("_observations", []).append(...)`; on a state carrying BOTH a
            # re-derivation control and a variants gate, a control that created a dict would
            # make the gate raise
            #     AttributeError: 'dict' object has no attribute 'append'
            # at the last state.
            fields.setdefault("_observations", []).append({"kind": "rederived", "values": observed})
        return True, ""

    def _verdict_inputs(self, st) -> set[str]:
        """The fields a state's verdict depends on -- THE one list, used both to invalidate a
        cached verdict (update) and to tell the model what re-triggers the check (the replay
        note). Two copies of this rule is how a refusal comes to name fields that do not
        actually unlock it.

        `requires`, plus whatever the negative control compares -- often a field from an
        EARLIER state (e.g. a control reading `rebuild_digest`, recorded at an earlier state) --
        plus the re-derivation inputs: fixing largest_run_cmd after a refusal must re-run the
        control, and editing it after a pass must not keep the verdict.
        """
        inputs = set(st.requires or []) | {f for f in (getattr(st, "discriminates", None) or {}).values() if f}
        for d in (getattr(st, "rederives", None) or {}).values():
            if isinstance(d, dict):
                if d.get("cmd"):
                    inputs.add(str(d["cmd"]))
                if d.get("normalize"):
                    inputs.add(str(d["normalize"]))
        inputs |= set((getattr(st, "rederives", None) or {}).keys())
        # THE FIELDS THE STATE'S DECLARED CONTRACT READS. A contract step names fields two
        # ways -- `field:` clauses and `{placeholders}` in its `run:` -- and the gate reads
        # them whether or not the state's `requires` lists them. Left out, a field the
        # validator reads but that is not a trigger would be re-recorded as the refusal asked
        # and still served the OLD verdict as a REPLAY, and a state with `requires: []` and a
        # corroboration gate reading `clone_path` would say "(none declared) ... re-record one
        # of them", an instruction that could not be followed. Declared, not inferred: the
        # instance's own contract.
        inputs |= self._contract_fields(st)
        # ...AND THE FIELDS THE VALIDATOR DECLARES IT READS. A variants gate may read
        # `variant_run_cmds`, a field a LATER state requires, so a corrected value would
        # replay the old refusal and the replay note would send the model to re-record an
        # unrelated field with its old value to force the re-run. A validator says what it
        # reads (`fn.reads`); that is a declaration, and it counts exactly like `requires`.
        inputs |= self._validator_reads(st)
        return inputs

    def _validator_reads(self, st) -> set[str]:
        fn = self.validators.get(st.name)
        return {str(f) for f in (getattr(fn, "reads", None) or ())}

    def _contract_fields(self, st) -> set[str]:
        """Every field the instance's declared contract steps for `st` read, by the executor's
        own grammar (`field_contract.fields_read`). The spec and the instance do not change
        during a run, so the answer is computed once per state per Worklist."""
        cache = self.__dict__.setdefault("_contract_fields_cache", {})
        if st.name in cache:
            return cache[st.name]
        inst = (getattr(self.ctx, "instance", None) or {}) if self.ctx is not None else {}
        steps = list((inst.get("state_contracts") or {}).get(st.name) or [])
        if "corroboration_gate" in str(getattr(st, "verify", "") or ""):
            steps += list(inst.get("corroboration_contract") or [])
        out: set[str] = set()
        for step in steps:
            out |= _contract_fields_read(step)
        cache[st.name] = out
        return out

    def _replay_note(self, st) -> str:
        """Appended to a refusal served from cache: say it is a replay, and name the way out.

        The way out matters more than the label. The gates that freeze are exactly the ones
        whose verdict depends on the SANDBOX's disk, not on any recorded field (the variants
        gate above all) -- the agent fixes the scripts on disk, no input field changes, and the
        cache can never invalidate, and a verdict that re-recording could not touch would send
        the agent rewriting correct setup scripts. Re-recording one of the state's own inputs (the same
        value counts -- invalidation keys on field NAMES) is the documented lever that forces
        the check to run again.
        """
        inputs = ", ".join(sorted(self._verdict_inputs(st))) or "(none declared)"
        return (
            f" NOTE: this verdict is a REPLAY of the check's last run, not a fresh "
            f"execution -- the check re-runs only when one of its own inputs is "
            f"re-recorded: {inputs}. Recording any other field cannot change it. If you "
            f"have fixed the problem outside those fields (in the sandbox, on disk), "
            f"re-record one of them -- the same value is fine -- to make the check run "
            f"again."
        )

    def _validated(self, c: dict, st) -> bool:
        """Has the harness confirmed this state's evidence? Cached: validators may shell out."""
        fn = self.validators.get(st.name)
        bounded = any(
            isinstance(d, dict) and derive.bounds_of(d) for d in (getattr(st, "computes", None) or {}).values()
        )
        if (
            fn is None
            and not (getattr(st, "discriminates", None) or {})
            and not (getattr(st, "rederives", None) or {})
            and not bounded
        ):
            return True
        cache = c.setdefault("_checked", {})
        if st.name in cache:
            entry = cache[st.name]
            ok = bool(entry[0])
            if ok:
                return True
            # A REPLAYED refusal is still a refusal the model is shown, and it must COUNT.
            # Otherwise `_strikes` would stay frozen, STUCK_DEFER/STUCK_BLOCK would never fire,
            # the systemic detector (fed from _strike) would stay blind, and the refusal text
            # would say "fix and re-record" while re-recording non-input fields was
            # structurally unable to change the verdict.
            # Escalate from `base` (entry[2], the check's own words), never from the served
            # message -- stacking suffix on suffix is how a record grows without bound.
            base = str(entry[2] if len(entry) > 2 else entry[1])
            # THE SAME TEST THE LIVE PATH APPLIES. This must look at as much of `base` as the
            # live classification below does, since a variants-gate fault's mark can sit after a
            # layout description of several hundred characters; otherwise it would be excused
            # live and then replayed through _strike on the next non-input record, counting a
            # strike for the harness's own dead sandbox.
            if self._harness_origin_text(base):
                # A cached harness fault EXPIRES instead of replaying. The served advice is
                # "fix the environment" -- a fix this cache cannot see, because invalidation keys
                # only on the state's input fields. Replaying would re-count the same dead
                # daemon toward BLOCKED and never notice the environment healing. The entry was
                # served when it was written; here it is dropped and the check RE-RUNS -- a
                # healed environment passes, a still-broken one records a fresh fault, so the
                # starvation guard counts real failed attempts, never replays.
                del cache[st.name]
            else:
                served = self._strike(c, st.name, base + self._replay_note(st), fresh=False)
                cache[st.name] = [False, served, base]
                self._note_refusal(c, st, served, base, origin="candidate", served="replay")
                return False
        # The negative control runs FIRST and independently of any validator. It is the one
        # check that cannot be satisfied by recording well-shaped values, so a state carrying it
        # is gated even when the task wrote no verify: at all.
        ok, detail = self._discriminates(c, st)
        # The re-derivation control, second and equally validator-independent: recorded
        # execution results are re-executed by the harness before any validator reads them.
        if ok:
            ok2, detail2 = self._rederives(c, st)
            if not ok2:
                ok, detail = ok2, detail2
        # a computed quantity outside its declared bound (the spec's number) refuses here
        if ok:
            ok3, detail3 = self._bounds(c, st)
            if not ok3:
                ok, detail = ok3, detail3
        # ORIGIN, not just outcome. A failure the HARNESS caused -- a dead
        # container, a docker hiccup, a full disk, a crash in the validator itself -- is not
        # evidence about the candidate, and charging it to the candidate discards work
        # that was never judged. The harness retries itself once (a dead sandbox is
        # revivable), and if it fails again the fault is recorded WITHOUT a strike.
        harness_origin = False
        if ok and fn is not None:
            for _attempt in (1, 2):
                try:
                    ok, detail = fn(c["fields"], self.ctx)
                    harness_origin = False
                    break
                except Exception as e:  # noqa: BLE001 - a broken validator must not kill the run
                    harness_origin = True
                    ok, detail = False, f"validator error: {type(e).__name__}: {e}"
        if not ok and not harness_origin:
            # a control that self-identifies as the harness's own fault counts the same way
            # NEAR the start, not AT it. Every control wraps its failures -- the
            # re-derivation one prefixes "[<field>] ", the variants gate numbers its
            # problems -- so an exact startswith would miss a marker the seam had correctly
            # placed. The marker is ours and cannot be produced by a candidate's
            # command output, so finding it in the opening of the detail is enough.
            harness_origin = self._harness_origin_text(str(detail))
        if not ok:
            # Bound the check's OWN words first, escalate second. If the escalation were
            # appended and then truncated with everything else, "attempt 3 ... deferred" could
            # fall off the end exactly on the long refusals where the model most needs it.
            base = str(detail)[:REFUSAL_DETAIL_CHARS]
            detail = self._harness_fault(c, st.name, base) if harness_origin else self._strike(c, st.name, base)
            self._note_refusal(c, st, detail, base, origin="harness" if harness_origin else "candidate", served="fresh")
        else:
            (c.get("_strikes") or {}).pop(st.name, None)
        # 1000, not 300: validation_error() serves this cached detail to the model as
        # "here is what to fix", and the longest rederive refusals run ~880 chars -- the
        # sandbox-revival one carries a full digest, the recorded value, what a rebuilt
        # sandbox does and does not hold, and the fix; the non-reproducible one carries two
        # digests and the list of things that vary between runs. Cut at 300 or 600 either
        # loses its fix clause, which is the half that makes it a refusal worth sending.
        #
        # Refusals cache [False, served, base]: `served` is what the model sees (escalation
        # included, never truncated after the fact); `base` is the check's own words, kept so
        # a REPLAY of this verdict re-escalates from the source instead of stacking suffixes.
        cache[st.name] = [True, str(detail)[:2000]] if ok else [False, str(detail), base]
        # WHAT WAS OBSERVED, if the validator left any. Certification records
        # {"exit": 1, "expected": "nonzero"} and is therefore replayable: a candidate rule can be
        # re-evaluated against a run that already happened. Recording only prose --
        # "buggy(fails as required)" -- would leave no rule change measurable against it.
        #
        # Popped either way. On a refusal the candidate is retried, and a stale observation from
        # the failed attempt would be read as evidence for the next one; on success it must not
        # reach a delivered record or travel out as part of a lead.
        observed = c["fields"].pop("_observations", None)
        if ok:
            # Snapshot WHAT was proven, at the moment it was proven. The workspace it came from
            # is mutable (Maven overwrites its surefire report on every checkout), so the record
            # -- not the file -- is the evidence from here on.
            c.setdefault("evidence", {})[st.name] = {
                "detail": str(detail)[:2000],
                "fields": {r: c["fields"].get(r) for r in (st.requires or [])},
            }
            if observed:
                c["evidence"][st.name]["observations"] = observed
        if not ok and not harness_origin:
            # 200, not 80: diagnosis is read off refusal history -- digest prefixes compared
            # across cycles -- and a full-digest refusal cut at 80 ends before its first digest
            # does.
            #
            # Not for a harness fault: _harness_fault already wrote its own line, and a second
            # one spelled ":REFUSED:" would make the fault COUNT where it must not --
            # worklist_drop's veto reads ":REFUSED:" as "a check ran and refused this
            # candidate" and would flip a blocked booking over a dead sandbox to REJECTED.
            c["history"].append(f"{st.name}:REFUSED:{str(detail)[:2000]}")
        return bool(ok)

    # ── the typed log: a gate's verdict as a record ──────────────────────────────
    def _events(self):
        ctx = self.ctx
        if ctx is None:
            return None
        return getattr(ctx, "events", None) or getattr(getattr(ctx, "agent", None), "events", None)

    def _note_refusal(self, c: dict, st, served_text: str, base: str, *, origin: str, served: str) -> None:
        """Write the `refusal` record (see the debug-log plan, section 4) and leave,
        for the loop, the site and expectation of the message the model is about to see.

        The fields ARE the classification: origin (harness or candidate), served (fresh or a
        replay), the strike this refusal counted as and the escalation it caused, so no text
        markers are needed to carry the origin through the log."""
        from agentcore.observability import sites as _sites

        esc = getattr(self, "_last_escalation", None) or {}
        key = next((k for k, cand in self.store["candidates"].items() if cand is c), "")
        site = (
            "worklist.gate.harness_fault"
            if origin == "harness"
            else "worklist.gate.replay"
            if served == "replay"
            else "worklist.gate.refuse"
        )
        events = self._events()
        if events is not None:
            try:
                events.emit(
                    "on_refusal",
                    site=site,
                    gate=str(getattr(st, "verify", "") or "controls"),
                    state=st.name,
                    candidate=key,
                    fingerprint=_refusal_shape(base),
                    origin=origin,
                    served=served,
                    strike=int(esc.get("strike", 0) or 0),
                    escalation=str(esc.get("escalation", "none")),
                    text=served_text,
                )
            except Exception:  # noqa: BLE001 - the log must never change a verdict
                pass
        scratch = getattr(self.ctx, "scratch", None) if self.ctx is not None else None
        if isinstance(scratch, dict):
            expects = None
            if origin == "candidate":
                # within=10, not 2: a variants refusal is answered by rewriting check
                # scripts, rebuilding a variant and testing both in the sandbox before anything
                # is re-recorded -- many honest cycles that a two-cycle window would close as
                # "ignored".
                try:
                    expects = _sites.records_one_of(self._verdict_inputs(st), within=10)
                except Exception:  # noqa: BLE001
                    expects = None
            scratch["_shown"] = {"site": site, "expects": expects}

    @staticmethod
    def _harness_origin_text(detail: str) -> bool:
        """Does this refusal say it is the HARNESS's own fault? ONE test, applied to a live
        verdict and to a cached one alike -- two copies of it disagreed on the window.

        NEAR the start, not AT it: every control wraps its failures (the re-derivation one
        prefixes "[<field>] ", the variants gate puts its layout description first), so the
        mark can sit several hundred characters in. The markers are ours and cannot come
        from a candidate's command output, so finding one in the opening is enough."""
        head = str(detail)[:2000]
        return any(m in head for m in ("HARNESS ERROR", "HARNESS FAULT", "validator error:"))

    def _harness_fault(self, c: dict, state: str, detail: str) -> str:
        """Record a failure the harness caused. Never strikes, never defers, never blocks.

        Counted separately so a broken environment is visible in the record (silence
        would hide it), and phrased so the agent does not go looking for a defect in
        evidence that was never examined."""
        faults = c.setdefault("_harness_faults", {})
        n = faults[state] = int(faults.get(state, 0)) + 1
        self._last_escalation = {
            "strike": 0,
            "escalation": "terminate" if n >= self._knob("stuck_block", STUCK_BLOCK) else "none",
        }
        c["history"].append(f"{state}:HARNESS FAULT (not counted against this candidate): {detail[:2000]}")
        # POOL IT ACROSS CANDIDATES. Excusing harness faults from strikes is right, and it
        # had a side effect: the cross-candidate fingerprint is fed from _strike, so the
        # systemic detector was blind to exactly the class it should convict -- a gate the
        # harness itself says is broken, hit by every candidate. Fingerprinting records the
        # shape and announces it at SYSTEMIC_AT; it never strikes, defers or blocks. The
        # origin travels as a flag, so _emit_systemic can tell the harness's own fault from
        # a correct gate three candidates fail the same way without re-reading the text.
        self._note_refusal_fingerprint(c, state, detail, harness=True)
        # Not counted, but not infinite either: a PERMANENTLY broken gate (a validator
        # that cannot be imported) can never pass, and a run that keeps re-attempting it
        # starves. Harness faults get their own escalation at the BLOCK threshold -- far
        # above the strike deferral at 3, which is what killed candidates over transient
        # container deaths -- and the terminal is BLOCKED, which by definition stays out
        # of the sample's denominator: nothing ever judged this candidate.
        if n >= self._knob("stuck_block", STUCK_BLOCK):
            c["state"] = BLOCKED
            c["fields"]["_reason"] = (
                f"{state}: the HARNESS failed {n} times here (never the candidate's evidence): {detail[:2000]}"
            )
            c["history"].append(
                f"{BLOCKED}:harness failed {n} times at {state}; set aside so the run "
                f"can spend its budget elsewhere -- NOT counted against the sample"
            )
            return (
                f"HARNESS FAULT at {state}, {n} times: the environment or the gate "
                f"itself is broken, so this candidate is set aside (BLOCKED, not "
                f"rejected -- nothing ever judged it). Detail: {detail}"
            )
        return (
            f"HARNESS FAULT at {state} -- the environment failed, not your evidence, "
            f"so this does not count against this candidate ({n} so far). It was "
            f"not cached, so the check re-runs on your next record. Detail: {detail} -- if it repeats, the sandbox or "
            f"the docker daemon is the thing to fix, not the record."
        )

    # How many DISTINCT candidates must hit the same refusal before it stops looking like their
    # fault. Three: two can be coincidence in a small sample, three unrelated candidates failing
    # one gate the same way is a statement about the gate.
    SYSTEMIC_AT = gates.SYSTEMIC_AT

    def _knob(self, name: str, fallback: int) -> int:
        """A gate threshold the SPEC may set (`worklist.stuck_block` and friends, config/schema.py),
        falling back to the constant in agentcore/agent/gates.py -- which is also the schema's
        default, so a spec that says nothing behaves exactly as before. A declared 0 or a
        non-number reads as "not set": none of these gates means anything at zero."""
        try:
            v = int(getattr(self.spec, name, None) or 0)
        except (TypeError, ValueError):
            v = 0
        return v if v > 0 else int(fallback)

    def _note_refusal_fingerprint(self, c: dict, state: str, detail: str, harness: bool = False) -> bool:
        """Record this refusal's shape across candidates; True once it looks SYSTEMIC.

        A gate whose report-discovery patterns cannot see a legitimate report (e.g. a Gradle
        report, TEST-*.xml) refuses good candidates every time, and nothing in a run would
        say "this refusal is systemic" unless a person read trajectories.

        A refusal that every candidate hits is evidence about the harness, not about the
        candidates. This does not change what the gate decides (the gate is still authoritative
        and the strike still counts): it makes the pattern VISIBLE the moment it forms, in the
        candidate's own refusal message and in the run's events, where an operator sees it in
        seconds instead of cycles.
        """
        # The shape of a refusal, not its instance-specific detail: the first sentence with
        # paths, ids and digests stripped, so "no report under /workspace/a/_verify_buggy" and
        # "...under /work/b/_verify_fixed" are one fingerprint.
        # NORMALISE, THEN TRUNCATE. Cutting first made the fingerprint depend on how long each
        # candidate's paths were, so three identical refusals produced three different shapes
        # and the pattern never formed -- the detector failing exactly as the thing it detects.
        # THE FIRST *PROBLEM*, not the header. A gate that opens with "2 problem(s) with the
        # declared variants" fingerprints every one of its refusals identically, so unrelated
        # failures group together and the announcement points at nothing. When the message
        # enumerates its problems, the first one is the shape.
        shape = _refusal_shape(detail)
        fps = self.store.setdefault("refusal_shapes", {})
        entry = fps.setdefault(f"{state}::{shape}", {"candidates": [], "count": 0})
        entry["count"] += 1
        if harness:
            # The harness's own fault (an unloadable validator, a sweep that could not verify
            # itself, an environment that failed). A shape with this flag that goes systemic
            # is the harness convicting itself; see _emit_systemic.
            entry["harness"] = True
        key = next((k for k, cand in self.store["candidates"].items() if cand is c), "")
        if key and key not in entry["candidates"]:
            entry["candidates"].append(key)
        # POOL ACROSS WORKERS. Under fan-out each worker may hold ONE candidate, so a
        # per-store count would cap at 1 and SYSTEMIC_AT=3 would be structurally unreachable.
        # The evidence lives one directory up; so does the pool file. Append-only
        # JSONL (short single-line writes are atomic on POSIX), read back on every refusal.
        pooled = set(entry["candidates"])
        shared = self._shared_shapes_path()
        if shared and key:
            fp = f"{state}::{shape}"
            written = entry.setdefault("pooled_written", [])
            try:
                if key not in written:
                    with open(shared, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"fp": fp, "candidate": key}) + "\n")
                    written.append(key)
                with open(shared, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if rec.get("fp") == fp and rec.get("candidate"):
                            pooled.add(str(rec["candidate"]))
            except OSError:  # the pool is observability; losing it must not change verdicts
                pass
        if len(pooled) < self._knob("systemic_at", self.SYSTEMIC_AT):
            return False
        if not entry.get("announced"):
            entry["announced"] = True
            self._emit_systemic(state, shape, entry, pooled=len(pooled))
        return True

    def systemic_shapes(self, key: str) -> dict:
        """The refusal shapes THIS harness has found systemic across candidates, per state.

        Used by the drop path to tell a blocked booking that disputes a harness-wide fault
        apart from one that merely asserts the gate is wrong: the flag is written by
        _strike from _note_refusal_fingerprint's cross-candidate count, never by an agent.
        """
        return dict((self.get(key) or {}).get("_systemic") or {})

    def _shared_shapes_path(self) -> str | None:
        """The run-level refusal-shape pool, when this worker is part of a fan-out.

        Active only under a fanout parent (fanout.json beside the worker dirs): a solo run's
        own store already sees every candidate, and a file nobody else reads is noise.
        """
        out_dir = str((getattr(self.ctx, "scratch", None) or {}).get("out_dir") or "")
        if not out_dir:
            return None
        parent = os.path.dirname(os.path.abspath(out_dir))
        if not os.path.exists(os.path.join(parent, "fanout.json")):
            return None
        return os.path.join(parent, "refusal_shapes.jsonl")

    def _emit_systemic(self, state: str, shape: str, entry: dict, pooled: int | None = None) -> None:
        """Say it once, loudly, where an operator watching the run will see it."""
        n = pooled if pooled is not None else len(entry["candidates"])
        across = "" if n == len(entry["candidates"]) else " (pooled across this run's workers)"
        msg = (
            f"SYSTEMIC REFUSAL at {state}: {n} unrelated candidates{across} "
            f"have now been refused the same way ({entry['count']} times in total) -- "
            f"{shape!r}. Three unrelated candidates failing one gate identically is evidence "
            f"about the gate or its teaching, not about the candidates. Check the harness "
            f"before spending more cycles here."
        )
        print(f"[worklist] {msg}", flush=True)
        events = self._events()
        try:
            if events is not None:
                # THE SITE, PASSED. `worklist.systemic` is registered and pointed at this
                # function; the emit must name it, or every systemic-refusal record would land
                # with `site: None` and could not be ranked by site.
                events.emit(
                    "on_intervention",
                    site="worklist.systemic",
                    kind="systemic_refusal",
                    state=state,
                    shape=shape,
                    candidates=n,
                    refusals=entry["count"],
                    detail=msg,
                )
        except Exception:  # noqa: BLE001 - a report must never break the run
            pass
        # DETECTED, THEN WHAT. When the systemic shape is one the harness itself authored as
        # a fault -- `_harness_fault` flagged the entry -- the harness has now
        # said two things at once: "my gate is broken" and "every candidate hits it". Nothing
        # the agent records can pass that gate, so every further cycle is paid for nothing.
        # The flag is read by the loop at cycle end, which ends the
        # run with a HarnessFault; it is never raised from here, because a tool's exceptions
        # are caught by the registry and would reach the model as one more refusal.
        # A systemic shape the harness did NOT author (three candidates failing a correct
        # gate the same way) is announced, not acted on.
        if entry.get("harness"):
            self.store["_harness_fault"] = {"state": state, "shape": shape, "candidates": n, "refusals": entry["count"]}

    def _strike(self, c: dict, state: str, detail: str, fresh: bool = True) -> str:
        """Count a refusal at `state` and escalate once a candidate is clearly stuck.

        Per-candidate refusal counts at a single state can be very large (one abbreviated SHA
        that could never resolve can consume more than half of a run). The validator is right
        every time; the run starves anyway, because current() serves the furthest-along
        candidate and a stuck one is served forever.

        `defer()` is the escape, but it needs the MODEL to notice the loop.
        Same lesson as the rest of this design: what depends on the model
        noticing does not happen, so the harness notices.

        Returns the (possibly escalated) message. Escalating matters as much as the counter: a
        refusal that reads identically the 1st and 40th time gives the model no evidence that
        repeating itself is failing.
        """
        strikes = c.setdefault("_strikes", {})
        n = strikes[state] = int(strikes.get(state, 0)) + 1
        # For the refusal record (_note_refusal): which strike this was and what it caused.
        self._last_escalation = {"strike": n, "escalation": "none"}
        systemic = self._note_refusal_fingerprint(c, state, detail)

        # The non-blocking advisor: at the defer threshold --
        # once per state per candidate -- a model reads the state's declarations, the
        # refusal history and the candidate's fields, and its advice rides the deferral
        # and escalation messages. The gate still decides everything; the advisor only
        # shortens the path to satisfying it or honestly abandoning it. Model round-trips
        # dominate wall time, so cycles spent iterating blind between refusals are the
        # scarcest thing a run spends.
        # Never on a REPLAY: the advisor explains a refusal PATTERN, and a replayed verdict
        # is by construction the same evidence it already saw (or would have seen) fresh --
        # a model call per replay is cost without information. The replay note carries the
        # way out instead. `>=` with the once-guard
        # (not `==`): with replays counting, n can pass STUCK_DEFER on replays and a LATER
        # fresh refusal must still be able to gather the advice.
        # ...and once per SHAPE, not once forever: advice cached at the threshold would
        # ride every later refusal verbatim, including one it contradicts (advice that a
        # check script is missing, stapled to "the check PASSED with a no-op binary").
        # Advice about a refusal the candidate no longer gets is disinformation with the
        # harness's voice. Same shape, cached;
        # a refusal of a NEW shape at/past the threshold re-asks with the new evidence.
        shape = _refusal_shape(detail)
        advice_shapes = c.setdefault("_advice_shape", {})
        if (
            fresh
            and n >= self._knob("stuck_defer", STUCK_DEFER)
            and (state not in (c.get("_advice") or {}) or advice_shapes.get(state) != shape)
        ):
            advice = self._advise(c, state, detail)
            if advice:
                c.setdefault("_advice", {})[state] = advice
                advice_shapes[state] = shape
            elif advice_shapes.get(state) != shape:
                # The re-ask failed (provider down, empty reply). The advice on file was
                # written for the OLD shape, and serving it under the new refusal would
                # contradict it: drop it, and a later fresh refusal asks again.
                (c.get("_advice") or {}).pop(state, None)
                advice_shapes.pop(state, None)
        advice = (c.get("_advice") or {}).get(state, "")
        advice_suffix = f" ADVISOR: {advice}" if advice else ""
        # ...AND THE AGENT IS TOLD. The `systemic` finding must reach the candidate's own
        # refusal message, not only operator stdout and the event log, or an agent keeps
        # treating a gate-wide fault as its own defect.
        if systemic:
            # Recorded on the candidate so a later blocked booking can be told apart from
            # an agent merely asserting "the gate is wrong": this flag is the HARNESS's own
            # finding across candidates, and cannot be claimed into existence.
            c.setdefault("_systemic", {})[state] = shape
            advice_suffix += (
                " SYSTEMIC: other candidates are being refused with this same shape. If "
                "your evidence looks right to you, this may be the GATE rather than your "
                "work. There is no free-text channel here, so say it the way the harness "
                "can hear: call worklist_drop with blocked=true and a reason naming what "
                "you believe the gate is getting wrong. Because the harness has already "
                "found this shape across candidates, that booking stays BLOCKED rather "
                "than being counted as your failure, and an advisor reviews the reason. "
                "Otherwise gather something genuinely different, or mine another "
                "candidate: repeating the same evidence will not move this one."
            )

        if n >= self._knob("stuck_block", STUCK_BLOCK):
            self._last_escalation["escalation"] = "terminate"
            # Set the row terminal directly rather than via terminate(), which looks the candidate
            # up by key -- a candidate whose identity fields are incomplete has no stable key and
            # must still be able to reach a terminal state instead of looping forever.
            #
            # REJECTED, not BLOCKED: this branch fires only after N refusals at one state,
            # which is the definitive checks-ran-and-refused case -- evidence about the
            # CANDIDATE, in the sample's denominator. Booking BLOCKED here would violate the
            # drop-time veto's principle through a side door. The one exception is a
            # refusal that self-identifies as the HARNESS's own fault (a validator that
            # could not load, or crashed) -- no check examined the candidate, so BLOCKED
            # stands.
            harness_fault = self._harness_origin_text(detail)
            terminal = BLOCKED if harness_fault else REJECTED
            c["state"] = terminal
            c["fields"]["_reason"] = f"{state} refused {n} times: {detail[:2000]}"
            c["history"].append(f"{terminal}:stuck at {state} after {n} attempts")
            if harness_fault:
                return (
                    f"{detail} -- refused {n} times by a check that could not run, so "
                    f"this candidate is BLOCKED (a pipeline fault, not candidate "
                    f"evidence) and will not be offered again. Report it and move on." + advice_suffix
                )
            return (
                f"{detail} -- refused {n} times now, so this candidate is REJECTED "
                f"(its checks ran and refused it; it stays in the sample's denominator) "
                f"and will not be offered again. That is a recorded outcome, not a "
                f"failure: move on." + advice_suffix
            )

        if n >= self._knob("stuck_defer", STUCK_DEFER):
            self._last_escalation["escalation"] = "defer"
            c["deferred"] = int(c.get("deferred", 0)) + 1
            c["history"].append(f"auto-deferred after {n} refusals at {state}")
            key = next((k for k, cand in self.store["candidates"].items() if cand is c), "")
            head = (
                f"{detail} -- this is attempt {n} at {state} and the same thing has been "
                f"refused each time, so this candidate is deferred"
            )
            if not self._defer_moved(key):
                return head + "." + self._nothing_to_defer_to() + advice_suffix
            return (
                head + " and another is now current. Repeating the same evidence will not "
                "work; either gather something genuinely different, or leave it deferred "
                "and mine other candidates." + advice_suffix
            )
        # The systemic notice does NOT wait for the defer threshold: it is worth most on the
        # first refusal of the third candidate to hit the same shape, before this one spends
        # eight strikes rediscovering what the harness already knows.
        return detail + (advice_suffix if systemic else "")

    def _defer_moved(self, key: str) -> bool:
        """Did deferring `key` actually hand the agent something else?

        `current()` serves the least-deferred ACTIVE candidate, so deferring the only one
        returns it immediately. Asked in one place because the answer is needed in three --
        the `worklist_defer` tool, the strike threshold and the slow-cycle threshold -- and
        the first fix corrected only the tool.
        """
        nxt = self.current()
        return bool(nxt) and nxt[0] != key

    @staticmethod
    def _nothing_to_defer_to() -> str:
        """What to say instead. "Mine other candidates" is not actionable when there are none."""
        return (
            " There is nothing else to work on, so this changed nothing and the same "
            "candidate is still current: deferring only helps once other candidates exist. "
            "Either worklist_add more of them, or worklist_drop this one with the reason "
            "if you believe it cannot be made to work at all."
        )

    def _advise(self, c: dict, state: str, detail: str) -> str:
        """One model call, at the defer threshold, that explains the refusal PATTERN.

        Non-blocking by construction: the return value is prose appended to messages the
        agent already receives; nothing here can accept, reject or advance a candidate.
        A broken or absent model yields no advice and changes nothing else."""
        model = getattr(self.ctx, "model", None) if self.ctx is not None else None
        if model is None:
            return ""
        st = next((s for s in self.spec.states if getattr(s, "name", None) == state), None)
        decls = {}
        if st is not None:
            for k in ("requires", "requires_true", "records", "rederives", "discriminates", "instruction"):
                v = getattr(st, k, None)
                if v:
                    decls[k] = v
        refusals = [h.split(":REFUSED:", 1)[1] for h in c.get("history", []) if h.startswith(f"{state}:REFUSED:")] + [
            str(detail)
        ]
        fields = {k: str(v)[:2000] for k, v in (c.get("fields") or {}).items() if not k.startswith("_")}
        prompt = (
            "You advise a builder agent inside a benchmark-construction harness. The agent "
            f"is stuck at state {state}, refused {len(refusals)} time(s) by a mechanical "
            "check that is authoritative and will not be bypassed. Read the state's "
            "declarations, the refusal messages and the recorded fields, then explain the "
            "most likely reason the refusals KEEP happening and the single most concrete "
            "next action. If the refusals indicate the check itself is measuring wrong "
            "(a harness fault), say so plainly and advise deferring. At most 120 words. "
            "First line exactly 'FAULT: candidate' or 'FAULT: method' or 'FAULT: harness'.\n\n"
            f"STATE DECLARATIONS:\n{decls}\n\nREFUSALS (oldest first):\n"
            + "\n".join(f"- {r[:400]}" for r in refusals[-6:])
            + f"\n\nRECORDED FIELDS:\n{fields}"
        )
        try:
            resp = model.query([{"role": "user", "content": prompt}])
            return (getattr(resp, "content", "") or "").strip()[:2000]
        except Exception:
            return ""

    def validation_error(self, key: str) -> str | None:
        """The refusal HOLDING a candidate, so the model can be told what to fix.

        In SPEC order, not `_checked`'s dict order. `_recompute` breaks at the
        first refusing state, and a re-run verdict is re-inserted LAST: after a candidate
        falls back because an earlier gate re-ran on a changed input and refused, dict order
        would serve the stale refusal of a LATER state while the render's `_last_refusal` (spec
        order) named the earlier one -- two readers of one cache disagreeing about which
        check the agent has to satisfy.
        """
        c = self.store["candidates"].get(key) or {}
        checked = c.get("_checked") or {}
        names = [s.name for s in self.spec.states]
        for name in names + [n for n in checked if n not in names]:
            entry = checked.get(name)
            if entry and not entry[0]:  # [ok, served] or [ok, served, base]
                return f"{name}: {entry[1]}"
        return None

    def recheck(self, key: str) -> None:
        """Drop cached verdicts so re-recorded evidence gets re-verified."""
        c = self.store["candidates"].get(key)
        if c:
            c.pop("_checked", None)
            self._recompute(key)

    # ── queries ──────────────────────────────────────────────────────────────
    def get(self, key: str) -> dict | None:
        return self.store["candidates"].get(key)

    def active(self) -> list[tuple[str, dict]]:
        """Candidates still in play, in insertion order."""
        term = set(self.spec.terminal or []) | {REJECTED, BLOCKED}
        return [
            (k, self.store["candidates"][k])
            for k in self.store["order"]
            if self.store["candidates"][k]["state"] not in term
        ]

    def current(self) -> tuple[str, dict] | None:
        """The candidate the model should work on now.

        Deliberately the *furthest-along* active candidate, not the first. Finishing one
        candidate is worth more than nudging twenty forward by one step each, and it directly
        counters the observed thrash between projects.
        """
        act = self.active()
        if not act:
            return None
        idx = {s.name: i for i, s in enumerate(self.spec.states)}
        # Least-deferred first, then furthest-along: finishing a candidate still beats spreading
        # effort, but a deferred one yields to others until they have been tried too.
        return min(act, key=lambda kv: (int(kv[1].get("deferred", 0)), -idx.get(kv[1]["state"], -1)))

    def tick_current(self) -> str:
        """Charge ONE cycle to whichever candidate is current. Returns a nudge, or "".

        Called once per cycle from the loop -- the only place that knows a cycle happened. The
        counter is per (candidate, state): advancing resets it, because progress is exactly what
        it is measuring the absence of. Time spent is not counted against a candidate that is
        moving, however long each step takes.
        """
        return self.tick()[0]

    def tick(self) -> "Nudge":
        """`tick_current`, plus whether the nudge is a REPEAT and which SITE said it.

        A repeat is the same candidate (or the same empty list) re-nudged at a later multiple
        of slow_nudge. The first nudge and a deferral are the consequence of the model's last
        reply; a repeat is a status line, and the loop records it without claiming the
        echoed-turn slot (a repeat would otherwise replace its tool output). The site is the registered emitter id for the log."""
        slow = self._knob("slow_nudge", SLOW_NUDGE)
        cur = self.current()
        if cur is None:
            msg = self._tick_empty()
            return Nudge(
                msg,
                bool(msg) and int(self.store.get("_empty_cycles") or 0) > slow,
                "worklist.empty.nudge" if msg else "",
            )
        self.store["_empty_cycles"] = 0
        key, c = cur
        # Reset on advance. Stored alongside the count so a resumed run cannot lose the pairing
        # and start charging a fresh state with an old candidate's total.
        if c.get("_cycles_state") != c["state"]:
            c["_cycles_state"] = c["state"]
            c["_cycles"] = 0
            # Progress re-arms the slow-defer: one-shot per (candidate,
            # state), not per lifetime -- a candidate that advanced and then stalled
            # again would otherwise monopolize the run with nothing but nudges.
            c.pop("_slow_deferred", None)
        n = c["_cycles"] = int(c.get("_cycles") or 0) + 1
        state = c["state"]

        if n >= self._knob("slow_defer", SLOW_DEFER) and not c.get("_slow_deferred"):
            c["_slow_deferred"] = True
            self.defer(key)
            c["history"].append(f"auto-deferred after {n} cycles at {state}")
            head = (
                f"{key} has now held your attention for {n} cycles at {state} without "
                f"reaching the next state, so it has been DEFERRED"
            )
            if not self._defer_moved(key):
                return Nudge(head + "." + self._nothing_to_defer_to(), False, "worklist.slow.deferred")
            return Nudge(
                head + " and another candidate is current. Nothing is lost -- every field "
                "it has proven is kept and it will be offered again once the others have "
                "been tried. If you believe it cannot be made to work at all, "
                "worklist_drop it with the reason; otherwise leave it and mine "
                "something else.",
                False,
                "worklist.slow.deferred",
            )
        if n >= slow and n % slow == 0:
            return Nudge(
                f"{key} has been the current candidate for {n} cycles and is still at "
                f"{state}. That is a long time for one candidate. Either do something "
                f"materially different from what the last {slow} cycles tried, or "
                f"worklist_defer it and work on another -- repeating the same approach has "
                f"not moved it.",
                n > slow,
                "worklist.slow.nudge",
            )
        return Nudge("", False, "")

    def _tick_empty(self) -> str:
        """Charge a cycle to having NO candidate at all. Returns a nudge, or "".

        The sibling of the slow-candidate counter above, for the case it cannot see. A candidate
        nothing, so a long run of these cycles is the more expensive of the two and the
        one nothing else watches. Same threshold, because the harness already calls
        now the only one nothing watched. Same threshold, because the harness already calls
        SLOW_NUDGE cycles without reaching the next state too long, and this is that with less
        to show for it.
        """
        n = self.store["_empty_cycles"] = int(self.store.get("_empty_cycles") or 0) + 1
        slow = self._knob("slow_nudge", SLOW_NUDGE)
        if n < slow or n % slow:
            return ""
        first = self.spec.states[0]
        wants = ", ".join(first.requires or []) or "the fields it names"
        return (
            f"You have gone {n} cycles with no candidate in the worklist at all. Whatever "
            f"you have been examining, the harness cannot see it: a candidate that is "
            f"registered keeps every field it proves, is carried across a resume, and is "
            f"walked forward by the checks -- work done outside the worklist is none of "
            f"that, and is lost when this run ends. Registering is not a claim that the "
            f"candidate is good; {first.name}, the first state, asks only for {wants}. "
            f"worklist_add it as soon as you have those and prove the rest afterwards, so "
            f"that what you have already established survives."
        )

    def unmet(self, key: str) -> list[str]:
        """Fields still needed to leave the current state -- using the SAME `_absent` test the
        state machine uses.

        A bare `r not in fields` check would let an empty value record as present, producing the
        self-contradictory nudge "still DISCOVERED ... it still needs: (nothing)" with no
        validator message either: a dead end that reads as a harness bug.
        """
        c = self.store["candidates"].get(key)
        if c is None:
            return []
        nxt = self.next_state_after(c["state"])
        if nxt is None:
            return []
        decls = self.computes()
        out: list[str] = []
        # A COMPUTED field is never asked for by name. Naming it is how an agent ends up
        # implementing a median in a container shell: it was told the field was missing, so it
        # produced one. Ask for the observations instead -- the thing it can actually go and
        # measure -- and ask once per source, however many quantities are computed from it.
        awaiting: dict[tuple[str, ...], list[str]] = {}
        for r in nxt.requires or []:
            if not _absent(c["fields"].get(r)):
                continue
            srcs = derive.sources_of(decls[r]) if r in decls else []
            if not srcs:
                out.append(r)
                continue
            missing = tuple(s for s in srcs if _absent(c["fields"].get(s)))
            if missing:
                awaiting.setdefault(missing, []).append(r)
            else:
                why = "; ".join(self.derive_problems(key)) or "see the note on your last record"
                out.append(f"{r} (the harness computes this from {', '.join(srcs)} and could not: {why})")
        for missing, derived in awaiting.items():
            out.append(
                f"{' and '.join(missing)} (raw observations -- the harness computes "
                f"{' and '.join(derived)} from {'them' if len(missing) > 1 else 'it'}; "
                f"do not compute {'them' if len(derived) > 1 else 'it'} yourself)"
            )
        # ASK ONCE. A field can reach BOTH branches above: required in its own right, and the
        # source of something computed (e.g. `timings_ms`). The record envelope -- which
        # renders this list as the LITERAL JSON reply to send -- would then show it as two
        # keys, the second of them a long sentence in the position of a field name. Showing
        # the exact thing to send is the point of that block, so a malformed object defeats it.
        # The annotated form wins because it carries the name AND the reason not to do the
        # arithmetic, so the bare copy loses nothing.
        annotated = {
            src
            for entry in out
            if " (raw observations" in entry
            for src in re.split(r"\s+and\s+|,\s*", entry.split(" (raw observations", 1)[0])
            if src
        }
        return [e for e in out if e not in annotated]

    def defer(self, key: str) -> None:
        """Send a candidate to the back of the queue without discarding its progress.

        current() serves the FURTHEST-ALONG candidate, so one stuck near the end (e.g. a real
        bug-fix commit with no issue URL) was served forever and every other candidate became
        unreachable. The only escape was dropping it and losing six completed steps.
        """
        c = self.store["candidates"].get(key)
        if c is not None:
            # BEHIND EVERY OTHER ACTIVE CANDIDATE, not one notch back. current() serves the
            # least-deferred candidate and breaks ties by progress, so "+1" on the candidate
            # that is furthest along could tie with the others and hand the same candidate
            # straight back, with a reply saying it was the only one.
            # Deferring means: the others first.
            others = [int(o.get("deferred", 0)) for k, o in self.active() if k != key]
            c["deferred"] = max([int(c.get("deferred", 0))] + others) + 1
            c["history"].append("deferred")

    def next_state_after(self, state: str):
        names = [s.name for s in self.spec.states]
        if state not in names:
            return None
        i = names.index(state)
        return self.spec.states[i + 1] if i + 1 < len(self.spec.states) else None

    def histogram(self) -> dict[str, int]:
        """Where the run's candidates actually are. The artifact that turns 'it
        underperformed' into 'they all died at SOURCE_READY'."""
        out: dict[str, int] = {}
        for k in self.store["order"]:
            st = self.store["candidates"][k]["state"]
            out[st] = out.get(st, 0) + 1
        return out

    def delivered(self) -> list[dict]:
        final = (self.spec.terminal or [None])[0]
        return [
            self.store["candidates"][k]["fields"]
            for k in self.store["order"]
            if self.store["candidates"][k]["state"] == final
        ]


# Every spelling of "no" a model actually sends. These agents speak JSON through a text
# protocol and quote their booleans routinely, so `requires_true` must not accept the STRING
# "false" as true: only the JSON literal `false` and the int 0 would reject. A candidate
# honestly reporting {"build_ok": "false"} must not advance on a self-graded state with no
# validator behind it.
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "off", "none", "null", "nil", "f"})


def _is_false(v: Any) -> bool:
    """Did the model say NO? Recognised negatives only.

    Deliberately not a whitelist of affirmatives: live records carry values such as `sha_resolved: "ok"` and
    `is_bug_fix: "yes -- ISSUE-123"`, and refusing those would break honest work. Anything
    that is not a recognised negative and not empty is the model saying something affirmative.
    """
    if isinstance(v, str):
        return v.strip().lower() in _FALSE_WORDS
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return not v
    return not v


def _absent(v: Any) -> bool:
    """Is a required field effectively missing?

    Empty containers and blank strings count as absent: an instance carrying
    `triggering_test: []` is exactly as unusable as one that omits the field, and treating it as
    present is how empty-but-well-shaped records could pass the gates.
    """
    if v is None:
        return True
    if isinstance(v, (str, list, tuple, dict, set)):
        return len(v) == 0
    return False
