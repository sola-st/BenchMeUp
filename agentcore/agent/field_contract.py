"""Declared field contracts: a task states its lead-field rules in its spec; this
module enforces them without knowing what any field means.

WHY THIS EXISTS: no task-specific gates written as code. Task-coded checks collapse into a
handful of structural families -- the same rule re-typed with different constants (a URL
field must look like a printed clone URL; an identity field must be atomic; a command must
invoke the program it measures). Constants-in-copies drift. Task content belongs in the spec, where it is versioned, regenerated and readable;
this module is the one place the SHAPES are implemented.

A CONTRACT is a list of clauses, each `{field, require, ...}`:

    field    the lead field the clause reads
    require  the predicate, in a small grammar (below)
    resolve  optional {spelling: value} map applied BEFORE the predicate; the resolved
             value is written back to the record (resolve-don't-refuse: a build FILE
             name states its build system mechanically)
    teach    optional sentence appended to the refusal (the worked example). {field}
             references interpolate the lead's own recorded values, so a refusal can
             carry its answer ("record it as {org}/{repo}"); unknown names stay
             literal
    optional true = an empty value passes this clause (the field is conditional;
             presence is someone else's rule)

The requires this grammar knows, beyond everything `admissibility.parse` accepts
(one of [...], comparisons, recency, booleans):

    is non-empty
    is a clone url               https?://... or scp-style with '@'
    is a full http url           a tracker link, never '#123' or a bare key
    is a git sha                 7-40 hex chars
    is an atomic name            contains no '/'
    is a qualified name          contains a '/' (owner/name)
    is a relative path to one file
    is one line naming no absolute path   no newline; no '/'-rooted token ('/'
        inside a relative path is fine)
    is a positive integer
    is not a pull request url
    is a command invoking {other_field}
    is distinct across leads
    resolves to a tracker item that is not a pull request   (live fetch; a fetch
        that fails stays transient and is re-judged where the next state fetches)

Universal TEXT conventions live here by the same precedent as admissibility's
true/false spellings: what a shell command line looks like, which flags are
version/help banners, what an http URL is. Which FIELDS exist, which vocabularies
apply and every threshold are spec content and arrive in the clauses. A conformance
test pins that this file contains no task words.
"""

from __future__ import annotations

import os
import re
from typing import Any

from agentcore.agent import admissibility as _adm

_URL = re.compile(r"^https?://\S+$")
_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_PATHISH = re.compile(r"^[\w./+-]{1,200}$")
# A '/'-ROOTED token: '/' at start-of-string or right after a shell delimiter opens
# an absolute path. A '/' INSIDE a token (utilities/magick) is a relative path's own
# separator and never refuses -- git-peel caution, only ROOTED tokens count.
_ABS_ROOTED = re.compile(r"(?:^|[\s;|&('\"=])(/[^\s;|&)'\"]*)")
_FIELD_REF = re.compile(r"^is a command invoking \{([a-z][a-z0-9_]*)\}$")
# A hex digest of a standard width: the record holds a real hash of real bytes.
_HEX_DIGEST = re.compile(r"^[0-9a-f]{32,128}$")
# The spellings a false value actually arrives in (same family as admissibility's).
_FALSEY = frozenset({"", "false", "no", "n", "0", "off", "none", "null", "failed"})


def _instance_text(ctx) -> str:
    """Everything the instance's spec says, as one string, for reading a stated
    threshold or token back out. The keys are the instance schema every task
    shares, not any one task's vocabulary."""
    inst = getattr(ctx, "instance", {}) or {}
    parts = [str(inst.get(k) or "") for k in ("criteria", "task", "task_description")]
    for k in ("criteria_list", "quality_gates_list"):
        v = inst.get(k)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
    return " ".join(parts)


def _stated_number(ctx, pattern: str, default=None):
    """A numeric bound THE SPEC states, or `default` when it states none. A copy of
    the number in code is a second source of truth that drifts; reading it back
    means the spec is the only place the number lives."""
    if not pattern:
        return default
    m = re.search(pattern, _instance_text(ctx), re.I)
    if not m:
        return default
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return default


def _stated_token(ctx, spec: dict) -> str:
    """What the spec declares as its reference value for a declared token grammar:
    the literal "none" when the spec's none-statement matches (read FIRST -- a spec
    that says its reference carries no such token means that, whatever tokens it
    goes on to mention as targets), else the first declared token inside a sentence
    the declared statement-pattern marks, else "" (the spec states nothing, and
    nothing is invented for it here)."""
    text = _instance_text(ctx)
    none_rx = str(spec.get("none_pattern") or "")
    if none_rx and re.search(none_rx, text, re.I):
        return "none"
    token_rx = str(spec.get("token_pattern") or "")
    stmt_rx = str(spec.get("statement_pattern") or "")
    if not token_rx or not stmt_rx:
        return ""
    for sentence in re.split(r"[.;\n]", text):
        if re.search(stmt_rx, sentence, re.I):
            m = re.search(token_rx, sentence)
            if m:
                return m.group(0)
    return ""


# Flags that ask a binary to print metadata about itself instead of doing work; a
# sole-argument -v/-V/version/help is the same ask spelled differently.
_BANNER_FLAGS = ("--version", "-version", "--help", "--usage")
_BANNER_SOLE = (["version"], ["help"], ["-v"], ["-V"])
_SHELLY = "/|<>=$'\""


def _non_empty(value: Any) -> bool:
    return bool(str(value or "").strip())


class _Literal(dict):
    def __missing__(self, key):  # unknown {name} stays literal, never a KeyError
        return "{" + key + "}"


def _refuse(clause: dict, why: str, fields: dict | None = None) -> tuple[bool, str]:
    teach = str(clause.get("teach") or "").strip()
    if teach and fields:
        try:
            teach = teach.format_map(_Literal({k: str(v) for k, v in fields.items()}))
        except Exception:  # noqa: BLE001 - a malformed teach must not eat the refusal
            pass
    return False, (why + (" " + teach if teach else ""))


def _command_verdict(
    clause: dict, cmd: str, target_name: str, target_value: str, fields: dict
) -> tuple[bool, str] | None:
    """Everything a command-line field must satisfy: it invokes the named value in
    invocation position, reads as a command rather than prose, and asks for work
    rather than a banner. Returns None when all hold."""
    words = cmd.split()
    if target_value:
        want = target_value.lower()
        toks = [t.strip("'\"") for t in words]
        i = 0
        while i < len(toks) and "=" in toks[i] and "/" not in toks[i].split("=", 1)[0]:
            i += 1  # skip leading VAR=value env assignments
        first = toks[i].lower() if i < len(toks) else ""
        basenames = [t.lower().rstrip("/").rsplit("/", 1)[-1] for t in toks if "/" in t]
        if want not in first and not any(want in b for b in basenames):
            return _refuse(
                clause,
                fields=fields,
                why=(
                    f"{clause['field']} {cmd[:80]!r} never INVOKES {target_value!r} (the "
                    f"recorded {target_name}) -- its name must sit in the invocation "
                    f"position: the first token, or a path token like ./src/{target_value}, "
                    f"not inside a quoted argument."
                ),
            )
    shelly = any(w.startswith("-") or any(ch in w for ch in _SHELLY) for w in words)
    if len(words) >= 8 and not shelly:
        return _refuse(
            clause,
            fields=fields,
            why=(f"{clause['field']} {cmd[:80]!r} reads as a prose sentence, not a command line."),
        )
    args_l = [t.lower().strip("'\"") for t in words[1:]]
    if any(t in _BANNER_FLAGS for t in args_l) or args_l in _BANNER_SOLE:
        return _refuse(
            clause,
            fields=fields,
            why=(
                f"{clause['field']} {cmd[:80]!r} asks for a version/help banner -- a "
                f"banner prints metadata, not work, so there is no runtime for a "
                f"measurement to hold onto. Name the command that makes "
                f"{target_value or 'the binary'} do its real work on an input you saw "
                f"or can generate."
            ),
        )
    return None


def _others(ctx, fields: dict, fname: str) -> list[str]:
    """The same field's values on every other lead in the working set; [] when no
    working set is reachable -- a broken snapshot must never block registration."""
    try:
        ws = getattr(getattr(ctx, "agent", None), "working_set", None) or getattr(ctx, "working_set", None)
        snap = ws.snapshot() if hasattr(ws, "snapshot") else {}
        cands = ((snap.get("worklist") or {}).get("candidates") or {}).values()
        vals = [
            str((c.get("fields") or {}).get(fname) or "").strip()
            for c in cands
            if (c.get("fields") or {}) is not fields
        ]
        return [v for v in vals if v]
    except Exception:  # noqa: BLE001
        return []


def evaluate_clause(clause: dict, fields: dict, ctx=None) -> tuple[bool, str]:
    """(ok, why) for one declared clause against one lead's fields. Fails closed on a
    require this grammar cannot read -- an unreadable rule enforced as a silent pass
    is how a declared control stops existing."""
    fname = str(clause.get("field") or "").strip()
    if not fname:
        return False, "a contract clause must name its `field`"
    require = " ".join(str(clause.get("require") or "").strip().split())
    value = str(fields.get(fname) if fields.get(fname) is not None else "").strip()

    resolve = clause.get("resolve") or {}
    if resolve and value:
        key = value.lower().rstrip("/").rsplit("/", 1)[-1]
        resolved = resolve.get(key)
        if resolved is not None and str(resolved) != value:
            fields[fname] = value = str(resolved)

    low = require.lower()

    if low == "is non-empty":
        if not value:
            return _refuse(clause, fields=fields, why=f"record {fname} -- it must be observed and written down.")
        return True, ""

    # every remaining predicate needs a recorded value first
    if not value:
        if clause.get("optional"):
            return True, ""
        return _refuse(
            clause,
            fields=fields,
            why=f"record {fname}: nothing recorded, and {require!r} cannot hold for an empty value.",
        )

    if low == "is a clone url":
        if not _URL.match(value) and "@" not in value:
            return _refuse(
                clause,
                fields=fields,
                why=f"{fname} must be the clone URL as a real source printed it (got {value[:60]!r}).",
            )
        return True, ""

    if low == "is a full http url":
        if not _URL.match(value):
            return _refuse(
                clause,
                fields=fields,
                why=f"{fname} {value[:60]!r} must be a full http(s) link, never a bare number or key.",
            )
        return True, ""

    if low == "is a git sha":
        if not _SHA.match(value.lower()):
            return _refuse(
                clause,
                fields=fields,
                why=f"{fname} {value[:40]!r} must be a git hash exactly "
                f"as real output printed it -- never one you did not see.",
            )
        return True, ""

    if low == "is an atomic name":
        if "/" in value:
            return _refuse(clause, fields=fields, why=f"{fname} {value[:60]!r} must name one thing, with no '/'.")
        return True, ""

    if low == "is a qualified name":
        if "/" not in value:
            return _refuse(clause, fields=fields, why=f"{fname} {value[:60]!r} must be the qualified owner/name form.")
        return True, ""

    if low == "is a relative path to one file":
        if not _PATHISH.match(value) or value.startswith("/"):
            return _refuse(
                clause,
                fields=fields,
                why=f"{fname} {value[:60]!r} must be a relative path "
                f"inside the repository, exactly as a real listing "
                f"printed it -- never a path you did not see.",
            )
        if value.rstrip("/") in ("", "."):
            return _refuse(
                clause,
                fields=fields,
                why=f"{fname} {value!r} is the repository root, which "
                f"is not one artifact -- it is however many the "
                f"repository holds.",
            )
        return True, ""

    if low == "is one line naming no absolute path":
        # The staged-replay convention (declared per-task via this clause): the
        # consumer replays this value inside ITS OWN copy of the tree, so the value
        # must be one line (a multi-line value renders to a file path that cannot
        # execute) with every path relative to that copy (a '/'-rooted token edits
        # a tree the consumer never reads).
        if "\n" in value or "\r" in value:
            return _refuse(
                clause,
                fields=fields,
                why=(
                    f"{fname} {value[:60]!r} spans more than one line -- record it as "
                    f"exactly ONE line (join steps with && or ;)."
                ),
            )
        rooted = _ABS_ROOTED.search(value)
        if rooted:
            return _refuse(
                clause,
                fields=fields,
                why=(
                    f"{fname} {value[:60]!r} roots {rooted.group(1)[:40]!r} at '/' -- "
                    f"every path in it must be relative to the tree it is replayed in, "
                    f"never an absolute path."
                ),
            )
        return True, ""

    if low == "is a plain absolute path":
        # a file:// URI is the same path in an envelope -- resolve it, never refuse it
        if value.startswith("file://"):
            value = value[len("file://") :]
            fields[fname] = value
        if not _SAFE_PATH.match(value):
            return _refuse(
                clause,
                fields=fields,
                why=(
                    f"record {fname} as a plain absolute path (letters, digits, '.', '_', "
                    f"'-', '/' only) -- got {value[:60]!r}."
                ),
            )
        return True, ""

    _ratio = _RATIO_CLAUSE.match(low)
    if _ratio:
        # A numeric relation between two recorded fields, declared in the spec and
        # evaluated blind here: a task may demand a runtime that GROWS with input size,
        # and testing that at LEAD time -- before a clone and a build are paid for --
        # needs "field A >= N x field B".
        factor = float(_ratio.group(1))
        other = _ratio.group(2)
        other_val = str(fields.get(other) or "")
        try:
            a, b = float(value), float(other_val)
        except (TypeError, ValueError):
            return _refuse(
                clause,
                fields=fields,
                why=(
                    f"{fname} ({value[:30]!r}) and {other} ({other_val[:30]!r}) must both "
                    f"be recorded numbers before their ratio can prove anything."
                ),
            )
        if b <= 0 or a < factor * b:
            shown = int(factor) if factor.is_integer() else factor
            return _refuse(
                clause,
                fields=fields,
                why=(
                    f"record {fname} at least {shown} times {other} -- got {a:g} vs "
                    f"{b:g}. A value that does not scale past its baseline proves the "
                    f"relation this spec demands does not hold for this candidate."
                ),
            )
        return True, ""

    if low == "is a number":
        try:
            float(value)
        except (TypeError, ValueError):
            return _refuse(clause, fields=fields, why=f"{fname} {value[:30]!r} must be a number.")
        return True, ""

    if low == "is affirmed":
        if value.lower() in _FALSEY:
            return _refuse(clause, fields=fields, why=(f"{fname} is not set to a true value (got {value[:30]!r})."))
        return True, ""

    if low == "is a hex digest":
        if not _HEX_DIGEST.match(value.lower()):
            return _refuse(clause, fields=fields, why=(f"{fname} {value[:40]!r} must be a hex hash of real output."))
        return True, ""

    if low == "is a hex digest of non-empty output":
        from agentcore.agent.digests import is_empty_digest

        if not _HEX_DIGEST.match(value.lower()):
            return _refuse(clause, fields=fields, why=(f"{fname} {value[:40]!r} must be a hex hash of real output."))
        if is_empty_digest(value.lower()):
            return _refuse(clause, fields=fields, why=(f"{fname} is the digest of EMPTY output, which proves nothing."))
        return True, ""

    if low == "is at least the spec-stated minimum":
        spec = clause.get("spec") or {}
        bound = _stated_number(ctx, str(spec.get("pattern") or ""), spec.get("default"))
        if bound is None:
            return True, ""  # the spec states no bound, so none is enforced
        # `scale` is the DECLARED conversion between the unit the spec's sentence uses and
        # the unit the field records ("at least 1 second" on a millisecond field:
        # scale 1000). Declared by the formalizer, which sees both; never read off a
        # field's name. It applies to the number the pattern captured, not to a default,
        # which is already written in the field's unit.
        pattern_hit = bool(spec.get("pattern")) and _stated_number(ctx, str(spec.get("pattern")), None) is not None
        if pattern_hit:
            try:
                bound = float(bound) * float(spec.get("scale") or 1)
            except (TypeError, ValueError):
                pass
        try:
            got = float(value)
        except (TypeError, ValueError):
            got = -1.0
        if got < float(bound):
            return _refuse(
                clause,
                fields=fields,
                why=(
                    f"record {fname} >= {int(bound) if float(bound).is_integer() else bound}"
                    f" -- the number this benchmark's spec asks for "
                    f"(got {value[:30] or 'nothing'!r})."
                ),
            )
        return True, ""

    if low == "is a plain container name":
        if not _SAFE_NAME.match(value):
            return _refuse(
                clause,
                fields=fields,
                why=(f"{fname} must be the plain name of a container you created (got {value[:60]!r})."),
            )
        return True, ""

    if low.startswith("matches "):
        pattern = require[len("matches ") :]
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return False, (f"contract clause for {fname} declares an unreadable pattern ({e}) -- fix the spec.")
        if not rx.search(value):
            return _refuse(clause, fields=fields, why=(f"{fname} {value[:60]!r} does not match the declared shape."))
        return True, ""

    if low == "is a positive integer":
        try:
            if int(value) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return _refuse(clause, fields=fields, why=f"{fname} {value[:30]!r} must be a positive integer.")
        return True, ""

    if low == "is not a pull request url":
        from agentcore.tools.evidence import is_pull_request_url

        if is_pull_request_url(value):
            return _refuse(
                clause,
                fields=fields,
                why=f"{fname} {value[:80]!r} is a pull request link -- a pull request is the change, not the report.",
            )
        return True, ""

    m = _FIELD_REF.match(low)
    if m:
        target = m.group(1)
        bad = _command_verdict(clause, value, target, str(fields.get(target) or "").strip(), fields)
        return bad if bad is not None else (True, "")

    if low == "resolves to a tracker item that is not a pull request":
        # An /issues/N link can be a PR in disguise: GitHub's issues endpoint answers
        # for PR numbers. The fetch is memoised per URL for the run; an unreachable
        # tracker stays transient -- only a CONFIRMED pull refuses here.
        from agentcore.tools import evidence

        def _sh(cmd, _ctx=ctx):
            env = getattr(_ctx, "env", None)
            if env is None:
                return 127, "no environment available"
            r = env.execute(cmd, timeout=180)
            return r.returncode, (r.output or "") + (r.stderr or "")

        info, _detail = evidence.read_tracker_item(value, _sh)
        if isinstance(info, dict) and info.get("exists") and info.get("kind") == "pull":
            return _refuse(
                clause,
                fields=fields,
                why=(
                    f"{value} is a PULL REQUEST wearing an /issues/ URL -- the item itself "
                    f"says it is a PR. A PR is the change, not the report: record the ISSUE "
                    f"it closes; if there is none, this lead does not qualify and no clone "
                    f"will change that."
                ),
            )
        return True, ""

    if low == "is distinct across leads":
        others = _others(ctx, fields, fname)
        dupes = sum(1 for o in others if o == value)
        if dupes >= 2 and dupes * 2 >= len(others):
            return _refuse(
                clause,
                fields=fields,
                why=f"{fname} is identical on {dupes} earlier lead(s) "
                f"({value[:80]!r}) -- that is a template, not "
                f"something you observed about THIS one.",
            )
        return True, ""

    pred = _adm.parse(require)
    if pred is not None:
        ok, why = _adm.evaluate(pred, value)
        return (True, "") if ok else _refuse(clause, f"{fname}: {why}")

    return False, (
        f"contract clause for {fname} states {require!r}, which this "
        f"grammar cannot read -- an unreadable rule must be fixed in the "
        f"spec, not silently skipped."
    )


def evaluate(clauses: list, fields: dict, ctx=None) -> tuple[bool, str]:
    """First failing clause wins; all passing reads as the one acceptance sentence."""
    for clause in clauses or []:
        if not isinstance(clause, dict):
            return False, f"a contract clause must be an object, got {str(clause)[:60]!r}"
        ok, why = evaluate_clause(clause, fields, ctx)
        if not ok:
            return False, why
    return True, "the declared contract holds"


def derived_fields(inst: dict) -> set[str]:
    """Every field some corroboration step DERIVES (run/timed `derive` and `save`, a
    marker_search's derive, a tracker's kind_into): the harness's numbers and names, never
    the miner's to type. An entry clause over one of these is judged after the derivation
    (corroboration_gate), never at the door (entry_gate): judged at the door
    it would demand typed values, and the derived ones would never be re-judged."""
    out: set[str] = set()
    for st in (inst or {}).get("corroboration_contract") or []:
        if not isinstance(st, dict):
            continue
        for k in ("derive", "save"):
            v = st.get(k)
            if isinstance(v, str) and v.strip():
                out.add(v.strip())
            elif isinstance(v, (list, tuple)):
                out.update(str(x).strip() for x in v if str(x).strip())
        ms = st.get("marker_search")
        if isinstance(ms, dict) and str(ms.get("derive") or "").strip():
            out.add(str(ms["derive"]).strip())
        if isinstance(st.get("tracker"), str):
            into = str(((st.get("policy") or {}).get("kind_into")) or "").strip()
            if into:
                out.add(into)
    return out


def _timed_provenance(inst: dict, fields: dict, fname: str) -> str:
    """Where a refused derived number came from: the exact command the harness
    timed, where, how many times, and the median it derived -- so a miner reading "got 2"
    sees what ran, and can tell a harness bug from a command that ran on nothing."""
    for st in (inst or {}).get("corroboration_contract") or []:
        if not (isinstance(st, dict) and "timed" in st and str(st.get("derive") or "") == fname):
            continue
        try:
            cmd = _render_cmd(str(st["timed"]), fields)
            cwd = _render_cmd(str(st.get("cwd") or ""), fields) if st.get("cwd") else "(the harness's cwd)"
        except KeyError:
            return ""
        reps = st.get("repetitions") or 3
        got = fields.get(fname)
        return (
            f" HOW THE HARNESS GOT {fname} = {got} ms: it ran {cmd} in {cwd}, {reps} times, "
            f"wall-clock taken inside your container, and derived the median. A command "
            f"that finishes in milliseconds on a large input is not consuming that input "
            f"-- the recorded command must read the file the recipe produced (name its "
            f"path in the command), and the recipe must write that same path."
        )
    return ""


def _clauses_over_derived(inst: dict) -> tuple[list, list]:
    """(clauses the door judges, clauses the corroboration gate judges after its steps)."""
    derived = derived_fields(inst)
    door, after = [], []
    for cl in (inst or {}).get("entry_contract") or []:
        if isinstance(cl, dict) and str(cl.get("field") or "").strip() in derived:
            after.append(cl)
        else:
            door.append(cl)
    return door, after


def entry_gate(fields: dict, ctx) -> tuple[bool, str]:
    """The one entry-state gate every task shares: the generic prelude (admitted
    source pool, diversity caps, the exclusion roster over the fields the instance
    names) followed by the instance's declared `entry_contract`. Wire it as
    `verify: agentcore.agent.field_contract:entry_gate`; there is no per-task code
    to write, and a task that declares no contract FAILS CLOSED -- a missing
    declaration must read as "declare it", never as an open door."""
    from agentcore.agent.diversity import check_caps
    from agentcore.agent.exclusions import check_candidate
    from agentcore.agent.sources import from_an_admitted_source

    inst = getattr(ctx, "instance", {}) or {}

    ok, why = from_an_admitted_source(fields, ctx)
    if not ok:
        return False, why
    ok, why = check_caps(fields, ctx)
    if not ok:
        return False, why
    for fname in inst.get("exclusions_on") or []:
        value = str(fields.get(fname) or "").strip()
        if value:
            ok, why = check_candidate(value, inst)
            if not ok:
                return False, why

    clauses = inst.get("entry_contract") or []
    if not clauses:
        return False, (
            "HARNESS/SPEC FAULT: this task's instance declares no "
            "`entry_contract`, so no lead can be judged. Declare the "
            "entry contract in the spec (it flows through the derived "
            "brief); an undeclared contract is a missing rule, not a "
            "permissive one."
        )
    # a clause over a field the corroboration steps derive is judged after the derivation,
    # by corroboration_gate -- at the door there is nothing honest to judge yet
    door, _after = _clauses_over_derived(inst)
    return evaluate(door, fields, ctx)


# ── corroboration: declared steps the harness EXECUTES ───────────────────────
#
# The entry contract judges shapes offline; corroboration re-runs the model's
# claims against the world -- a declared command in the agent's environment, a
# declared classifier over a diff, a declared policy over a live tracker item.
# Same rule as above: every command template, marker family, pattern and word
# list arrives in the declaration; this code knows only git-agnostic mechanics
# (run, parse lines, match, derive) and the universal tracker shape.
#
# A step is one of:
#   {field, require, ...}                  an offline clause (evaluate_clause)
#   {run, require?, save?, derive?, optional?, teach?}
#       run      command template; {field} interpolates SHELL-QUOTED recorded
#                values -- a template naming an unrecorded field refuses
#       require  "exit zero" (default) | "exit zero and some output line" |
#                "output is a git sha" | "output lines are exactly [{field}]" |
#                "some output line ends with one of [a, b]"
#       save     store the first output line under this name (for `accept`)
#       derive   write the first output line into this lead field
#       optional true = a failed step is skipped, never refused (derive skipped)
#   {run, classify: {pattern, exclude_pattern?, require_suffix?}, matching_into?,
#    keep?, teach_none_match?, teach_all_match?}
#       parses `--name-status` output, splits paths by the declared classifier,
#       requires BOTH classes non-empty, stores the matching paths
#   {marker_search: {clone, path, system, families, derive}, teach_missing?,
#    teach_mismatch?}
#       walks the repository root and every prefix of fields[path] via
#       `git ls-tree`, looking for a declared marker file; prefers the family
#       declared for fields[system]; a marker from a DIFFERENT family refuses
#   {tracker: <url field>, policy: {jira_types?, label_words, text_pattern,
#    accept_closed?, kind_into}}
#       fetches the item live; transient stays transient; a pull request
#       refuses; then the declared acceptance cascade decides

import shlex as _shlex

_CMD_REF = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_EXACT_REF = re.compile(r"^output lines are exactly \[\{([a-z][a-z0-9_]*)\}\]$")
_MATCHES_REF = re.compile(r"^output matches (.+)$")
_COMMIT_REF = re.compile(r"^last output line is the commit \{([a-z][a-z0-9_]*)\}$")
_ENDS_REF = re.compile(r"^some output line ends with one of \[(.*)\]$")


#: `output <value-form>`: a run step whose require describes its OUTPUT in the vocabulary the
#: FIELD grammar already reads ("is a number", "is non-empty", "is a hex digest"). A contract
#: has both grammars in it, one line apart, and a run step's output line IS a value, so
#: 'output is a number', 'output is non-empty' and 'output is a date' must read the same way
#: the same words do in a field clause. The harness resolves notation and refuses only what no
#: grammar reads.
_OUTPUT_VALUE = re.compile(r"^output (is .+)$")


def _output_value_clause(require: str) -> dict | None:
    """The field clause a run step's `output <value-form>` means, or None."""
    m = _OUTPUT_VALUE.match(" ".join(str(require or "").split()))
    if not m:
        return None
    clause = {"field": "output", "require": m.group(1)}
    _, why = evaluate_clause(clause, {"output": "probe-value"})
    return clause if "cannot read" not in str(why) else None


def run_require_readable(require: str) -> bool:
    """Would `_run_step` read this require? Asked of `_run_step` ITSELF, on a probe.

    Listing the executor's forms by hand would be a mirror that drifts. The executor is run on a fake command whose
    output is `probe-value`; the only answer that means "unreadable" is the grammar's own
    refusal sentence, so a new branch in `_run_step` is readable the moment it exists.
    """
    r = " ".join(str(require or "exit zero").strip().split())
    verdict = _run_step({"run": "probe", "require": r}, {}, lambda cmd, timeout=180: (0, "probe-value\n"), {})
    return verdict is None or "cannot read" not in str(verdict[1])


def _sh_runner(ctx):
    def run(cmd: str, timeout: int = 180):
        env = getattr(ctx, "env", None)
        if env is None:
            return 127, "no environment available"
        r = env.execute(cmd, timeout=timeout)
        return r.returncode, (r.output or "") + (r.stderr or "")

    return run


# git's object-peel suffixes (`<rev>^{commit}` etc.) are UNIVERSAL git notation
# that happens to look like this grammar's placeholders. The harness resolves
# notation, it does not refuse it: a `{commit}` immediately preceded by `^` is a
# peel, left literal, not a missing-field placeholder.
_GIT_PEEL = re.compile(r"\^\{(commit|tree|tag|blob|object)\}")


def fields_read(step: dict) -> set[str]:
    """Every field one contract step reads: its `field:` clause and the `{placeholder}` names
    in its `run:` and `require:` text, by exactly the grammar `_render_cmd` executes (a git
    peel is not a placeholder). The worklist uses this to make every such field recordable
    and a verdict input, so the vocabulary and the executor cannot disagree."""
    if not isinstance(step, dict):
        return set()
    out: set[str] = set()
    if step.get("field"):
        out.add(str(step["field"]))
    # `run:` is executed with the fields; `require:` is compared against them ("last output
    # line is the commit {parent_commit}"). Both read the field.
    for text in (str(step.get("run") or ""), str(step.get("require") or "")):
        if text:
            out |= set(_CMD_REF.findall(_GIT_PEEL.sub("", text)))
    return out


def _render_cmd(template: str, fields: dict) -> str:
    open_s, close_s = "\x00PL\x00", "\x00PR\x00"
    guarded = _GIT_PEEL.sub(lambda m: "^" + open_s + m.group(1) + close_s, template)

    def repl(m):
        name = m.group(1)
        value = str(fields.get(name) if fields.get(name) is not None else "").strip()
        if not value:
            raise KeyError(name)
        return _shlex.quote(value)

    rendered = _CMD_REF.sub(repl, guarded)
    return rendered.replace(open_s, "{").replace(close_s, "}")


def _classifier(spec: dict):
    flags = re.I if spec.get("ignore_case", True) else 0
    pat = re.compile(str(spec.get("pattern") or ""), flags)
    exc = re.compile(spec["exclude_pattern"], flags) if spec.get("exclude_pattern") else None
    sufs = tuple(str(s).lower() for s in (spec.get("require_suffix") or ()))

    def is_match(p: str) -> bool:
        q = p.lower() if spec.get("ignore_case", True) else p
        if sufs and not q.endswith(sufs):
            return False
        if exc is not None and exc.search(q):
            return False
        return bool(pat.search(q))

    return is_match


def _word_matched_labels(labels, words: set) -> set:
    out = set()
    for raw in labels or ():
        text = str(raw)
        if {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t} & words:
            out.add(text.lower())
    return out


def _tracker_step(step: dict, fields: dict, run) -> tuple[bool, str] | None:
    from agentcore.tools import evidence

    url = str(fields.get(step["tracker"]) or "").strip()
    policy = step.get("policy") or {}
    info, detail = evidence.read_tracker_item(url, lambda cmd: run(cmd))
    if info is None:
        return False, (
            f"TRANSIENT: could not reach the tracker at {url} "
            f"({str(detail)[:120]}). This says NOTHING about the lead itself "
            f"-- do not block or drop it. Use worklist_defer and retry this "
            f"delivery later; trackers rate-limit and recover."
        )
    if not info.get("exists"):
        return False, f"the tracker has no item at {url}"
    if info.get("kind") == "pull" and not policy.get("accept_pull"):
        return False, (
            f"{url} is a PULL REQUEST, not a tracker item. A PR is the "
            f'change, and its title saying "fix" is the author describing '
            f"their own patch -- it is not a report that something was "
            f"broken. Link the ISSUE the PR closes; if there is no reported "
            f"issue, this lead does not qualify."
        )
    kind_into = str(policy.get("kind_into") or "").strip()
    title = str(info.get("title") or "")
    if policy.get("accept_identified"):
        # The declared bar is "a reported issue exists here": structure the tracker
        # itself asserts -- an item type, labels, or a recognised item kind. A page
        # that merely loads has a title too; that is not an item.
        identified = bool(
            str(info.get("type") or "").strip() or info.get("labels") or info.get("kind") in ("issue", "jira")
        )
        if not identified:
            return False, (
                f"{url} is readable but nothing there identifies it as a tracker item "
                f"-- no issue type, no labels ({title[:60]!r}). A reported issue means "
                f"an entry in a tracker, not any page that happens to load."
            )
        if kind_into:
            fields[kind_into] = f"identified:{title[:60]}"
        return None
    jira_types = policy.get("jira_types")
    if jira_types is not None and info.get("kind") == "jira":
        itype = str(info.get("type") or "")
        if itype.lower() not in {str(t).lower() for t in jira_types}:
            return False, (
                f"the linked JIRA issue is a {itype or 'typeless item'}, not "
                f"one of {list(jira_types)} -- this lead does not point at a "
                f"reported problem of the declared kind"
            )
        if kind_into:
            fields[kind_into] = f"jira:{itype}"
        return None
    labels = {str(x).lower() for x in (info.get("labels") or [])}
    words = {str(w).lower() for w in (policy.get("label_words") or [])}
    hit = _word_matched_labels(labels, words)
    text_pat = re.compile(policy["text_pattern"], re.I) if policy.get("text_pattern") else None
    if hit:
        if kind_into:
            fields[kind_into] = "labels:" + ",".join(sorted(hit))
    elif text_pat is not None and (text_pat.search(title) or text_pat.search(str(info.get("body") or "")[:2000])):
        if kind_into:
            fields[kind_into] = "text:" + title[:60]
    elif policy.get("accept_closed") and str(info.get("state") or "").lower() == "closed":
        if kind_into:
            fields[kind_into] = "closed:" + title[:60]
    else:
        return False, (
            f"the linked item {title[:80]!r} (labels: {sorted(labels) or 'none'}, "
            f"state: {info.get('state')!r}) does not read as the reported "
            f"problem the declaration requires"
        )
    return None


def _marker_search_step(step: dict, fields: dict, run) -> tuple[bool, str] | None:
    ms = step["marker_search"]
    clone = str(fields.get(ms["clone"]) or "").strip()
    path = str(fields.get(ms["path"]) or "").strip().rstrip("/")
    system = str(fields.get(ms["system"]) or "").strip().lower()
    families = {str(k).lower(): tuple(v) for k, v in (ms.get("families") or {}).items()}
    all_markers = {m for fam in families.values() for m in fam}
    wanted = set(families.get(system, ()))

    code, out = run(f"git -C {_shlex.quote(clone)} ls-tree --name-only HEAD 2>&1")
    root_files = set(out.strip().splitlines()) if code == 0 else set()
    prefixes = [""]
    parts = path.split("/")
    for i in range(1, len(parts) + 1):
        prefixes.append("/".join(parts[:i]))
    hit = ""
    for pref in prefixes:
        candidates = root_files if not pref else set()
        if pref:
            code, out = run(f"git -C {_shlex.quote(clone)} ls-tree --name-only HEAD -- {_shlex.quote(pref + '/')} 2>&1")
            if code == 0:
                candidates = {ln.rsplit("/", 1)[-1] for ln in out.strip().splitlines()}
        found = (candidates & wanted) or (candidates & all_markers)
        if found:
            hit = (pref + "/" if pref else "") + sorted(found)[0]
            break
    if not hit:
        return False, (
            _step_text(str(step.get("teach_missing") or ""), fields)
            or (
                f"no declared marker file ({sorted(all_markers)[:3]}...) found "
                f"at the repository root or along {path!r}."
            )
        )
    if wanted and hit.rsplit("/", 1)[-1] not in wanted:
        return False, (
            f"you recorded {ms['system']} {system!r}, but the only marker "
            f"found at the root or along {path!r} is {hit!r}, which belongs "
            f"to a different family. " + _step_text(str(step.get("teach_mismatch") or ""), fields)
        )
    derive = str(ms.get("derive") or "").strip()
    if derive:
        fields[derive] = hit
    return None


def _step_text(text: str, fields: dict) -> str:
    try:
        return text.format_map(_Literal({k: str(v) for k, v in fields.items()}))
    except Exception:  # noqa: BLE001 - a malformed template must not eat the message
        return text


_NOT_FOUND = re.compile(r"(?:^|[\s:])([A-Za-z0-9_./+-]+): (?:command )?not found", re.MULTILINE)


def _ran(cmd: str, limit: int = 300) -> str:
    """What the harness actually ran, values substituted. A step that verifies
    a recorded string by searching for it prints nothing when the search fails, so without the
    rendered command the miner could not tell which check had run nor what string was
    searched for. The rendered command is the one thing the candidate's record cannot show."""
    text = " ".join(str(cmd or "").split())
    if not text:
        return ""
    if len(text) > limit:
        text = text[:limit] + " ..."
    return f" The harness ran: {text}"


def _whose_machine(code, out: str) -> str:
    """The sentence a failed corroboration command owes the miner when the shell said a
    tool is NOT FOUND: the steps run in the miner's OWN container, exactly as
    recorded, so the missing tool is the miner's to install and to rehearse, not a pipeline
    fault."""
    m = _NOT_FOUND.search(str(out or ""))
    prog = m.group(1) if m else None
    if prog is None and int(code or 0) != 127:
        return ""
    what = f"`{prog}`" if prog else "a tool the command needs"
    return (
        f" {what} is not installed in YOUR container: the harness runs your recorded "
        f"commands there, exactly as recorded, so this is not a pipeline fault. Install "
        f"it yourself (apt-get install ... in your container), rehearse the exact "
        f"recorded command until it runs, then record again."
    )


def _timed_step(step: dict, fields: dict, run, saves: dict, ctx=None) -> tuple[bool, str] | None:
    """THE HARNESS'S CLOCK. `{"timed": "{cmd_field}", "cwd": "{path_field}",
    "repetitions": 3, "derive": "<field>", "timeout": <s>}`: the recorded command runs
    `repetitions` times, each under the measurement lane, the wall-clock taken INSIDE the
    sandbox by the wall clock of agentcore.wallclock around a shell that is handed the command whole (no docker
    round trip in the timed region), every run must exit zero, and
    the integer median in milliseconds is derived. A number the agent's command prints is
    the agent's number; this one is the harness's, which is what a ratio clause may be trusted to compare."""

    def bail(why):
        if step.get("optional"):
            return None
        teach = _step_text(str(step.get("teach") or "").strip(), fields)
        return False, (why + (" " + teach if teach else ""))

    derive = str(step.get("derive") or "").strip()
    if not derive:
        return False, (
            "HARNESS/SPEC FAULT: a `timed` corroboration step derives nothing -- "
            "declare `derive: <field>` for the median it produces; fix the spec, "
            "never skip the rule."
        )
    try:
        reps = int(step.get("repetitions") or 3)
        if reps <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return False, (
            f"HARNESS/SPEC FAULT: `timed` step declares repetitions "
            f"{str(step.get('repetitions'))[:20]!r}; it must be a positive integer."
        )
    try:
        cmd_q = _render_cmd(str(step["timed"]), fields)
        cwd_q = _render_cmd(str(step.get("cwd") or ""), fields) if step.get("cwd") else ""
    except KeyError as e:
        return bail(f"this step's command names {{{e.args[0]}}}, which is not recorded -- record it first.")
    timeout = None
    if step.get("timeout") is not None:
        try:
            timeout = int(step["timeout"])
            if timeout <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return False, (
                f"corroboration step declares timeout "
                f"{str(step['timeout'])[:40]!r}, which is not a positive "
                f"number of seconds -- fix the spec, never skip the rule."
            )
    else:
        _op = os.environ.get("AGENTCORE_CORROBORATION_TIMEOUT", "").strip()
        try:
            timeout = int(float(_op)) if _op else None
        except ValueError:
            timeout = None
        if timeout is not None and timeout <= 0:
            timeout = None
    # The wrapper: cd (quoted), then the clock around `sh -c <recorded command, one word>`.
    # The recorded command's own stdout is discarded (its runtime is the observation); its
    # exit status is the wrapper's.
    # The command's stdout is discarded (its runtime is the observation); its STDERR TAIL is
    # kept after the elapsed time, so a refusal can show what the shell actually said.
    prefix = f"cd {cwd_q} && " if cwd_q else ""
    from agentcore.wallclock import NOW_MS_SH

    wrapper = (
        f'{prefix}__e=$(mktemp); __t0=$({NOW_MS_SH}); sh -c {cmd_q} >/dev/null 2>"$__e"; '
        f'__rc=$?; __t1=$({NOW_MS_SH}); echo $((__t1-__t0)); tail -c 400 "$__e"; '
        f'rm -f "$__e"; exit $__rc'
    )
    from agentcore import measurement_lane

    samples: list[int] = []
    for i in range(reps):
        with measurement_lane.hold(f"corroboration timed {derive} [{i + 1}/{reps}]"):
            code, out = run(wrapper, timeout) if timeout is not None else run(wrapper)
        lines = [ln.strip() for ln in str(out).strip().splitlines() if ln.strip()]
        if code != 0:
            said = "\n".join(lines[1:])[-300:] if len(lines) > 1 else (lines[0] if lines else "")
            m_nf = _NOT_FOUND.search(said)
            where = ""
            if m_nf and not m_nf.group(1).startswith("/") and ("/" in m_nf.group(1) or m_nf.group(1).startswith(".")):
                where = (
                    f" `{m_nf.group(1)}` is a relative path, resolved from the step's cwd "
                    f"({cwd_q or 'the harness cwd'}): record the command so that it runs from "
                    f"there, or name the binary by an absolute path."
                )
            return bail(
                f"the timed command failed (exit {code}) on repetition {i + 1}"
                f"{': ' + said if said else ''}" + where + _whose_machine(code, said)
            )
        try:
            samples.append(int(float(lines[0])))
        except (IndexError, ValueError):
            return bail(
                f"the timing wrapper printed no elapsed time on repetition {i + 1} "
                f"({str(out).strip()[-120:] or 'no output'}); date, tr, printf and "
                f"cut must be available in the sandbox."
            )
    samples.sort()
    fields[derive] = samples[len(samples) // 2]
    if step.get("save"):
        saves[str(step["save"])] = fields[derive]
    return None


def _run_step(step: dict, fields: dict, run, saves: dict, ctx=None) -> tuple[bool, str] | None:
    def bail(why):
        if step.get("optional"):
            return None
        teach = _step_text(str(step.get("teach") or "").strip(), fields)
        return False, (why + (" " + teach if teach else ""))

    try:
        cmd = _render_cmd(step["run"], fields)
    except KeyError as e:
        return bail(f"this step's command names {{{e.args[0]}}}, which is not recorded -- record it first.")
    if step.get("timeout") is None:
        # THE OPERATOR'S DEFAULT FOR A STEP THAT DECLARES NOTHING. Steps without a `timeout`
        # would otherwise get the runner's 180 s, which refuses slow builds. A launch-time
        # knob for a machine, never a spec edit; a declared timeout still wins.
        _op = os.environ.get("AGENTCORE_CORROBORATION_TIMEOUT", "").strip()
        _op_timeout = None
        if _op:
            try:
                _op_timeout = int(float(_op))
            except ValueError:
                _op_timeout = None
        if _op_timeout and _op_timeout > 0:
            code, out = run(cmd, timeout=_op_timeout)
        else:
            code, out = run(cmd)
    else:
        try:
            declared_timeout = int(step["timeout"])
            if declared_timeout <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return False, (
                f"corroboration step declares timeout "
                f"{str(step['timeout'])[:40]!r}, which is not a positive "
                f"number of seconds -- fix the spec, never skip the rule."
            )
        code, out = run(cmd, declared_timeout)
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]

    if "classify" in step:
        if code != 0:
            return bail(f"the command failed: {out.strip()[:160]}" + _whose_machine(code, out))
        paths = [ln.split("\t", 1)[-1] for ln in out.strip().splitlines() if "\t" in ln]
        match = _classifier(step["classify"])
        hits = [p for p in paths if match(p)]
        rest = [p for p in paths if not match(p)]
        if not hits:
            return bail(
                _step_text(str(step.get("teach_none_match") or ""), fields)
                or f"no path in the diff ({len(paths)} changed) matches the declared classifier."
            )
        if not rest:
            return bail(
                _step_text(str(step.get("teach_all_match") or ""), fields)
                or "every changed path matches the declared classifier; both kinds must land together."
            )
        into = str(step.get("matching_into") or "").strip()
        if into:
            fields[into] = hits[: int(step.get("keep") or 5)]
        return None

    require = " ".join(str(step.get("require") or "exit zero").split())
    first = lines[0] if lines else ""
    if require == "exit zero":
        if code != 0:
            return bail(
                f"the command failed: {out.strip()[:160] or 'no output'}." + _ran(cmd) + _whose_machine(code, out)
            )
    elif require == "exit zero and some output line":
        if code != 0 or not lines:
            return bail(f"the command failed or printed nothing: {out.strip()[:160] or 'no output'}.")
    elif require == "output is a git sha":
        if code != 0 or not re.fullmatch(r"[0-9a-f]{7,40}", first):
            note = ""
            low_out = str(out or "").lower()
            if (
                "needed a single revision" in low_out
                or "unknown revision" in low_out
                or "bad revision" in low_out
                or "not a valid object" in low_out
            ):
                # THE CLONE IS THE SOURCE OF TRUTH FOR A REVISION: a forty-digit hash padded from a
                # short prefix passes every entry clause (forty hex digits IS a sha) and is
                # refused here. Say whose record it is and where the only honest value
                # comes from.
                revs = re.findall(r"\b[0-9a-f]{7,40}\b", cmd)
                rev = next((r for r in revs if r != first), revs[0] if revs else "the recorded revision")
                where = re.search(r"-C\s+('?)(\S+?)\1(?:\s|$)", cmd)
                clone = where.group(2) if where else "the recorded clone"
                note = (
                    f" The revision {rev} does not exist in the clone at {clone}: this is "
                    f"YOUR record, not the gate. A hash that was typed, remembered, or "
                    f"padded to forty digits resolves nowhere; record exactly what "
                    f"`git -C {clone} rev-parse HEAD` prints in your container."
                )
            return bail(f"expected a git hash, got: {out.strip()[:120] or 'no output'}." + note)
    else:
        m = _EXACT_REF.match(require)
        if m:
            expected = str(fields.get(m.group(1)) or "").strip().rstrip("/")
            if code != 0 or lines != [expected]:
                return bail(
                    f"expected exactly {expected!r}, got {len(lines)} line(s)"
                    f" ({', '.join(lines[:3])}{', ...' if len(lines) > 3 else ''})."
                )
        else:
            m = _ENDS_REF.match(require)
            m_rx = _MATCHES_REF.match(require)
            m_commit = _COMMIT_REF.match(require)
            if m:
                sufs = tuple(s.strip() for s in m.group(1).split(",") if s.strip())
                if code != 0 or not any(ln.lower().endswith(sufs) for ln in lines):
                    return bail(f"no output line ends with one of {list(sufs)} ({len(lines)} line(s)).")
            elif m_rx:
                if code != 0 or not re.search(m_rx.group(1), out):
                    return bail(f"the output does not match {m_rx.group(1)!r} ({out.strip()[-120:] or 'no output'!r}).")
            elif m_commit:
                want = str(fields.get(m_commit.group(1)) or "")
                got = lines[-1] if lines else ""
                if code != 0 or not want or got[:40] != want[:40]:
                    return bail(f"expected the commit {want[:12] or '?'}, the output shows {got[:12] or '?'}.")
            elif (clause := _output_value_clause(require)) is not None:
                # The field grammar decides, on the first output line. Exit zero is still
                # required: a rule about the output of a command that failed decides nothing.
                if code != 0:
                    return bail(
                        f"the command failed: {out.strip()[:160] or 'no output'}."
                        + _ran(cmd)
                        + _whose_machine(code, out)
                    )
                # Against the lead's REAL fields and ctx, not a throwaway dict: a peer form
                # ("is at least 2 times field probe_baseline_ms") must see its peer, and a
                # ctx form ("is distinct across leads") must see the other leads -- without
                # them the first refuses every candidate and the second passes every one.
                ok, why = evaluate_clause(clause, {**fields, "output": first}, ctx)
                if not ok:
                    return bail(f"the output does not satisfy {require!r}: {why}")
            else:
                return False, (
                    f"corroboration step states {require!r}, which this "
                    f"grammar cannot read -- fix the spec, never skip the rule."
                )
    if step.get("save"):
        saves[str(step["save"])] = first
    if step.get("derive"):
        fields[str(step["derive"])] = first
    return None


_TIMES_FIELD = re.compile(r"\btimes\s+field\s+([A-Za-z_][A-Za-z0-9_]*)")


def timed_fields(inst: dict) -> set[str]:
    """The fields whose values a ratio clause of the entry contract compares -- the numbers
    that must be MEASURED cleanly, because a ratio between them is what will be judged.

    Declared, never inferred: a clause `<field> is at least N times field <other>` names both
    sides, and both are timed fields. No field name is spelled here; a benchmark with no
    ratio clause has no timed fields, and a step deriving one of these holds the host-wide
    measurement lane while it runs (measurement_lane). Everything else -- clones, builds,
    digests -- runs wide."""
    out: set[str] = set()
    for cl in (inst or {}).get("entry_contract") or []:
        if not isinstance(cl, dict):
            continue
        req = str(cl.get("require") or "")
        others = _TIMES_FIELD.findall(req)
        if not others:
            continue
        out.update(others)
        this = str(cl.get("field") or "").strip()
        if this:
            out.add(this)
    return out


def _evaluate_steps(steps: list, fields: dict, ctx, saves: dict) -> tuple[bool, str] | None:
    """The one step dispatcher both gate entry points share; first failure wins,
    None means every step held."""
    run = _sh_runner(ctx)
    timed = timed_fields(getattr(ctx, "instance", {}) or {})
    for step in steps:
        if not isinstance(step, dict):
            return False, f"a corroboration step must be an object, got {str(step)[:60]!r}"
        if "tracker" in step:
            verdict = _tracker_step(step, fields, run)
        elif "marker_search" in step:
            verdict = _marker_search_step(step, fields, run)
        elif "resolve_shas" in step:
            verdict = _resolve_shas_step(step, fields, ctx)
        elif "report" in step:
            verdict = _report_step(step, fields, ctx)
        elif "diff_check" in step:
            verdict = _diff_check_step(step, fields, ctx)
        elif "shared" in step:
            verdict = _shared_step(step, fields, ctx)
        elif "digests" in step:
            verdict = _digests_step(step, fields, ctx)
        elif "stated_reference" in step:
            verdict = _stated_reference_step(step, fields, ctx)
        elif "measurements" in step:
            verdict = _measurements_step(step, fields, ctx)
        elif "comparisons" in step:
            verdict = _comparisons_step(step, fields, ctx)
        elif "variant_cmds" in step:
            verdict = _variant_cmds_step(step, fields, ctx)
        elif "exclusions_on" in step:
            verdict = _exclusions_on_step(step, fields, ctx)
        elif "timed" in step:
            # the harness's own clock, in the sandbox the step names or the shared env
            step_run = _sandbox_runner(ctx, fields, str(step["sandbox"])) if step.get("sandbox") else run
            verdict = _timed_step(step, fields, step_run, saves, ctx=ctx)
        elif "run" in step:
            # a step naming a sandbox field runs THERE -- the same filesystem the
            # evidence was produced on, never a different machine's
            step_run = _sandbox_runner(ctx, fields, str(step["sandbox"])) if step.get("sandbox") else run
            derived = step.get("derive")
            derived = (
                [str(derived)]
                if isinstance(derived, str)
                else [str(d) for d in derived]
                if isinstance(derived, (list, tuple))
                else []
            )
            timed_here = [d for d in derived if d in timed]
            if timed_here:
                # this step produces a number a ratio clause will judge: a timed
                # execution, queued on the host-wide measurement lane
                from agentcore import measurement_lane

                with measurement_lane.hold(f"corroboration {' '.join(timed_here)}"):
                    verdict = _run_step(step, fields, step_run, saves, ctx=ctx)
            else:
                verdict = _run_step(step, fields, step_run, saves, ctx=ctx)
        elif "field" in step:
            ok, why = evaluate_clause(step, fields, ctx)
            verdict = None if ok else (False, why)
        else:
            return False, (f"unreadable step {str(step)[:80]!r} -- fix the spec, never skip the rule.")
        if verdict is not None:
            return verdict
    return None


def corroboration_gate(fields: dict, ctx) -> tuple[bool, str]:
    """The one corroboration gate every task shares: the instance's declared step
    list, executed in order; the first failing step refuses. Wire it as
    `verify: agentcore.agent.field_contract:corroboration_gate`. A task that
    declares no steps FAILS CLOSED."""
    inst = getattr(ctx, "instance", {}) or {}
    steps = inst.get("corroboration_contract") or []
    if not steps:
        return False, (
            "HARNESS/SPEC FAULT: this task's instance declares no "
            "`corroboration_contract`, so no delivery can be judged. "
            "Declare the steps in the spec; an undeclared contract is a "
            "missing rule, not a permissive one."
        )
    saves: dict = {}
    verdict = _evaluate_steps(steps, fields, ctx, saves)
    if verdict is not None:
        return verdict
    # the entry contract's clauses over fields these steps DERIVED are judged now, on the
    # derived values -- the floor and the ratio over the harness's own timings
    _door, after = _clauses_over_derived(inst)
    if after:
        for cl in after:
            if not isinstance(cl, dict):
                continue
            ok, why = evaluate_clause(cl, fields, ctx)
            if not ok:
                return False, why + _timed_provenance(inst, fields, str(cl.get("field") or ""))
    accept = str(inst.get("corroboration_accept") or "the declared corroboration holds")
    pool = {k: str(v) for k, v in fields.items()}
    pool.update(saves)
    try:
        accept = accept.format_map(_Literal(pool))
    except Exception:  # noqa: BLE001
        pass
    return True, accept


# ── step kinds the collector chains need ───────────────
#
# Same contract as every step above: the mechanics here are git/report/text
# universals; every field name, pattern, word list and path arrives declared.

#: "is at least 10 times field scale_probe_small_ms" -- declared factor, declared peer.
_RATIO_CLAUSE = re.compile(r"^is at least ([0-9]+(?:\.[0-9]+)?) times field ([a-z_][a-z0-9_]*)$")

_SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./-]{1,200}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


def _sandbox_runner(ctx, fields: dict, sandbox_field: str):
    """Commands run in the candidate's NAMED sandbox when one is recorded (rebuilt
    from its recipe when needed), else in the agent's own environment -- the same
    discipline collector_gates._run_in enforces, shared here."""
    from agentcore.agent.collector_gates import _run_in

    sandbox = str(fields.get(sandbox_field) or "").strip() if sandbox_field else ""

    def run(cmd: str, timeout: int = 600):
        return _run_in(ctx, sandbox, cmd, timeout=timeout)

    return run


def _resolve_shas_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """Both declared commit fields must exist in a real clone; abbreviations and
    ancestor expressions are RESOLVED and written back, never refused. A fabricated hash still refuses: resolution runs against
    the real repository."""
    import shlex as _sh

    from agentcore.tools import gitsha

    spec = step["resolve_shas"]
    sha_fields = [str(f) for f in (spec.get("fields") or [])]
    clone_field = str(spec.get("clone") or "")
    sandbox_field = str(spec.get("sandbox") or "")
    run = _sandbox_runner(ctx, fields, sandbox_field)

    clone = str(fields.get(clone_field) or "").strip()
    if not clone and spec.get("clone_fallback_from"):
        base = str(fields.get(spec["clone_fallback_from"]) or "").strip()
        clone = ("/work/" + base.rsplit("/", 1)[-1]) if base else ""

    for field in sha_fields:
        sha = str(fields.get(field) or "").strip().lower()
        if not sha:
            return False, f"{field} is missing"
        if gitsha.is_full(sha):
            continue
        if not (gitsha.is_abbreviated(sha) or gitsha.is_ancestor_expr(sha)):
            return False, (
                f"{field}={sha!r} is not a usable commit hash "
                f"({gitsha.MIN_PREFIX}-40 hex characters expected, optionally "
                f"followed by ^ or ~N)"
            )
        if not clone or not _SAFE_PATH.match(clone):
            return False, (
                f"record {clone_field} as a plain absolute path to the clone, "
                f"so an abbreviated commit can be resolved against the real "
                f"repository"
            )
        full, detail = gitsha.resolve(sha, runner=lambda c: run(c, 120), repo_path=clone)
        if full is None:
            return False, f"{field}: {detail}"
        fields[field] = full  # the harness fixes it up; the model does not have to

    if not clone or not _SAFE_PATH.match(clone):
        return False, (
            f"record {clone_field} as a plain absolute path to the clone (letters, digits, '.', '_', '-', '/' only)"
        )
    sandbox = str(fields.get(sandbox_field) or "").strip() if sandbox_field else ""
    if sandbox and not _SAFE_NAME.match(sandbox):
        return False, f"{sandbox_field} must be the plain name of a container you created"

    q = _sh.quote
    probes = " && ".join(f"git cat-file -e {q(str(fields[f]))}^{{commit}}" for f in sha_fields)
    code, out = run(f"cd {q(clone)} && {probes} && echo OK_ALL_RESOLVE", 180)
    if out.strip().splitlines()[-1:] == ["OK_ALL_RESOLVE"]:
        return None
    if code == 127:
        return False, f"could not reach the clone: {out.strip()[:140]}"
    return False, (
        f"could not resolve {' and '.join(sha_fields)} in {clone}: "
        f"{out.strip()[-140:]}. Clone the repository, record {clone_field} "
        f"(and {sandbox_field or 'sandbox'}, if you cloned in one)."
    )


_REPORT_KEEP = 40000


def _report_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """The oracle is parsed from a machine-readable report, never from prose.

    Declared: which field names the report path, which side the evidence archives
    under, and what the report must show -- "failures" (>=1 <failure>; an <error>-only
    run is an environment problem, not a bug witness) or "clean" (>=1 test, zero
    failures and errors). The report format is declared too; junit-xml is the one the
    shared reader speaks."""

    from agentcore import reports
    from agentcore.tools import evidence

    spec = step["report"]
    path_field = str(spec["path_field"])
    side = str(spec.get("side") or path_field)
    expect = str(spec.get("expect") or "failures")
    fmt = str(spec.get("format") or "junit-xml")
    teach = _step_text(str(step.get("teach") or "").strip(), fields)

    report_path = str(fields.get(path_field) or "").strip()
    if not report_path:
        return False, (f"record {path_field}: the path to the {fmt} report the run produced. " + teach)
    sandbox_field = str(spec.get("sandbox") or "")
    sandbox = str(fields.get(sandbox_field) or "").strip() if sandbox_field else ""
    if sandbox and not _SAFE_NAME.match(sandbox):
        return False, f"{sandbox_field} must be the plain name of a container you created"

    from agentcore.agent.collector_gates import _run_in

    code, out, resolved = evidence.read_reports(
        # capture_limit=0: a report is parsed whole; a window cut into the middle of a
        # JUnit XML is 'not a readable report', not a long one.
        lambda box, cmd: _run_in(ctx, box, cmd, timeout=120, capture_limit=0),
        sandbox,
        report_path,
        str(fields.get(str(spec.get("clone") or "")) or "").strip(),
    )
    if code != 0 or not out.strip():
        return False, (
            f"could not read {resolved}. Run the check and record the path of the report it produced. " + teach
        )
    fields[path_field] = resolved

    r = reports.parse_report(fmt, out)
    if r is None:
        return False, f"{report_path} is not a readable {fmt} report"
    kinds = set()
    if r.failures:
        kinds.add("failure")
    if r.errors:
        kinds.add("error")
    messages = " | ".join(r.failed_names + r.error_names)[:300]

    if r.tests < 1:
        return False, (
            "the report shows 0 tests executed -- your selector matched "
            "nothing, so nothing was witnessed. Fix the selector and re-run."
        )
    if expect == "failures":
        if "failure" not in kinds:
            if "error" in kinds:
                return False, (
                    f"the check ERRORED rather than failing an assertion "
                    f"({messages[:120]}). A compile/import/setup error is not "
                    f"evidence of the defect -- it usually means the check "
                    f"depends on what the fix introduces. Reject this "
                    f"candidate."
                )
            return False, ("the report shows no failing case, so the defect is not witnessed there.")
    elif expect == "clean":
        if r.failures or r.errors:
            return False, (
                f"the check still fails ({r.failures} failure(s), "
                f"{r.errors} error(s)) -- this is not the passing side of "
                f"the transition."
            )
    else:
        return False, (
            f"report step declares expect={expect!r}, which this grammar "
            f"cannot read -- fix the spec, never skip the rule."
        )

    archive_report_evidence(
        fields,
        side,
        {"tests": r.tests, "failures": r.failures, "errors": r.errors, "kinds": kinds, "messages": messages},
        out,
    )
    return None


def archive_report_evidence(fields: dict, side: str, parsed: dict, report_text: str) -> None:
    """Copy the report evidence INTO the record at the moment it is verified: counts,
    kinds, messages, a fingerprint, and the (capped) report itself. Maven and pytest
    overwrite their report location, so by delivery the path usually holds only the
    other side's run -- the record must carry what was actually seen."""
    import hashlib

    fields[f"{side}_evidence"] = {
        "tests": parsed.get("tests", 0),
        "failures": parsed.get("failures", 0),
        "errors": parsed.get("errors", 0),
        "kinds": sorted(parsed.get("kinds") or []),
        "messages": str(parsed.get("messages") or "")[:400],
        "report_sha256": hashlib.sha256((report_text or "").encode("utf-8", "replace")).hexdigest(),
        "report_bytes": len(report_text or ""),
        "report_xml": (report_text or "")[:_REPORT_KEEP],
        "report_truncated": len(report_text or "") > _REPORT_KEEP,
    }


def _shared_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """One of the harness's existing generic gates, declared by name."""
    name = str(step["shared"])
    if name == "caps":
        from agentcore.agent.diversity import check_caps

        ok, why = check_caps(fields, ctx)
    elif name == "exclusions":
        from agentcore.agent.collector_gates import candidate_is_admissible

        ok, why = candidate_is_admissible(fields, ctx)
    elif name == "sandbox_exists":
        from agentcore.agent.sandboxes import sandbox_exists

        ok, why = sandbox_exists(fields, ctx)
    elif name == "sandbox_exclusive":
        from agentcore.agent.sandboxes import sandbox_is_exclusive

        ok, why = sandbox_is_exclusive(fields, ctx)
    elif name == "env_recipe":
        from agentcore.tools import evidence

        ok, why = evidence.env_recipe_from_sandbox(fields, ctx)
    elif name == "variants":
        from agentcore.agent.collector_gates import variants_are_materialized

        ok, why = variants_are_materialized(fields, ctx)
    elif name == "session":
        from agentcore.agent.collector_gates import consumer_session_runs

        ok, why = consumer_session_runs(fields, ctx)
    else:
        return False, (
            f"shared step names {name!r}, which is not a gate the harness defines -- fix the spec, never skip the rule."
        )
    return None if ok else (False, why)


def _diff_check_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """The recorded changed-path lists must match a re-run diff, split by the declared
    classifier, and the change must carry declared defect evidence: the commit message
    matches the declared pattern, or the declared tracker item corroborates it."""
    import shlex as _sh

    from agentcore.tools import evidence

    spec = step["diff_check"]
    # Two declared modes: with src_field/test_field the RECORDED lists are verified
    # against the re-run diff; without them the diff itself is classified (a task may
    # not ask the model to record paths at all).
    src_field = str(spec.get("src_field") or "")
    test_field = str(spec.get("test_field") or "")
    recorded = bool(src_field and test_field)
    src = test = None
    if recorded:
        src = evidence.as_paths(fields.get(src_field))
        test = evidence.as_paths(fields.get(test_field))
        if not isinstance(src, list) or not src:
            return False, (
                f"record {src_field}: the main-source paths the diff actually "
                f"printed, as a list. Not a yes/no -- the harness re-runs the "
                f"same diff and compares."
            )
        if not isinstance(test, list) or not test:
            return False, (f"record {test_field}: the test paths the diff actually printed, as a list.")

    clone_field = str(spec.get("clone") or "")
    if not clone_field:
        return False, "diff_check must declare its clone field -- fix the spec."
    clone = str(fields.get(clone_field) or "").strip()
    if not _SAFE_PATH.match(clone):
        return False, f"record {clone_field} as a plain absolute path to the clone"
    run = _sandbox_runner(ctx, fields, str(spec.get("sandbox") or ""))
    q = _sh.quote
    old_v = str(fields.get(str(spec["old"])) or "")
    new_v = str(fields.get(str(spec["new"])) or "")
    code, out = run(f"cd {q(clone)} && git diff --name-only {q(old_v)} {q(new_v)}", 120)
    if code != 0:
        return False, f"could not re-run the diff: {out.strip()[-140:]}"
    printed = [ln.strip() for ln in out.splitlines() if ln.strip()]
    match = _classifier(spec.get("test_classify") or {})
    if recorded:
        missing = [p for p in list(src) + list(test) if p not in set(printed)]
        if missing:
            return False, (
                f"these paths are not in the diff between the two commits: "
                f"{missing[:3]}. Record what `git diff --name-only` actually "
                f"printed."
            )
        if not any(match(p) for p in test):
            return False, (f"none of {test_field} matches the declared test-path shape: {test[:3]}")
        if not any(p not in test for p in src):
            return False, (
                f"{src_field} must include at least one non-test file -- a "
                f"change that only touches tests fixes nothing."
            )
    else:
        if not printed:
            return False, (
                "the diff changes no files at all, so there is nothing for a "
                "test to witness. Check the two revisions are the right way "
                "round."
            )
        hits = [p for p in printed if match(p)]
        rest = [p for p in printed if not match(p)]
        if not hits:
            return False, (
                f"this change touches no test file ({len(rest)} file(s) "
                f"changed: {', '.join(rest[:4])}). Without a test change "
                f"nothing can witness the defect. Drop this candidate."
            )
        if not rest:
            return False, (
                f"this change touches only test files "
                f"({', '.join(hits[:4])}), so it contains no fix. Drop this "
                f"candidate."
            )

    def _derive_true():
        for f in spec.get("derive_true") or []:
            fields[str(f)] = "True"

    ev = spec.get("defect_evidence") or {}
    if not ev:
        _derive_true()
        return None
    msg_pat = re.compile(ev["message_pattern"], re.I) if ev.get("message_pattern") else None
    code, msg = run(f"cd {q(clone)} && git log -1 --format=%B {q(new_v)}", 60)
    if code != 0:
        return False, (
            f"could NOT read the change's message ({str(msg).strip()[-140:]}). "
            f"This is not a judgment about the candidate -- the check has "
            f"not run."
        )
    if msg_pat is not None and msg_pat.search(str(msg)):
        _derive_true()
        return None
    tracker_field = str(ev.get("tracker") or "")
    url = str(fields.get(tracker_field) or "").strip() if tracker_field else ""
    if url.startswith(("http://", "https://")):
        verdict = _tracker_step({"tracker": tracker_field, "policy": ev.get("policy") or {}}, fields, run)
        if verdict is None:
            _derive_true()
            return None
        return verdict
    pointer = re.compile(ev["pointer_pattern"]) if ev.get("pointer_pattern") else None
    hint = (
        f" The message references a tracker item -- record {tracker_field} pointing at it so the harness can read it."
        if pointer is not None and pointer.search(str(msg))
        else ""
    )
    return False, (
        "nothing shows this change fixes a DEFECT. Its message describes no fix, and no "
        f"{tracker_field or 'tracker link'} is recorded. The change may still be a real "
        f"fix with its description in the tracker rather than the message -- if so, "
        f"record the link. If it is a feature or a refactor, drop the candidate: the "
        f"spec admits neither." + hint
    )


def _teach_or(step_or_spec: dict, key: str, fields: dict, default: str) -> str:
    """The engine's generic refusal, with the declared teaching APPENDED -- the same
    convention _refuse applies to clauses: the declaration adds task-specific
    guidance, it never replaces the mechanics' own account of what failed."""
    text = str(step_or_spec.get(key) or "").strip()
    return (default + " " + _step_text(text, fields)) if text else default


def _digests_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """A declared list field must hold >= N identical hex digests of NON-EMPTY
    output, optionally equal to a second declared digest field (the clean-rebuild
    control: identical output within one build proves the binary is stable, not
    the thing that produced it). The empty-output family is universal -- hashes
    of zero bytes agree perfectly and verify nothing."""
    from agentcore.agent.digests import all_empty
    from agentcore.tools import evidence

    spec = step["digests"]
    fname = str(spec["field"])
    min_count = int(spec.get("min_count") or 1)
    raw = evidence.as_paths(fields.get(fname)) or fields.get(fname)
    if not isinstance(raw, list) or len(raw) < min_count:
        return False, _teach_or(
            step,
            "teach_list",
            fields,
            f"record {fname}: a list of at least {min_count} hashes "
            f"of the output from {min_count} separate runs on the "
            f"same input.",
        )
    bad = [d for d in raw if not _HEX_DIGEST.match(str(d).strip().lower())]
    if bad:
        return False, f"{fname} must be hex hashes of real output; got {bad[:2]}"
    if len({str(d).strip().lower() for d in raw}) != 1:
        return False, _teach_or(
            step,
            "teach_vary",
            fields,
            f"the runs recorded in {fname} did not produce identical output, so nothing can be validated against it.",
        )

    equal_to = str(spec.get("equal_to") or "")
    pool = [*raw] + ([fields.get(equal_to)] if equal_to else [])
    if all_empty(pool):
        return False, _teach_or(
            step,
            "teach_empty",
            fields,
            f"every run recorded in {fname} hashed EMPTY output -- "
            f"the digest of zero bytes, so the matching runs prove "
            f"nothing. Capture the real output first.",
        )
    if equal_to:
        other = str(fields.get(equal_to) or "").strip().lower()
        if not other:
            return False, _teach_or(
                step,
                "teach_equal_missing",
                fields,
                f"record {equal_to}: the same hash taken again from a clean rebuild with the same configuration.",
            )
        if not _HEX_DIGEST.match(other):
            return False, f"{equal_to} must be a hex hash of real output"
        if other != str(raw[0]).strip().lower():
            return False, _teach_or(
                step,
                "teach_equal_differs",
                fields,
                f"{equal_to} does not match {fname}: the output changed after a clean rebuild. Not admissible.",
            )
    return None


def _stated_reference_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """A declared field must agree with what THE SPEC calls its reference, under a
    declared token grammar. When the spec's none-statement matches, the field must
    carry no such token; when the spec names a token as the reference, the field
    must carry it; when the spec states neither, nothing is enforced -- a spec that
    does not say does not get a value invented for it here."""
    spec = step["stated_reference"]
    fname = str(spec["field"])
    value = str(fields.get(fname) or "").strip()
    stated = _stated_token(ctx, spec)
    if not stated:
        return None
    if stated == "none":
        m = re.search(str(spec.get("token_pattern") or ""), value)
        if m:
            pool = dict(fields)
            pool["found"] = m.group(0)
            return False, _teach_or(
                step,
                "teach_none",
                pool,
                f"{fname} {value[:70]!r} carries {m.group(0)!r}, but "
                f"the spec declares its reference carries none. "
                f"Re-record {fname} without it.",
            )
        return None
    if not re.search(rf"{re.escape(stated)}\b", value):
        pool = dict(fields)
        pool["stated"] = stated
        return False, _teach_or(
            step,
            "teach_missing",
            pool,
            f"{fname} {value[:70]!r} does not contain {stated}, which is the reference this benchmark declares.",
        )
    return None


def _measurements_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """A declared list of measured records: each entry names where it lives, the
    exact command that reproduces it, and its measured cost -- cost is MEASURED,
    never assumed from a size. The costliest entry is DERIVED by the harness into
    declared fields, so "largest" cannot quietly mean "most bytes"."""
    import json as _json

    spec = step["measurements"]
    fname = str(spec["field"])
    f_name = str(spec.get("name") or "name")
    f_path = str(spec["path"])
    f_cmd = str(spec["cmd"])
    f_cost = str(spec["cost"])
    f_size = str(spec.get("size") or "")
    min_entries = int(spec.get("min_entries") or 1)

    raw = fields.get(fname)
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except (ValueError, TypeError):
            raw = None
    if not isinstance(raw, list) or not raw:
        return False, _teach_or(
            step,
            "teach_shape",
            fields,
            f"record {fname}: a JSON list with one entry per item, "
            f"each carrying {f_name!r}, {f_path!r}, {f_cmd!r} and "
            f"a measured {f_cost!r}.",
        )
    entries = [e for e in raw if isinstance(e, dict)]
    if len(entries) < min_entries:
        return False, _teach_or(step, "teach_min", fields, f"{fname} must carry at least {min_entries} entries.")
    costs = []
    for e in entries:
        name = e.get(f_name)
        if not str(e.get(f_path) or "").strip():
            return False, f"entry {name!r} in {fname} records no {f_path}"
        if not str(e.get(f_cmd) or "").strip():
            return False, _teach_or(
                step,
                "teach_cmd",
                {**fields, "entry": str(name)},
                f"entry {name!r} in {fname} records no {f_cmd}: the "
                f"exact command that reproduces THIS entry. Without "
                f"it nobody can re-run it, and the record is a claim "
                f"rather than an observation.",
            )
        try:
            costs.append(float(e.get(f_cost)))
        except (TypeError, ValueError):
            return False, _teach_or(
                step,
                "teach_cost",
                {**fields, "entry": str(name)},
                f"entry {name!r} in {fname} records no {f_cost}: its measured cost. Cost is measured, not assumed.",
            )
        if f_size:
            try:
                int(e.get(f_size))
            except (TypeError, ValueError):
                return False, (
                    f"entry {name!r} in {fname} records no {f_size}; it "
                    f"decides nothing but a consumer reading the record "
                    f"should see what each entry is"
                )
    if len(set(costs)) < 2 and min_entries >= 2:
        return False, _teach_or(
            step,
            "teach_spread",
            fields,
            f"every entry in {fname} records the same cost "
            f"({costs[0]:g}). Add entries until the costs actually "
            f"spread, or there is nothing to see between them.",
        )

    derive = spec.get("derive") or {}
    if derive:
        top = max(entries, key=lambda e: float(e.get(f_cost)))
        if derive.get("path_into"):
            fields[str(derive["path_into"])] = str(top.get(f_path))
        if derive.get("cost_into"):
            fields[str(derive["cost_into"])] = float(top.get(f_cost))
        if derive.get("cmd_into"):
            fields[str(derive["cmd_into"])] = str(top.get(f_cmd))
    return None


def _tolerance_of(mode_value) -> float | None:
    """The numeric tolerance a declared comparison-mode field states (`tol=<x>`),
    or None when it compares exactly."""
    m = re.search(r"tol\s*=\s*([0-9.eE+-]+)", str(mode_value or ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _comparisons_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """A declared list of alternative runs judged against a reference output.

    Each entry carries a label, a status from the declared enum, and -- when it
    worked -- its own output check and its own timing. Entries that did NOT work
    are kept with a `detail` (a run that breaks under one configuration is a
    finding, not noise); they just do not count as a bar to clear, so at least one
    entry carrying a declared token must have worked and been timed. The best such
    entry is DERIVED into declared fields. Numeric floors and ceilings are read
    from the spec, never restated here."""
    import json as _json

    from agentcore.tools import evidence

    spec = step["comparisons"]
    fname = str(spec["field"])
    f_label = str(spec["label"])
    f_status = str(spec.get("status") or "status")
    statuses = [str(s) for s in (spec.get("statuses") or [])]
    ok_status = str(spec.get("ok_status") or (statuses[0] if statuses else "ok"))
    f_detail = str(spec.get("detail") or "detail")
    f_digest = str(spec.get("digest") or "")
    f_dev = str(spec.get("deviation") or "")
    f_cost = str(spec.get("cost") or "")
    f_spread = str(spec.get("spread") or "")
    f_runs = str(spec.get("runs") or "")
    token_rx = str(spec.get("token_pattern") or "")
    shape = _teach_or(
        step,
        "teach_shape",
        fields,
        f"record {fname} as a JSON list, one entry per alternative, each "
        f"carrying {f_label!r} and {f_status!r} from "
        f"{'/'.join(statuses) or 'the declared enum'}.",
    )

    comps = fields.get(fname)
    if isinstance(comps, str):
        try:
            comps = _json.loads(comps)
        except (ValueError, TypeError):
            comps = None
    if not isinstance(comps, list):
        return False, shape

    ref = ""
    for rf in spec.get("reference_from") or []:
        v = fields.get(str(rf))
        got = evidence.as_paths(v) or ([v] if v else [])
        if got:
            ref = str(got[0]).strip().lower()
            break
    tol = _tolerance_of(fields.get(str(spec.get("mode_from") or "")))
    min_runs = _stated_number(
        ctx,
        str((spec.get("min_runs_stated") or {}).get("pattern") or ""),
        (spec.get("min_runs_stated") or {}).get("default"),
    )
    max_spread = _stated_number(
        ctx,
        str((spec.get("max_spread_stated") or {}).get("pattern") or ""),
        (spec.get("max_spread_stated") or {}).get("default"),
    )

    eligible, seen_tokens = [], set()
    for i, e in enumerate(comps):
        if not isinstance(e, dict):
            return False, shape
        label = str(e.get(f_label) or "").strip()
        if not label:
            return False, f"entry #{i + 1} in {fname} records no {f_label}. {shape}"
        status = str(e.get(f_status) or "").strip().lower()
        if statuses and status not in statuses:
            return False, (
                f"entry {label!r} in {fname} has {f_status} "
                f"{status or 'nothing'!r}. Use exactly one of "
                f"{'/'.join(statuses)} -- free text would let anything "
                f"through, meaning whatever it was meant to mean."
            )
        token = re.search(token_rx, label) if token_rx else None
        if token:
            seen_tokens.add(token.group(0))

        if status != ok_status:
            if not str(e.get(f_detail) or "").strip():
                return False, (
                    f"entry {label!r} in {fname} is recorded {status!r} with "
                    f"no {f_detail}. Record what happened -- that difference "
                    f"is a finding, and is the reason these are kept rather "
                    f"than dropped."
                )
            continue

        if f_digest and tol is None:
            dig = str(e.get(f_digest) or "").strip().lower()
            if not _HEX_DIGEST.match(dig):
                return False, (
                    f"entry {label!r} in {fname} is recorded {ok_status!r} "
                    f"but has no {f_digest}. Run it and hash the output, "
                    f"exactly as for the reference."
                )
            # ref == "" refuses too: with no reference recorded there is nothing an
            # ok-entry's output can have matched, so "ok" is nobody's observation.
            if dig != ref:
                return False, _teach_or(
                    step,
                    "teach_digest_differs",
                    {**fields, "entry": label},
                    f"entry {label!r} in {fname} is recorded {ok_status!r}, but its "
                    f"{f_digest} is not the reference's. That is an entry whose output "
                    f"DIFFERS: record it with the status that says so, and a "
                    f"{f_detail}. Calling it {ok_status!r} hides the one finding the "
                    f"oracle exists to catch.",
                )
        elif f_dev and tol is not None:
            try:
                dev = float(e.get(f_dev))
            except (TypeError, ValueError):
                return False, (
                    f"entry {label!r} in {fname} is compared under a "
                    f"tolerance, so record {f_dev}: the largest difference "
                    f"between its output and the reference's. Without it, "
                    f"'within tolerance' is nobody's observation."
                )
            if dev > tol:
                return False, (
                    f"entry {label!r} in {fname} deviates by {dev:g}, outside "
                    f"the declared tolerance {tol:g} -- its output differs "
                    f"from the reference's. Record it with the status that "
                    f"says so."
                )

        cost = None
        if f_cost:
            try:
                cost = float(e.get(f_cost))
            except (TypeError, ValueError):
                return False, (
                    f"entry {label!r} in {fname} records no {f_cost}. Each "
                    f"entry offered needs its OWN figure, or there is "
                    f"nothing to compare against."
                )
        if f_spread:
            try:
                sp = float(e.get(f_spread))
            except (TypeError, ValueError):
                return False, (
                    f"entry {label!r} in {fname} records no {f_spread}. A "
                    f"figure with no variance says nothing about whether a "
                    f"difference is real."
                )
            if max_spread is not None and sp > float(max_spread):
                return False, (
                    f"entry {label!r} in {fname} has {f_spread}={sp:g}, above "
                    f"the {float(max_spread):g} this benchmark's spec sets."
                )
        if f_runs and min_runs is not None:
            try:
                runs = int(e.get(f_runs) or 0)
            except (TypeError, ValueError):
                runs = 0
            if runs < int(min_runs):
                return False, (
                    f"entry {label!r} in {fname} records {runs or 'no'} "
                    f"{f_runs}; this benchmark's spec asks for at least "
                    f"{int(min_runs)}."
                )
        if token and cost is not None:
            eligible.append((label, cost))

    if spec.get("require_eligible", True) and not eligible:
        return False, _teach_or(
            step,
            "teach_none_eligible",
            fields,
            f"no entry in {fname} gives a bar to clear: at least one "
            f"must carry a declared token in its {f_label}, have "
            f"{f_status} {ok_status!r}, and record its {f_cost}.",
        )
    if spec.get("require_stated_tokens"):
        baseline = _stated_token(ctx, spec)
        stated = []
        for m in re.finditer(token_rx, _instance_text(ctx)):
            t = m.group(0)
            if t != baseline and t not in stated:
                stated.append(t)
        missing = [t for t in stated if t not in seen_tokens]
        if missing:
            return False, _teach_or(
                step,
                "teach_stated_missing",
                {**fields, "missing": ", ".join(missing)},
                f"this benchmark's spec names {', '.join(missing)} "
                f"among the entries {fname} must offer, and no entry "
                f"records it. Record it -- with a not-{ok_status} "
                f"status and a {f_detail} if it did not work. "
                f"Leaving it out is the only wrong answer.",
            )
    derive = spec.get("derive") or {}
    if derive and eligible:
        best_label, best_cost = min(eligible, key=lambda x: x[1])
        if derive.get("label_into"):
            fields[str(derive["label_into"])] = best_label
        if derive.get("cost_into"):
            fields[str(derive["cost_into"])] = best_cost
    return None


def _variant_cmds_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """Each declared variant's command, proven in the fresh copy the variants gate
    just rebuilt. A recorded command that only works in the agent's own working
    directory is a claim; running it in the rebuilt copy -- and requiring it to
    reproduce the reference output -- makes it an observation."""
    import json as _json

    spec = step["variant_cmds"]
    fname = str(spec["field"])
    declared = [
        str(v.get("name"))
        for v in ((getattr(ctx, "instance", None) or {}).get("artifact_variants") or [])
        if isinstance(v, dict) and v.get("name")
    ]
    if not declared:
        return None

    raw = fields.get(fname)
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except (ValueError, TypeError):
            raw = None
    if not isinstance(raw, dict):
        return False, _teach_or(
            step,
            "teach_shape",
            fields,
            f"record {fname}: a JSON object with one entry per "
            f"variant, each the exact command that runs THAT "
            f"variant's build in its own copy, with "
            f"{{{spec.get('placeholder') or 'input_path'}}} where "
            f"the input goes.",
        )
    missing = [v for v in declared if not str(raw.get(v) or "").strip()]
    if missing:
        return False, f"{fname} records no command for variant(s): {', '.join(missing)}"

    placeholder = str(spec.get("placeholder") or "input_path")
    input_value = str(fields.get(str(spec.get("input") or "")) or "").strip()
    ref = ""
    for rf in spec.get("reference_from") or []:
        from agentcore.tools import evidence

        v = fields.get(str(rf))
        got = evidence.as_paths(v) or ([v] if v else [])
        if got:
            ref = str(got[0]).strip().lower()
            break
    tol = _tolerance_of(fields.get(str(spec.get("mode_from") or "")))
    root = str(fields.get(str(spec.get("root") or "")) or "").strip()
    run = _sandbox_runner(ctx, fields, str(spec.get("sandbox") or ""))

    for name in declared:
        template = str(raw[name])
        cmd = template.replace("{" + placeholder + "}", input_value)
        leftover = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", cmd)
        if leftover:
            return False, (
                f"{fname}[{name}] still contains placeholder(s) "
                f"{sorted(set(leftover))} after substituting "
                f"{{{placeholder}}}; that is the only placeholder a "
                f"consumer fills."
            )
        fresh = f"{root}/_verify_{name}"
        wrapped = f"cd {fresh} && ( {cmd} ) 2>&1 | sha256sum | awk '{{print $1}}'"
        code, out = run(wrapped, 900)
        got = (out or "").strip().splitlines()[-1].strip().lower() if (out or "").strip() else ""
        if code != 0 or not _HEX_DIGEST.match(got):
            tail = (out or "").strip()[-200:]
            return False, (
                f"{fname}[{name}] did not run in that variant's fresh copy "
                f"({fresh}): exit {code}, {tail or 'no output'}. The command "
                f"must work in the copy the harness rebuilds, not in your "
                f"working directory."
            )
        if tol is None and ref and got != ref:
            return False, _teach_or(
                step,
                "teach_differs",
                {**fields, "entry": name},
                f"{fname}[{name}] reproduces {got[:12]}... in the "
                f"fresh copy but the reference output is "
                f"{ref[:12]}.... A variant whose command does not "
                f"reproduce the reference is either the wrong "
                f"command or a broken build.",
            )
    return None


def _exclusions_on_step(step: dict, fields: dict, ctx) -> tuple[bool, str] | None:
    """Whatever THIS spec excludes, matched mechanically over the declared fields --
    the same rule the entry gate applies, available to any state's contract."""
    from agentcore.agent.exclusions import check_candidate

    inst = getattr(ctx, "instance", {}) or {}
    for fname in step.get("exclusions_on") or []:
        value = str(fields.get(str(fname)) or "").strip()
        if value:
            ok, why = check_candidate(value, inst)
            if not ok:
                return False, why
    return None


def _state_gate(state_name: str):
    """A verify-able gate bound to one state's declared steps: the instance carries
    `state_contracts: {STATE: [steps...]}` and agent.yaml wires
    `verify: agentcore.agent.field_contract:state__STATE`. Fails closed when the
    state declares nothing."""

    def gate(fields: dict, ctx) -> tuple[bool, str]:
        inst = getattr(ctx, "instance", {}) or {}
        steps = (inst.get("state_contracts") or {}).get(state_name)
        if not steps:
            return False, (
                f"HARNESS/SPEC FAULT: no declared contract for state "
                f"{state_name!r} (`state_contracts` in the instance). An "
                f"undeclared contract is a missing rule, not a permissive "
                f"one."
            )
        saves: dict = {}
        verdict = _evaluate_steps(steps, fields, ctx, saves)
        if verdict is not None:
            return verdict
        accept = str((inst.get("state_accepts") or {}).get(state_name) or "the declared contract for this state holds")
        pool = {k: str(v) for k, v in fields.items()}
        pool.update(saves)
        try:
            accept = accept.format_map(_Literal(pool))
        except Exception:  # noqa: BLE001
            pass
        return True, accept

    gate.__name__ = f"state__{state_name}"
    return gate


def __getattr__(name: str):
    if name.startswith("state__"):
        return _state_gate(name[len("state__") :])
    raise AttributeError(name)
