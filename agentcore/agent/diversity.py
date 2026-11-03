"""How many delivered instances may share one repository, one language, one anything.

WHY. Asking in the prompt for a few instances per source is not enough: cloning a repo,
installing its toolchain and getting its tests to run costs many turns, while taking another
instance out of the same checkout costs almost none. A target of N instances therefore pulls
the agent toward the repository already open, so the limit is enforced here instead of asked for. What it deliberately does NOT do is say where
to look next -- the task spec describes what a source has to be, and choosing one is the agent's
job.

Caps are declared in agent.yaml, not written into this file:

    worklist:
      caps:
        - dimension: repository
          keys: [upstream_url, repo_url]
          limit: 4
        - dimension: language
          keys: [language]
          limit: 12

Several can apply at once and a candidate has to satisfy all of them. Nothing here knows what any
task calls its fields, because a cap that assumed one field name would quietly do nothing on the
two collectors that use a different one.

Checked at two gates, for the reason audit_delivered exists next to the per-candidate checks:

    the first gate      early, before a container gets built for a candidate that could never be
                        delivered. Catches the usual case, where instances are finished one by one.
    the delivery gate   the one that actually holds. A validator's answer is cached per candidate,
                        so a batch registered before anything was delivered would pass the first
                        gate once and never be asked again.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from agentcore.agent import gates

# Used when a task declares a cap without a limit (WorklistSpec.cap_default_limit fills it
# into a loaded spec; this is the fallback for caps read without the model). One home:
# agentcore/agent/gates.py.
DEFAULT_LIMIT = gates.CAP_DEFAULT_LIMIT

# Fallback name for the state a candidate must reach before its source is charged a slot.
# Anything short of that cost its source nothing, so it must not consume one. The REAL name is
# read off the task's WorklistSpec.terminal when it can be reached -- the terminal state is
# per-task configuration, and a collector that renamed its success state would otherwise
# have a cap that counted nothing.
DELIVERED = "DELIVERED"

# scp-style git address: [user@]host:path -- what `git remote -v` prints for SSH remotes.
_SCP_LIKE = re.compile(r"^(?:[\w.-]+@)([\w.-]+):(.+)$")


def _field(cap, name):
    """One field off a cap, whether it is a CapSpec or a plain dict.

    Dicts are read with .get FIRST. getattr(some_dict, "keys") returns the built-in method, which
    is truthy and not the configured list -- a cap read that way iterates a bound method and
    matches nothing, which is a cap that silently does not exist.
    """
    if isinstance(cap, dict):
        return cap.get(name)
    return getattr(cap, name, None)


class ResolvedCap(NamedTuple):
    """One cap's declared fields, read once. `limit` already carries DEFAULT_LIMIT when the
    declaration left it blank; callers that treat a non-positive limit as "no cap" still say
    so themselves, because SourcePool deliberately keeps the keys of an unlimited cap."""

    keys: list
    limit: int
    urls: bool
    dimension: str


def resolve_cap(cap):
    """A cap as (keys, limit, urls, dimension), or None when it declares no keys.

    One reader shared by worklist_render, worklist_tools, sources and this module, so the
    dict-vs-CapSpec workaround `_field` exists for lives in one place.
    """
    keys = _field(cap, "keys")
    if not keys:
        return None
    # A scalar `keys: org` is a one-key cap, not three one-character keys. CapSpec validates
    # this away, but the dict path exists precisely for yaml loaded WITHOUT the model, and
    # list("org") there yields ['o','r','g'] -- value_of reads nothing and the cap silently
    # admits everything, the same "cap that does not exist" _field guards against.
    if isinstance(keys, str):
        keys = [keys]
    # int() is deliberately unguarded: every reader this replaced raised on a malformed
    # `limit:`, and silently substituting the default would enforce 4 for a spec that asked
    # for 10 without telling anyone.
    raw = _field(cap, "limit")
    limit = DEFAULT_LIMIT if raw in (None, "") else int(raw)
    urls = _field(cap, "urls")
    return ResolvedCap(
        list(keys), limit, True if urls is None else bool(urls), str(_field(cap, "dimension") or "source")
    )


def success_from_spec(spec) -> str:
    """The task's own success state off a WorklistSpec (or a plain dict carrying `terminal`).

    The ctx-free half of `terminal_from`, for the engines that hold a spec rather than a ctx
    and were each spelling `(spec.terminal or ["DELIVERED"])[0]` by hand.
    """
    term = spec.get("terminal") if isinstance(spec, dict) else getattr(spec, "terminal", None)
    if not term:
        return DELIVERED
    # A scalar `terminal: VERIFIED` names one state, not the letter "V" (watch_runs reads the
    # same shape defensively). term[0] on a str would be its first letter.
    return str(term) if isinstance(term, str) else str(term[0])


def _normalise(value: str, urls: bool) -> str:
    s = (value or "").strip().rstrip("/")
    if not s:
        return ""
    if urls:
        had_scheme = "://" in s
        if had_scheme:
            s = s.split("://", 1)[1]
        scp = _SCP_LIKE.match(s)
        if scp:  # git@github.com:x/y.git -> github.com/x/y.git
            s = scp.group(1) + "/" + scp.group(2)
        if s.lower().endswith(".git"):
            s = s[:-4]
        parts = [p for p in s.strip("/").split("/") if p]
        # A leading hostname is recognised by its dot, scheme or no scheme: the model writes
        # `github.com/x/y` straight out of prose as readily as a clone URL, and the two must
        # not open separate buckets.
        if len(parts) >= 3 and "." in parts[0]:
            return "/".join(parts[-2:]).lower()
        if had_scheme and len(parts) >= 2:
            return "/".join(parts[-2:]).lower()
        if scp or had_scheme:
            return "/".join(parts).lower()
    return s.lower()


def value_of(fields: dict, keys, *, urls: bool = True) -> str:
    """This candidate's value on one dimension, normalised so it compares equal to itself.

    One repository, one bucket, however it is spelt: with or without a scheme
    (https://github.com/x/y vs github.com/x/y), as an scp-style remote (git@github.com:x/y.git),
    with a .git suffix, a trailing slash, or different case. Each unmerged spelling multiplies
    the cap -- four spellings of one repo and a cap of 4 permit 16.

    WHICH KEY WINS. Taking the first non-empty key would leave a spelling hole: a cap may
    declare `keys: [repo, repo_url, project]` where `repo` is sometimes a bare name, and a bare
    name has no owner for `_normalise` to canonicalise, so `scrapy` and `scrapy/scrapy` would
    be two buckets against one limit. So under `urls`, a key whose value SURVIVES
    canonicalisation (it carries an owner) is preferred over one that does not: an ownerless
    value is a weaker spelling of the same address, not a different address, which is what
    declaring `urls` already asserts about these fields. The first non-empty value remains the
    answer when no key carries an owner -- a `language: c` dimension has none and never will.
    """
    first = ""
    for k in keys or ():
        v = fields.get(k)
        if not (isinstance(v, str) and v.strip()):
            continue
        out = _normalise(v, urls)
        if not out:
            continue
        if urls and "/" in out:
            return out
        first = first or out
    return first


def _spec_worklists(ctx) -> list:
    """Every worklist spec reachable from ctx, in preference order."""
    found = []
    for path in (("agent", "spec"), ("spec",)):
        obj = ctx
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        wl = getattr(obj, "worklist", None) if obj is not None else None
        if wl is not None:
            found.append(wl)
    return found


def caps_from(ctx) -> list:
    """The caps this task declared, or [] if it declared none or cannot be reached."""
    for wl in _spec_worklists(ctx):
        caps = getattr(wl, "caps", None)
        if caps:
            return list(caps)
    return []


def terminal_from(ctx) -> str:
    """The task's own success state -- the first entry of WorklistSpec.terminal -- or the
    conventional name when the spec cannot be reached."""
    for wl in _spec_worklists(ctx):
        if getattr(wl, "terminal", None):
            return success_from_spec(wl)
    return DELIVERED


# States whose candidates cost the agent nothing and release their diversity slot: a
# failed attempt must not lock its source out (that would punish honest trying). Every
# OTHER state -- delivered or still in flight -- holds a slot. Counting
# only DELIVERED at admission time would let two leads still in vetting each pass while
# the count stood just under the cap, then both land. A reservation closes that gap.
_DEAD_STATES = ("REJECTED", "BLOCKED")


def _counts(
    ws, keys, urls: bool, success: str, *, reserving: bool = False, exclude_fields: dict | None = None
) -> dict[str, int]:
    """Per-value candidate counts. success-state only by default; with reserving=True,
    every non-dead candidate counts (delivered + in flight), except the candidate being
    checked itself (exclude_fields) -- a registered lead re-validated mid-flight must
    not refuse itself for holding its own slot.

    `ws` is a working set OR the worklist store itself. The store form exists so the prompt's
    capacity line can be computed by THIS function rather than by a second implementation --
    two implementations could disagree (the gate refusing a source the line advertised as
    open), so there is one counter.
    """
    store = ws if isinstance(ws, dict) else getattr(ws, "worklist", None)
    if not isinstance(store, dict):
        return {}
    counts: dict[str, int] = {}
    excluded_self = False
    for cand in (store.get("candidates") or {}).values():
        state = cand.get("state")
        if reserving:
            if state in _DEAD_STATES:
                continue
            # Self-exclusion: at most ONE candidate, and only an in-flight one -- the
            # candidate under check is by definition not delivered yet, and excluding
            # by bare field-equality would also drop DELIVERED twins, reopening the
            # very overshoot this reservation exists to close.
            if (
                not excluded_self
                and state != success
                and exclude_fields is not None
                and (cand.get("fields") or {}) == exclude_fields
            ):
                excluded_self = True
                continue
        elif state != success:
            continue
        v = value_of(cand.get("fields") or {}, keys, urls=urls)
        if v:
            counts[v] = counts.get(v, 0) + 1
    return counts


def check_caps(fields: dict, ctx, *, caps=None) -> tuple[bool, str]:
    """(ok, detail). Refuses when any declared cap is already full for this candidate's value.

    Admits when the value cannot be read. The gate that requires that field will say so in words
    naming the field; a second refusal here would only tell the agent that something it cannot
    see is full, and a refusal it cannot act on costs as much as a wrong one.
    """
    caps = caps if caps is not None else caps_from(ctx)
    ws = getattr(ctx, "working_set", None)
    if not caps or ws is None:
        return True, ""
    success = terminal_from(ctx)

    for cap in caps:
        rc = resolve_cap(cap)
        if rc is None or rc.limit <= 0:
            continue
        keys, limit, urls, dim = rc

        value = value_of(fields, keys, urls=urls)
        if not value:
            continue
        counts = _counts(ws, keys, urls, success, reserving=True, exclude_fields=fields)
        n = counts.get(value, 0)
        if n < limit:
            continue
        delivered_n = _counts(ws, keys, urls, success).get(value, 0)
        in_flight = n - delivered_n
        others = sorted(k for k in counts if k != value)
        return False, (
            f"{value} has already given {delivered_n} delivered instances"
            + (f" and holds {in_flight} more still in flight" if in_flight else "")
            + f", which fills the limit ({limit}) for "
            f"any one {dim}. Nothing is wrong with this candidate -- it is fine -- and another "
            f"one from {value} will be refused in the same way. Find a DIFFERENT {dim} and bring "
            f"a candidate from that. "
            f"Already used: {', '.join(others) or 'nothing else yet'}. "
            f"The spec says what a {dim} has to look like; which one you pick is your call."
        )
    return True, ""


def capacity_summary(ctx) -> str:
    """One dashboard line: where the caps still have room, and what is FULL.

    A refusal message names the problem repo; this line names the REMAINING ROOM, so the
    agent can see which sources are still open.
    Reservation semantics match check_caps (in-flight holds a slot).
    """
    caps = caps_from(ctx)
    ws = getattr(ctx, "working_set", None)
    if not caps or ws is None:
        return ""
    parts, room = [], {}
    success = terminal_from(ctx)  # fixed across caps, as in check_caps
    for cap in caps:
        rc = resolve_cap(cap)
        if rc is None or rc.limit <= 0:
            continue
        keys, limit, urls, dim = rc
        counts = _counts(ws, keys, urls, success, reserving=True)
        full = sorted(k for k, n in counts.items() if n >= limit)
        open_ = sorted((k, limit - n) for k, n in counts.items() if n < limit)
        seg = f"{dim} cap {limit}: "
        seg += ("FULL: " + ", ".join(full[:6]) + ". ") if full else ""
        if open_:
            seg += "room left: " + ", ".join(f"{k}({r})" for k, r in open_[:6])
        elif counts:
            seg += "every used " + dim + " is full -- bring a NEW one"
        else:
            # Not "every used X is full": this branch is also reached with an EMPTY
            # worklist, where that statement would be unearned.
            seg += f"nothing registered yet -- up to {limit} per {dim}"
        parts.append(seg)
        room[dim] = _pool_room(ctx, keys, urls, limit, counts)
    line = " | ".join(parts)

    # THE AGGREGATE, not only the per-source room. `sources.gate` computes exactly this at the
    # door and then never runs again, so a pool that becomes too small mid-flight would be
    # discovered only by running out, one cap refusal at a time.
    target = _target_of(ctx)
    done = _delivered_count(ctx)
    if target and room:
        dim, left = min(room.items(), key=lambda kv: kv[1])
        if done + left < target:
            line += (
                f" || YOUR POOL CANNOT REACH THE TARGET: {done} delivered and room for "
                f"{left} more across every {dim} you have, which is {done + left} against "
                f"{target}. No candidate can fix that -- add more with sources_add."
            )
    return line


def _pool_room(ctx, keys, urls, limit, counts) -> int:
    """Slots left across the WHOLE declared pool, not only the sources already used.

    An untouched pool entry holds nothing and so appears in no count; leaving it out would
    under-report the ceiling and cry shortfall at a run with plenty of room.
    """
    ws = getattr(ctx, "working_set", None)
    snap = ws.snapshot() if hasattr(ws, "snapshot") else (ws if isinstance(ws, dict) else {})
    # ToolContext has no `spec` attribute -- the agent's spec hangs off ctx.agent, the
    # path _spec_worklists above already walks.
    _spec = getattr(getattr(ctx, "agent", None), "spec", None) or getattr(ctx, "spec", None)
    key = getattr(getattr(_spec, "sources", None), "key", None) or "source_pool"
    entries = [e for e in (snap.get(key) or []) if isinstance(e, dict)]
    if not entries:
        return 0
    seen, total = set(), 0
    for e in entries:
        v = value_of(e, keys, urls=urls)
        if not v or v in seen:
            continue
        seen.add(v)
        total += max(0, limit - counts.get(v, 0))
    return total


def _target_of(ctx) -> int:
    inst = getattr(ctx, "instance", None)
    try:
        return int((inst or {}).get("sample_size") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def _delivered_count(ctx) -> int:
    ws = getattr(ctx, "working_set", None)
    store = getattr(ws, "worklist", None)
    if not isinstance(store, dict):
        return 0
    term = terminal_from(ctx)
    return sum(1 for c in (store.get("candidates") or {}).values() if isinstance(c, dict) and c.get("state") == term)
