"""Enforce the spec's own exclusion list mechanically, instead of asking the model to obey prose.

WHY THIS EXISTS. Prose that lists excluded projects is not reliably obeyed: a model can read
"commons-csv is not excluded (the excluded one is Csv, not commons-csv)" and mine it anyway, and
a delivered instance that is really an excluded project is then counted as a success because
nothing checked. Prominence in the prompt is not the remedy. The constraint has to be a
harness-side check that runs on evidence the model records, like every other worklist verify.

NOTHING HERE NAMES A BENCHMARK. The framework must not know about any particular benchmark (that
is the whole generality claim), so the excluded names are EXTRACTED FROM THE INSTANCE SPEC at
runtime, keyed on the word "exclude" rather than on any suite. A spec that excludes nothing gets
no gate; a spec that excludes something else gets that enforced instead, for free.

KNOWN LIMITATION, stated rather than hidden: matching a spec's short id ("Csv") to a repository
name ("commons-csv") is a heuristic (see `violates`). It is not a proof: a spec id such as
`JacksonXml` against the real repo `jackson-dataformat-xml` is a case it does NOT catch. This
gate reduces a class of error; it does
not eliminate it, and it is deliberately biased toward false negatives, because a false positive
here refuses a legitimate candidate the run cannot recover.
"""

from __future__ import annotations

import re

# Where a spec might state its exclusions. Order is irrelevant; all are scanned and merged.
_SPEC_FIELDS = ("source_requirements", "criteria", "task_description", "task")

# "excluded: A, B, C" / "are excluded: ..." / "Excluded projects: ...". Keyed on the word, not on
# any benchmark. Stops at a sentence end so it cannot swallow the rest of the paragraph.
# `/` and `+` belong in this class because a spec may name repositories in owner/repo form, and
# leaving them out would stop the match at the first slash
# of "Excluded repositories: astropy/astropy, django/django, ...", so excluded_names() would
# return [] and check_candidate would answer "the spec excludes nothing" for EVERY repo. A
# broken parser here fails silently, not with an error.
# The FORBIDDING PHRASE. "exclude" is not the only wording a spec may use: a roster can be
# stated as "Project must NOT be one of the 12 repos already used by ...: astropy/astropy, ...".
# A parser that only recognises one phrasing fails silently (see the `/` note above).
#
# Each alternative must still be followed by a colon OR AN OPENING PARENTHESIS and a
# comma-separated list within the same sentence, which is what keeps ordinary prose out: "the
# program must not be one that never terminates" has neither, and no list.
#
# The parenthesis matters: a roster written "...already use (astropy/astropy, django/django,
# ...)" would return [] under a colon-only regex and nothing would be excluded.
_FORBIDS = r"(?:exclud\w*|(?:must|may|shall|can)\s+not\s+be\s+(?:one\s+of|among|any\s+of))"
_EXCL = re.compile(
    # `*` belongs in the class for the same reason `/` does: territory exclusions are
    # written "apache/*" (organisation syntax), and a class without `*` truncates the
    # capture at the first territory entry.
    _FORBIDS + r"[^:.(]{0,80}[:(]\s*([A-Za-z0-9][A-Za-z0-9 ,._/*+-]{2,600}?)\s*(?:\)|\.\s|\.$|;|\n|$)",
    re.IGNORECASE,
)

# Phrases that appear inside a captured list but are prose, not names.
_NOISE = re.compile(r"^(and|or|the|any|from|listed|following|set|of|projects?|etc)$", re.IGNORECASE)

# Below this length a name is matched only as a whole token, never as a substring: "Cli" as a
# substring would flag "client", "Time" would flag "timezone". Above it, substring matching is
# what catches `Chart` inside `jfreechart`.
_SUBSTRING_MIN = 5


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t}


# Segments that are hosting infrastructure rather than any part of a project's name.
_HOST_WORDS = {
    "github",
    "gitlab",
    "bitbucket",
    "com",
    "org",
    "net",
    "io",
    "https",
    "http",
    "www",
    "git",
    "sourceforge",
    "savannah",
    "gnu",
}


def _owner_and_repo(candidate: str) -> tuple[str, str]:
    """Split whatever identifies a candidate into (owner, repo).

    Callers hand this very different things: "apache/commons-lang", a clone URL, an scp-style
    "git@github.com:psf/requests.git", a bare "joda-time", and a URL followed by a
    program name in one string. Returns ("", <whole>) when no owner can be determined, which is
    the case a bare project id has to keep working in.
    """
    for word in str(candidate).split():
        w = word.strip().rstrip("/")
        if "://" in w:
            w = w.split("://", 1)[1]
        if w.startswith("git@") and ":" in w:
            w = w.split(":", 1)[1]
        if w.lower().endswith(".git"):
            w = w[:-4]
        if "/" not in w:
            continue
        parts = [p for p in w.split("/") if p and p.lower() not in _HOST_WORDS]
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        if parts:
            return "", parts[-1]
    whole = str(candidate).strip()
    if whole.lower().endswith(".git"):
        whole = whole[:-4]
    return "", whole


def excluded_names(instance: dict) -> list[str]:
    """Every name the instance spec says is excluded, in the spec's own spelling.

    Returns [] when the spec excludes nothing -- which must leave the caller with no gate at all,
    not an empty gate that rejects everything.
    """
    names: list[str] = []
    seen: set[str] = set()
    for field in _SPEC_FIELDS:
        text = instance.get(field)
        if not isinstance(text, str):
            continue
        for m in _EXCL.finditer(text):
            for raw in m.group(1).split(","):
                name = raw.strip(" .;")
                # A captured run can trail into prose ("Time) are excluded from the corpus");
                # keep only the leading name-like token sequence.
                name = re.sub(r"\s.*$", "", name) if " " in name and len(name.split()) > 2 else name
                if len(name) < 2 or _NOISE.match(name) or not re.match(r"^[A-Za-z]", name):
                    continue
                key = _squash(name)
                if key and key not in seen:
                    seen.add(key)
                    names.append(name)
    return names


def violates(candidate: str, excluded: list[str]) -> str | None:
    """Return the excluded name a candidate repo/program matches, or None.

    `candidate` is whatever identifies the thing being mined -- "apache/commons-lang",
    "https://github.com/apache/commons-csv.git", "joda-time". Owner prefixes and URL noise are
    stripped by tokenisation, so the caller does not have to normalise first.

    Three matching rules, from strongest to weakest:
      1. the excluded name equals a token of the candidate      commons-lang   ~ Lang
      2. it equals the candidate's squashed form                jackson-core   ~ JacksonCore
      3. it is a substring, only if long enough to be safe      jfreechart     ~ Chart
    """
    if not candidate or not excluded:
        return None
    owner, repo = _owner_and_repo(candidate)
    # Tokens come from the REPO and from any extra words (a lead may be "<url> <program>"), never
    # from the owner. Including the owner would make a spec excluding astropy/astropy refuse
    # astropy/astroquery, and pytest-dev/pytest refuse pytest-dev/pluggy: whole organisations of
    # admissible projects.
    toks = _tokens(repo) | {t for w in str(candidate).split() if "/" not in w and "://" not in w for t in _tokens(w)}
    toks -= _HOST_WORDS
    squashed = _squash(repo)

    for name in excluded:
        n = _squash(name)
        if not n:
            continue

        # An excluded name written as "owner/*" names an ORGANISATION -- every repo under
        # that owner is out. This is the fan-out territory syntax: bare names stay project
        # ids (including owners in bare-name matching would refuse whole
        # organisations), single "owner/repo" names stay one repository, and only
        # the explicit trailing "/*" claims the org. Bare names such as "Excluded: apache, google"
        # could never match an owner/repo-shaped candidate.
        if name.endswith("/*"):
            exc_owner = _squash(name[:-2])
            if exc_owner and owner and _squash(owner) == exc_owner:
                return name
            continue

        # An excluded name written as owner/repo names ONE repository. Both halves must match.
        # Comparing only the repo half is what produced the false positives above; comparing the
        # squashed whole ("pytestdevpytest") is what made the exclusion unmatchable before that.
        if "/" in name:
            exc_owner, exc_repo = name.rsplit("/", 1)
            if _squash(exc_repo) != squashed:
                continue
            # A candidate recorded without its owner is still the same project, so a bare "pytest"
            # matches "pytest-dev/pytest". With an owner present, it has to be the right one.
            if not owner or _squash(owner) == _squash(exc_owner):
                return name
            continue

        # A bare excluded name is a project id, matched against the repo only.
        #   1. equals a token of the repo         commons-lang   ~ Lang
        #   2. equals the repo's squashed form    jackson-core   ~ JacksonCore
        #   3. a substring, if long enough        jfreechart     ~ Chart
        if n in toks or n == squashed:
            return name
        if len(n) >= _SUBSTRING_MIN and n in squashed:
            return name
    return None


def check_candidate(candidate: str, instance: dict) -> tuple[bool, str]:
    """(ok, detail) -- the shape every worklist `verify:` uses.

    The rejection message corrects the SPECIFIC misreading that caused the real violation, since
    a model that concluded "Csv is not commons-csv" once will conclude it again from a message
    that only says "excluded".
    """
    excluded = excluded_names(instance or {})
    if not excluded:
        return True, "the spec excludes nothing"
    hit = violates(candidate, excluded)
    if hit:
        return False, (
            f"{candidate!r} is {hit!r}, which this benchmark's spec explicitly excludes. Match on "
            f"the REPOSITORY name, not the spec's short id -- {hit!r} on the exclusion list is the "
            f"repo you just picked. Drop this candidate and choose a project that is not on the "
            f"list: {', '.join(excluded)}."
        )
    return True, f"not among the {len(excluded)} excluded projects"
