"""Resolve an abbreviated git commit to its full 40-character hash.

WHY. `e806e6b` is a perfectly valid git prefix; a model that records one has recorded
something true, and refusing it would be a harness design error, not a model failure. The
clone is already present and `git rev-parse` answers in milliseconds, so the prefix is
resolved here instead of bounced back.

WHAT THE REFUSAL PROTECTS. A model may take a 7-character abbreviation and complete it with
33 invented characters. That guarantee survives: resolution runs against the
real repository, so a fabricated hash fails to resolve and is still refused, while a GENUINE
prefix succeeds. See tests/test_gitsha.py.
"""

from __future__ import annotations

import re
import shlex

_HEX = re.compile(r"^[0-9a-f]+$")
FULL_LEN = 40
# Below this, a prefix is too weak to be meaningful: 4 hex chars is 65k values and will collide in
# any real repository. git itself refuses fewer than 4.
MIN_PREFIX = 7
# `<hash>^`, `<hash>~1`, `<hash>^^` -- git's own way of naming an ancestor. A value such as
# parent_commit='e2eb7d1a...589^' is the correct parent of the fix commit, written exactly as git
# writes it, and must not be refused as "not a usable commit hash" merely because the string is
# not pure hex. Anchoring on a hash and letting `rev-parse --verify` do the work costs nothing: whatever
# resolves is real, which is the only property the refusal protects. Deliberately NOT
# accepting bare names like `main` or `HEAD~1` -- those resolve too, but to whatever the branch
# points at today, which is not something the model observed for this candidate.
_ANCESTOR = re.compile(r"^(?P<base>[0-9a-f]{%d,%d})(?P<walk>(?:\^\d*|~\d*)+)$" % (MIN_PREFIX, FULL_LEN))


def is_full(sha: str) -> bool:
    s = (sha or "").strip().lower()
    return len(s) == FULL_LEN and bool(_HEX.match(s))


def is_abbreviated(sha: str) -> bool:
    """A plausible prefix -- hex, shorter than full, long enough to be worth resolving."""
    s = (sha or "").strip().lower()
    return MIN_PREFIX <= len(s) < FULL_LEN and bool(_HEX.match(s))


def is_ancestor_expr(sha: str) -> bool:
    """`abc1234^`, `abc1234~2` -- a hash we can check, plus a walk git knows how to perform."""
    return bool(_ANCESTOR.match((sha or "").strip().lower()))


def resolve(sha: str, *, runner, repo_path: str) -> tuple[str | None, str]:
    """(full_sha, detail). None means it could not be resolved -- which is a real refusal.

    `runner` is a callable (cmd, ...) -> (returncode, output), so this works identically against
    the agent's own environment, a per-candidate container, or a plain subprocess in tests. The
    module deliberately does not know which of those it is talking to.

    Uses `^{commit}` so a prefix that resolves to a tree or tag is rejected rather than quietly
    accepted as a commit.
    """
    s = (sha or "").strip().lower()
    if is_full(s):
        # Still confirm it exists: a full-length hash can be fabricated too, and that is precisely
        # what this check guards against.
        code, out = runner(f"git -C {shlex.quote(repo_path)} rev-parse --verify {shlex.quote(s)}^{{commit}}")
        if code == 0 and is_full(out.strip().splitlines()[0] if out.strip() else ""):
            return out.strip().splitlines()[0], "already full, and it exists"
        return None, f"{s} is 40 characters but does not exist in this repository"
    if is_ancestor_expr(s):
        code, out = runner(f"git -C {shlex.quote(repo_path)} rev-parse --verify {shlex.quote(s)}^{{commit}}")
        first = (out or "").strip().splitlines()[0].strip() if (out or "").strip() else ""
        if code == 0 and is_full(first):
            return first, f"resolved {s} -> {first}"
        return None, (
            f"{s} does not resolve in this repository. Check the hash it is anchored "
            f"on, and that the repository actually has that ancestor."
        )
    if not is_abbreviated(s):
        return None, (f"{s!r} is not a usable commit hash (expected {MIN_PREFIX}-{FULL_LEN} hex characters)")

    code, out = runner(f"git -C {shlex.quote(repo_path)} rev-parse --verify {shlex.quote(s)}^{{commit}}")
    first = (out or "").strip().splitlines()[0].strip() if (out or "").strip() else ""
    if code == 0 and is_full(first):
        return first, f"resolved {s} -> {first}"

    low = (out or "").lower()
    if "ambiguous" in low:
        return None, (
            f"{s} is ambiguous in this repository -- it matches more than one object. Record more characters."
        )
    # The fabrication case: a prefix that simply is not in the repository.
    return None, (
        f"{s} does not exist in this repository. If you abbreviated a hash you saw in "
        f"real output, re-read that output; do not complete it from memory."
    )
