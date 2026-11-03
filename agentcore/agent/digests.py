"""Hashes that prove nothing, because they are the hash of no bytes at all.

A program that builds and runs but prints nothing yields identical output hashes on every run,
all equal to e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855, the SHA-256 of
the empty string. "Three runs produced identical output" is then true and meaningless.

This is kept separate from any one collector because the shape is general: any check comparing
hashes of captured output can be satisfied by capturing none, and every task that hashes anything
has the same hole.
"""

from __future__ import annotations

import hashlib

# The digest of b"" under each hash an agent is likely to reach for at a shell prompt. Computed
# rather than pasted -- a typo'd constant here would silently re-open the hole it exists to close.
EMPTY_DIGESTS = frozenset(
    hashlib.new(name, b"").hexdigest() for name in ("md5", "sha1", "sha224", "sha256", "sha384", "sha512")
)

# The same, for the empty-file forms some tools emit ("d41d8cd9...  -"). Callers strip before
# lookup, so this is only for readers wondering why a bare hex string is enough.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def is_empty_digest(value) -> bool:
    """True when this is the hash of zero bytes under any common algorithm."""
    return str(value or "").strip().lower() in EMPTY_DIGESTS


def all_empty(values) -> bool:
    """True when every value given is the hash of zero bytes (and there is at least one).

    All of them, not any: a single empty digest among real ones is a run that produced no output
    that time, which the existing "the runs disagreed" check already catches and describes better.
    """
    vals = [v for v in (values or []) if str(v or "").strip()]
    return bool(vals) and all(is_empty_digest(v) for v in vals)
