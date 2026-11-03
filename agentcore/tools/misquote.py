"""A value of the candidate, retyped wrongly, proves nothing about the candidate.

A worker may copy a commit hash with a few characters swapped, fetch the wrong hash and drop a
valid candidate. The harness knows the candidate's exact values; a drop whose reason cites a
near miss of one of them is refused with the exact value. Task-blind: any hex value of twelve characters or more, whatever it names.
"""

from __future__ import annotations

import re

_HEX = re.compile(r"(?<![0-9a-zA-Z])[0-9a-fA-F]{12,64}(?![0-9a-zA-Z])")
NEAR = 4  # characters that may differ before two hashes are simply different hashes


def _hex_values(fields: dict) -> dict[str, str]:
    """{hex value -> field name} over the candidate's string fields."""
    out: dict[str, str] = {}
    for name, v in (fields or {}).items():
        if isinstance(v, str):
            for h in _HEX.findall(v):
                out.setdefault(h.lower(), str(name))
    return out


def misquoted_value(text: str, fields: dict) -> str | None:
    """The refusal when `text` names a hex value that is a near miss of one of the candidate's
    values and equal to none of them; None otherwise."""
    known = _hex_values(fields)
    if not known:
        return None
    for tok in _HEX.findall(str(text or "")):
        t = tok.lower()
        if t in known:
            continue
        for h, name in known.items():
            if len(h) != len(t):
                continue
            diff = [i for i, (a, b) in enumerate(zip(h, t)) if a != b]
            if 0 < len(diff) <= NEAR:
                pos = ", ".join(str(i + 1) for i in diff)
                return (
                    f"The reason names {tok}. That is not a value of this candidate: its {name} is "
                    f"{h}, which differs at character{'s' if len(diff) > 1 else ''} {pos}. A retyped "
                    f"value proves nothing about the candidate. Copy {name} exactly as the candidate "
                    f"line gives it, check again, and drop only if the exact value fails."
                )
    return None
