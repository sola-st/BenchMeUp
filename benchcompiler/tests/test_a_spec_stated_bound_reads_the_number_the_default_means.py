"""The number a spec-stated pattern reads from the canonical must be the number its default means.

A clause declares `spec.pattern` (read from the spec's text) and `spec.default` (used when the
pattern finds nothing). The checker reads the bound as the engine will (pattern, times `spec.scale`,
over the canonical's criteria text) and compares it to the default: a mismatch is refused with
`scale` named; a consistent pair is accepted.
"""

from __future__ import annotations

import json

from benchcompiler.spec import check_canonical

CLAUSE = "is at least the spec-stated minimum"


def _canonical(spec):
    base = json.load(open("benchcompiler/examples/boca_flags.canonical.json"))["canonical_spec"]
    cs = json.loads(json.dumps(base))
    cs["criteria"] = [
        "the costliest input must bring the default build's median runtime to "
        "at least 1 second (median over at least 5 runs)"
    ]
    cs["entry_contract"] = [c for c in cs["entry_contract"] if c.get("require") != CLAUSE]
    cs["entry_contract"].append({"field": "probe_baseline_ms", "require": CLAUSE, "teach": "floor", "spec": spec})
    return cs


def _mine(cs):
    return [
        (d.code, d.detail[:220], d.fix_hint[:120])
        for d in check_canonical(cs)
        if d.field == "entry_contract" and "spec-stated" in d.detail
    ]


def test_a_pattern_that_reads_one_when_the_default_means_a_thousand_is_refused():
    got = _mine(_canonical({"pattern": r"at least (\d+) second", "default": 1000}))
    print("\n[verbose]", got)
    assert got and got[0][0] == "UNREADABLE_CONTRACT", got
    assert "scale" in (got[0][1] + got[0][2])


def test_a_declared_scale_that_makes_them_agree_passes():
    got = _mine(_canonical({"pattern": r"at least (\d+) second", "default": 1000, "scale": 1000}))
    print("\n[verbose]", got)
    assert got == []


def test_a_pattern_that_finds_nothing_in_the_text_is_refused_too():
    """The bound must be readable from THIS canonical's text, or the default is all
    that ever runs and the pattern is decoration."""
    got = _mine(_canonical({"pattern": r"at least (\d+) fortnight", "default": 1000}))
    print("\n[verbose]", got)
    assert got and got[0][0] == "UNREADABLE_CONTRACT", got
