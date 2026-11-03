"""`is at least the spec-stated minimum` must say where the spec states it.

The clause reads the bound from the spec's own text at runtime via `spec: {pattern, default}`.
Without a `pattern` no bound is read and the clause enforces nothing, so at formalize time a spec-
stated-minimum clause without a `spec.pattern` capturing one number is UNREADABLE_CONTRACT (model-
authored).
"""

from __future__ import annotations

import json

from benchcompiler.spec import check_canonical

CLAUSE = "is at least the spec-stated minimum"


def _canonical(clause_extra):
    base = json.load(open("benchcompiler/examples/boca_flags.canonical.json"))["canonical_spec"]
    cs = json.loads(json.dumps(base))
    cs["entry_contract"] = [c for c in cs["entry_contract"] if c.get("require") != CLAUSE]
    cs["entry_contract"].append(
        {"field": "probe_baseline_ms", "require": CLAUSE, "teach": "the floor the spec states", **clause_extra}
    )
    return cs


def _mine(cs):
    return [
        (d.code, d.detail[:200])
        for d in check_canonical(cs)
        if d.field == "entry_contract" and "spec-stated" in d.detail
    ]


def test_a_clause_without_a_pattern_is_refused_and_says_what_to_declare():
    got = _mine(_canonical({}))
    print("\n[verbose]", got)
    assert got and got[0][0] == "UNREADABLE_CONTRACT", got
    assert "pattern" in got[0][1]


def test_a_pattern_that_captures_no_number_is_refused():
    got = _mine(_canonical({"spec": {"pattern": "at least one second"}}))
    print("\n[verbose]", got)
    assert got and got[0][0] == "UNREADABLE_CONTRACT", got


def test_a_clause_with_a_capturing_pattern_passes():
    got = _mine(_canonical({"spec": {"pattern": r"at least (\d+) second", "default": 1000, "scale": 1000}}))
    print("\n[verbose]", got)
    assert got == []
