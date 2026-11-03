"""Whether the sampler pre-verifies runtime is the spec's decision, not a demand the harness makes.

A canonical that projects no runtime screening is a legitimate division of labour and passes. One
that projects some must project every pair its criteria state (a floor and a speedup), or the half
it kept is arbitrary.
"""

from __future__ import annotations

import json

from benchcompiler.spec import check_canonical

CODE = "MISSING_VIABILITY_PROJECTION"

CRITERIA = [
    # both triggers of the check: a floor ("at least N second") and a comparison ("speedup")
    "the costliest input must bring the default build's median runtime to at least 1 second",
    "the optimized build must achieve a measurable speedup over the default build there",
]


def _canonical(entry_clauses):
    base = json.load(open("benchcompiler/examples/boca_flags.canonical.json"))["canonical_spec"]
    cs = json.loads(json.dumps(base))
    cs["criteria"] = CRITERIA
    cs["entry_contract"] = [
        {"field": "repository_url", "require": "is a clone url"},
        {"field": "commit_hash", "require": "is a git sha"},
        *entry_clauses,
    ]
    cs["corroboration_contract"] = [
        {"field": "clone_path", "require": "is non-empty"},
        {
            "run": "git -C {clone_path} rev-parse --verify {commit_hash}^{commit}",
            "require": "output is a git sha",
            "derive": "resolved_commit",
        },
    ]
    cs["sampling"] = {**(cs.get("sampling") or {}), "lead_requires": ["repository_url", "commit_hash", "clone_path"]}
    return cs


def _mine(cs):
    return [(d.code, d.detail[:200]) for d in check_canonical(cs) if d.code == CODE]


def test_a_sampler_that_screens_cheaply_is_not_asked_for_a_probe():
    got = _mine(_canonical([]))
    print("\n[verbose]", got)
    assert got == [], "the harness demanded runtime screening the spec never asked for"


def test_a_sampler_that_screens_on_one_pair_must_screen_on_both():
    """A canonical projecting some runtime screening must project every pair its criteria state."""
    got = _mine(
        _canonical(
            [
                {"field": "probe_baseline_ms", "require": "is at least 1.05 times field probe_optimized_ms"},
            ]
        )
    )
    print("\n[verbose]", got)
    assert got, "a half-projected screening passed"
    assert "growth" in got[0][1] or "small" in got[0][1]


def test_both_pairs_projected_passes():
    got = _mine(
        _canonical(
            [
                {"field": "probe_large_ms", "require": "is at least 10 times field probe_small_ms"},
                {"field": "probe_large_ms", "require": "is at least 1.05 times field probe_optimized_ms"},
            ]
        )
    )
    print("\n[verbose]", got)
    assert got == []
