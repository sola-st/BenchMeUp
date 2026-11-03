"""A spec must say how a finished instance is checked, or nothing downstream can certify it.

`artifact_variants` (named variants, each with a `check_cmd` and an `expect`) lets `bench merge`
render the interface mechanically and `bench certify` re-execute every instance against it. Without
a declared contract a model would have to invent the interface, which cannot be verified. The
declaration is required, as `admissibility` is for what goes in; this governs whether what comes out
can be verified.
"""

from __future__ import annotations

import pytest

from benchcompiler.spec import DIAGNOSTIC_CODES, check_verifiability

GOOD = {
    "artifact_variants": [
        {
            "name": "fixed",
            "description": "tree at the fix commit",
            "check_cmd": "mvn -q -Dtest={triggering_test} test",
            "expect": "pass",
        },
        {
            "name": "buggy",
            "description": "tree at the parent commit",
            "check_cmd": "mvn -q -Dtest={triggering_test} test",
            "expect": "fail",
        },
    ]
}


def _codes(raw):
    return {d.code for d in check_verifiability(raw)}


def test_the_codes_are_registered():
    assert "MISSING_VARIANTS" in DIAGNOSTIC_CODES
    assert "UNCHECKABLE_VARIANTS" in DIAGNOSTIC_CODES


def test_a_raw_spec_declaring_no_variants_is_NOT_refused_here():
    """Whether a variant contract applies depends on `oracle_strategy`, which the formalizer
    assigns and a raw spec does not carry; `check_canonical` requires it of the oracles that
    contrast two states of the artifact (see
    test_variants_required_only_where_they_mean_something.py).
    """
    assert "MISSING_VARIANTS" not in _codes({"benchmark_name": "x"})


def test_a_declared_contract_passes():
    assert _codes(GOOD) == set()


# ── what makes a variant checkable ───────────────────────────────────────────────────────────


def test_a_variant_with_no_check_command_is_refused():
    """A named variant nobody can run leaves the certifier nothing to drive."""
    bad = {"artifact_variants": [{"name": "fixed", "expect": "pass"}]}
    assert "UNCHECKABLE_VARIANTS" in _codes(bad)


def test_a_variant_with_no_expected_outcome_is_refused():
    """`expect` is the polarity; without it the polarity would have to be guessed from the English
    description and could be inverted.
    """
    bad = {"artifact_variants": [{"name": "fixed", "check_cmd": "make test"}]}
    assert "UNCHECKABLE_VARIANTS" in _codes(bad)


@pytest.mark.parametrize("expect", ["passes", "true", "0", "ok", ""])
def test_expect_must_be_pass_or_fail(expect):
    """Read by `_variant_expectations`, which understands exactly two words."""
    bad = {"artifact_variants": [{"name": "v", "check_cmd": "make test", "expect": expect}]}
    assert "UNCHECKABLE_VARIANTS" in _codes(bad)


def test_a_variant_with_no_name_is_refused():
    """The name is what `rebuild-<variant>` / `check-<variant>` are built from."""
    bad = {"artifact_variants": [{"check_cmd": "make test", "expect": "pass"}]}
    assert "UNCHECKABLE_VARIANTS" in _codes(bad)


def test_at_least_one_variant_must_be_expected_to_pass():
    """A contract where everything is expected to fail cannot demonstrate a working baseline,
    so a green run and a broken artifact are indistinguishable."""
    bad = {"artifact_variants": [{"name": "buggy", "check_cmd": "make test", "expect": "fail"}]}
    assert "UNCHECKABLE_VARIANTS" in _codes(bad)


def test_the_refusal_can_be_acted_on():
    """The requirement lives at the canonical stage, so the message is checked there and is
    actionable.
    """
    from benchcompiler.spec import check_canonical

    d = [x for x in check_canonical({"oracle_strategy": "fail-to-pass-tests"}) if x.code == "MISSING_VARIANTS"][0]
    assert d.field == "artifact_variants"
    assert d.fix_hint and "check_cmd" in d.fix_hint and "expect" in d.fix_hint


# ── it must not name a task ──────────────────────────────────────────────────────────────────

BANNED = ["maven", "mvn", "pytest", "java", "python", "gcc", "commit", "test", "compile"]


def test_the_check_names_no_ecosystem():
    """Whatever this refuses, it must refuse for a contest archive and a corpus directory the
    same way it does for a git repository."""
    import inspect
    import re

    from benchcompiler import spec as _spec

    src = inspect.getsource(_spec.check_verifiability).lower()
    hits = [w for w in BANNED if re.search(rf"\b{w}s?\b", src)]
    assert not hits, f"task-specific vocabulary in a generic gate: {hits}"
