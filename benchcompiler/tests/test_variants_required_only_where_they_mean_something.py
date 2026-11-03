"""Require a variant contract of the benchmarks that can have one, and only those.

The requirement is a canonical-stage check keyed on `oracle_strategy`: an oracle whose ground truth
comes from contrasting two or more states of the artifact (before/after a fix, clean/injected,
candidate/reference) needs `rebuild-<name>`/`check-<name>` variants; oracles such as llm-judge,
human-annotation, external-authority and rerun-labeling do not. The raw-stage gate keeps only the
well-formedness rules, which apply whenever variants are declared.
"""

from __future__ import annotations

import pytest

from benchcompiler.registry import ORACLE_PATTERNS, VARIANT_ORACLE_PATTERNS
from benchcompiler.spec import check_canonical, check_verifiability

# `results:` is part of a well-formed variant: what a check is judged on is declared, never left to
# its exit code. This fixture is about a different rule, so it declares the evidence.
CONTRACT = [
    {"name": "fixed", "check_cmd": "run {t}", "expect": "pass", "results": "junit-xml", "expect_failures": 0},
    {"name": "buggy", "check_cmd": "run {t}", "expect": "fail", "results": "junit-xml", "expect_failures": 1},
]


def _canon(strategy, variants=None):
    # solver_input and (with variants) negative_control are asked of every canonical spec; this file is
    # about a different rule, so the fixture declares them.
    d = {
        "oracle_strategy": strategy,
        "task": "t",
        # Asked of every canonical spec: the formalizer's own output must say what the benchmark is.
        "candidate_definition": "one fix commit",
        "instance_input": "a buggy tree",
        "instance_output": "a patch",
        "data_source": "fix commits",
        "oracle_detail": "the project's own test suite decides",
        "solver_input": {
            "kind": "workdir",
            "path": "{clone_path}",
            "describes": "the solver edits the checked-out tree",
        },
        "negative_control": {"break_cmd": "corruption_cmd", "check": "buggy", "describes": "undo the fix"},
        # Asked of every canonical spec: admissibility is also checked on the way out, because a model
        # resolving prose sources produces it too.
        "admissibility": [{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"}],
    }
    if variants is not None:
        d["artifact_variants"] = variants
        # a declared report kind needs its discovery patterns; the fixture declares them like every block above
        d["report_discovery"] = {"junit-xml": ["*surefire-reports/TEST-*.xml", "TEST-*.xml"]}
    return d


def _codes(d):
    return {x.code for x in check_canonical(d)}


# ── the classification itself ────────────────────────────────────────────────────────────────


def test_every_classified_pattern_is_a_real_pattern():
    """A typo here would silently exempt a whole class of benchmark from the requirement."""
    unknown = sorted(set(VARIANT_ORACLE_PATTERNS) - set(ORACLE_PATTERNS))
    assert not unknown, unknown


@pytest.mark.parametrize("strategy", ["fail-to-pass-tests", "execution-replay", "injection", "reference-equivalence"])
def test_a_contrast_oracle_must_declare_variants(strategy):
    """Ground truth from contrasting two states of the artifact -- which is what
    rebuild-<name>/check-<name> expresses and the certifier re-executes."""
    assert strategy in VARIANT_ORACLE_PATTERNS
    assert "MISSING_VARIANTS" in _codes(_canon(strategy))
    assert _codes(_canon(strategy, CONTRACT)) == set()


@pytest.mark.parametrize("strategy", ["llm-judge", "human-annotation", "external-authority"])
def test_a_non_executing_oracle_is_not_asked_for_shell_commands(strategy):
    """Judge-style oracles score an answer; there is no variant to rebuild."""
    assert strategy not in VARIANT_ORACLE_PATTERNS
    assert "MISSING_VARIANTS" not in _codes(_canon(strategy))


def test_rerun_labeling_executes_but_still_has_no_variants():
    """The case that proves the discriminator is contrast, not execution. It runs the artifact N
    times under controlled variation and reads ground truth off the distribution -- one
    artifact, no second state to compare against."""
    assert "rerun-labeling" not in VARIANT_ORACLE_PATTERNS
    assert "MISSING_VARIANTS" not in _codes(_canon("rerun-labeling"))


def test_an_unclassifiable_oracle_is_flagged_rather_than_guessed():
    """`other` means the author invented a pattern. Silently exempting it would let anyone skip
    the requirement by declining to classify; silently requiring it would forbid the escape
    hatch the registry deliberately provides."""
    codes = _codes(_canon("other"))
    assert "UNCLASSIFIED_ORACLE" in codes
    assert "MISSING_VARIANTS" not in codes


# ── declared variants are still well-formed, whatever the pattern ────────────────────────────


def test_a_malformed_contract_is_refused_even_for_a_judge_oracle():
    """Well-formedness is not conditional: if a spec declares variants at all, they must be
    runnable. Only the REQUIREMENT depends on the pattern."""
    bad = [{"name": "x", "check_cmd": "run", "expect": "maybe"}]
    assert "UNCHECKABLE_VARIANTS" in _codes(_canon("llm-judge", bad))


def test_a_contract_that_navigates_away_is_refused_at_this_stage_too():
    bad = [{"name": "x", "check_cmd": "cd /w && run", "expect": "pass"}]
    assert "CHECK_LEAVES_VARIANT" in _codes(_canon("fail-to-pass-tests", bad))


# ── the raw stage stops requiring what it cannot judge ───────────────────────────────────────


def test_the_raw_gate_no_longer_demands_variants():
    """The raw stage has no `oracle_strategy` to consult, so it does not require variants (that
    would force every benchmark into one verification style).
    """
    assert "MISSING_VARIANTS" not in {d.code for d in check_verifiability({"task": "t"})}


def test_the_raw_gate_still_refuses_a_malformed_contract():
    """What it CAN judge without the strategy: whether a declared contract is runnable."""
    raw = {"artifact_variants": [{"name": "x", "check_cmd": "run", "expect": "maybe"}]}
    assert "UNCHECKABLE_VARIANTS" in {d.code for d in check_verifiability(raw)}
