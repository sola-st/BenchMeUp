"""One source filter among good criteria is caught.

A criterion whose grammatical subject is the source (e.g. "Sources must be Java projects ...") is a
source filter; criteria about the instance are not, even when they mention "project". Source filters
belong in admissibility, not in construction criteria, so one in an otherwise good list is flagged.
"""

from __future__ import annotations

import pytest

from benchcompiler.spec import check_canonical

ADMIS = [
    {"name": "language", "observe": "GET /repos/{o}/{p} -> language", "require": "one of [java]"},
    {"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"},
]

GOOD = [
    "The fix commit must add or modify at least one triggering test which fails on the "
    "commit's parent and passes on the commit itself, confirmed by an actual test run.",
    "Both the buggy (parent) and fixed revisions must build successfully with the project's configured build tool.",
]


def _canon(criteria):
    return {
        "oracle_strategy": "external-authority",
        "task": "t",
        "criteria": criteria,
        "admissibility": ADMIS,
        "solver_input": {"kind": "workdir", "path": ".", "describes": "a tree"},
    }


def _problems(criteria):
    return [d for d in check_canonical(_canon(criteria)) if d.code == "CRITERIA_ARE_SOURCE_FILTERS"]


@pytest.mark.parametrize(
    "bad",
    [
        "Sources must be Java projects (JVM) and not one of the excluded projects already "
        "present in Defects4J: Chart, Cli, Closure, Codec.",
        "Project admissibility filters (see admissibility below) must be satisfied.",
        "Project must be a Python project and must not be one of the 12 excluded repositories: "
        "astropy/astropy, django/django.",
        "Repositories must have at least 100 stars.",
        "The repository must not be a fork.",
    ],
)
def test_one_source_filter_among_good_criteria_is_caught(bad):
    """One source filter among otherwise good criteria."""
    problems = _problems(GOOD + [bad])
    print(f"  {bad[:70]!r}\n    -> {[d.detail[:90] for d in problems]}")
    assert problems, bad


def test_the_refusal_quotes_the_offending_criterion():
    """A refusal naming no criterion cannot be acted on when four of five are fine."""
    bad = "Project must be a Python project and must not be one of the 12 excluded repositories."
    [d] = _problems(GOOD + [bad])
    print(d.detail)
    assert "Project must be a Python project" in d.detail


@pytest.mark.parametrize(
    "ok",
    GOOD
    + [
        "The instance must originate from a real, merged pull request that closes an issue.",
        "The PR's own added/modified test must fail on the pre-fix snapshot and pass on the merge "
        "commit, confirmed by an actual test run.",
        "Both the pre-fix and post-fix snapshots must be installable and able to run the project's test harness.",
        "The optimized build must reproduce the default build's output byte-for-byte on every recorded input.",
    ],
)
def test_a_criterion_about_the_instance_is_untouched(ok):
    """Criteria about the instance that mention 'project' are not source filters; what makes a
    source filter is being about the source.
    """
    problems = _problems([ok])
    print(f"  {ok[:70]!r} -> {[d.detail[:60] for d in problems]}")
    assert not problems, ok


def test_a_spec_with_no_admissibility_names_is_not_second_guessed():
    """Nothing to compare against means no judgement to make."""
    canonical = _canon(GOOD + ["Sources must be Java projects."])
    canonical["admissibility"] = []
    canonical["admissibility_exemption"] = "not applicable: a fixed published corpus"
    assert not [d for d in check_canonical(canonical) if d.code == "CRITERIA_ARE_SOURCE_FILTERS"]


# ── the shipped corpus, stated loudly ────────────────────────────────────────────────────────
