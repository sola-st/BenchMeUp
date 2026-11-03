"""A spec that states nothing about what a source must be is rejected.

`admissibility` is a required part of `data`: at least one clause, every clause measurable. The
dimensions and thresholds belong to the spec and differ per task; the formalizer only requires that
they exist and can be checked, using the same predicate parser the harness enforces with.
"""

from __future__ import annotations

from benchcompiler import spec as specmod

REPO_CLAUSES = [
    {"name": "stars", "observe": "GET /repos/{org}/{project} -> stargazers_count", "require": ">= 100"},
    {"name": "commit_count", "observe": "git rev-list --count HEAD in a shallow clone", "require": ">= 100"},
    {"name": "last_commit", "observe": "GET /repos/{org}/{project} -> pushed_at", "require": "within 12 months"},
]

CONTEST_CLAUSES = [
    {"name": "entrants", "observe": "the round's scoreboard page -> participant count", "require": ">= 500"},
]


def _raw(admissibility=None, data="Real, public projects with a version-control history."):
    d = {
        "task": "repair",
        "instance": "one bug-fix commit",
        "data": data,
        "correctness": "tests that fail before the fix and pass after it",
        "interface": "patch",
        "scale": {"instance_count": 60},
    }
    if admissibility is not None:
        d["admissibility"] = admissibility
    return d


def _codes(diags):
    return [d.code if hasattr(d, "code") else d["code"] for d in diags]


# ── silence is a rejection ──────────────────────────────────────────────────────────────────


def test_a_spec_with_no_admissibility_is_rejected():
    """A spec that names a source type but no admissibility clause."""
    diags = specmod.check_admissibility(_raw())
    assert diags, "a spec stating no floor for its sources must not be accepted"
    assert "MISSING_ADMISSIBILITY" in _codes(diags)


def test_the_rejection_says_what_to_add():
    d = specmod.check_admissibility(_raw())[0]
    hint = (d.fix_hint if hasattr(d, "fix_hint") else d["fix_hint"]).lower()
    assert "name" in hint and "observe" in hint and "require" in hint, hint


def test_an_empty_list_counts_as_silence():
    assert _codes(specmod.check_admissibility(_raw([]))) == ["MISSING_ADMISSIBILITY"]


# ── clauses must be measurable ──────────────────────────────────────────────────────────────


def test_a_well_formed_set_of_clauses_is_accepted():
    assert specmod.check_admissibility(_raw(REPO_CLAUSES)) == []


def test_an_unmeasurable_threshold_is_rejected():
    bad = [dict(REPO_CLAUSES[0], require="should be popular")]
    diags = specmod.check_admissibility(_raw(bad))
    assert "UNMEASURABLE_ADMISSIBILITY" in _codes(diags)


def test_the_offending_clause_is_named():
    bad = [REPO_CLAUSES[0], dict(REPO_CLAUSES[1], require="a lot")]
    d = specmod.check_admissibility(_raw(bad))[0]
    detail = d.detail if hasattr(d, "detail") else d["detail"]
    assert "commit_count" in detail, detail


def test_a_clause_missing_its_observe_is_rejected():
    """Without an `observe` the value is composed from the description rather than fetched."""
    bad = [{k: v for k, v in REPO_CLAUSES[0].items() if k != "observe"}]
    assert _codes(specmod.check_admissibility(_raw(bad))) == ["UNMEASURABLE_ADMISSIBILITY"]


def test_every_problem_is_reported_at_once():
    """All problems in a rejected spec are reported in one pass."""
    bad = [dict(REPO_CLAUSES[0], require="lots"), dict(REPO_CLAUSES[1], name="commit count")]
    diags = specmod.check_admissibility(_raw(bad))
    joined = " ".join((d.detail if hasattr(d, "detail") else d["detail"]) for d in diags)
    assert "lots" in joined and "commit count" in joined, joined


# ── the criteria belong to the task, not to us ──────────────────────────────────────────────


def test_a_task_whose_sources_are_not_repositories_is_equally_valid():
    assert specmod.check_admissibility(_raw(CONTEST_CLAUSES)) == []


def test_no_dimension_is_required_by_name():
    """No particular dimension (stars, commits, ...) is demanded, only that something measurable is
    stated.
    """
    assert specmod.check_admissibility(_raw([CONTEST_CLAUSES[0]])) == []


def test_the_diagnostic_codes_are_registered():
    assert "MISSING_ADMISSIBILITY" in specmod.DIAGNOSTIC_CODES
    assert "UNMEASURABLE_ADMISSIBILITY" in specmod.DIAGNOSTIC_CODES


def test_it_uses_the_same_parser_the_harness_enforces_with():
    """There is one definition of "measurable": a predicate the formalizer accepts must be
    evaluable by the gate.
    """
    from agentcore.agent import admissibility as adm

    for clause in REPO_CLAUSES:
        assert adm.parse(clause["require"]) is not None
    assert specmod.check_admissibility(_raw(REPO_CLAUSES)) == []
