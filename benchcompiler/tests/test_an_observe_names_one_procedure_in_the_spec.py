"""An `observe` may not offer two ways to obtain the same value.

`clause_problems` refuses an observe that is empty, in the blocklist, shorter than 12 characters, or
that offers alternatives ("X, or Y"), because alternatives yield different numbers with no record of
which was used. No particular shape is required: natural-language instructions that are followable
and deterministic are accepted.
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from benchcompiler.spec import check_admissibility, observe_problems


def _codes(raw):
    return {d.code for d in check_admissibility(raw)}


def _clause(observe):
    return {"name": "commit_count", "observe": observe, "require": ">= 1000"}


@pytest.mark.parametrize(
    "observe",
    [
        "git rev-list --count HEAD in a shallow clone, or the contributors API",
        "GET /repos/{o}/{p} -> stargazers_count or the stars badge",
        "the round's scoreboard page, OR the organiser's announcement post",
    ],
)
def test_two_procedures_for_one_value_are_refused(observe):
    problems = observe_problems(_clause(observe))
    print(observe, "->", problems)
    assert problems, observe
    assert "UNMEASURABLE_ADMISSIBILITY" in _codes({"admissibility": [_clause(observe)]})


def test_the_refusal_says_to_pick_one():
    """The refusal says what to change."""
    text = " ".join(observe_problems(_clause("git rev-list --count HEAD in a shallow clone, or the contributors API")))
    print(text)
    assert "one" in text.lower(), text
    assert "same number" in text.lower() or "different" in text.lower(), text


@pytest.mark.parametrize(
    "observe",
    [
        "git rev-list --count HEAD in a full clone",
        "GET https://api.github.com/repos/{org}/{project} -> stargazers_count",
        "GET https://api.github.com/repos/{org}/{project} -> pushed_at (YYYY-MM-DD)",
        "GET https://api.github.com/search/issues?q=repo:{org}/{project}+is:issue+is:closed+label:bug -> total_count",
        "GET https://api.github.com/repos/{org}/{project}/contents/ -> the root listing; report "
        "the FIRST file name in it that, lowercased, is one of the required set",
        "the origin remote's default branch head",  # `or` inside a word must not match
    ],
)
def test_one_procedure_still_passes(observe):
    """Observes shipped in this repo that are not disjunctions, and a word containing `or`, are
    accepted.
    """
    problems = observe_problems(_clause(observe))
    print(observe[:60], "->", problems)
    assert problems == [], observe


def test_what_clause_problems_already_refused_still_is():
    """The existing rules still apply unchanged."""
    for observe in ("", "look at it", "n/a", "short"):
        assert "UNMEASURABLE_ADMISSIBILITY" in _codes({"admissibility": [_clause(observe)]}), observe


def test_the_prompts_own_worked_example_is_now_refused():
    """A vague procedure ("look at the project") is refused with UNMEASURABLE_ADMISSIBILITY even
    when it is longer than the blocklisted phrases and the length floor.
    """
    problems = observe_problems(_clause("look at the project"))
    print(problems)
    assert problems
    assert "UNMEASURABLE_ADMISSIBILITY" in _codes({"admissibility": [_clause("look at the project")]})


def test_a_procedure_that_merely_mentions_looking_is_untouched():
    """Detection is anchored at the start of the phrase, so a verb inside a real procedure is not
    refused.
    """
    assert (
        observe_problems(
            _clause(
                "GET https://api.github.com/repos/{o}/{p}/contents/ -> the root listing; look at the "
                "first name in it that is a build file"
            )
        )
        == []
    )


# ── the shipped corpus, measured ─────────────────────────────────────────────────────────────


def _shipped_clauses():
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for path in sorted(glob.glob(os.path.join(here, "..", "examples", "*.canonical.json"))):
        with open(path) as fh:
            doc = json.load(fh)
        for clause in (doc.get("canonical_spec") or {}).get("admissibility") or []:
            out.append((os.path.basename(path), clause))
    return out


def test_the_corpus_this_was_measured_on_is_still_there():
    clauses = _shipped_clauses()
    print(f"{len(clauses)} clauses over {len({n for n, _ in clauses})} lockfiles")
    assert len(clauses) >= 15, clauses


def test_no_shipped_canonical_spec_offers_a_choice_of_procedure():
    """The shipped raw specs state the field with one procedure ("in a full clone"); checked over
    all lockfiles.
    """
    offenders = [(name, c.get("name"), c.get("observe")) for name, c in _shipped_clauses() if observe_problems(c)]
    for row in offenders:
        print("OFFENDER", row)
    assert not offenders, offenders
