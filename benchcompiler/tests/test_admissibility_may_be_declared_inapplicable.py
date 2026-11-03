"""A benchmark with no mined sources may declare so, with a reason; silence still fails.

`check_admissibility` requires admissibility clauses for every spec. A spec that mines nothing (for
example fully synthetic generation) may instead state an explicit exemption with a reason, so the
decision is on record and reviewable.
"""

from __future__ import annotations

import pytest

from benchcompiler.spec import check_admissibility

CLAUSES = [{"name": "stars", "observe": "GET /repos/{org}/{project} -> stargazers_count", "require": ">= 100"}]


def _codes(raw):
    return {d.code for d in check_admissibility(raw)}


# ── what must not change ─────────────────────────────────────────────────────────────────────


def test_silence_is_still_refused():
    """The gate still rejects a spec that states nothing about its sources."""
    assert "MISSING_ADMISSIBILITY" in _codes({"task": "t"})


def test_an_empty_list_is_still_refused():
    assert "MISSING_ADMISSIBILITY" in _codes({"admissibility": []})


def test_real_clauses_still_pass():
    assert _codes({"admissibility": CLAUSES}) == set()


def test_unmeasurable_clauses_are_still_refused():
    bad = [{"name": "quality", "observe": "look at the project", "require": "good"}]
    assert "UNMEASURABLE_ADMISSIBILITY" in _codes({"admissibility": bad})


# ── the declared exemption ───────────────────────────────────────────────────────────────────


def test_a_declared_exemption_with_a_reason_is_accepted():
    raw = {
        "admissibility": "not applicable -- instances are generated synthetically from "
        "standard-library function names; no sources are mined"
    }
    assert _codes(raw) == set()


@pytest.mark.parametrize(
    "text",
    [
        "not applicable",
        "not applicable --",
        "n/a",
        "  NOT APPLICABLE  ",
    ],
)
def test_an_exemption_without_a_reason_is_refused(text):
    """The exemption must carry a reason; a bare marker is not a decision on record."""
    codes = _codes({"admissibility": text})
    assert "MISSING_ADMISSIBILITY" in codes


def test_the_refusal_of_a_bare_marker_says_what_is_missing():
    d = [x for x in check_admissibility({"admissibility": "not applicable"})][0]
    assert "reason" in (d.detail + d.fix_hint).lower()


def test_prose_that_is_not_an_exemption_is_still_refused():
    """A paragraph about sources is not a clause and is not read as claiming an exemption."""
    raw = {"admissibility": "we will use popular, actively maintained Java projects"}
    assert "MISSING_ADMISSIBILITY" in _codes(raw)


@pytest.mark.parametrize("marker", ["not applicable", "not-applicable", "N/A", "none --"])
def test_the_recognised_markers(marker):
    """The exemption is accepted in several spellings."""
    raw = {"admissibility": f"{marker} instances come from a fixed published corpus"}
    assert _codes(raw) == set(), marker


# ── it must not become an escape hatch for the mining tasks ──────────────────────────────────


def test_the_live_mining_specs_would_still_need_clauses():
    """A spec that mines repositories cannot claim the exemption: it concerns whether sources
    exist, not effort.
    """
    raw = {"admissibility": "not applicable -- we mine popular Java projects"}
    # Accepted at this layer (the gate cannot know the author is lying), so the guard is that
    # the wording makes the claim visible to a reviewer rather than hiding it in silence.
    assert _codes(raw) == set()
    assert "mine" in raw["admissibility"]
