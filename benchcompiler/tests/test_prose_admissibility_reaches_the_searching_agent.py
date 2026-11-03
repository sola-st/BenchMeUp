"""A prose `admissibility:` reaches the searching agent as the clauses the model derived.

Raw clauses win only when the raw spec ships them as a list; a prose string falls through to the
clauses the formalizer derived from it, so the agent searches with a floor.
"""

from __future__ import annotations

from benchcompiler.search import search_instance

CANONICAL = {
    "canonical_spec": {
        "task": "t",
        "candidate_definition": "one warning",
        "data_source": "d",
        "sampling": {"lead_key": ["clone_url"], "normalize": {"clone_url": "git_url"}},
        "admissibility": [
            {
                "name": "stargazers_count",
                "observe": "GET /repos/{org}/{project} -> stargazers_count",
                "require": ">= 1000",
            },
            {"name": "is_fork", "observe": "GET /repos/{org}/{project} -> fork", "require": "is false"},
        ],
    }
}


def _inst(raw_admissibility):
    raw = {"task": "t", "data": "d", "admissibility": raw_admissibility}
    return search_instance(raw, CANONICAL, benchmark_name="x", target=5)


def test_prose_admissibility_falls_through_to_the_derived_clauses():
    inst = _inst("public Java projects, popular and actively maintained, not forks")
    assert [a["name"] for a in inst["admissibility"]] == ["stargazers_count", "is_fork"]
    assert inst["admissibility"][0]["require"] == ">= 1000"


def test_clauses_the_raw_spec_ships_still_win():
    mine = [{"name": "commit_count", "observe": "git rev-list --count HEAD", "require": ">= 1000"}]
    assert [a["name"] for a in _inst(mine)["admissibility"]] == ["commit_count"]


def test_silence_stays_silent():
    assert _inst(None)["admissibility"] == [a for a in CANONICAL["canonical_spec"]["admissibility"]] or [
        a["name"] for a in _inst(None)["admissibility"]
    ] == ["stargazers_count", "is_fork"]
