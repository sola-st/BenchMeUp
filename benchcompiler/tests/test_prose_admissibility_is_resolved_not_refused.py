"""Admissibility written as prose reaches the model; only a clause list is carried verbatim.

`admissibility` that is already a structured clause list is carried verbatim into the canonical spec
and validated by the parser. Prose is the fuzzy sentence the term table exists to resolve, so it
goes to the model and comes back as clauses. Silence is still refused mechanically. A declared
exemption is not a clause list and must not be wired into the admissibility gate, where
`adm.evaluate` would raise.
"""

from __future__ import annotations

import pytest

from benchcompiler import formalizer
from benchcompiler.spec import check_admissibility, check_canonical, mechanical_precheck

CLAUSES = [
    {"name": "stars", "observe": "GET /repos/{org}/{project} -> stargazers_count", "require": ">= 100"},
]

RESOLVED = [
    {"name": "stars", "observe": "GET /repos/{org}/{project} -> stargazers_count", "require": ">= 1000"},
    {"name": "last_commit", "observe": "GET /repos/{org}/{project} -> pushed_at", "require": "within 12 months"},
]

PROSE = "popular, actively maintained Java projects that are not forks"
EXEMPTION = "not applicable: instances are generated synthetically from library signatures"


def _raw(admissibility, **over):
    d = {
        "task": "repair",
        "instance": "one bug-fix commit",
        "data": "one fix commit",
        "correctness": "the added test fails at the parent and passes at the fix",
        "interface": "patch",
        "scale": {"instance_count": 60},
        "admissibility": admissibility,
        "solver_input": {"kind": "workdir", "path": ".", "describes": "the edited source tree a solver submits"},
    }
    d.update(over)
    return d


def _codes(diags):
    return {d.code for d in diags}


def _llm(canonical_extra=None):
    def call(*a, **k):
        cs = {
            "task": "repair",
            "instance_input": "a buggy tree",
            "instance_output": "a patch",
            "data_source": "one fix commit",
            "candidate_definition": "one fix commit",
            "oracle_strategy": "external-authority",
            "oracle_detail": "the project's own test suite decides",
            "interface": "patch",
            "scale": {"instance_count": "60", "compute_budget": "10h", "human_budget": "0"},
            "criteria": ["the added test fails at the parent commit and passes at the fix"],
            "quality_gates": [],
            "examples": [],
            "constraints": [],
            "assumptions": [],
            "delegations": [],
        }
        cs.update(canonical_extra or {})
        return {"outcome": "ACCEPT", "canonical_spec": cs}

    return call


# ── 1. prose is a question for the model, not a rejection ────────────────────────────────────


def test_prose_admissibility_is_not_refused_before_the_model():
    """Prose admissibility is not rejected before the model sees it."""
    codes = _codes(mechanical_precheck(_raw(PROSE)))
    print("mechanical_precheck on prose ->", sorted(codes))
    assert "MISSING_ADMISSIBILITY" not in codes


def test_silence_is_still_refused_before_the_model():
    """Silence is refused, since nothing can resolve it without inventing a floor."""
    raw = _raw(PROSE)
    del raw["admissibility"]
    assert "MISSING_ADMISSIBILITY" in _codes(mechanical_precheck(raw))


def test_an_unmeasurable_clause_list_is_still_refused_before_the_model():
    """A structured block is the author's own machine-checkable statement, carried verbatim.
    It must be valid as written; there is no model resolution to wait for."""
    bad = [{"name": "quality", "observe": "look at the project", "require": "good"}]
    assert "UNMEASURABLE_ADMISSIBILITY" in _codes(mechanical_precheck(_raw(bad)))


def test_the_prompt_asks_the_model_to_resolve_prose_into_clauses():
    prompt = formalizer.build_prompt(_raw(PROSE))
    print("prose prompt admissibility mentions:", prompt.count("admissibility"))
    assert formalizer.ADMISSIBILITY_SLOT.strip() in prompt, "the model has no output slot to resolve prose into"


def test_the_prompt_does_not_offer_the_slot_when_clauses_were_stated():
    """Carried verbatim means the model's copy would be discarded -- asking for it invites a
    paraphrase that silently loses."""
    prompt = formalizer.build_prompt(_raw(CLAUSES))
    assert formalizer.ADMISSIBILITY_SLOT.strip() not in prompt


def test_the_model_resolution_is_taken_when_the_raw_block_is_prose(monkeypatch):
    monkeypatch.setattr(formalizer, "call_llm_json", _llm({"admissibility": RESOLVED}))
    result = formalizer.formalize(_raw(PROSE), model="stub", judge=False)
    assert result.accepted, [str(d) for d in result.diagnostics]
    print("carried admissibility ->", result.canonical_spec.admissibility)
    assert result.canonical_spec.admissibility == RESOLVED


def test_a_clause_list_is_still_carried_verbatim(monkeypatch):
    """The rule that must not break: the model does not get to re-word a structured block."""
    monkeypatch.setattr(formalizer, "call_llm_json", _llm({"admissibility": RESOLVED}))
    result = formalizer.formalize(_raw(CLAUSES), model="stub", judge=False)
    assert result.accepted, [str(d) for d in result.diagnostics]
    assert result.canonical_spec.admissibility == CLAUSES


def test_a_model_that_does_not_resolve_prose_is_refused(monkeypatch):
    """Otherwise prose becomes a way to ship no floor at all -- worse than the rejection."""
    monkeypatch.setattr(formalizer, "call_llm_json", _llm())
    result = formalizer.formalize(_raw(PROSE), model="stub", judge=False)
    print("outcome:", result.outcome, [str(d) for d in result.diagnostics])
    assert not result.accepted
    assert "MISSING_ADMISSIBILITY" in _codes(result.diagnostics)


def test_a_model_resolution_is_held_to_the_same_parser(monkeypatch):
    bad = [{"name": "quality", "observe": "look at the project", "require": "good"}]
    monkeypatch.setattr(formalizer, "call_llm_json", _llm({"admissibility": bad}))
    result = formalizer.formalize(_raw(PROSE), model="stub", judge=False)
    assert not result.accepted
    assert "UNMEASURABLE_ADMISSIBILITY" in _codes(result.diagnostics)


# ── 2. an exemption is not a clause list ─────────────────────────────────────────────────────


def test_an_exemption_is_not_carried_as_admissibility(monkeypatch):
    monkeypatch.setattr(formalizer, "call_llm_json", _llm())
    result = formalizer.formalize(_raw(EXEMPTION), model="stub", judge=False)
    assert result.accepted, [str(d) for d in result.diagnostics]
    d = result.canonical_spec.to_dict()
    print("emitted keys ->", sorted(d))
    assert d.get("admissibility", []) == [], "prose reached the field the harness evaluates"
    assert EXEMPTION in str(d.get("admissibility_exemption", "")), (
        "the exemption vanished -- it is a decision on record and must survive to the artifact"
    )


def test_whatever_is_carried_can_be_evaluated_by_the_gate_that_gets_it(monkeypatch):
    """`instantiate` wires `candidate_is_admissible` whenever this field is truthy, and the gate
    calls `adm.evaluate` on each entry; an exemption string must not reach it.
    """
    from agentcore.agent import admissibility as adm

    for block in (EXEMPTION, PROSE, CLAUSES):
        monkeypatch.setattr(formalizer, "call_llm_json", _llm({"admissibility": RESOLVED}))
        result = formalizer.formalize(_raw(block), model="stub", judge=False)
        assert result.accepted, [str(d) for d in result.diagnostics]
        carried = result.canonical_spec.admissibility
        assert isinstance(carried, list), f"{block!r} carried a {type(carried).__name__}"
        for clause in carried:
            adm.evaluate(adm.parse(clause.get("require")), "1")  # must not raise


def test_the_exemption_still_needs_a_reason():
    assert "MISSING_ADMISSIBILITY" in _codes(mechanical_precheck(_raw("not applicable")))


def test_check_admissibility_itself_is_unchanged():
    """The raw-level verdict is covered elsewhere; here only when `mechanical_precheck` consults it
    is tested.
    """
    assert "MISSING_ADMISSIBILITY" in _codes(check_admissibility({"admissibility": PROSE}))
    assert check_admissibility({"admissibility": CLAUSES}) == []


@pytest.mark.parametrize(
    "carried,expect",
    [
        (CLAUSES, set()),
        ([], {"MISSING_ADMISSIBILITY"}),
        ([{"name": "q", "observe": "look at it", "require": "good"}], {"UNMEASURABLE_ADMISSIBILITY"}),
    ],
)
def test_check_canonical_now_guards_the_field_it_hands_downstream(carried, expect):
    canonical = {
        "oracle_strategy": "external-authority",
        "criteria": ["c"],
        "admissibility": carried,
        "solver_input": {"kind": "workdir", "path": ".", "describes": "a tree"},
    }
    codes = _codes(check_canonical(canonical))
    print(carried, "->", sorted(codes))
    assert expect <= codes


def test_an_exempt_canonical_needs_no_clauses():
    canonical = {
        "oracle_strategy": "external-authority",
        "criteria": ["c"],
        "admissibility": [],
        "admissibility_exemption": EXEMPTION,
        "solver_input": {"kind": "workdir", "path": ".", "describes": "a tree"},
    }
    assert "MISSING_ADMISSIBILITY" not in _codes(check_canonical(canonical))
