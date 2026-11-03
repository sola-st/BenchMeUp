"""A delegation on a field where the formalizer prompt forbids it is rejected.

DELEGATE is valid on a named set of fields and invalid as a whole-field answer for task, instance,
data or correctness (INVALID_DELEGATION). `check_canonical` verifies this against
`registry.DELEGABLE_WHOLE_FIELDS`.
"""

from __future__ import annotations

import pytest

from benchcompiler.registry import DELEGABLE_WHOLE_FIELDS
from benchcompiler.spec import check_canonical

ADMIS = [{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"}]


def _canon(delegations):
    return {
        "task": "decide whether a test is flaky",
        "candidate_definition": "one pytest test at one frozen snapshot",
        "instance_input": "a repo snapshot and a test id",
        "instance_output": "a label",
        "data_source": "pytest tests",
        "oracle_strategy": "llm-judge",
        "oracle_detail": "200 reruns decide",
        "criteria": ["the label matches the rerun-derived label"],
        "admissibility": ADMIS,
        "delegations": delegations,
        "solver_input": {"kind": "workdir", "path": ".", "describes": "a tree"},
    }


def _codes(delegations):
    return [d.code for d in check_canonical(_canon(delegations))]


def _ok(field="interface"):
    """`scale` is not the default field here: DELEGATE is permitted on scale's instance count (a
    sub-item), not on the whole field. DELEGABLE_WHOLE_FIELDS is {constraints, interface,
    quality, references}.
    """
    return {
        "field": field,
        "term": "the submission channel",
        "resolved_to": "a CLI verb",
        "rationale": "the spec said DELEGATE and the artifact ships a CLI",
    }


def test_the_premise_holds():
    """A well-formed delegation on a delegable field must pass, or everything below is noise."""
    assert _codes([_ok()]) == []
    assert _codes([]) == []


@pytest.mark.parametrize("field", ["task", "instance", "data", "correctness"])
def test_a_delegation_on_a_field_that_cannot_be_delegated_is_refused(field):
    """DELEGATE is invalid on the fields naming what the benchmark is for or its core correctness
    mechanism.
    """
    codes = _codes([_ok(field)])
    print(f"  delegating {field!r} -> {codes}")
    assert "INVALID_DELEGATION" in codes, field


def test_the_refusal_names_the_field_and_what_may_be_delegated():
    d = [x for x in check_canonical(_canon([_ok("correctness")])) if x.code == "INVALID_DELEGATION"][0]
    text = d.detail + d.fix_hint
    print(text)
    assert "correctness" in text
    assert any(f in text for f in DELEGABLE_WHOLE_FIELDS), text


@pytest.mark.parametrize("missing", ["resolved_to", "rationale"])
def test_a_delegation_that_resolved_to_nothing_is_refused(missing):
    """A delegation with no recorded resolution is a silent default and is rejected."""
    bad = {k: v for k, v in _ok().items() if k != missing}
    codes = _codes([bad])
    print(f"  missing {missing!r} -> {codes}")
    assert "INVALID_DELEGATION" in codes, missing


@pytest.mark.parametrize("field", sorted(DELEGABLE_WHOLE_FIELDS))
def test_every_field_the_registry_permits_is_accepted(field):
    """The vocabulary comes from `registry.DELEGABLE_WHOLE_FIELDS`, read rather than copied, so
    the rule and the registry cannot drift into disagreeing."""
    assert _codes([_ok(field)]) == [], field


def test_a_sub_item_delegation_is_not_refused():
    """The prompt permits DELEGATE on sub-items within criteria/quality, which arrive with a
    dotted or qualified field. Refusing those would forbid a shape the prompt offers."""
    assert _codes([_ok("criteria.dedup")]) == []
    assert _codes([_ok("quality.rerun_count")]) == []
    # The one the prompt calls out by name: scale's instance count, not scale.
    assert _codes([_ok("scale.instance_count")]) == []
    assert "INVALID_DELEGATION" in _codes([_ok("scale")]), (
        "the whole `scale` field is not delegable -- only its instance count"
    )
