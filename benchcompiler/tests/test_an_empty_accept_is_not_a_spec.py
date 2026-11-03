"""A model reply of `{"outcome":"ACCEPT","canonical_spec":{}}` must not produce a spec.

`check_canonical` rejects a canonical spec whose required prose fields (`candidate_definition`,
`task`, `oracle_detail`, `data_source`, `instance_input`, `instance_output`) are empty, so a
truncated or degraded reply is not accepted as a spec. The check lives in `check_canonical` because
it concerns the formalizer's answer, not the researcher's prose.
"""

from __future__ import annotations

import pytest

from benchcompiler import formalizer
from benchcompiler.spec import check_canonical

ADMIS = [{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"}]

#: Every prose field a later stage reads and cannot do its job without.
LOAD_BEARING = ["task", "candidate_definition", "instance_input", "instance_output", "data_source", "oracle_detail"]


def _canon(**over):
    d = {
        "task": "decide whether a test is flaky",
        "candidate_definition": "one pytest test at one frozen snapshot",
        "instance_input": "a repo snapshot and a test id",
        "instance_output": "a label in {flaky, stable}",
        "data_source": "pytest tests in Python projects",
        "oracle_strategy": "llm-judge",
        "oracle_detail": "200 reruns in a fixed container decide the label",
        "criteria": ["the label matches the rerun-derived label"],
        "admissibility": ADMIS,
        "solver_input": {
            "kind": "parameter",
            "name": "label",
            "describes": "the verdict",
            "probe_values": ["flaky", "stable"],
        },
        "parameters": [{"name": "label", "kind": "string", "default": "stable"}],
    }
    d.update(over)
    return d


def _codes(canonical):
    return [d.code for d in check_canonical(canonical)]


def test_a_fully_populated_spec_is_accepted():
    """The two-sided half first: this fixture must pass, or every assertion below is vacuous."""
    print(_codes(_canon()))
    assert _codes(_canon()) == []


@pytest.mark.parametrize("field", LOAD_BEARING)
def test_an_empty_load_bearing_field_is_refused(field):
    codes = _codes(_canon(**{field: ""}))
    print(f"  {field} empty -> {codes}")
    assert "MISSING_INFO" in codes, field


@pytest.mark.parametrize("field", LOAD_BEARING)
def test_the_refusal_names_the_field(field):
    d = [x for x in check_canonical(_canon(**{field: ""})) if x.code == "MISSING_INFO"]
    assert d and any(field in x.field or field in x.detail for x in d), [str(x) for x in d]


@pytest.mark.parametrize("field", LOAD_BEARING)
def test_whitespace_is_not_content(field):
    assert "MISSING_INFO" in _codes(_canon(**{field: "   \n  "})), field


def test_the_wholly_empty_reply_is_refused(monkeypatch):
    """The shape this exists for, end to end through `formalize`."""
    monkeypatch.setattr(formalizer, "call_llm_json", lambda *a, **k: {"outcome": "ACCEPT", "canonical_spec": {}})
    raw = {
        "task": "t",
        "instance": "i",
        "data": "d",
        "correctness": "c",
        "interface": "x",
        "scale": "10",
        "admissibility": ADMIS,
        "solver_input": {"kind": "workdir", "path": ".", "describes": "a tree"},
    }
    result = formalizer.formalize(raw, model="stub")
    print(f"  outcome={result.outcome} codes={sorted({d.code for d in result.diagnostics})}")
    assert not result.accepted, "an empty reply produced a canonical spec"
    assert "MISSING_INFO" in {d.code for d in result.diagnostics}


def test_a_truncated_reply_is_refused_too(monkeypatch):
    """The realistic version: the model answered, and got cut off partway."""
    monkeypatch.setattr(
        formalizer,
        "call_llm_json",
        lambda *a, **k: {
            "outcome": "ACCEPT",
            "canonical_spec": {
                "task": "decide whether a test is flaky",
                "instance_input": "a repo snapshot and a test id",
            },
        },
    )
    raw = {
        "task": "t",
        "instance": "i",
        "data": "d",
        "correctness": "c",
        "interface": "x",
        "scale": "10",
        "admissibility": ADMIS,
        "solver_input": {"kind": "workdir", "path": ".", "describes": "a tree"},
    }
    result = formalizer.formalize(raw, model="stub")
    print(f"  outcome={result.outcome} codes={sorted({d.code for d in result.diagnostics})}")
    assert not result.accepted
