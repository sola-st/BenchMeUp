"""Notation the harness can resolve is not a verdict on the spec.

A lead_key field with no entry in sampling.normalize gets the generic treatment (`none`), and a
delegation of a whole field the raw spec states is resolved. Both are resolved before the canonical
is checked, and each resolution is recorded in the canonical's assumptions.
"""

from __future__ import annotations

import copy

from benchcompiler.formalizer import resolve_notation

RAW = {
    "task": "t",
    "instance": "i",
    "data": "d",
    "correctness": "c",
    "interface": "if",
    "scale": "First pass: ~15-20 instances.\n",
    "criteria": "x",
    "constraints": "-",
}

CS = {
    "sampling": {
        "lead_key": ["repo_url", "fix_commit", "pr_number"],
        "normalize": {"repo_url": "git_url", "stale_field": "none"},
    },
    "delegations": [
        {"field": "scale", "term": "scale", "resolved_to": "20", "rationale": "not stated"},
        {"field": "constraints", "term": "x", "resolved_to": "y", "rationale": "delegable"},
        {"field": "criteria.dedup", "term": "d", "resolved_to": "e", "rationale": "sub-item"},
    ],
    "assumptions": [{"field": "other", "resolved_to": "kept", "reason": "the model's own"}],
}


def test_missing_normal_forms_and_stated_field_delegations_are_resolved_on_record():
    cs, notes = resolve_notation(RAW, copy.deepcopy(CS))
    print("[verbose] notes:", notes)
    assert cs["sampling"]["normalize"] == {"repo_url": "git_url", "fix_commit": "none", "pr_number": "none"}
    assert [d["field"] for d in cs["delegations"]] == ["constraints", "criteria.dedup"]
    fields = [a["field"] for a in cs["assumptions"]]
    assert fields[0] == "other" and "sampling.normalize" in fields and "delegations" in fields
    assert any("fix_commit" in a["resolved_to"] and "pr_number" in a["resolved_to"] for a in cs["assumptions"])
    assert any("scale" in a["reason"] for a in cs["assumptions"])
    assert len(notes) == 2


def test_nothing_to_resolve_changes_nothing():
    cs = {"sampling": {"lead_key": ["a"], "normalize": {"a": "none"}}, "delegations": [], "assumptions": []}
    out, notes = resolve_notation(RAW, copy.deepcopy(cs))
    assert out == cs and notes == []
    # a delegation of a field the raw spec does NOT state stays for the checker to judge
    cs2 = {"delegations": [{"field": "scale", "term": "s", "resolved_to": "r", "rationale": "q"}]}
    out, notes = resolve_notation({"task": "t"}, copy.deepcopy(cs2))
    assert out["delegations"] == cs2["delegations"] and notes == []
