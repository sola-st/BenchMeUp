"""An unexpected key in a model's reply must not crash the formalizer.

`build_canonical` constructs dataclasses from the model's dicts; a reply with an extra or missing
key, or a bare string where a dict is expected, yields a typed diagnostic rather than an exception.
"""

from __future__ import annotations

import pytest

from benchcompiler import formalizer
from benchcompiler.formalizer import build_canonical

ADMIS = [{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"}]

BASE = {
    "task": "t",
    "instance_input": "i",
    "instance_output": "o",
    "data_source": "d",
    "candidate_definition": "c",
    "oracle_strategy": "llm-judge",
    "oracle_detail": "od",
    "interface": "cli",
    "scale": {"instance_count": "10", "compute_budget": "1h", "human_budget": "0"},
    "criteria": ["the label matches the rerun-derived label"],
}


def _cs(**over):
    d = dict(BASE)
    d.update(over)
    return d


def test_the_exact_reply_that_crashed():
    """`term` belongs to Delegation and was written on an Assumption."""
    cs = _cs(
        assumptions=[
            {
                "field": "data",
                "term": "popular",
                "resolved_to": "stars >= 1000",
                "reason": "resolved through the term table",
            }
        ]
    )
    canonical = build_canonical({"admissibility": ADMIS}, cs)
    print("  assumptions ->", [a.__dict__ for a in canonical.assumptions])
    assert canonical.assumptions[0].resolved_to == "stars >= 1000"
    assert canonical.assumptions[0].field == "data"


@pytest.mark.parametrize(
    "block,extra",
    [
        ("assumptions", {"term": "popular"}),
        ("assumptions", {"rationale": "because"}),
        ("assumptions", {"confidence": 0.9}),
        ("delegations", {"reason": "because"}),
        ("delegations", {"note": "n"}),
    ],
)
def test_a_stray_key_is_dropped_not_raised(block, extra):
    entry = {"field": "interface", "resolved_to": "a CLI verb", "reason": "r", "term": "t", "rationale": "r"}
    entry = {
        k: v
        for k, v in entry.items()
        if k
        in (
            ("field", "resolved_to", "reason")
            if block == "assumptions"
            else ("field", "term", "resolved_to", "rationale")
        )
    }
    entry.update(extra)
    canonical = build_canonical({"admissibility": ADMIS}, _cs(**{block: [entry]}))
    got = getattr(canonical, block)
    print(f"  {block} + {extra} -> {[x.__dict__ for x in got]}")
    assert len(got) == 1


@pytest.mark.parametrize("block", ["assumptions", "delegations"])
def test_a_missing_key_does_not_raise_either(block):
    """A reply missing a key is also a diagnostic rather than a traceback."""
    canonical = build_canonical({"admissibility": ADMIS}, _cs(**{block: [{"field": "interface"}]}))
    assert len(getattr(canonical, block)) == 1


@pytest.mark.parametrize("block", ["assumptions", "delegations"])
def test_a_non_object_entry_is_skipped(block):
    """A bare string where a dict is expected is handled."""
    canonical = build_canonical({"admissibility": ADMIS}, _cs(**{block: ["popular means stars >= 1000", None]}))
    assert isinstance(getattr(canonical, block), list)


def test_formalize_returns_a_diagnostic_rather_than_raising(monkeypatch):
    """End to end: the stage returns a diagnostic, never a traceback."""
    monkeypatch.setattr(
        formalizer,
        "call_llm_json",
        lambda *a, **k: {
            "outcome": "ACCEPT",
            "canonical_spec": _cs(
                assumptions=[{"field": "data", "term": "popular", "resolved_to": "stars >= 1000", "reason": "r"}],
                solver_input={},
            ),
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
    print(f"  outcome={result.outcome}")
    assert result.outcome in ("ACCEPT", "REJECT")
