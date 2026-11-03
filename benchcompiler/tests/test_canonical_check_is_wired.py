"""`check_canonical` must actually run when a spec is formalized.

Formalizing a spec whose oracle needs a variant contract, and which has none, does not come back
ACCEPT.
"""

from __future__ import annotations

from benchcompiler import formalizer
from benchcompiler.spec import CanonicalSpec, check_canonical

# A raw spec that clears mechanical_precheck, so these tests reach the LLM seam and the
# canonical check rather than stopping at the front door.
_RAW = {
    "task": "t",
    "instance": "i",
    "data": "d",
    "correctness": "c",
    "interface": "if",
    "scale": "10",
    "admissibility": [{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"}],
    # Asked of every spec, whatever its oracle: a benchmark whose task cannot be attempted through
    # its own interface is not a benchmark.
    "solver_input": {"kind": "workdir", "path": "{clone_path}", "describes": "the solver edits the checked-out tree"},
}


def _canonical(**over):
    base = dict(
        spec_id="s1",
        task="t",
        instance_input="i",
        instance_output="o",
        data_source="d",
        candidate_definition="c",
        oracle_strategy="fail-to-pass-tests",
        oracle_detail="od",
        interface="if",
        scale_instance_count="10",
        scale_compute_budget="1 CPU-hours",
        scale_human_budget="0",
        # Declared: how a solver submits, and how an instance is broken on purpose. Both are asked of every canonical spec.
        solver_input={"kind": "workdir", "path": "{clone_path}", "describes": "the solver edits the checked-out tree"},
        negative_control={"break_cmd": "corruption_cmd", "check": "buggy", "describes": "undo the fix"},
        # Checked on the canonical: this is the field `instantiate` wires the admissibility gate onto, so it
        # must be something `adm.evaluate` can be called with.
        admissibility=[{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"}],
    )
    base.update(over)
    return CanonicalSpec(**base)


def test_it_accepts_a_canonical_spec_object_not_only_a_dict():
    """The formalizer holds a CanonicalSpec, so the check accepts it directly rather than requiring
    a dict conversion.
    """
    codes = {d.code for d in check_canonical(_canonical())}
    assert "MISSING_VARIANTS" in codes


def test_a_judge_oracle_object_passes():
    assert check_canonical(_canonical(oracle_strategy="llm-judge")) == []


# ── the wiring ───────────────────────────────────────────────────────────────────────────────


def test_formalize_runs_the_canonical_check(monkeypatch):
    """A contrast oracle with no variant contract must not be ACCEPTed."""
    monkeypatch.setattr(
        formalizer,
        "call_llm_json",
        lambda *a, **k: {
            "outcome": "ACCEPT",
            "canonical_spec": {
                "task": "t",
                "instance_input": "i",
                "instance_output": "o",
                "data_source": "d",
                "candidate_definition": "c",
                "oracle_strategy": "fail-to-pass-tests",
                "oracle_detail": "od",
                "interface": "if",
                "scale": {"instance_count": "10", "compute_budget": "1 CPU-hours"},
            },
        },
        raising=False,
    )
    res = formalizer.formalize(_RAW, model="stub", judge=False)
    assert res.outcome == "REJECT", res.outcome
    assert "MISSING_VARIANTS" in {d.code for d in res.diagnostics}


def test_formalize_still_accepts_a_spec_whose_oracle_needs_no_contract(monkeypatch):
    """The gate must not become a blanket refusal."""
    monkeypatch.setattr(
        formalizer,
        "call_llm_json",
        lambda *a, **k: {
            "outcome": "ACCEPT",
            "canonical_spec": {
                "task": "t",
                "instance_input": "i",
                "instance_output": "o",
                "data_source": "d",
                "candidate_definition": "c",
                "oracle_strategy": "llm-judge",
                "oracle_detail": "od",
                "interface": "if",
                "scale": {"instance_count": "10", "compute_budget": "1 CPU-hours"},
            },
        },
        raising=False,
    )
    res = formalizer.formalize(_RAW, model="stub", judge=False)
    assert res.outcome == "ACCEPT", [str(d) for d in res.diagnostics]
    assert res.canonical_spec is not None


def test_the_rejection_carries_the_diagnostic_not_just_a_verdict():
    """The refusal is actionable."""
    d = [x for x in check_canonical(_canonical()) if x.code == "MISSING_VARIANTS"][0]
    assert "fail-to-pass-tests" in d.detail, "name the strategy that triggered it"
    assert d.fix_hint and "check_cmd" in d.fix_hint
