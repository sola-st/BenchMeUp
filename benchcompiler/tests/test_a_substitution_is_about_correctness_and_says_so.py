"""`--auto-substitute` acts only on correctness blockers and records the original spec id.

`formalize_with_fallback` substitutes an oracle only when a correctness code is raised against the
correctness field, read per diagnostic (a `NON_OPERATIONAL_CORRECTNESS` against `data` does not
qualify). The `spec_id` names the spec the researcher submitted; the substitution is recorded in
`substitution`.
"""

from __future__ import annotations

from benchcompiler import formalizer

ADMIS = [{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"}]

RAW = {
    "task": "t",
    "instance": "i",
    "data": "one fix commit",
    "correctness": "it should work",
    "interface": "cli",
    "scale": "10",
    "admissibility": ADMIS,
    "solver_input": {"kind": "workdir", "path": ".", "describes": "a tree"},
}


def _accept_reply():
    return {
        "outcome": "ACCEPT",
        "canonical_spec": {
            "task": "t",
            "instance_input": "i",
            "instance_output": "o",
            "data_source": "one fix commit",
            "candidate_definition": "one fix commit",
            "oracle_strategy": "external-authority",
            "oracle_detail": "the suite decides",
            "interface": "cli",
            "scale": {"instance_count": "10", "compute_budget": "1h", "human_budget": "0"},
            "criteria": ["the added test fails at the parent and passes at the fix"],
            "quality_gates": [],
            "examples": [],
            "constraints": [],
            "assumptions": [],
            "delegations": [],
        },
    }


def _reject_reply(code, field):
    return {"outcome": "REJECT", "diagnostics": [{"code": code, "field": field, "detail": "d", "fix_hint": "f"}]}


class _Scripted:
    """Replies in order; records how many oracle proposals were asked for."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, model, prompt, *a, **k):
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


# ── the gate reads code and field together ──


def test_a_correctness_code_on_another_field_does_not_trigger_substitution(monkeypatch):
    """NON_OPERATIONAL_CORRECTNESS raised against `data` does not trigger substitution: an oracle
    cannot fix the data field.
    """
    llm = _Scripted(_reject_reply("NON_OPERATIONAL_CORRECTNESS", "data"))
    monkeypatch.setattr(formalizer, "call_llm_json", llm)
    proposed = []
    monkeypatch.setattr(
        formalizer,
        "propose_oracle_strategies",
        lambda *a, **k: (
            proposed.append(1)
            or {
                "proposals": [
                    {"oracle_strategy": "external-authority", "mechanism": "m", "human_budget_implication": "0h"}
                ]
            }
        ),
    )
    result = formalizer.formalize_with_fallback(RAW, model="stub", judge=False)
    print(f"  outcome={result.outcome} proposals_requested={len(proposed)}")
    assert result.outcome == "REJECT"
    assert not proposed, "an oracle was proposed to fix a problem with `data`"


def test_a_correctness_code_on_the_correctness_field_still_substitutes(monkeypatch):
    """The two-sided half: the path this function exists for must keep working."""
    llm = _Scripted(_reject_reply("NON_OPERATIONAL_CORRECTNESS", "correctness"), _accept_reply())
    monkeypatch.setattr(formalizer, "call_llm_json", llm)
    monkeypatch.setattr(
        formalizer,
        "propose_oracle_strategies",
        lambda *a, **k: {
            "proposals": [
                {
                    "oracle_strategy": "external-authority",
                    "mechanism": "the suite decides",
                    "human_budget_implication": "0h",
                }
            ]
        },
    )
    result = formalizer.formalize_with_fallback(RAW, model="stub", judge=False)
    print(f"  outcome={result.outcome}")
    assert result.outcome == "ACCEPT_SUBSTITUTED"


def test_a_non_correctness_blocker_alongside_it_still_blocks(monkeypatch):
    """If the data field is also broken, that is fixed first."""
    llm = _Scripted(
        {
            "outcome": "REJECT",
            "diagnostics": [
                {"code": "NON_OPERATIONAL_CORRECTNESS", "field": "correctness", "detail": "d", "fix_hint": "f"},
                {"code": "MISSING_INFO", "field": "data", "detail": "d", "fix_hint": "f"},
            ],
        }
    )
    monkeypatch.setattr(formalizer, "call_llm_json", llm)
    proposed = []
    monkeypatch.setattr(formalizer, "propose_oracle_strategies", lambda *a, **k: proposed.append(1) or None)
    assert formalizer.formalize_with_fallback(RAW, model="stub", judge=False).outcome == "REJECT"
    assert not proposed


# ── the id names the spec the researcher submitted ──


def test_a_substituted_accept_keeps_the_original_spec_id(monkeypatch):
    """`--lock` records the sha of the original file, so the spec_id is hashed from the original
    raw spec, not the substituted one.
    """
    from benchcompiler.formalizer import _spec_id

    llm = _Scripted(_reject_reply("NON_OPERATIONAL_CORRECTNESS", "correctness"), _accept_reply())
    monkeypatch.setattr(formalizer, "call_llm_json", llm)
    monkeypatch.setattr(
        formalizer,
        "propose_oracle_strategies",
        lambda *a, **k: {
            "proposals": [
                {
                    "oracle_strategy": "external-authority",
                    "mechanism": "the suite decides",
                    "human_budget_implication": "0h",
                }
            ]
        },
    )
    result = formalizer.formalize_with_fallback(RAW, model="stub", judge=False)
    print(f"  spec_id={result.canonical_spec.spec_id} original={_spec_id(RAW)}")
    assert result.outcome == "ACCEPT_SUBSTITUTED"
    assert result.canonical_spec.spec_id == _spec_id(RAW), (
        "the id names the substituted spec, which is not the one on disk"
    )


def test_what_was_substituted_is_still_recorded(monkeypatch):
    """Keeping the original id still records that a substitution happened (ACCEPT_SUBSTITUTED)."""
    llm = _Scripted(_reject_reply("NON_OPERATIONAL_CORRECTNESS", "correctness"), _accept_reply())
    monkeypatch.setattr(formalizer, "call_llm_json", llm)
    monkeypatch.setattr(
        formalizer,
        "propose_oracle_strategies",
        lambda *a, **k: {
            "proposals": [
                {
                    "oracle_strategy": "external-authority",
                    "mechanism": "the suite decides",
                    "human_budget_implication": "0h",
                }
            ]
        },
    )
    result = formalizer.formalize_with_fallback(RAW, model="stub", judge=False)
    assert result.substitution and result.substitution.get("mechanism")
    assert result.original_diagnostics, "the rejection that caused it was dropped"
