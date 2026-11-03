"""A REJECT can only be about what the researcher wrote.

When the model accepts and `check_canonical` refuses blocks the model itself writes (`sampling`,
`entry_contract`, `corroboration_contract`, `normalize`), the retry loop runs. If the model then
answers REJECT restating those diagnostics, the result is another retry, not a verdict on the spec
(`MODEL_AUTHORED_CODES` marks these). If the model never produces a canonical that passes its
checks, the result is REJECT with a `(system)` diagnostic saying the formalizer could not write one.
"""

from __future__ import annotations

from benchcompiler import formalizer as fz

CLAUSES_OK = [{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 1000"}]

PROSE = "popular, actively maintained Python projects"


def _raw(**over):
    d = {
        "task": "repair",
        "instance": "one commit",
        "data": "one fix commit",
        "correctness": "the added test fails at the parent and passes at the fix",
        "interface": "patch",
        "scale": {"instance_count": 60},
        "admissibility": PROSE,
        "solver_input": {"kind": "workdir", "path": ".", "describes": "the edited tree"},
    }
    d.update(over)
    return d


def _accept(admissibility=CLAUSES_OK):
    cs = {
        "task": "t",
        "instance_input": "i",
        "instance_output": "o",
        "data_source": "d",
        "candidate_definition": "c",
        "oracle_strategy": "external-authority",
        "oracle_detail": "the suite decides",
        "interface": "patch",
        "scale": {"instance_count": "60", "compute_budget": "1h", "human_budget": "0"},
        "criteria": ["the added test fails at the parent and passes at the fix"],
        "quality_gates": [],
        "examples": [],
        "constraints": [],
        "assumptions": [],
        "delegations": [],
        "admissibility": admissibility,
    }
    return {"outcome": "ACCEPT", "canonical_spec": cs}


def _reject(*codes, field="sampling"):
    return {
        "outcome": "REJECT",
        "diagnostics": [
            {
                "code": c,
                "field": field,
                "detail": "entry_contract lacks the ratio clause",
                "fix_hint": "add a clause relating the two probe fields",
            }
            for c in codes
        ],
    }


class _Model:
    """Replies in sequence, and records every prompt it was given."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, model, prompt, *a, **k):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


def test_a_reject_for_model_authored_blocks_buys_another_round(monkeypatch):
    """The model rejects for its own `sampling`/contract blocks, then, told so, produces a
    canonical that passes.
    """
    llm = _Model(_reject("UNREADABLE_CONTRACT", "MISSING_VIABILITY_PROJECTION"), _accept())
    monkeypatch.setattr(fz, "call_llm_json", llm)
    result = fz.formalize(_raw(), model="stub", judge=False)
    print(f"\n[verbose] rounds={len(llm.prompts)} outcome={result.outcome}")
    assert result.outcome == "ACCEPT", [d.code for d in result.diagnostics]
    assert len(llm.prompts) == 2, "the reject should have bought a round, not ended the run"


def test_the_retry_says_whose_blocks_those_are(monkeypatch):
    llm = _Model(_reject("UNREADABLE_CONTRACT"), _accept())
    monkeypatch.setattr(fz, "call_llm_json", llm)
    fz.formalize(_raw(), model="stub", judge=False)
    second = llm.prompts[1]
    print(f"\n[verbose] retry preamble:\n{second[:1200]}")
    lowered = second.lower()
    assert "you write" in lowered or "your own" in lowered or "yours to write" in lowered, (
        "the retry must say the rejected blocks are the model's own, or the model has no reason to answer differently"
    )


def test_a_reject_the_researcher_can_act_on_is_still_final(monkeypatch):
    """Compile-or-reject is unchanged for anything the researcher actually wrote."""
    llm = _Model(_reject("MISSING_INFO", field="scale"), _accept())
    monkeypatch.setattr(fz, "call_llm_json", llm)
    result = fz.formalize(_raw(), model="stub", judge=False)
    print(
        f"\n[verbose] rounds={len(llm.prompts)} outcome={result.outcome} codes={[d.code for d in result.diagnostics]}"
    )
    assert result.outcome == "REJECT"
    assert len(llm.prompts) == 1, "a researcher-authored blocker is not retried"


def test_rounds_that_never_converge_say_it_was_the_formalizer(monkeypatch):
    llm = _Model(_reject("UNREADABLE_CONTRACT"))
    monkeypatch.setattr(fz, "call_llm_json", llm)
    result = fz.formalize(_raw(), model="stub", judge=False)
    fields = [d.field for d in result.diagnostics]
    details = " ".join(d.detail for d in result.diagnostics).lower()
    print(f"\n[verbose] rounds={len(llm.prompts)} outcome={result.outcome} fields={fields}")
    assert result.outcome == "REJECT"
    assert "(system)" in fields, (
        "the last word must name the formalizer, not leave the researcher holding "
        "diagnostics about blocks their spec does not contain"
    )
    assert "formalizer" in details or "could not" in details


def test_a_reject_about_a_block_the_raw_spec_ships_is_final(monkeypatch):
    """UNREADABLE_CONTRACT is model-authored, but a raw spec that ships its own
    `corroboration_contract` makes that block the researcher's: retrying cannot change it.
    """
    raw = _raw(corroboration_contract=[{"run": "x", "require": "output looks about right"}])
    llm = _Model(_reject("UNREADABLE_CONTRACT", field="corroboration_contract"), _accept())
    monkeypatch.setattr(fz, "call_llm_json", llm)
    result = fz.formalize(raw, model="stub", judge=False)
    print(f"\n[verbose] rounds={len(llm.prompts)} outcome={result.outcome}")
    assert result.outcome == "REJECT" and len(llm.prompts) == 1


def test_the_retry_names_only_blocks_the_raw_spec_lacks(monkeypatch):
    llm = _Model(_reject("UNREADABLE_CONTRACT", field="sampling.normalize"), _accept())
    monkeypatch.setattr(fz, "call_llm_json", llm)
    fz.formalize(_raw(), model="stub", judge=False)
    second = llm.prompts[1]
    i = second.find("appear nowhere in the raw spec")
    print(f"\n[verbose] {second[max(0, i - 200) : i + 40]}")
    assert "sampling appear nowhere" in second
    assert "entry_contract" not in second[max(0, i - 200) : i], (
        "the list is derived from the diagnostics, not restated by hand"
    )


def test_zero_rounds_is_a_reject_not_a_crash(monkeypatch):
    llm = _Model(_accept())
    monkeypatch.setattr(fz, "call_llm_json", llm)
    result = fz.formalize(_raw(), model="stub", rounds=0, judge=False)
    print(f"\n[verbose] {result.outcome} {[d.field for d in result.diagnostics]}")
    assert result.outcome == "REJECT" and "(system)" in [d.field for d in result.diagnostics]
