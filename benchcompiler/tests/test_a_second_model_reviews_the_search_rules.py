"""The search rules are reviewed by a second model call before they are accepted.

The generator answers with a canonical spec; once it passes the mechanical checks, a judge
compares it with the specification. A refusal goes back to the generator as feedback and the
loop runs again; when the rounds are spent the researcher gets the feedback, not the rules.
"""

from __future__ import annotations

import json
import os

from benchcompiler import formalizer, rules_audit
from benchcompiler.spec import load_raw_spec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = load_raw_spec(os.path.join(ROOT, "specs", "d4j", "spec.yaml"))
REPLY = json.load(open(os.path.join(ROOT, "specs", "d4j", "canonical.json")))
ACCEPT = json.dumps({"omissions": [], "ambiguities": [], "verdict": "accept", "fix": ""})
REFUSE = json.dumps(
    {
        "omissions": [],
        "verdict": "refuse",
        "ambiguities": [{"term": "popular", "gap": "no star threshold"}],
        "fix": "state a minimum star count in admissibility",
    }
)


def _script(monkeypatch, judge_answers):
    generator_prompts, judge_prompts = [], []

    def generator(model, prompt, **kw):
        generator_prompts.append(prompt)
        return json.loads(json.dumps(REPLY))

    answers = iter(judge_answers)

    def judge(model, prompt, **kw):
        judge_prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(formalizer, "call_llm_json", generator)
    monkeypatch.setattr(rules_audit, "call_llm_json", judge)
    return generator_prompts, judge_prompts


def test_a_refusal_goes_back_to_the_generator_and_the_next_round_is_accepted(monkeypatch):
    gen, jud = _script(monkeypatch, [REFUSE, ACCEPT])
    result = formalizer.formalize(RAW, "m")
    print("  generator calls:", len(gen), " judge calls:", len(jud), " outcome:", result.outcome)
    assert result.outcome == "ACCEPT"
    assert len(gen) == 2 and len(jud) == 2
    assert "no star threshold" in gen[1] and "state a minimum star count" in gen[1]
    assert "YOUR PREVIOUS ATTEMPT" in gen[1]
    assert RAW["task"].split()[0] in jud[0]


def test_rules_the_judge_keeps_refusing_go_back_to_the_researcher(monkeypatch):
    gen, jud = _script(monkeypatch, [REFUSE] * 3)
    result = formalizer.formalize(RAW, "m", rounds=3)
    print("  outcome:", result.outcome, "|", str(result.diagnostics[0])[:100])
    assert result.outcome == "REJECT"
    assert len(gen) == 3
    assert any("no star threshold" in d.detail for d in result.diagnostics)


def test_a_review_that_cannot_be_read_is_not_an_acceptance(monkeypatch):
    gen, jud = _script(monkeypatch, ["not json", "still not json"])
    accepted, why = rules_audit.judge_rules(RAW, REPLY["canonical_spec"], "m")
    print("  ->", accepted, why)
    assert not accepted and "could not be read" in why


def test_an_accept_that_lists_an_ambiguity_is_malformed():
    answer, why = rules_audit.parse_judge_answer(
        json.dumps({"omissions": [], "ambiguities": [{"term": "t", "gap": "g"}], "verdict": "accept", "fix": ""})
    )
    assert answer is None and "accepts while listing" in why


def test_the_review_can_be_switched_off(monkeypatch):
    gen, jud = _script(monkeypatch, [])
    result = formalizer.formalize(RAW, "m", judge=False)
    assert result.outcome == "ACCEPT" and len(jud) == 0
