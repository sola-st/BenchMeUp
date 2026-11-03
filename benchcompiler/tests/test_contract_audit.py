"""The drawn config is audited against the spec before it is accepted, and the audit's refusal
is fed back to the drawer. No model: the model call is scripted. The answer parser is
two-sided: an acceptance that lists a gap, or a refusal without a fix, is not an audit."""

from __future__ import annotations

import json
import os

import pytest

from benchcompiler import contract_audit, contract_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _without_harness_claims(stages: list[dict]) -> list[dict]:
    """The stage plan with any sentence about what the harness does replaced."""
    from benchcompiler.stage_prompt import harness_claims

    out = []
    for s in stages:
        s = dict(s)
        for f in ("goal", "done_when"):
            if harness_claims(str(s.get(f) or "")):
                s[f] = f"the files of stage {s['name']!r} are written and work in a fresh container"
        out.append(s)
    return out


def _answer_shape(drawn: dict) -> dict:
    """A drawn contract.json, back in the shape the drawer answers with."""
    out = {
        k: drawn.get(k)
        for k in ("variants", "observable", "oracle", "negative_control", "solver_input", "measure", "predicates")
    }
    out["verbs"] = {
        n: {"args": v["args"], "does": (drawn.get("_does") or {}).get(n, f"{n} does its job")}
        for n, v in drawn["verbs"].items()
    }
    out["instance_fields"] = drawn.get("_instance_fields") or []
    out["stages"] = _without_harness_claims(
        [
            {"name": s["name"], "files": s["files"], "goal": s.get("goal", "g"), "done_when": s.get("done_when", "d")}
            for s in drawn.get("_stages") or []
        ]
    )
    return out


@pytest.fixture
def boca_answer():
    with open(os.path.join(ROOT, "agents", "boca_worker", "contract.json"), encoding="utf-8") as f:
        return _answer_shape(json.load(f))


REFUSAL = {
    "claims": [
        {
            "claim": "the test is applied to both snapshots",
            "realized_by": [],
            "realized": False,
            "gap": "no variant setup applies the test",
        }
    ],
    "verdict": "refuse",
    "fix": "add a step {verb: apply_patch, args: {patch: /bench/test_patch}} to prefix and postfix",
}
ACCEPT = {
    "claims": [
        {
            "claim": "two builds of one program",
            "realized_by": ["variants.default", "variants.optimized"],
            "realized": True,
            "gap": "",
        }
    ],
    "verdict": "accept",
    "fix": "",
}


def test_parse_is_two_sided():
    assert contract_audit.parse_audit_answer(json.dumps(REFUSAL))[1] == ""
    assert contract_audit.parse_audit_answer(json.dumps(ACCEPT))[1] == ""
    bad = dict(ACCEPT, claims=[dict(ACCEPT["claims"][0], realized=False, gap="x")])
    assert "unrealized claim is a refusal" in contract_audit.parse_audit_answer(json.dumps(bad))[1]
    assert "without a fix" in contract_audit.parse_audit_answer(json.dumps(dict(REFUSAL, fix="")))[1]
    assert "no claims" in contract_audit.parse_audit_answer(json.dumps(dict(ACCEPT, claims=[])))[1]
    assert "no verdict" in contract_audit.parse_audit_answer("{}")[1]


def test_the_audit_refusal_reaches_the_next_draw(monkeypatch, boca_answer):
    calls = []
    script = [json.dumps(boca_answer), json.dumps(REFUSAL), json.dumps(boca_answer), json.dumps(ACCEPT)]

    def fake(model, prompt, **kw):
        calls.append((kw.get("site"), prompt))
        return script.pop(0)

    monkeypatch.setattr(contract_prompt, "call_llm_json", fake)
    monkeypatch.setattr(contract_audit, "call_llm_json", fake)
    raw = {"name": "boca", "task": "find flags that make the program faster"}
    contract, log = contract_prompt.draw_contract(raw, "scripted", identity_fields=boca_answer["instance_fields"])
    print("[verbose]\n" + "\n".join(log))
    assert contract is not None
    assert [s for s, _ in calls] == ["contract", "contract.audit", "contract", "contract.audit"]
    # the second draw saw the audit's gap and fix, as feedback from the checker
    second_draw = calls[2][1]
    assert "REFUSED BY THE CHECKER" in second_draw
    assert "no variant setup applies the test" in second_draw and "apply_patch" in second_draw
    assert "does not realize the spec" in second_draw
    # the audit prompt showed the spec and the config with its does sentences
    audit_prompt = calls[1][1]
    assert "find flags that make the program faster" in audit_prompt and '"does"' in audit_prompt


def test_an_unreadable_audit_is_a_refusal_not_an_acceptance(monkeypatch, boca_answer):
    script = [json.dumps(boca_answer), "not json", "still not json"]
    monkeypatch.setattr(contract_prompt, "call_llm_json", lambda m, p, **kw: script.pop(0))
    monkeypatch.setattr(contract_audit, "call_llm_json", lambda m, p, **kw: script.pop(0))
    contract, log = contract_prompt.draw_contract(
        {"name": "x", "task": "t"}, "scripted", rounds=1, identity_fields=boca_answer["instance_fields"]
    )
    print("[verbose]\n" + "\n".join(log))
    assert contract is None
    assert any("not accepted without an audit" in ln for ln in log)


def test_the_audit_prompt_holds_identity_predicates_to_the_definition():
    """A generated predicate `test -f /work/repo/{test_file_path}` is evaluated in a clone at the
    base commit, where it rejects every PR that adds its test file; the audit must refuse it.
    """
    contract = {
        "verbs": {},
        "variants": {},
        "observable": {},
        "oracle": {},
        "negative_control": None,
        "solver_input": None,
        "measure": None,
        "predicates": [{"run": "test -f /work/repo/{test_file_path}", "require": "exit zero"}],
    }
    text = contract_audit.build_audit_prompt({"candidate_definition": "a PR that adds a test"}, contract)
    low = text.lower()
    assert "identity predicate" in low
    assert "every candidate the definition admits" in low
    assert "adds or modifies" in low and "setup.sh" in low and "git cat-file -e {commit}:{path}" in text
    # one claim per predicate, so the rule is applied and not only stated
    assert "one claim per identity predicate" in low
    assert "holds for every candidate the definition admits" in " ".join(low.split())
