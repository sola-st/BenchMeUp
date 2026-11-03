"""`bench worker` with a fake model provider: the system behaves as if it calls the real model.

`litellm.completion` is replaced below the drawer's own retry and refusal logic, so every prompt,
parse, checker refusal and retry runs as in a real draw. The scripted provider answers the config
prompt three times (garbage, a config naming the source file field differently from the canonical's
identity, a correct one) and the stage-section prompts with one harness claim on stage 1 before a
clean answer. The test asserts the retries happened and the worker written at the end is loadable
and carries what was drawn.
"""

from __future__ import annotations

import copy
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKER = os.path.join(ROOT, "agents", "boca_worker")
SPEC = os.path.join(ROOT, "benchcompiler", "examples", "boca_flags.yaml")
CANONICAL = os.path.join(ROOT, "specs", "boca", "canonical.json")
with open(CANONICAL, encoding="utf-8") as _f:
    # the identity is the formal spec's (for example repo_url, commit_sha, program_name); the test
    # reads it rather than naming it
    IDENTITY = [str(k) for k in json.load(_f)["canonical_spec"]["sampling"]["lead_key"]]
WRONG_FIELD = IDENTITY[-1]


def install_fake_provider(monkeypatch, reply):
    """Replace litellm.completion with `reply(prompt, messages) -> text`; returns the call log."""
    import litellm

    calls: list[tuple[str, str]] = []

    def completion(*args, **kwargs):
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        text = reply(prompt, messages)
        calls.append((prompt, text))
        return litellm.ModelResponse(
            id=f"fake-{len(calls)}",
            model=str(kwargs.get("model") or "fake"),
            choices=[{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": text}}],
            usage={
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(text) // 4,
                "total_tokens": (len(prompt) + len(text)) // 4,
            },
        )

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda *a, **k: 0.0)
    return calls


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


def _good_config_answer() -> dict:
    """A correct answer to the config prompt, assembled from a recorded draw."""
    with open(os.path.join(WORKER, "contract.json"), encoding="utf-8") as f:
        c = json.load(f)
    verbs = {
        name: {"args": spec["args"], "does": (c.get("_does") or {}).get(name) or f"{name} does its job"}
        for name, spec in c["verbs"].items()
    }
    return {
        "verbs": verbs,
        "variants": c["variants"],
        "observable": c["observable"],
        "oracle": c["oracle"],
        "negative_control": c["negative_control"],
        "solver_input": c["solver_input"],
        "measure": c["measure"],
        "predicates": c["predicates"],
        "instance_fields": c["_instance_fields"],
        "numbers_quoted": c.get("_numbers_quoted") or {},
        "stages": _without_harness_claims(
            [
                {"name": s["name"], "files": s["files"], "goal": s["goal"], "done_when": s["done_when"]}
                for s in c["_stages"]
            ]
        ),
    }


def _wrong_identity(answer: dict) -> dict:
    """The same config with the last identity field named differently."""
    w = copy.deepcopy(answer)
    w["instance_fields"] = [f if f != WRONG_FIELD else f"main_{WRONG_FIELD}" for f in w["instance_fields"]]
    w["predicates"] = [
        {"run": p["run"].replace("{" + WRONG_FIELD + "}", "{main_" + WRONG_FIELD + "}"), "require": p["require"]}
        for p in w["predicates"]
    ]
    return w


def _sections(i: int) -> dict:
    where = " ".join(
        [f"You are building an instance of the compiler-flags benchmark as scripts; this is stage {i}."] * 7
    )
    goal = " ".join([f"Write the files of stage {i} so that the checks of this stage pass in a fresh container."] * 5)
    traps = [
        "A package present in your sandbox but not in the Dockerfile is missing in the fresh container.",
        "A build that ignores the flags it is given produces the same program for every version.",
        "Output that carries a timestamp or a path is never stable.",
    ]
    return {"where_you_are": where, "goal": goal, "traps": traps}


@pytest.mark.timeout(300)
def test_bench_worker_draws_through_refusals_and_writes_a_loadable_worker(tmp_path, monkeypatch):
    good = _good_config_answer()
    wrong = _wrong_identity(good)
    seen = {"config": 0, "stage": {}}

    def reply(prompt: str, messages) -> str:
        if "=== THE CONFIGURATION TO AUDIT ===" in prompt:  # the audit of an accepted shape
            return json.dumps(
                {
                    "claims": [
                        {
                            "claim": "two builds of one program agree",
                            "realized_by": ["variants"],
                            "realized": True,
                            "gap": "",
                        }
                    ],
                    "verdict": "accept",
                    "fix": "",
                }
            )
        if "=== OUTPUT SCHEMA ===" in prompt:  # the config prompt
            n = seen["config"]
            seen["config"] += 1
            return ["this is not json {", json.dumps(wrong), json.dumps(good)][min(n, 2)]
        m = re.search(r"Stage (\d+) of (\d+)", prompt)  # a stage-section prompt
        assert m, "an unexpected prompt reached the provider"
        i = int(m.group(1))
        n = seen["stage"].get(i, 0)
        seen["stage"][i] = n + 1
        s = _sections(i)
        if i == 1 and n == 0:
            s = dict(s, traps=s["traps"] + ["The harness will check that the commit exists in the clone."])
        return json.dumps(s)

    calls = install_fake_provider(monkeypatch, reply)
    from benchcompiler import worker

    out_dir = str(tmp_path / "boca_fake_worker")
    log: list[str] = []
    result = worker.emit_worker(
        SPEC, CANONICAL, out_dir, model="fake/provider-model", sample_size=3, name="boca_fake_worker", log=log.append
    )
    print("\n".join(log))

    # the retries happened where the checker refused
    assert seen["config"] == 3, seen
    # a harness claim in the stage text is noted, not refused
    assert all(seen["stage"][i] == 1 for i in (1, 2, 3, 4)), seen
    # the garbage answer is retried inside the JSON call itself, so the draw log shows only
    # the checker's refusal of the second answer
    assert any(f"instance_fields must include the identity fields ['{WRONG_FIELD}']" in ln for ln in log)
    assert any("noted, not refused" in ln and "describes what the harness" in ln for ln in log)
    # every call went through the provider seam, none through the network
    assert len(calls) == 3 + 1 + 4  # three draws, one audit of the accepted shape, four stage sections

    # the worker on disk is loadable and carries what was drawn
    import sys

    sys.path.insert(0, ROOT)
    from agentcore.config.loader import load_agent_spec
    from agentcore.tools import recipes_tools as rt

    spec = load_agent_spec(os.path.join(out_dir, "agent.yaml"))
    assert list(spec.worklist.key) == IDENTITY
    assert [st.name for st in spec.worklist.states][0] == "REGISTERED" and len(spec.worklist.states) == 5
    assert spec.stages == [] and spec.oracle.grade == "agentcore.tools.recipes_tools:finish_gate"
    with open(os.path.join(out_dir, "contract.json"), encoding="utf-8") as f:
        drawn = json.load(f)
    assert len(drawn["_stage_sections"]) == 4 and drawn["_drawn"]["model"] == "fake/provider-model"
    assert "_stage_prompts" not in drawn
    c = rt.load_family_contract(out_dir)
    from agentcore import recipes

    text = recipes.stage_instruction(recipes.stages_for(c)[0], c)
    assert "this is stage 1" in text  # the noted harness claim stays in the text
    assert "git -C /work/repo rev-parse --verify {" in text  # the config's predicate, rendered now
    with open(os.path.join(out_dir, "instances", "instances.jsonl"), encoding="utf-8") as f:
        inst = json.loads(f.readline())
    assert inst["identity_field"] == "repo_url" and inst["sample_size"] == 3
    assert result["key"] == IDENTITY
