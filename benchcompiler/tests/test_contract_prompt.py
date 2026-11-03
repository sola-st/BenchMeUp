"""The prompt that derives a family's instance contract, offline.

Checks the plumbing: the prompt renders from every example spec, a valid answer parses into the
engine's contract with the `does` sentences kept, and a bad answer is refused with the field named
so the retry loop has something to say.
"""

from __future__ import annotations

import glob
import json
import os

from benchcompiler import contract_prompt as cp
from benchcompiler.spec import load_raw_spec

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")

GOOD = json.JSONDecoder().raw_decode(cp.EXAMPLE[cp.EXAMPLE.index("{") :])[
    0
]  # the first worked example; a second follows


def test_the_prompt_renders_for_every_example_spec():
    for path in sorted(glob.glob(os.path.join(EXAMPLES, "*.yaml"))):
        text = cp.build_contract_prompt(load_raw_spec(path))
        assert "=== THE SPEC" in text and "RULES" in text and "WORKED EXAMPLE" in text
        assert len(text) < 40_000, path


def test_the_worked_example_parses_and_keeps_does():
    contract, why = cp.parse_contract_answer(json.dumps(GOOD))
    assert not why, why
    assert contract["_does"]["run"].startswith("run <dir>'s program")
    assert contract["_instance_fields"] == ["repo_url", "commit_hash", "main_source_file"]
    assert contract["measure"]["floor_ms"] == 1000


def test_a_missing_does_or_a_solver_placeholder_is_refused_by_name():
    bad = json.loads(json.dumps(GOOD))
    del bad["verbs"]["break"]["does"]
    _, why = cp.parse_contract_answer(json.dumps(bad))
    assert "break" in why and "does" in why
    bad = json.loads(json.dumps(GOOD))
    bad["variants"]["default"]["setup"][0]["args"]["flags"] = "<solver_input>"
    _, why = cp.parse_contract_answer(json.dumps(bad))
    assert "solver placeholder" in why


def test_the_recorded_draws_still_validate():
    from agentcore import recipes

    n = 0
    for path in sorted(glob.glob(os.path.join(EXAMPLES, "drawn_contracts", "*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        core = {
            k: d.get(k)
            for k in (
                "verbs",
                "variants",
                "observable",
                "oracle",
                "negative_control",
                "solver_input",
                "measure",
                "predicates",
            )
        }
        recipes.validate_contract(core, where=path)
        n += 1
    assert n >= 5


def test_a_plan_that_lumps_building_and_running_into_one_stage_is_refused():
    bad = json.loads(json.dumps(GOOD))
    bad["stages"] = [
        bad["stages"][0],
        {
            "name": "write everything at once",
            "files": ["build.sh", "run.sh", "gen_inputs.sh", "break.sh"],
            "goal": "write it all",
            "done_when": "it passes",
        },
    ]
    _, why = cp.parse_contract_answer(json.dumps(bad))
    print(why)
    assert "run.sh" in why and "own stage" in why


def test_a_stage_may_ship_data_files_but_not_undeclared_scripts():
    plan = json.loads(json.dumps(GOOD))
    plan["stages"][2]["files"].append("reference_output.txt")
    contract, why = cp.parse_contract_answer(json.dumps(plan))
    assert not why, why
    plan["stages"][2]["files"].append("helper.sh")
    _, why = cp.parse_contract_answer(json.dumps(plan))
    assert "helper.sh" in why and "not a declared verb" in why
