"""A stage plan's goal that describes the harness is noted for a human's look, not refused, and "build
system" is not the harness.

A sentence about what "the system" or "the checker" will do is noted; "libraries that the project's
build system requires" does not match the rule.
"""

from __future__ import annotations

import json
import os

from benchcompiler import contract_prompt
from benchcompiler.stage_prompt import harness_claims

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKER = os.path.join(ROOT, "agents", "boca_worker")


def _answer_from(contract: dict) -> dict:
    verbs = {
        name: {"args": spec["args"], "does": (contract.get("_does") or {}).get(name) or f"{name} runs"}
        for name, spec in contract["verbs"].items()
    }
    return {
        "verbs": verbs,
        "variants": contract["variants"],
        "observable": contract["observable"],
        "oracle": contract["oracle"],
        "negative_control": contract["negative_control"],
        "solver_input": contract["solver_input"],
        "measure": contract["measure"],
        "predicates": contract["predicates"],
        "instance_fields": contract["_instance_fields"],
        "numbers_quoted": contract.get("_numbers_quoted") or {},
        "stages": [
            {"name": s["name"], "files": s["files"], "goal": s["goal"], "done_when": s["done_when"]}
            for s in contract["_stages"]
        ],
    }


def test_a_build_system_is_not_the_harness():
    assert (
        harness_claims(
            "A common mistake is to install only the compiler but forget libraries "
            "that the project's build system requires."
        )
        == []
    )
    assert (
        harness_claims(
            "Use the file system layout the project expects; the operating system package list is in the Dockerfile."
        )
        == []
    )
    assert harness_claims("The system will verify that the image builds.") != []
    assert harness_claims("The checker builds the image and runs setup.sh in a fresh container.") != []


def test_a_stage_goal_that_describes_the_harness_is_noted_with_the_stage_named():
    with open(os.path.join(WORKER, "contract.json"), encoding="utf-8") as f:
        contract = json.load(f)
    answer = _answer_from(contract)
    for st in answer["stages"]:
        st["goal"] = "Write the files of this stage so that they work in a fresh container."
        st["done_when"] = "the files are written"
    answer["stages"][0]["goal"] = (
        "Install the compiler in the Dockerfile. The checker builds the image and runs setup.sh in a fresh container."
    )
    identity = contract["_instance_fields"]
    parsed, why = contract_prompt.parse_contract_answer(json.dumps(answer), identity)
    print("[verbose] why:", why, "notes:", parsed.get("_draw_notes"))
    assert why == "" and parsed is not None
    notes = " ".join(parsed["_draw_notes"])
    assert answer["stages"][0]["name"] in notes and "harness" in notes.lower()
    assert "The checker builds the image" in notes
    # without the sentence there is nothing to note
    answer["stages"][0]["goal"] = (
        "Install the compiler in the Dockerfile and clone the repository at the commit in setup.sh."
    )
    parsed, why = contract_prompt.parse_contract_answer(json.dumps(answer), identity)
    assert why == "" and parsed["_draw_notes"] == []
