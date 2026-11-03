"""The model writes a stage's judgment sections; the harness writes the facts and assembles.
The model's sections are refused for invented scripts, later-stage files, recording talk, or
descriptions of what the harness checks."""

from __future__ import annotations

import os

import pytest

from benchcompiler import stage_prompt as sp

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.join(HERE, "..", "..", "tests", "fixtures", "recipes", "synth", "contract.yaml")


@pytest.fixture
def contract():
    from agentcore import recipes

    return recipes.load_contract(SYNTH)


GOOD = {
    "where_you_are": " ".join(["You are building an instance of a compiler-flags benchmark as scripts."] * 8),
    "goal": " ".join(["Make build.sh honour the flags it is given so both versions compile."] * 5),
    "traps": [
        "A Makefile that hardcodes -O2 ignores your flags; pass CFLAGS on the make command line.",
        "The output directory may not exist; create it first.",
        "Do not run the program here; only compile.",
    ],
}


def test_good_sections_pass_and_assemble_into_the_harness_frame(contract):
    assert sp.validate_stage_sections(GOOD, contract, 1) == ""
    text = sp.assemble_stage_prompt(contract, 1, GOOD)
    print(text)
    for h in (
        "Where you are",
        "Goal of this stage",
        "What to write into bench/",
        "How it is checked",
        "Traps",
        "Done when",
        "Example of a passing report for this stage",
    ):
        assert h in text
    assert "build.sh <dir> <flags>" in text and "both versions build [variants_build]" in text
    assert "bench_deliver" in text and "hardcodes -O2" in text
    assert "run.sh" not in text.split("Traps")[0].split("What to write")[1]


def test_each_defect_in_the_model_sections_is_named(contract):
    bad = dict(GOOD, traps=GOOD["traps"] + ["Later you will write run.sh and gen_inputs.sh."])
    why = sp.validate_stage_sections(bad, contract, 1)
    assert "run.sh belongs to a later stage" in why and "gen_inputs.sh belongs" in why
    bad = dict(GOOD, traps=GOOD["traps"] + ["Also write helper.sh to wrap make."])
    assert "helper.sh is not a script" in sp.validate_stage_sections(bad, contract, 1)
    # recording talk is noted, not refused
    bad = dict(GOOD, goal=GOOD["goal"] + " Record the median runtime of each build.")
    assert sp.validate_stage_sections(bad, contract, 1) == ""
    assert any("record the" in n for n in sp.notes_on_stage_sections(bad, contract, 1))
    fine = dict(GOOD, traps=GOOD["traps"] + ["Do not record the runtime anywhere."])
    assert sp.validate_stage_sections(fine, contract, 1) == "" and sp.notes_on_stage_sections(fine, contract, 1) == []
    bad = dict(GOOD, where_you_are="Short.")
    assert "where_you_are has 1 words" in sp.validate_stage_sections(bad, contract, 1)
    assert "not a JSON object" in sp.validate_stage_sections("text", contract, 1)


def test_the_request_carries_the_stage_facts_and_the_reading_of_record(contract):
    text = sp.build_stage_prompt_request(contract, 1, {"task": "t", "criteria": "each input records its median"})
    assert "build.sh <dir> <flags>" in text and "both versions build" in text
    assert "never mention them" in text and "the system records it" in text
    assert "Do not describe" in text and "drop the sentence" in text
