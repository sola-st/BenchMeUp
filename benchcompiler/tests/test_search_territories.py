"""Parallel searching agents split the space by TERRITORIES the model draws from the family's
own brief (star bands, languages, domains: whichever fits), never by a rule in code. Each
attempt's brief carries its territory and names its siblings'. A hand-written territory
list is accepted the same way."""

from __future__ import annotations

import json
import os
import sys

from benchcompiler.tests.test_worker_emit_with_fake_provider import install_fake_provider

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPEC = os.path.join(ROOT, "benchcompiler", "examples", "boca_flags.yaml")
CANONICAL = os.path.join(ROOT, "specs", "boca", "canonical.json")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOOD = {
    "dimension": "repository stars",
    "territories": [
        "Repositories with 100 to 999 stars.",
        "Repositories with 1,000 to 2,999 stars.",
        "Repositories with 3,000 to 9,999 stars.",
        "Repositories with 10,000 stars or more.",
    ],
}


def _agent(tmp_path):
    import shutil

    from benchcompiler import search

    out = str(tmp_path / "agents" / "boca_search")
    search.emit_search_agent(SPEC, CANONICAL, out, target=1000, model="example/model", log=lambda *_: None)
    # the prompts are shared: agents/_shared beside the agent, as in the real tree
    shutil.copytree(os.path.join(ROOT, "agents", "_shared"), str(tmp_path / "agents" / "_shared"))
    return out


def test_territories_are_drawn_from_the_brief_and_checked(tmp_path, monkeypatch):
    from benchcompiler import search

    agent = _agent(tmp_path)
    answers = [
        json.dumps({"territories": ["too few"]}),  # refused: 1 of 4
        json.dumps({"dimension": "x", "territories": ["a b c d e"] * 4}),  # refused: not distinct
        json.dumps(GOOD),
    ]
    calls = install_fake_provider(monkeypatch, lambda prompt, messages: answers.pop(0))
    log = []
    t = search.draw_territories(agent, 4, "fake/model", log=log.append)
    print("[verbose] log:", log)
    assert t["territories"] == GOOD["territories"] and t["dimension"] == "repository stars"
    assert len(calls) == 3 and "refused" in log[0] and "refused" in log[1]
    # the prompt is the family's own brief
    assert (
        "CHEAP ON PURPOSE" in calls[0][0] and "stars: >= 100" in calls[0][0] and "exactly 4 territories" in calls[0][0]
    )
    assert "Your previous answer was refused" in calls[1][0]


def test_each_attempt_carries_its_territory_and_its_siblings(tmp_path):
    from benchcompiler import search

    agent = _agent(tmp_path)
    inst = search.attempt_instance(
        agent,
        known_from=[],
        target=250,
        territory=GOOD["territories"][1],
        siblings=[t for j, t in enumerate(GOOD["territories"]) if j != 1],
    )
    assert inst["territory"] == "Repositories with 1,000 to 2,999 stars." and len(inst["sibling_territories"]) == 3
    assert inst["sample_size"] == 250
    # rendered into the brief
    from agentcore.config.loader import load_agent_spec
    from agentcore.run.single import build_agent

    spec = load_agent_spec(
        os.path.join(agent, "agent.yaml"),
        {
            "env": {"type": "local", "image": None},
            "models": {"main": {"name": "scripted", "class": "scripted", "kwargs": {"script": []}}},
        },
    )
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    a = build_agent(spec, agent, instance=inst, workspace=ws, out_dir=str(tmp_path / "out"))
    a._scratch["out_dir"] = str(tmp_path / "out")
    text = a.prompts.render("system", **a._template_vars())
    i = text.index("YOUR TERRITORY")
    block = text[i : i + 500]
    print("[verbose] territory block:\n" + block)
    assert "Repositories with 1,000 to 2,999 stars." in block
    assert "Other agents search the same benchmark in parallel" in block
    assert "Repositories with 10,000 stars or more." in block and "1,000 to 2,999" not in block.split("parallel")[1]
    assert '"found_by"' in text
    # no territory: no block
    solo = search.attempt_instance(agent, known_from=[], target=1000)
    a2 = build_agent(spec, agent, instance=solo, workspace=ws, out_dir=str(tmp_path / "out2"))
    a2._scratch["out_dir"] = str(tmp_path / "out2")
    assert "YOUR TERRITORY" not in a2.prompts.render("system", **a2._template_vars())


def test_check_territories_names_what_is_wrong():
    from benchcompiler.search import check_territories

    assert check_territories(None, 2) == "the answer is not a JSON object"
    assert "exactly 2" in check_territories({"territories": ["one two three four"]}, 2)
    assert "four words" in check_territories({"territories": ["short", "one two three four"]}, 2)
    assert "distinct" in check_territories({"territories": ["one two three four", "One two three four "]}, 2)
    assert check_territories(GOOD, 4) == ""
