"""`bench search-agent`, `bench stock-freeze` and `bench handover`: the searching agent is
emitted from the spec's own words, the frozen stock is the union on the identity, and the
hand-over fills the workers' candidate pools whole repositories at a time."""

from __future__ import annotations

import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPEC = os.path.join(ROOT, "benchcompiler", "examples", "boca_flags.yaml")
CANONICAL = os.path.join(ROOT, "specs", "boca", "canonical.json")
WORKER = os.path.join(ROOT, "agents", "boca_worker")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _cand(repo, sha, program="prog", **extra):
    return {"repo_url": repo, "commit_sha": sha, "program_name": program, **extra}


def test_the_searching_agent_is_the_specs_own_words(tmp_path):
    from benchcompiler import search

    out = str(tmp_path / "boca_search")
    r = search.emit_search_agent(
        SPEC,
        CANONICAL,
        out,
        target=50,
        model="example/model",
        known_from=[str(tmp_path / "earlier")],
        log=lambda *_: None,
    )
    assert r["identity"] == ["repo_url", "commit_sha", "program_name"]
    inst = json.loads(open(os.path.join(out, "instances", "instances.jsonl")).readline())
    raw = yaml.safe_load(open(SPEC))
    assert inst["data_source"] == raw["data"].strip()  # verbatim
    assert [a["name"] for a in inst["admissibility"]] == [a["name"] for a in raw["admissibility"]]
    assert inst["constraints"] == "- C programs only"
    # The spread sentence is rendered from the canonical's own caps, so it is checked against them
    # rather than against a copy of their wording.
    caps = (json.load(open(CANONICAL))["canonical_spec"].get("sampling") or {}).get("caps") or []
    assert inst["sample_size"] == 50
    assert inst["spread"], "the brief must tell the agent how candidates may be spread"
    for cap in caps:
        assert str(cap["limit"]) in inst["spread"] and all(k in inst["spread"] for k in cap["keys"]), (
            f"the spread sentence drops the cap {cap!r} the canonical declares"
        )
    assert inst["known_from"] == [str(tmp_path / "earlier")]
    cfg = yaml.safe_load(open(os.path.join(out, "agent.yaml")))
    assert "worklist" not in cfg and cfg["tools"][-1] == "stock_verify"
    assert cfg["oracle"]["grade"] == "agentcore.tools.search_tools:finish_gate"
    assert cfg["budgets"]["meters"][0] == {"kind": "cost_usd", "limit": 2.0, "charge_on": "cycle"}
    # loadable by the runner, prompts included
    from agentcore.config.loader import load_agent_spec

    spec = load_agent_spec(os.path.join(out, "agent.yaml"))
    assert str(spec.prompts["system"]).endswith("search_system.j2")
    rec = json.load(open(os.path.join(out, "search.json")))
    assert rec["identity"] == r["identity"] and rec["target"] == 50
    # The check confirms the definition, not only the repository-level rules.
    from agentcore.agent.prompts import PromptStore

    shared = os.path.join(ROOT, "agents", "_shared")
    text = PromptStore(shared, {"system": "search_system.j2"}).render("system", instance=inst)
    print("[verbose]", [ln for ln in text.splitlines() if "definition" in ln][:4])
    # the definition is judged by the agent per candidate (evidence + judgment recorded on the line);
    # the check script confirms what a script can confirm
    assert "judgment" in text and "in your own words" in text.lower()
    assert "A CANDIDATE IS SOMETHING YOU HAVE LOOKED AT" in text
    assert "is not a judgment" in text
    assert "what a script CAN confirm" in text
    assert "the mechanical part of the definition" in text and "judged by YOU" in text
    assert "exits 0 only when every line passes" in text


def test_the_frozen_stock_is_the_union_on_the_identity_and_append_only(tmp_path):
    from benchcompiler import search

    agent = str(tmp_path / "boca_search")
    search.emit_search_agent(SPEC, CANONICAL, agent, target=5, model="example/model", log=lambda *_: None)
    a, b = "a" * 40, "b" * 40
    run1 = tmp_path / "run1" / "stock"
    run2 = tmp_path / "run2" / "stock"
    run1.mkdir(parents=True), run2.mkdir(parents=True)
    (run1 / "candidates.jsonl").write_text(
        json.dumps(_cand("https://github.com/x/alpha", a, found="README"))
        + "\n"
        + json.dumps(_cand("https://github.com/x/beta", b))
        + "\n"
        + '{"repo_url": "https://github.com/x/bad"}\n'
    )
    # the live stock was wiped after a failed check: the file is empty, and only the copy kept at its
    # verify still holds what the agent found.
    (run2 / "verify_001.candidates.jsonl").write_text(
        json.dumps(_cand("https://github.com/X/ALPHA.git", a, found="docs"))
        + "\n"  # the same alpha
        + json.dumps(_cand("https://github.com/x/gamma", "c" * 40))
        + "\n"
        + json.dumps(_cand("https://github.com/x/wiped", "e" * 40))
        + "\n"
    )
    (run2 / "candidates.jsonl").write_text("")
    out = str(tmp_path / "stock")
    m = search.freeze_stock([str(tmp_path / "run1"), str(tmp_path / "run2")], out, agent_dir=agent, log=print)
    print("[verbose] manifest runs:", json.dumps(m["runs"]))
    assert m["count"] == 4
    assert m["runs"][str(tmp_path / "run1")] == {
        "files": 1,
        "lines": 3,
        "distinct": 2,
        "duplicates": 0,
        "malformed": 1,
        "new_to_stock": 2,
    }
    # one file, not two: the wiped live stock contributes nothing and the NEWEST copy stands in
    assert m["runs"][str(tmp_path / "run2")]["new_to_stock"] == 2 and m["runs"][str(tmp_path / "run2")]["files"] == 1
    lines = [json.loads(ln) for ln in open(os.path.join(out, "candidates.jsonl"))]
    assert [ln["repo_url"] for ln in lines] == [
        "https://github.com/x/alpha",
        "https://github.com/x/beta",
        "https://github.com/x/gamma",
        "https://github.com/x/wiped",
    ]
    assert lines[0]["found"] == "README"  # the first spelling is kept
    # a second freeze keeps what is frozen first and adds what is new
    run3 = tmp_path / "run3"
    run3.mkdir()
    (run3 / "candidates.jsonl").write_text(
        json.dumps(_cand("https://github.com/x/beta", b))
        + "\n"
        + json.dumps(_cand("https://github.com/x/delta", "d" * 40))
        + "\n"
    )
    m2 = search.freeze_stock([str(run3)], out, agent_dir=agent, log=lambda *_: None)
    assert m2["count"] == 5 and m2["runs"][str(run3)]["new_to_stock"] == 1
    # every candidate keeps the run it came from across freezes (the known-set reads it)
    origins = json.load(open(os.path.join(out, "manifest.json")))["origin"]
    by_repo = {json.loads(ln)["repo_url"]: ln for ln in open(os.path.join(out, "candidates.jsonl"))}
    from agentcore.tools.search_tools import key_of

    fields, norm = ["repo_url", "commit_sha", "program_name"], {"repo_url": "git_url"}
    assert origins[key_of(json.loads(by_repo["https://github.com/x/alpha"]), fields, norm)] == str(tmp_path / "run1")
    assert origins[key_of(json.loads(by_repo["https://github.com/x/delta"]), fields, norm)] == str(run3)
    lines = [json.loads(ln) for ln in open(os.path.join(out, "candidates.jsonl"))]
    assert (
        lines[0]["repo_url"] == "https://github.com/x/alpha" and lines[-1]["repo_url"] == "https://github.com/x/delta"
    )


def test_the_handover_fills_the_workers_pools_whole_repositories_at_a_time(tmp_path):
    from benchcompiler import search

    stock = tmp_path / "candidates.jsonl"
    rows = [
        _cand("https://github.com/x/alpha", "1" * 40, "a.c"),
        _cand("https://github.com/x/alpha", "2" * 40, "b.c"),
        _cand("https://github.com/x/beta", "3" * 40),
        _cand("https://github.com/x/gamma", "4" * 40),
        _cand(
            "https://github.com/x/delta",
            "5" * 40,
            found_by="scan",
            evidence="README line",
            rule_id="UNKNOWN",
            analyzer_name="sonar",
        ),
        {"repo_url": "https://github.com/x/nokey"},
    ]
    stock.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = str(tmp_path / "fleet")
    m = search.handover(str(stock), WORKER, out, workers=2, target=3, launch=False, log=print)
    assert m["workers"] == 2 and m["candidates"] == 5 and m["total_target"] == 3
    pools = []
    for i in range(2):
        inst = json.loads(open(os.path.join(out, f"worker_{i}.instances.jsonl")).readline())
        pools.append(inst["candidates"])
        assert inst["sample_size"] in (1, 2)
        assert "candidates" in inst and all("repo_url" in c for c in inst["candidates"])
    # a candidate reaches the worker with everything the search recorded: the identity it
    # registers, and the rest as the search's notes (the brief says they are leads, not facts)
    delta = next(c for p in pools for c in p if c["repo_url"].endswith("delta"))
    assert set(delta) == {
        "repo_url",
        "commit_sha",
        "program_name",
        "found_by",
        "evidence",
        "rule_id",
        "analyzer_name",
    }, delta
    alpha_in = [i for i, p in enumerate(pools) if any(c["repo_url"].endswith("alpha") for c in p)]
    assert len(alpha_in) == 1, "one repository's candidates go to one worker"
    assert sum(len(p) for p in pools) == 5
    assert "--instance" in m["commands"][0] and "construct" in m["commands"][0]
    assert os.path.isfile(os.path.join(out, "handover.json"))
