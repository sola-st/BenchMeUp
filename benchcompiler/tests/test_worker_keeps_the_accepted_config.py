"""`bench worker` keeps the accepted family config when the stage text cannot be drawn, and draws only
the stage text when given that config back.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKER = os.path.join(ROOT, "agents", "boca_worker")
SPEC = os.path.join(ROOT, "benchcompiler", "examples", "boca_flags.yaml")
CANONICAL = os.path.join(ROOT, "specs", "boca", "canonical.json")


def _accepted_config() -> dict:
    with open(os.path.join(WORKER, "contract.json"), encoding="utf-8") as f:
        c = json.load(f)
    return {k: v for k, v in c.items() if k not in ("_stage_sections", "_drawn")}


def test_the_accepted_config_survives_a_failed_stage_draw_and_is_reused(tmp_path, monkeypatch):
    from benchcompiler import contract_prompt, stage_prompt, worker

    config = _accepted_config()
    n = len(config["_stages"])
    out = str(tmp_path / "boca_worker")

    monkeypatch.setattr(
        contract_prompt, "draw_contract", lambda *a, **k: (copy.deepcopy(config), ["round 1: accepted"])
    )
    monkeypatch.setattr(
        stage_prompt,
        "draw_stage_prompts",
        lambda contract, raw, model, **k: ([None] * n, ["stage 1 round 1: refused -- rate limit"]),
    )
    with pytest.raises(RuntimeError):
        worker.emit_worker(SPEC, CANONICAL, out, model="fake/model", sample_size=2, log=print)
    kept_path = os.path.join(out, "contract.accepted.json")
    assert os.path.isfile(kept_path), "the accepted config must be kept on disk"
    kept = json.load(open(kept_path, encoding="utf-8"))
    print("[verbose] kept keys:", sorted(kept))
    assert kept["verbs"] == config["verbs"] and kept["_stages"] == config["_stages"]
    assert "_stage_sections" not in kept
    assert not os.path.isfile(os.path.join(out, "contract.json")), "no worker is written without stage text"

    # given back the kept config, only the stage text is drawn
    def no_draw(*a, **k):
        raise AssertionError("the config was drawn again")

    monkeypatch.setattr(contract_prompt, "draw_contract", no_draw)
    secs = [{"where_you_are": "w " * 60, "goal": "g " * 50, "traps": ["a", "b", "c"]}] * n

    def stage_draw(contract, raw, model, **k):
        contract["_stage_sections"] = secs
        return secs, ["stage 1 round 1: accepted"]

    monkeypatch.setattr(stage_prompt, "draw_stage_prompts", stage_draw)
    r = worker.emit_worker(SPEC, CANONICAL, out, model="fake/model", sample_size=2, contract_json=kept_path, log=print)
    written = json.load(open(os.path.join(out, "contract.json"), encoding="utf-8"))
    assert written["_stage_sections"] == secs and written["verbs"] == config["verbs"]
    assert r["key"] == config["_instance_fields"][: len(r["key"])]
