"""Regression reproductions for durable LLM exchange and usage logging."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentcore.observability.stage_log import StageLog
from benchcompiler import llm


@pytest.mark.xfail(reason="open defect: LLM log directory reuse", strict=False)
def test_reusing_a_log_directory_after_process_restart_does_not_overwrite(monkeypatch, tmp_path):
    monkeypatch.setenv(llm.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(llm, "_LOG_N", {})
    messages = [{"role": "user", "content": "prompt"}]

    llm._log_to_dir("draw", "model", messages, "first answer", 0, True, None)
    monkeypatch.setattr(llm, "_LOG_N", {})  # model a fresh process using the same run directory
    llm._log_to_dir("draw", "model", messages, "second answer", 0, True, None)

    answers = sorted((tmp_path / "draw").glob("*.answer.txt"))
    assert len(answers) == 2
    assert {path.read_text() for path in answers} == {"first answer", "second answer"}


def _response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        "_claude_code": {
            "cost_usd": 0.25,
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "models_billed": ["claude-opus-5"],
        },
    }


@pytest.mark.xfail(reason="open defect: usage/cost of failed draws", strict=False)
@pytest.mark.parametrize(
    ("content", "repaired", "expected"),
    [
        ("", None, None),
        ("not JSON", "still not JSON", None),
        ("{broken}", '{"recovered": true}', {"recovered": True}),
    ],
    ids=["empty-answer", "invalid-json", "repaired-json"],
)
def test_unsuccessful_and_repaired_draws_keep_usage_and_cost(monkeypatch, tmp_path, content, repaired, expected):
    monkeypatch.setenv(llm.LOG_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(llm, "_LOG_N", {})
    monkeypatch.setattr(llm, "_litellm", lambda: object())
    monkeypatch.setattr(llm, "_route", lambda model: (model, {}))
    monkeypatch.setattr(llm, "_completion", lambda litellm, model: lambda **kwargs: _response(content))
    if repaired is not None:
        monkeypatch.setattr(llm, "repair_json", lambda text: repaired)

    result = llm.call_llm_json("claude-code/claude-opus-5", "prompt", retries=0, site="draw")

    assert result == expected
    records = list((tmp_path / "draw").glob("*.json"))
    assert len(records) == 1
    meta = json.loads(records[0].read_text())
    assert meta["prompt_tokens"] == 11
    assert meta["completion_tokens"] == 7
    assert meta["total_tokens"] == 18
    assert meta["cost_usd"] == pytest.approx(0.25)
    assert meta["cli_usage"] == {"input_tokens": 11, "output_tokens": 7}


@pytest.mark.xfail(reason="open defect: usage objects not logged", strict=False)
def test_usage_objects_are_logged_as_well_as_plain_dicts():
    """Sibling path: LiteLLM commonly exposes usage as a model object, not a dict."""
    usage = SimpleNamespace(prompt_tokens=13, completion_tokens=5, total_tokens=18)
    response = SimpleNamespace(usage=usage)

    assert llm.usage_of(response) == {
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
    }


@pytest.mark.xfail(reason="open defect: stage log usage/cost", strict=False)
def test_the_structured_stage_log_keeps_usage_and_cost(monkeypatch, tmp_path):
    """Sibling sink: JSONL is the run record, not merely the optional readable copy."""
    monkeypatch.delenv(llm.LOG_DIR_ENV, raising=False)
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cost_usd": 0.25,
    }

    with StageLog(str(tmp_path), "formalizer"):
        llm._record(
            "formalizer.formalize",
            "model",
            [{"role": "user", "content": "p"}],
            '{"ok": true}',
            0,
            parsed=True,
            usage=usage,
        )

    records = [json.loads(line) for line in (tmp_path / "formalizer.jsonl").read_text().splitlines() if line.strip()]
    call = next(record for record in records if record.get("kind") == "llm_call")
    assert call["usage"] == usage
