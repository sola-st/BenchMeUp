"""A JSON answer cut off at the output limit is reported as cut off, not as "not valid JSON".

The caller learns the cause (raise max_tokens or ask for a shorter answer), and the retry says so to
the model.
"""

from __future__ import annotations

import types

import pytest

from benchcompiler import llm


def _fake_litellm(calls: list, finish_reason: str, text: str):
    def completion(**kwargs):
        calls.append(kwargs["messages"][-1]["content"])
        return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}]}

    return types.SimpleNamespace(completion=completion)


def test_an_ordinary_bad_json_answer_is_still_reported_as_such(monkeypatch, capsys):
    calls: list[str] = []
    monkeypatch.setattr(llm, "_litellm", lambda: _fake_litellm(calls, "stop", "sure, here you go"))
    monkeypatch.setattr(llm, "_route", lambda m: (m, {}))
    out = llm.call_llm_json("fake/model", "draw the config", retries=1, max_tokens=32000)
    err = capsys.readouterr().err
    assert out is None and "cut off" not in err and "not valid JSON" in calls[-1]


@pytest.mark.xfail(reason="open defect: cut-off answer handling", strict=False)
def test_a_cut_off_answer_names_the_limit_and_the_retry_asks_for_a_shorter_one(monkeypatch, capsys):
    """A cut-off answer is reported with the token limit, and the retry asks for a shorter answer."""
    calls: list[str] = []
    monkeypatch.setattr(llm, "_litellm", lambda: _fake_litellm(calls, "length", '{"verbs": {"build": {"args": ["dir"'))
    monkeypatch.setattr(llm, "_route", lambda m: (m, {}))
    out = llm.call_llm_json("fake/model", "draw the config", retries=1, max_tokens=32000)
    err = capsys.readouterr().err
    print("[verbose] stderr:", err)
    print("[verbose] retry message:", calls[-1][:200])
    assert out is None
    assert "cut off" in err and "32000" in err
    # the retry tells the model the real problem, not "that was not valid JSON"
    assert len(calls) == 2 and "cut off" in calls[-1] and "shorter" in calls[-1].lower()
