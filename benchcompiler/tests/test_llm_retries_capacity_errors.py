"""A provider capacity error (429 rate limit, 529 overloaded) in the draw path is retried with backoff
and does not count as a refused round.
"""

from __future__ import annotations

import types

from benchcompiler import llm


class _RateLimitError(Exception):
    status_code = 429


def _fake_litellm(calls: list, failures: int, text: str, exc):
    def completion(**kwargs):
        calls.append(kwargs["messages"][-1]["content"])
        if len(calls) <= failures:
            raise exc
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}

    return types.SimpleNamespace(completion=completion, RateLimitError=_RateLimitError)


def test_a_rate_limit_is_retried_with_backoff_and_the_answer_arrives(monkeypatch, capsys):
    calls: list[str] = []
    waits: list[float] = []
    exc = _RateLimitError(
        'OpenrouterException - {"error":{"message":"Provider returned error","code":429,'
        '"metadata":{"raw":"vendor/model-a is temporarily rate-limited upstream"}}}'
    )
    monkeypatch.setattr(llm, "_litellm", lambda: _fake_litellm(calls, 2, '{"ok": true}', exc))
    monkeypatch.setattr(llm, "_route", lambda m: (m, {}))
    monkeypatch.setattr(llm.time, "sleep", lambda s: waits.append(s))
    out = llm.call_llm_json("fake/model", "draw", retries=0, max_tokens=100)
    err = capsys.readouterr().err
    print("[verbose] calls:", len(calls), "waits:", waits, "stderr:", err)
    assert out == {"ok": True}
    assert len(calls) == 3 and len(waits) == 2 and all(0 < w <= 60 for w in waits)
    assert "capacity" in err.lower() or "rate" in err.lower()


def test_a_permanent_error_is_not_retried(monkeypatch, capsys):
    calls: list[str] = []
    waits: list[float] = []
    exc = ValueError('{"error":{"message":"Invalid request","code":400}}')
    monkeypatch.setattr(llm, "_litellm", lambda: _fake_litellm(calls, 5, '{"ok": true}', exc))
    monkeypatch.setattr(llm, "_route", lambda m: (m, {}))
    monkeypatch.setattr(llm.time, "sleep", lambda s: waits.append(s))
    out = llm.call_llm_json("fake/model", "draw", retries=0, max_tokens=100)
    assert out is None and len(calls) == 1 and waits == []


def test_patience_runs_out_and_the_failure_is_named(monkeypatch, capsys):
    calls: list[str] = []
    exc = _RateLimitError("rate-limited upstream")
    monkeypatch.setattr(llm, "_litellm", lambda: _fake_litellm(calls, 999, '{"ok": true}', exc))
    monkeypatch.setattr(llm, "_route", lambda m: (m, {}))
    clock = {"t": 0.0}
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(llm.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(llm, "CAPACITY_RETRY_SECONDS", 120)
    out = llm.call_llm_json("fake/model", "draw", retries=0, max_tokens=100)
    err = capsys.readouterr().err
    print("[verbose] calls:", len(calls), "stderr:", err)
    assert out is None and len(calls) >= 3
    assert "giving up" in err.lower() and "120" in err
