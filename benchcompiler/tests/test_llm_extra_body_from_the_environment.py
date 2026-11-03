"""LLM_EXTRA_BODY, a JSON object, is merged into the request body of every direct OpenRouter call,
beside the provider block when there is one, so a campaign can turn a model's reasoning off for
helper calls too.
"""

import json

from benchcompiler import llm

BODY = {"reasoning": {"enabled": False}, "provider": {"allow_fallbacks": True}}


def test_the_body_reaches_a_direct_openrouter_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    monkeypatch.setenv("LLM_EXTRA_BODY", json.dumps(BODY))
    model, kwargs = llm._route_litellm("openrouter/vendor/model-a")
    print("route ->", model, {k: v for k, v in kwargs.items() if k != "api_key"})
    assert kwargs.get("extra_body") == BODY


def test_a_provider_order_and_the_body_are_both_kept(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "DeepInfra")
    monkeypatch.setenv("LLM_EXTRA_BODY", json.dumps({"reasoning": {"enabled": False}}))
    _, kwargs = llm._route_litellm("openrouter/vendor/model-a")
    print("route kwargs ->", {k: v for k, v in kwargs.items() if k != "api_key"})
    assert kwargs["extra_body"]["reasoning"] == {"enabled": False}
    assert kwargs["extra_body"]["provider"]["order"] == ["DeepInfra"]


def test_unset_changes_nothing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("LLM_EXTRA_BODY", raising=False)
    _, kwargs = llm._route_litellm("openrouter/vendor/model-a")
    assert "extra_body" not in kwargs
