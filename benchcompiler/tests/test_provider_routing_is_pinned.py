"""Provider routing is the account's decision; the code pins a provider only when told to.

No provider name appears in the routing code. A provider pin for reproducibility can be requested
through the environment.
"""

from __future__ import annotations

from benchcompiler.llm import _proxy_kwargs


def test_an_openrouter_call_names_no_provider_by_default(monkeypatch):
    """No provider name in code; the account's enabled list decides."""
    monkeypatch.setenv("OPENROUTER_KEY", "k")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    kw = _proxy_kwargs("openrouter/vendor/model-a")
    print(f"\n[verbose] routing kwargs: {kw}")
    assert "extra_body" not in kw, (
        "a provider named in code outlives the model it was chosen for -- google-vertex "
        "refused every vendor call for as long as it stood"
    )


def test_no_provider_name_is_hardcoded_anywhere_in_the_router():
    """No provider literal in the routing code under any name: a provider is an operational choice
    that belongs to whoever runs the campaign.
    """
    import inspect

    from benchcompiler import llm

    src = inspect.getsource(llm._proxy_kwargs)
    live = "\n".join(l.split("#")[0] for l in src.splitlines())
    for name in ("google-vertex", "google-ai-studio", "deepinfra", "streamlake", "novita"):
        assert name not in live.lower(), f"{name!r} is hardcoded in the router"


def test_the_pin_is_available_when_someone_spells_it(monkeypatch):
    """A campaign that needs one provider every time can request it."""
    monkeypatch.setenv("OPENROUTER_KEY", "k")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "deepinfra,streamlake")
    kw = _proxy_kwargs("openrouter/vendor/model-a")
    print(f"\n[verbose] {kw['extra_body']}")
    assert kw["extra_body"]["provider"]["order"] == ["deepinfra", "streamlake"]
    assert kw["extra_body"]["provider"]["allow_fallbacks"] is False, (
        "a spelled pin still refuses rather than silently serving from somewhere else"
    )


def test_any_is_still_accepted_as_an_explicit_opt_out(monkeypatch):
    """Spelling the pin explicitly keeps working."""
    monkeypatch.setenv("OPENROUTER_KEY", "k")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "any")
    kw = _proxy_kwargs("openrouter/vendor/model-a")
    print(f"\n[verbose] {kw}")
    assert "extra_body" not in kw


def test_non_openrouter_models_are_untouched(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://proxy")
    kw = _proxy_kwargs("openai/model-a")
    print(f"\n[verbose] {kw}")
    assert "extra_body" not in kw and kw["api_base"] == "http://proxy"


def test_openrouter_key_serves_every_call_even_with_the_proxy_configured(monkeypatch):
    """All calls go straight to OpenRouter, even when litellm proxy variables are present in the
    environment; the `openai/openrouter/<id>` spelling stays valid (it appears in recording
    cache keys).
    """
    from benchcompiler.llm import _route

    monkeypatch.setenv("LITELLM_BASE_URL", "http://proxy")
    monkeypatch.setenv("LITELLM_KEY", "proxy-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_KEY", "or-key")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "any")

    send_model, kw = _route("openai/openrouter/vendor/model-a")
    print(f"\n[verbose] {send_model} {kw}")
    assert send_model == "openrouter/vendor/model-a"
    assert kw.get("api_key") == "or-key"
    assert "api_base" not in kw, "the proxy must not serve this call"

    # a caller that explicitly points elsewhere still wins
    monkeypatch.setenv("OPENAI_API_BASE", "http://mine")
    send_model, kw = _route("openai/openrouter/vendor/model-a")
    assert send_model.startswith("openai/"), "explicit caller env is respected"


def test_the_github_credential_alias_is_mapped_at_the_cli_door():
    """The CLI resolves the GITHUB_TOKEN alias: GH_FINEGRAINED_TOKEN from .env is mapped to
    GITHUB_TOKEN.
    """
    from benchcompiler.bench_cli import map_credential_aliases

    env = {"GH_FINEGRAINED_TOKEN": "tok-abc"}
    note = map_credential_aliases(env)
    assert env.get("GITHUB_TOKEN") == "tok-abc" and "taken from" in note
    env2 = {"GITHUB_TOKEN": "explicit", "GH_FINEGRAINED_TOKEN": "other"}
    assert map_credential_aliases(env2) == "" and env2["GITHUB_TOKEN"] == "explicit"
    assert "WARNING" in map_credential_aliases({})
