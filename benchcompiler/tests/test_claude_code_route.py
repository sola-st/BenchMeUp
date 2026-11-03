"""`call_llm_json` transports a `claude-code/<id>` model through the CLI.

The CLI is picked for such a model, receives the prompt, its answer is read, and transient errors go
through the same retry loop as any provider. Host tier, no network: the CLI is a fake binary that
records what it was given.
"""

from __future__ import annotations

import json
import os
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchcompiler import llm  # noqa: E402

HELP = (
    "--print --output-format --model --effort --max-budget-usd --system-prompt "
    "--disallowed-tools --strict-mcp-config --mcp-config --safe-mode "
    "--disable-slash-commands --setting-sources --settings "
    "--exclude-dynamic-system-prompt-sections --bare --version --help"
)


def _fake(tmp_path, answer: str) -> str:
    path = tmp_path / "claude"
    record = tmp_path / "call.json"
    body = json.dumps(
        {
            "type": "result",
            "is_error": False,
            "result": answer,
            "total_cost_usd": 0.001,
            "usage": {"input_tokens": 240, "output_tokens": 12},
            "modelUsage": {"claude-haiku-4-5": {"costUSD": 0.001}},
        }
    )
    path.write_text(f"""#!/usr/bin/env python3
import json, sys
argv = sys.argv[1:]
if "--help" in argv:
    print({HELP!r}); sys.exit(0)
json.dump({{"argv": argv, "stdin": sys.stdin.read()}}, open({str(record)!r}, "w"))
sys.stdout.write({body!r})
""")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_a_claude_code_model_is_drawn_through_the_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BINARY", _fake(tmp_path, '{"verdict": "ok"}'))
    out = llm.call_llm_json("claude-code/claude-haiku-4-5", "answer in JSON", max_tokens=32000)
    call = json.load(open(tmp_path / "call.json", encoding="utf-8"))
    print("[verbose]", out, call["argv"][:8])
    assert out == {"verdict": "ok"}
    assert call["stdin"] == "answer in JSON"  # the prompt goes on stdin
    assert call["argv"][call["argv"].index("--model") + 1] == "claude-haiku-4-5"
    assert "--disallowed-tools" in call["argv"]  # the isolation travels with it


def test_a_transient_cli_failure_is_retried_by_the_same_patience_loop(monkeypatch):
    calls: list[int] = []

    def flaky(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("capacity: Overloaded")
        return {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]}

    monkeypatch.setattr(llm, "_completion", lambda litellm, model: flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "_route", lambda m: (m, {}))
    assert llm.call_llm_json("claude-code/claude-opus-5", "p") == {"ok": True}
    assert len(calls) == 2


def test_a_session_limit_is_retried_like_any_capacity_error(monkeypatch):
    """Session-limit errors from the CLI are provider capacity errors, so the retry loop waits them
    out up to its deadline instead of failing on the first refusal.
    """
    from agentcore.models import claude_code as cc

    calls: list[int] = []

    def invoke(messages, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise cc.ClaudeCodeError("You've hit your session limit · resets 8:20am (UTC)", "session_limit")
        return {
            "content": '{"ok": true}',
            "finish_reason": "stop",
            "cost_usd": 0.0,
            "usage": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "raw": {},
        }

    monkeypatch.setattr(cc, "invoke", invoke)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "_route", lambda m: (m, {}))
    out = llm.call_llm_json("claude-code/claude-opus-5", "p")
    print("[verbose]", out, len(calls))
    assert out == {"ok": True}
    assert len(calls) == 2


def test_the_route_asks_for_no_proxy_kwargs():
    """The CLI takes neither a base URL nor a key: it carries its own authentication."""
    model, kwargs = llm._route("claude-code/claude-opus-5")
    assert model == "claude-code/claude-opus-5" and kwargs == {}
