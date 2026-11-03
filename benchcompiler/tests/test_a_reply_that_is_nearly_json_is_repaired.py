"""A reply that is JSON except for a backslash or control character spelling is repaired.

Inside a JSON string a control character and a backslash that begins no valid escape each have one
legal spelling. The repair is attempted only after the strict parse fails and is accepted only if
the repaired text parses; nothing outside a string is touched.
"""

from __future__ import annotations

import json

from benchcompiler.llm import repair_json


def test_a_regex_backslash_inside_a_string_is_repaired():
    text = '{"require": "output matches ^\\d{4}-\\d{2}$", "ok": true}'
    print(f"\n[verbose] raw: {text}")
    try:
        json.loads(text)
        raise AssertionError("this is supposed to be invalid JSON")
    except json.JSONDecodeError as e:
        print(f"[verbose] strict parse: {e}")
    got = json.loads(repair_json(text))
    print(f"[verbose] repaired -> {got}")
    assert got["require"] == r"output matches ^\d{4}-\d{2}$"
    assert got["ok"] is True


def test_a_raw_newline_inside_a_string_is_repaired():
    text = '{"detail": "line one\nline two", "n": 3}'
    got = json.loads(repair_json(text))
    print(f"\n[verbose] repaired -> {got}")
    assert got["detail"] == "line one\nline two"
    assert got["n"] == 3


def test_valid_escapes_and_structure_are_left_alone():
    original = {"a": 'quote " and backslash \\ and unicode \u00e9', "b": ["x", 1, None], "c": {"d": True}}
    text = json.dumps(original)
    assert json.loads(repair_json(text)) == original


def test_structure_outside_strings_is_never_touched():
    """A newline between fields is legal JSON whitespace and must stay whitespace."""
    text = '{\n  "a": 1,\n  "b": 2\n}'
    assert json.loads(repair_json(text)) == {"a": 1, "b": 2}


def test_a_truly_broken_reply_is_returned_unchanged():
    text = '{"a": 1, oops'
    assert repair_json(text) == text, "nothing to repair means nothing changed"


def test_the_caller_parses_a_repaired_reply(monkeypatch):
    """A call whose only fault was a backslash returns a dict."""
    from benchcompiler import llm

    reply = '{"outcome": "ACCEPT", "pattern": "^\\d+$"}'

    class _Resp(dict):
        pass

    monkeypatch.setattr(llm, "_route", lambda m: (m, {}))
    monkeypatch.setattr(
        llm,
        "_litellm",
        lambda: type(
            "L", (), {"completion": staticmethod(lambda **kw: {"choices": [{"message": {"content": reply}}]})}
        )(),
    )
    out = llm.call_llm_json("stub", "prompt")
    print(f"\n[verbose] call_llm_json -> {out}")
    assert out == {"outcome": "ACCEPT", "pattern": r"^\d+$"}
