"""A reply that is JSON except for a missing closing bracket at its end is repaired.

The brackets open at the end of the text are closed in the only order JSON allows. The repair is
attempted only after the strict parse fails and only when the text ends outside a string with
structures still open; a reply broken anywhere else is left for the conversational retry.
"""

from __future__ import annotations

import json

from benchcompiler.llm import repair_json

TRUNCATED = (
    '{"lessons": [{"kind": "early_sign", "scope": "family", "lesson": "No terminal command runs '
    'until bench/Dockerfile exists.", "evidence": [{"worker": "worker_0", "candidate": '
    '"iqss/dataverse@4272257d", "cycles": "1-1"}]}]'
)


def test_a_missing_final_brace_is_added():
    try:
        json.loads(TRUNCATED)
        raise AssertionError("this is supposed to be invalid JSON")
    except json.JSONDecodeError as e:
        print(f"\n[verbose] strict parse: {e}")
    got = json.loads(repair_json(TRUNCATED))
    print(f"[verbose] repaired -> {json.dumps(got)[:120]}")
    assert got["lessons"][0]["evidence"][0]["cycles"] == "1-1"
    assert len(got["lessons"]) == 1


def test_several_missing_closers_are_added_in_order():
    text = '{"a": [1, 2, {"b": [3'
    got = json.loads(repair_json(text))
    assert got == {"a": [1, 2, {"b": [3]}]}


def test_a_bracket_inside_a_string_does_not_count():
    text = '{"regex": "^[a-z", "n": 1'  # the [ inside the string is text, not structure
    got = json.loads(repair_json(text))
    assert got == {"regex": "^[a-z", "n": 1}


def test_a_complete_reply_is_left_alone():
    text = '{"a": [1, 2]}'
    assert repair_json(text) == text


def test_a_reply_broken_elsewhere_is_not_closed_into_nonsense():
    text = '{"a": 1 "b": 2}'  # a missing comma in the middle: not ours to guess
    try:
        json.loads(repair_json(text))
        raise AssertionError("must still fail: this is not a truncation")
    except json.JSONDecodeError:
        pass
