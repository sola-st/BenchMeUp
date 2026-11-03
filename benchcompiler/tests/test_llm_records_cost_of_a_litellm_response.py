"""A litellm reply is an object, not a dict: its `usage` is a Usage object and its cost sits in
`_hidden_params["response_cost"]`. `usage_of` reads both, so helper calls through the proxy are
logged with tokens and cost.
"""

import litellm

from benchcompiler.llm import usage_of


def _reply():
    r = litellm.ModelResponse(
        model="openrouter/vendor/model-a",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
        usage={"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
    )
    r._hidden_params["response_cost"] = 0.00042
    return r


def test_usage_and_cost_are_read_from_a_litellm_object():
    got = usage_of(_reply())
    print("usage_of(ModelResponse) ->", got)
    assert got.get("prompt_tokens") == 1200
    assert got.get("completion_tokens") == 300
    assert got.get("cost_usd") == 0.00042


def test_a_dict_reply_still_reads_as_before():
    got = usage_of({"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})
    print("usage_of(dict) ->", got)
    assert got == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
