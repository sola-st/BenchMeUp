"""Shared LLM JSON-calling helper, litellm-based.

Same pattern as study/workflows/paper_miner.py: ask for JSON-only output,
parse it, retry once with the parse error fed back in if it fails.
"""

import json
import os
import random
import re
import sys
import time


def _litellm():
    """litellm, imported lazily: `import litellm` calls load_dotenv() as a side effect, so
    importing this module would otherwise put .env secrets into os.environ. The proxy vars
    still load before the first completion; imports are pure."""
    try:
        import litellm
    except ImportError:
        print("Missing dependency: pip install litellm", file=sys.stderr)
        sys.exit(1)
    # Some models reject params others accept (e.g. only
    # temperature=1 is supported, not 0). Drop unsupported params instead of
    # erroring, rather than hardcoding per-model exceptions here.
    litellm.drop_params = True
    return litellm


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    return text


# No timeout meant a stalled request blocked forever. This is the JUDGE's call path, so a
# single hung request silently stalls a whole construct run -- and it did: a study call sat for
# 19 minutes with zero CPU while an identical probe answered in 1.1 seconds. Overridable because
# a big sample legitimately takes longer than a one-line probe.
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT") or 180)


def _route(model: str) -> tuple[str, dict]:
    """`claude-code/<id>` is transported by the CLI, which takes neither a base URL nor a key:
    it carries its own authentication (a `claude` login, or ANTHROPIC_API_KEY)."""
    from agentcore.models import claude_code as cc

    if cc.is_claude_code(model):
        return model, {}
    return _route_litellm(model)


def _route_litellm(model: str) -> tuple[str, dict]:
    """(model litellm actually sends, kwargs). Every call runs on the OpenRouter key.
    `openai/openrouter/<id>` names an OpenRouter model and is transported directly to
    OpenRouter as `openrouter/<id>` -- callers keep their string (it sits inside scenario
    recording cache keys), only the wire changes. Explicit
    caller env (OPENAI_API_BASE) still wins, as everywhere in this module."""
    if model.startswith("openai/openrouter/") and not os.environ.get("OPENAI_API_BASE"):
        model = model[len("openai/") :]
    return model, _proxy_kwargs(model)


def _proxy_kwargs(model: str) -> dict:
    """Route openai-compatible models through the configured OpenAI-compatible endpoint.

    The endpoint is set in .env (LITELLM_BASE_URL/LITELLM_KEY) and reached as
    `openai/<model name at the endpoint>`. Doing it here means callers need not export
    OPENAI_API_BASE/OPENAI_API_KEY themselves and cannot silently run a different model on a
    direct key. Explicit env (OPENAI_API_BASE set by the caller) still wins.
    """
    if model.startswith("openai/") and os.environ.get("LITELLM_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
        return {
            "api_base": os.environ["LITELLM_BASE_URL"],
            "api_key": os.environ.get("LITELLM_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        }
    # The direct route: .env spells the key OPENROUTER_KEY while litellm reads
    # OPENROUTER_API_KEY, so it is passed explicitly.
    if model.startswith("openrouter/"):
        kwargs = {}
        if not os.environ.get("OPENROUTER_API_KEY") and os.environ.get("OPENROUTER_KEY"):
            kwargs["api_key"] = os.environ["OPENROUTER_KEY"]
        # Which provider serves a model is the account's decision, not a pin in this file:
        # the account's enabled-provider list already restricts routing. A campaign that
        # needs a fixed provider order (endpoints differ, e.g. in quantization) spells it
        # in OPENROUTER_PROVIDER_ORDER; `allow_fallbacks: false` then refuses rather than
        # drifts. Every response names its provider, so a run records who answered it.
        order = os.environ.get("OPENROUTER_PROVIDER_ORDER", "")
        if order and order.lower() not in ("", "any", "default"):
            kwargs["extra_body"] = {
                "provider": {"order": [p.strip() for p in order.split(",") if p.strip()], "allow_fallbacks": False}
            }
        # A campaign's request body for helper calls: the agents get theirs from
        # models.main.kwargs.extra_body; a helper such as the lessons reader has no config, so
        # the launch exports LLM_EXTRA_BODY (JSON), e.g. {"reasoning": {"enabled": false}}
        # when the provider would otherwise reason by default. A provider order set above is kept.
        extra = os.environ.get("LLM_EXTRA_BODY", "").strip()
        if extra:
            body = json.loads(extra)
            if "provider" in kwargs.get("extra_body", {}):
                body = {**body, "provider": kwargs["extra_body"]["provider"]}
            kwargs["extra_body"] = body
        return kwargs
    return {}


#: Where to write a readable copy of every exchange. Set per run; unset means nothing extra
#: is written. `bench formalize --log-dir` and `bench worker --log-dir` set it.
LOG_DIR_ENV = "LLM_LOG_DIR"
_LOG_N: dict = {}


def usage_of(resp) -> dict:
    """Tokens and cost from one response, whatever transported it.

    A chat-completions `usage` block, plus the Claude Code CLI's own record when the call went
    through it (`_claude_code`), where the input count is already summed over the uncached,
    cache-write and cache-read fields."""
    out: dict = {}
    try:
        u = (resp or {}).get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
        # litellm may hand back a response object whose usage is an object too.
        if u is not None and not isinstance(u, dict):
            u = {
                k: getattr(u, k)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if getattr(u, k, None) is not None
            }
        if isinstance(u, dict):
            out.update({k: u[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in u})
        hp = getattr(resp, "_hidden_params", None)
        if isinstance(hp, dict) and hp.get("response_cost") is not None:
            out["cost_usd"] = float(hp["response_cost"])
        cc = (resp or {}).get("_claude_code") if isinstance(resp, dict) else None
        if isinstance(cc, dict):
            out["cost_usd"] = cc.get("cost_usd")
            out["cli_usage"] = cc.get("usage")
            out["models_billed"] = cc.get("models_billed")
    except Exception:  # noqa: BLE001 - a log must never break a draw
        pass
    return out


def _log_to_dir(
    site: str,
    model: str,
    messages: list,
    text: str,
    attempt: int,
    parsed: bool,
    error: str | None,
    usage: dict | None = None,
) -> None:
    """One exchange as files under $LLM_LOG_DIR/<site>/: NNN.prompt.txt, NNN.answer.txt and
    NNN.json. The prompt is the WHOLE conversation, because a retry's feedback is part of
    what the model was asked."""
    root = os.environ.get(LOG_DIR_ENV)
    if not root:
        return
    try:
        import datetime

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", site or "llm.call")
        d = os.path.join(root, safe)
        os.makedirs(d, exist_ok=True)
        n = _LOG_N[d] = _LOG_N.get(d, 0) + 1
        stem = os.path.join(d, f"{n:03d}")
        with open(f"{stem}.prompt.txt", "w", encoding="utf-8") as f:
            for m in messages:
                f.write(f"===== {str(m.get('role', '?')).upper()} =====\n{m.get('content', '')}\n\n")
        with open(f"{stem}.answer.txt", "w", encoding="utf-8") as f:
            f.write(text or "")
        meta = {
            "site": site,
            "model": model,
            "attempt": attempt,
            "parsed": bool(parsed),
            "error": error,
            "n_messages": len(messages),
            "prompt_chars": sum(len(str(m.get("content") or "")) for m in messages),
            "answer_chars": len(text or ""),
            **{k: v for k, v in (usage or {}).items() if v is not None},
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
    except Exception:  # noqa: BLE001 - a log must never break a draw
        pass


def _record(
    site: str,
    model: str,
    messages: list,
    text: str,
    attempt: int,
    parsed: bool,
    error: str | None = None,
    usage: dict | None = None,
) -> None:
    """Every exchange goes to the open stage log, if there is one. The whole conversation
    so far is the prompt: a retry's feedback is part of what the model was asked.

    Lazy and guarded: this module must import with nothing but benchcompiler on the path
    (test_importing_llm_loads_no_secrets), and a stage log can only be open when agentcore
    is importable, since that is where StageLog lives."""
    _log_to_dir(site, model, messages, text, attempt, parsed, error, usage)
    try:
        from agentcore.observability import stage_log
        from benchcompiler import stage_sites  # noqa: F401  (registers the stages' sites)
    except ImportError:
        return
    log = stage_log.current()
    if log is None:
        return
    try:
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        log.llm_call(site, model=model, prompt=prompt, response=text, attempt=attempt, parsed=parsed, error=error)
    except Exception:  # noqa: BLE001 - the log must never break a stage
        pass


def repair_json(text: str) -> str:
    """The string-aware JSON repair, owned by `agentcore.tools.parsing` so the action parser
    and this caller resolve the same accidents the same way. Imported at
    call time: this module must import with nothing but benchcompiler on the path."""
    from agentcore.tools.parsing import repair_json as _shared

    return _shared(text)


#: How long one call keeps retrying a provider capacity error (429 rate limit, 529
#: overloaded, 5xx, a dropped connection) before giving up, so a transient provider failure
#: is not charged as a refused round.
CAPACITY_RETRY_SECONDS = 600
CAPACITY_RETRY_MAX_WAIT = 60
_CAPACITY_WORDS = (
    "rate-limited",
    "rate limited",
    "rate_limit",
    "ratelimit",
    "overloaded",
    "temporarily",
    "try again later",
    "capacity",
    "too many requests",
)
_CAPACITY_CODES = (408, 425, 429, 500, 502, 503, 504, 529)


def _is_capacity_error(exc: BaseException) -> bool:
    """A provider capacity or connectivity failure, whatever class litellm wrapped it in:
    the status code, the exception's name and the message's own words decide. A 400 with
    "Invalid request" stays permanent."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    code = getattr(exc, "status_code", None)
    if any(w in name for w in ("ratelimit", "serviceunavailable", "apiconnection", "timeout", "internalserver")):
        return True
    if code in _CAPACITY_CODES:
        return True
    if any(f'"code":{c}' in text or f'"code": {c}' in text for c in _CAPACITY_CODES):
        return True
    return any(w in text for w in _CAPACITY_WORDS)


def _completion(litellm, model: str):
    """The function that performs one completion for this model.

    litellm for everything, except a `claude-code/<id>` model, which is the Claude Code CLI
    driven as an endpoint (agentcore.models.claude_code). Same signature, same response shape,
    so every caller and every retry above is unchanged."""
    from agentcore.models import claude_code as cc

    if not cc.is_claude_code(model):
        return litellm.completion

    def completion(**kwargs):
        messages = kwargs.get("messages") or []
        try:
            result = cc.invoke(
                messages, model=kwargs.get("model") or model, timeout=kwargs.get("timeout"), effort=kwargs.get("effort")
            )
        except cc.ClaudeCodeError as e:
            # The transient class and the subscription's session limit are retried by the
            # patience loop below, which reads the message; the permanent class must not be,
            # so it is named as it is.
            capacity = e.kind in ("transient", "session_limit")
            raise RuntimeError(f"{'capacity: ' if capacity else ''}{e.reason}") from e
        return cc.as_completion(result, model)

    return completion


def _complete_with_patience(completion, deadline: float, label: str, **kwargs):
    """One completion, retried with exponential backoff and jitter while the provider reports
    a capacity error and the deadline has not passed; other errors propagate."""
    n = 0
    while True:
        try:
            return completion(**kwargs)
        except Exception as e:  # noqa: BLE001 - classified below
            if not _is_capacity_error(e):
                raise
            n += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(
                    f"  giving up on {label} after {CAPACITY_RETRY_SECONDS}s of capacity errors: "
                    f"{type(e).__name__}: {str(e)[:200]}",
                    file=sys.stderr,
                )
                raise
            wait = min(2**n, CAPACITY_RETRY_MAX_WAIT)
            wait = min(wait * (0.5 + random.random() / 2), remaining)
            print(
                f"  capacity error from {label} ({type(e).__name__}: {str(e)[:120]}); retrying in "
                f"{wait:.0f}s ({remaining:.0f}s of patience left)",
                file=sys.stderr,
            )
            time.sleep(wait)


def call_llm_json(
    model: str,
    prompt: str,
    retries: int = 2,
    temperature: float = 0,
    timeout: float | None = None,
    site: str = "llm.call",
    max_tokens: int | None = None,
) -> dict | None:
    messages = [{"role": "user", "content": prompt}]
    text = ""
    fin = None
    litellm = _litellm()
    deadline = time.monotonic() + CAPACITY_RETRY_SECONDS
    for attempt in range(retries + 1):
        try:
            send_model, route_kwargs = _route(model)
            # A long structured answer (a family contract with stages) is cut off by the
            # provider's default output limit and then fails to parse; a caller that knows
            # its answer is long says so.
            extra = {"max_tokens": int(max_tokens)} if max_tokens else {}
            resp = _complete_with_patience(
                _completion(litellm, model),
                deadline,
                model,
                model=send_model,
                messages=messages,
                temperature=temperature,
                timeout=timeout or DEFAULT_TIMEOUT,
                **extra,
                **route_kwargs,
            )
            text = resp["choices"][0]["message"]["content"]
            fin = resp["choices"][0].get("finish_reason")
            if not text:
                # A reasoning model can spend the whole output budget thinking and answer
                # nothing (finish_reason "length").
                # That is a budget error to report, not an AttributeError on None.
                _record(
                    site,
                    model,
                    messages,
                    "",
                    attempt,
                    parsed=False,
                    error=f"empty answer (finish_reason={fin!r}); raise max_tokens",
                )
                print(
                    f"  LLM call returned no content (finish_reason={fin!r}); the output budget "
                    f"of {max_tokens or 'the provider default'} tokens was spent before an answer",
                    file=sys.stderr,
                )
                return None
            text = strip_code_fence(text)
            parsed = json.loads(text)
            _record(site, model, messages, text, attempt, parsed=True, usage=usage_of(resp))
            return parsed
        except json.JSONDecodeError as e:
            # Try the exact repair first: a backslash or a control character inside a string
            # is a spelling accident, not an answer, and asking again costs a whole call.
            try:
                parsed = json.loads(repair_json(text))
            except Exception:  # noqa: BLE001 - the strict error below is the real one
                parsed = None
            if parsed is not None:
                _record(site, model, messages, text, attempt, parsed=True, error=f"repaired: {e}")
                return parsed
            if fin == "length":
                # The object was cut off at the output limit; say so, rather than log a
                # generic parse failure and ask for the same answer again.
                limit = max_tokens or "the provider default"
                _record(site, model, messages, text, attempt, parsed=False, error=f"cut off at max_tokens={limit}: {e}")
                print(
                    f"  LLM answer was cut off at the output limit (max_tokens={limit}) and is "
                    f"not complete JSON; raise max_tokens or ask for a shorter answer",
                    file=sys.stderr,
                )
                if attempt < retries:
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your answer was cut off at the output limit of {limit} tokens before the "
                                f"JSON object was complete. Respond with a SHORTER JSON object (fewer words in "
                                f"every sentence, no repetition), and nothing else."
                            ),
                        }
                    )
                    continue
                return None
            _record(site, model, messages, text, attempt, parsed=False, error=f"JSON: {e}")
            if attempt < retries:
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": f"That was not valid JSON ({e}). Respond with ONLY the JSON object, no other text.",
                    }
                )
                continue
            print(f"  JSON parse failed after retries: {e}", file=sys.stderr)
            return None
        except Exception as e:
            _record(site, model, messages, text, attempt, parsed=False, error=f"{type(e).__name__}: {e}")
            # A timeout is worth one retry -- proxies stall transiently -- but not an endless
            # one, and the reason must reach the caller rather than looking like a refusal.
            if attempt < retries and "timeout" in f"{type(e).__name__} {e}".lower():
                print(f"  LLM call timed out, retrying ({attempt + 1}/{retries})", file=sys.stderr)
                continue
            print(f"  LLM call failed: {type(e).__name__}: {e}", file=sys.stderr)
            return None
    return None


#: How much of a refused attempt to show back. A designed lifecycle is a few thousand
#: characters; the cap exists only so a pathological reply cannot crowd out the instructions
#: it is being judged against. The largest real design is ~14k characters, so the cap admits
#: every real design whole.
PREVIOUS_ATTEMPT_CHARS = 20000


def retry_block(feedback: str, previous=None) -> str:
    """The retry preamble, in ONE place for all three propose-and-grade loops.

    Showing only the reasons for a refusal tells the model what is wrong with something it
    cannot see, so each round would be an independent attempt conditioned on a problem list.
    That is survivable for existential checks ("obligation 'run_cmd' appears in no state's
    requires" names the thing to ADD) but not for checks about what a design already CONTAINS:
    no model can drop a field from a state it is not looking at.

    Attempt first, reasons second, both in one block, so the reasons read as being about the
    JSON above them rather than as a fresh brief. Rendered as JSON, not a repr, because JSON is
    the format the reply is asked for -- the same lesson `test_context_material_is_readable_not_a_repr` pins at the other end of the pipeline.
    """
    if not feedback:
        return ""
    out = ["\nYOUR PREVIOUS ATTEMPT IS BELOW, FOLLOWED BY WHY IT WAS REFUSED."]
    if previous:
        rendered = json.dumps(previous, indent=1, default=str)
        if len(rendered) > PREVIOUS_ATTEMPT_CHARS:
            rendered = rendered[:PREVIOUS_ATTEMPT_CHARS] + "\n  ... (truncated)"
        out += ["This is what you emitted:", rendered]
    out += ["Fix ALL of these; everything not named below was accepted and should be kept:", feedback, ""]
    return "\n".join(out)
