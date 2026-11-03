"""Owned thin wrapper over litellm — the only model leaf we depend on.

Exposes exactly three concerns: query (chat completion), cost (USD accounting),
cache-control (provider prompt-cache breakpoints). No mini-swe-agent, no MSWEA_*
conventions, no startup banner.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from agentcore.protocols import ModelResponse

log = logging.getLogger("agentcore.model")

# How long to keep retrying a TRANSIENT provider failure before giving up on the call.
# Rationale: a short retry policy (a few waits of a few seconds) combined with loop.py's
# 5-consecutive-error abort would give a run a survival window of about a minute. Provider
# capacity blips (a shared-pool overload, not an account balance or per-key quota problem)
# routinely last minutes, so the cost of being patient (a few idle minutes) is far below the
# cost of being impatient (discarding hours of work and a mostly-unspent budget).
TRANSIENT_RETRY_SECONDS = float(os.getenv("AGENT_LLM_TRANSIENT_RETRY_SECONDS", "600"))
TRANSIENT_RETRY_MAX_WAIT = float(os.getenv("AGENT_LLM_TRANSIENT_MAX_WAIT", "60"))
# How long ONE request may take before it is given up on. Without it a call inherits "wait
# forever": a request on a socket the server has closed raises nothing, so the
# transient-retry layer above never engages and the wallclock meter, which advances per
# cycle, stays frozen. A timeout turns that silence into an error the retry layer can act
# on. The agent.yaml `timeout:` under `env:` is the DOCKER one and does not reach a model
# call; a model's own kwargs.timeout still wins over this default.
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("AGENT_LLM_REQUEST_TIMEOUT", "600"))
# Permanent failures (bad key, malformed request, context overflow) must NOT burn this budget:
# no amount of waiting fixes them, and retrying hides the real error behind a long stall.
PERMANENT_RETRY_ATTEMPTS = 1


def _litellm_exc_groups():
    """(transient, permanent) exception tuples, resolved lazily so importing this module
    never costs a litellm import."""
    import litellm

    transient = tuple(
        getattr(litellm, n)
        for n in ("RateLimitError", "ServiceUnavailableError", "InternalServerError", "APIConnectionError", "Timeout")
        if hasattr(litellm, n)
    )
    permanent = tuple(
        getattr(litellm, n)
        for n in (
            "AuthenticationError",
            "BadRequestError",
            "ContextWindowExceededError",
            "NotFoundError",
            "PermissionDeniedError",
        )
        if hasattr(litellm, n)
    )
    return transient, permanent


_CAPACITY_WORDS = (
    "rate-limited",
    "rate limited",
    "rate_limit",
    "overloaded",
    "engine_overloaded",
    "temporarily",
    "try again later",
    "capacity",
    "too many requests",
)
_CAPACITY_CODES = (408, 425, 429, 500, 502, 503, 504, 529)


def is_capacity_error(exc: BaseException) -> bool:
    """A provider capacity or connectivity failure hiding in a generic litellm.APIError.

    OpenRouter wraps 429 and 529 as APIError: the failure arrives as `APIError:
    OpenrouterException - {"error":{"code":529,...,"previous_errors":[{"code":429,...}]}}`,
    which neither the model layer's retry nor the loop's streak limit would count as
    transient without this check. The status code and the message's own
    words decide; a 400 with "Invalid request" stays permanent."""
    try:
        import litellm
    except Exception:
        return False
    if not isinstance(exc, getattr(litellm, "APIError", ())):
        return False
    code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if code in _CAPACITY_CODES:
        return True
    if any(f'"code":{c}' in text or f'"code": {c}' in text for c in _CAPACITY_CODES):
        return True
    return any(w in text for w in _CAPACITY_WORDS)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Honour a provider's own Retry-After hint when it sends one -- it knows better than our
    backoff curve. Tolerates the several shapes litellm/httpx surface headers in."""
    for attr in ("response", "llm_provider_response", "raw_response"):
        resp = getattr(exc, attr, None)
        headers = getattr(resp, "headers", None)
        if not headers:
            continue
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
            raw = None
            try:
                raw = headers.get(key)
            except Exception:
                pass
            if raw is None:
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if 0 < val <= 3600:
                return val
    return None


def provider_of(resp: Any) -> tuple:
    """(provider, generation id) of a completion, or (None, None) when it says nothing.

    WHICH ENDPOINT SERVED THE CALL IS PART OF WHAT RAN. One model on OpenRouter may be served
    by many endpoints whose quantization differs, so identical calls can be served by
    different ones. The response carries `provider` and an id; keeping them means the record
    can answer which endpoint produced a given reply.
    """
    if resp is None:
        return (None, None)
    get = resp.get if isinstance(resp, dict) else lambda k, d=None: getattr(resp, k, d)
    prov = get("provider", None)
    ident = get("id", None)
    return (str(prov) if prov else None, str(ident) if ident else None)


def _safe_cost(resp: Any) -> float:
    try:
        hp = getattr(resp, "_hidden_params", {}) or {}
        if hp.get("response_cost") is not None:
            return float(hp["response_cost"])
    except Exception:
        pass
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:
        return 0.0


class LitellmModel:
    """A single configured model. Matches the agentcore.protocols.Model protocol."""

    def __init__(
        self,
        name: str,
        *,
        temperature: float | None = None,
        cache_control: bool = False,
        kwargs: dict | None = None,
        on_retry=None,
    ):
        self.name = name
        self.temperature = temperature
        self.cache_control = cache_control
        self.extra_kwargs = dict(kwargs or {})
        self._cost = 0.0
        self.n_calls = 0
        # Optional callback(attempt, wait, remaining, error) so a long transient stall is
        # visible in the trajectory rather than looking like a hung process.
        self.on_retry = on_retry
        self.n_transient_retries = 0

    # ── cache control ────────────────────────────────────────────────────────
    def _apply_cache_control(self, messages: list[dict]) -> list[dict]:
        if not self.cache_control or not messages:
            return messages
        out = [dict(m) for m in messages]
        # mark the last system message and the last message as cache breakpoints
        for idx in {next((i for i, m in enumerate(out) if m.get("role") == "system"), None), len(out) - 1}:
            if idx is None:
                continue
            content = out[idx].get("content")
            if isinstance(content, str):
                out[idx]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return out

    # ── provider routing ─────────────────────────────────────────────────────
    def _proxy_kwargs(self) -> dict:
        """Route an `openai/`-prefixed model through LITELLM_BASE_URL, or {} for a direct one.

        Mirrors `benchcompiler.llm._proxy_kwargs` deliberately: the two packages must agree, or
        a run and the planner that set it up would silently use different providers. Without
        proxy routing, naming an `openai/...` model would send the call to OpenAI's real API
        with whatever OPENAI_API_KEY is exported.

        Returned as explicit call kwargs rather than left to litellm's own OPENAI_API_BASE
        environment handling, because the launch scripts already export OPENAI_API_KEY and
        OPENAI_BASE_URL for the agent's IN-CONTAINER cheap-LLM prefiltering. Explicit kwargs beat the environment, so the two uses cannot collide.

        Half-configured is treated as not configured: a base URL with no key would reach the
        proxy unauthenticated and 401 after the retry layer had spent its patience on it.
        """
        name = str(self.name or "")
        if name.startswith("openai/"):
            base = os.environ.get("LITELLM_BASE_URL", "").strip()
            key = (os.environ.get("LITELLM_KEY") or os.environ.get("LITELLM_API_KEY") or "").strip()
            return {"api_base": base, "api_key": key} if base and key else {}
        # The direct route: .env spells the key OPENROUTER_KEY while litellm reads
        # OPENROUTER_API_KEY. Mapping it here mirrors benchcompiler.llm exactly; with only
        # OPENROUTER_KEY exported the call would otherwise 401, which reads as an outage.
        if (
            name.startswith("openrouter/")
            and not os.environ.get("OPENROUTER_API_KEY")
            and os.environ.get("OPENROUTER_KEY")
        ):
            return {"api_key": os.environ["OPENROUTER_KEY"]}
        return {}

    def _apply_openrouter_fallbacks(self, call_kwargs: dict) -> None:
        """Let OpenRouter route around an overloaded provider instead of failing the call.

        OpenRouter fronts many providers per model. Asking for fallbacks explicitly means a
        capacity blip at one provider gets re-routed rather than surfaced as a hard error.

        Only touches `openrouter/*` models, and never overrides a `provider` block the config
        already set -- an explicit pin (e.g. for reproducibility, or to avoid a provider that
        quantizes differently) must win over this default.
        """
        if not str(call_kwargs.get("model", "")).startswith("openrouter/"):
            return
        body = dict(call_kwargs.get("extra_body") or {})
        if "provider" in body:
            return  # caller pinned providers deliberately; respect it
        provider: dict = {"allow_fallbacks": True}
        # allow_fallbacks alone may not route around a provider answering 429 for a long time.
        # OPENROUTER_IGNORE_PROVIDERS (comma-separated provider names) is the launcher's lever to exclude degraded providers without editing any agent yaml.
        ignore = [p.strip() for p in os.environ.get("OPENROUTER_IGNORE_PROVIDERS", "").split(",") if p.strip()]
        if ignore:
            provider["ignore"] = ignore
        body["provider"] = provider
        call_kwargs["extra_body"] = body

    # ── query ──────────────────────────────────────────────────────────────
    def _completion(self, messages: list[dict], **kwargs):
        """Call the provider, retrying TRANSIENT failures patiently and PERMANENT ones not at all.

        Deliberately hand-rolled rather than a `@retry` decorator: the two classes need
        different stopping rules (a wall-clock budget vs. fail-fast), the provider's own
        Retry-After must be able to override the backoff curve, and each wait needs to be
        observable -- a silent multi-minute stall is indistinguishable from a hang. `on_retry`
        lets the loop surface the wait as a real event instead.
        """
        import litellm

        transient, permanent = _litellm_exc_groups()
        deadline = time.monotonic() + TRANSIENT_RETRY_SECONDS
        attempt = 0
        while True:
            try:
                return litellm.completion(messages=messages, **kwargs)
            except permanent:
                raise  # no amount of waiting fixes these
            except Exception as e:
                if not (isinstance(e, transient) or is_capacity_error(e)):
                    raise
                attempt += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning(
                        "giving up on %s after %.0fs of transient failures: %s", self.name, TRANSIENT_RETRY_SECONDS, e
                    )
                    raise
                # exponential backoff, full jitter, clamped to what's left of the budget
                wait = _retry_after_seconds(e) or min(2**attempt, TRANSIENT_RETRY_MAX_WAIT)
                wait = min(wait * (0.5 + random.random() / 2), remaining)
                log.warning(
                    "transient %s from %s (attempt %d); retrying in %.1fs (%.0fs of patience left)",
                    type(e).__name__,
                    self.name,
                    attempt,
                    wait,
                    remaining,
                )
                self.n_transient_retries += 1
                if self.on_retry:
                    try:
                        self.on_retry(attempt=attempt, wait=wait, remaining=remaining, error=e)
                    except Exception:
                        pass  # observability must never break the call path
                time.sleep(wait)

    def query(self, messages: list[dict], **kwargs) -> ModelResponse:
        import litellm  # local import keeps package import fast

        litellm.drop_params = True
        call_kwargs: dict[str, Any] = {"model": kwargs.pop("model", None) or self.name}
        call_kwargs.update(self._proxy_kwargs())
        self._apply_openrouter_fallbacks(call_kwargs)
        if self.temperature is not None and "temperature" not in kwargs:
            call_kwargs["temperature"] = self.temperature
        call_kwargs.update(self.extra_kwargs)
        call_kwargs.update(kwargs)
        # Last, so a config's kwargs.timeout and a per-call one both win: the default only
        # fills the silence that let one dead socket hang a run for 33 minutes.
        call_kwargs.setdefault("timeout", DEFAULT_REQUEST_TIMEOUT)
        msgs = self._apply_cache_control(messages)

        resp = self._completion(msgs, **call_kwargs)
        self.n_calls += 1
        delta = _safe_cost(resp)
        self._cost += delta
        content, finish, source = "", "", "empty"
        try:
            msg = resp.choices[0].message
            finish = resp.choices[0].finish_reason or ""
            content = msg.content or ""
            source = "content" if content.strip() else "empty"
            if not content.strip():
                # THE ANSWER IN THE REASONING CHANNEL. Some models answer with
                # finish_reason "stop", content null, completion_tokens 0 -- and the whole
                # reply in `reasoning_content`. Read as empty, the loop would re-ask
                # without progress. The reasoning IS the reply; the parser
                # gets it, the record says where it came from.
                reasoning = (
                    getattr(msg, "reasoning_content", None)
                    or (getattr(msg, "provider_specific_fields", None) or {}).get("reasoning")
                    or ""
                )
                if str(reasoning).strip():
                    content, source = str(reasoning), "reasoning"
        except Exception:
            content, finish, source = "", "", "empty"
        prov, gen = provider_of(resp)
        out = ModelResponse(
            content=content, raw=_to_dict(resp), cost=delta, finish_reason=finish, provider=prov, generation_id=gen
        )
        try:
            out.source = source
        except Exception:  # noqa: BLE001 - a frozen ModelResponse still carries content
            pass
        return out

    def cost(self) -> float:
        return self._cost

    def get_template_vars(self) -> dict[str, Any]:
        return {"model_name": self.name, "model_cost": round(self._cost, 4), "n_model_calls": self.n_calls}


def _to_dict(resp: Any) -> dict:
    try:
        return resp.model_dump()  # litellm responses are pydantic
    except Exception:
        try:
            return dict(resp)
        except Exception:
            return {}


# Optional convenience: load API keys from a .env if python-dotenv is around.
def _maybe_load_dotenv():
    path = os.getenv("AGENT_ENV_FILE", ".env")
    if os.path.exists(path):
        try:
            import dotenv

            dotenv.load_dotenv(path)
        except Exception:
            pass


_maybe_load_dotenv()
