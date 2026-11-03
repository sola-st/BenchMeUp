"""The Claude Code CLI as a plain model endpoint: one prompt in, one text answer out.

WHY. The CLI carries its own authentication (an existing `claude` login through the system
keychain, or ANTHROPIC_API_KEY), so a machine that can run Claude Code can run this pipeline
with no provider key of its own. It is an alternative route to a provider-key endpoint.

WHAT IT COSTS TO GET RIGHT. Claude Code is a coding assistant, not an endpoint: by default it
wraps the prompt in its own system prompt, its tool schemas, the CLAUDE.md it finds by walking
up from the working directory, skills, plugins, hooks, MCP servers and per-machine blocks. A
six-word prompt was billed about 31,000 input tokens at the defaults and about 260 under the
isolation below -- and the difference is not only cost: whatever the harness injects is in the
model's context, so a measurement taken at the defaults is a measurement of something else.

WHAT THIS MODULE DOES NOT DO. There is no temperature, no top_p and no seed: `--effort`
(low|medium|high|xhigh|max) is the only sampling control, and those parameters are dropped
with a note rather than silently honoured. There is no max_tokens either.

Measured against 2.1.263 and 2.1.266 on Linux. Flag names and the tool inventory move between
versions, so nothing here is hardcoded blindly: the flags are probed from `--help` and the
deny-list can be re-derived with `python -m benchcompiler.claude_code_probe`.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

#: Every tool the CLI can put in the model's context. A bare name in a deny rule removes the
#: tool ENTIRELY, schema included, not just from the permission set. The list must be
#: complete: denying the obvious ones still left about 7,000 tokens of schemas.
#: Re-derive it on a new version (the probe asks the CLI to name its own tools).
TOOL_NAMES = (
    "Bash,BashOutput,Edit,Glob,Grep,KillShell,NotebookEdit,Read,Write,"
    "WebFetch,WebSearch,"
    "Agent,ListAgents,SendMessage,Task,TaskOutput,TaskStop,Workflow,ScheduleWakeup,Monitor,"
    "TaskCreate,TaskGet,TaskList,TaskUpdate,"
    "AskUserQuestion,EnterPlanMode,ExitPlanMode,ReportFindings,TodoWrite,PushNotification,Artifact,"
    "Skill,SlashCommand,ToolSearch,"
    "CronCreate,CronDelete,CronList,EnterWorktree,ExitWorktree,RemoteTrigger"
)

#: The environment the CLI runs under. CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC is not
#: cosmetic: at the defaults a SECOND model appears in the invocation's own usage, about 900
#: tokens a call, billed to the caller. CLAUDE_CODE_ENABLE_TASKS decides whether the Task*
#: schemas exist at all, so a host that sets it measured 3,288 tokens where one without it
#: measured 279 -- same prompt, same flags. Both are set here so the result does not depend on
#: the machine.
ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ENABLE_TASKS": "0",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "CI": "1",
}

EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = os.environ.get("CLAUDE_CODE_EFFORT") or "medium"
#: `high` effort on a large prompt runs for minutes; a timeout is a transient failure.
DEFAULT_TIMEOUT = float(os.environ.get("CLAUDE_CODE_TIMEOUT") or 900)
#: Each call is a full native process of a few hundred megabytes.
MAX_CONCURRENCY = int(os.environ.get("CLAUDE_CODE_CONCURRENCY") or 4)

_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENCY)
_LOCK = threading.Lock()
_FLAGS: dict[str, frozenset] = {}
_WORKDIR: dict[str, str] = {}

PREFIX = "claude-code/"


def is_claude_code(model: str) -> bool:
    """Whether a model string names the CLI (`claude-code/<model-id>`)."""
    return str(model or "").startswith(PREFIX)


def model_id(model: str) -> str:
    """The id to pass to --model: full model id or an alias."""
    name = str(model or "")
    return name[len(PREFIX) :] if name.startswith(PREFIX) else name


# ── the binary ──────────────────────────────────────────────────────────────────────────


def find_binary() -> str | None:
    """The CLI: CLAUDE_CODE_BINARY, then PATH, then the VS Code extension's copy.

    The extension's path carries a version number and moves when the extension updates, so
    the newest is taken and the version is recorded with the result. Install properly for
    anything reproducible."""
    explicit = os.environ.get("CLAUDE_CODE_BINARY")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    found = shutil.which("claude")
    if found:
        return found
    pattern = os.path.expanduser("~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude")
    paths = sorted(glob.glob(pattern), key=_version_key)
    return paths[-1] if paths else None


def _version_key(path: str) -> tuple:
    m = re.search(r"claude-code-(\d+)\.(\d+)\.(\d+)", path)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def version(binary: str) -> str:
    try:
        p = subprocess.run([binary, "--version"], capture_output=True, text=True, errors="replace", timeout=60)
        return (p.stdout or p.stderr).strip().splitlines()[0] if (p.stdout or p.stderr).strip() else ""
    except Exception:  # noqa: BLE001 - a version is a nicety, never a failure
        return ""


def supported_flags(binary: str) -> frozenset:
    """Every flag this CLI advertises, parsed once per process.

    An unknown flag is a hard failure, and the set differs between versions, so what is
    passed is the intersection of what is wanted and what is advertised."""
    with _LOCK:
        cached = _FLAGS.get(binary)
    if cached is not None:
        return cached
    try:
        p = subprocess.run([binary, "--help"], capture_output=True, text=True, errors="replace", timeout=60)
        flags = frozenset(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", (p.stdout or "") + (p.stderr or "")))
    except Exception:  # noqa: BLE001 - an unreadable help means "pass nothing optional"
        flags = frozenset()
    with _LOCK:
        _FLAGS[binary] = flags
    return flags


def workdir() -> str:
    """An empty scratch directory, one per process.

    The CLI discovers CLAUDE.md by walking UP from its working directory, so launching inside
    a project silently injects that project's memory into every prompt. Demonstrated: with a
    CLAUDE.md saying "the secret codename is ZEPHYR-9911", a plain run answers ZEPHYR."""
    with _LOCK:
        d = _WORKDIR.get("dir")
        if d and os.path.isdir(d):
            return d
        d = tempfile.mkdtemp(prefix="claude-code-endpoint-")
        _WORKDIR["dir"] = d
        return d


# ── the prompt ──────────────────────────────────────────────────────────────────────────


CONTINUE_INSTRUCTION = (
    "Write your next reply as the Assistant. Reply with the message content only -- "
    "no preamble about what you are about to do, and no '### Assistant' header."
)


def render(messages: list[dict]) -> tuple[str, str]:
    """(system text, prompt) from a message list.

    The CLI takes ONE string, so a conversation is flattened: system messages are hoisted to
    --system-prompt and the rest becomes a labelled transcript that ends by asking for the
    next assistant turn. Without that ending the model summarises the transcript instead of
    continuing it."""
    system = "\n\n".join(_text(m) for m in messages if m.get("role") == "system" and _text(m))
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) == 1 and rest[0].get("role") == "user":
        return system, _text(rest[0])
    parts = []
    for m in rest:
        label = "Assistant" if m.get("role") == "assistant" else "User"
        parts.append(f"### {label}\n{_text(m)}")
    parts.append(f"### Assistant\n{CONTINUE_INSTRUCTION}")
    return system, "\n\n".join(parts)


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):  # the content-block form
        return "\n".join(str(b.get("text") or "") for b in c if isinstance(b, dict)).strip()
    return str(c or "").strip()


# ── the invocation ──────────────────────────────────────────────────────────────────────


def build_argv(
    binary: str,
    *,
    model: str,
    system: str,
    effort: str | None = None,
    max_budget_usd: float | None = None,
    deny: str = TOOL_NAMES,
    bare: bool = False,
) -> list[str]:
    """The command line, holding only flags this CLI advertises.

    --system-prompt REPLACES Claude Code's own prompt; --append-system-prompt would leave the
    whole harness prompt in front of it. Under the replacement the model reports its system
    prompt as a one-line preamble plus exactly the given text."""
    have = supported_flags(binary)
    argv = [binary, "-p"]

    def add(flag: str, *values: str) -> None:
        if flag in have:
            argv.append(flag)
            argv.extend(values)

    add("--output-format", "json")
    add("--model", model_id(model))
    if effort:
        add("--effort", effort)
    if max_budget_usd is not None:
        add("--max-budget-usd", str(max_budget_usd))
    # ALWAYS PASSED, EMPTY IF THE CALLER HAS NONE. Omitting the flag does not mean "no
    # system prompt": it means Claude Code's own, which adds thousands of input tokens to a
    # short prompt.
    add("--system-prompt", system or "")
    add("--disallowed-tools", deny)
    add("--strict-mcp-config")
    add("--mcp-config", '{"mcpServers": {}}')  # '{}' alone is rejected: mcpServers is required
    add("--safe-mode")  # memory, skills, plugins, hooks -- keeps the keychain
    add("--disable-slash-commands")  # skills also resolve as /slash-commands
    add("--setting-sources", "")
    add("--settings", "{}")
    add("--exclude-dynamic-system-prompt-sections")
    if bare:
        # --bare is tidier still, and it SKIPS THE KEYCHAIN: on a `claude`-login install it
        # fails with "Not logged in - Please run /login". Only with an API key.
        add("--bare")
    return argv


def environment() -> dict:
    env = dict(os.environ)
    env.update(ISOLATION_ENV)
    return env


def summed_input_tokens(usage: dict) -> int:
    """Input tokens actually billed: uncached plus cache creation plus cache read.

    `input_tokens` alone is often single digits and under-reports enormously."""
    u = usage or {}
    return (
        int(u.get("input_tokens") or 0)
        + int(u.get("cache_creation_input_tokens") or 0)
        + int(u.get("cache_read_input_tokens") or 0)
    )


def failure_reason(stdout: str, stderr: str) -> str:
    """The reason, read from the JSON `result` field.

    NOT by scanning the whole body: on failure the CLI still prints a full result object whose
    session ids and token counts contain digit runs like 429 and 503, so a substring match
    over that blob reads a permanent failure as a rate limit and burns every retry on it."""
    for chunk in (stdout, stderr):
        text = (chunk or "").strip()
        if text.startswith("{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            for key in ("result", "error", "message"):
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ((stderr or "").strip() or (stdout or "").strip())[:1000]


_PERMANENT = (
    "not logged in",
    "please run /login",
    "there's an issue with the selected model",
    "unknown option",
    "unknown argument",
    "invalid mcp configuration",
)
# The subscription's own capacity: "You've hit your session limit · resets 8:20am (UTC)". It
# lifts at the stated time, so it is neither a defect nor a blip to retry in seconds.
_SESSION_LIMIT = ("session limit", "usage limit", "limit reached", "limit will reset")
_TRANSIENT = (
    "overloaded",
    "rate limit",
    "too many requests",
    "timed out",
    "timeout",
    "econnreset",
    "etimedout",
    "socket hang up",
    "service unavailable",
    "bad gateway",
)


def classify(reason: str) -> str:
    """ "permanent" (retrying cannot help), "session_limit" (wait for the stated reset),
    "transient" (retry with backoff), or "unknown"."""
    low = (reason or "").lower()
    if any(w in low for w in _PERMANENT):
        return "permanent"
    if any(w in low for w in _SESSION_LIMIT):
        return "session_limit"
    if any(w in low for w in _TRANSIENT):
        return "transient"
    return "unknown"


_RESET = re.compile(
    r"resets\s+(?:(?P<mon>[A-Za-z]{3})[a-z]*\.?\s+(?P<day>\d{1,2}),?\s+(?:at\s+)?)?"
    r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>am|pm)\s*(?:\((?P<tz>[^)]+)\))?",
    re.IGNORECASE,
)
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def seconds_until_reset(reason: str, now=None) -> float | None:
    """Seconds from `now` to the reset a session-limit message states ("resets 8:20am (UTC)",
    "resets Oct 16, 9am (Europe/Berlin)"); the next such moment, so a time already past today
    means tomorrow. None when the message names no reset this can read."""
    import datetime as dt

    m = _RESET.search(reason or "")
    if not m:
        return None
    tz = dt.timezone.utc
    if m.group("tz") and m.group("tz").strip().upper() not in ("UTC", "GMT"):
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(m.group("tz").strip())
        except Exception:
            return None
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(tz)
    hour = int(m.group("h")) % 12 + (12 if m.group("ap").lower() == "pm" else 0)
    minute = int(m.group("m") or 0)
    if m.group("mon"):
        try:
            month = _MONTHS.index(m.group("mon")[:3].lower()) + 1
        except ValueError:
            return None
        at = now.replace(month=month, day=int(m.group("day")), hour=hour, minute=minute, second=0, microsecond=0)
        if at <= now:
            at = at.replace(year=at.year + 1)
    else:
        at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if at <= now:
            at += dt.timedelta(days=1)
    return (at - now).total_seconds()


# The same patience the litellm route has (agentcore.models.litellm_model), read from the same
# variables, so one knob governs how long either route waits out a provider blip.
TRANSIENT_RETRY_SECONDS = float(os.getenv("AGENT_LLM_TRANSIENT_RETRY_SECONDS", "600"))
TRANSIENT_RETRY_MAX_WAIT = float(os.getenv("AGENT_LLM_TRANSIENT_MAX_WAIT", "60"))
#: After a stated reset, the limit is not always lifted to the second.
RESET_MARGIN_SECONDS = 60.0
_sleep = time.sleep


class ClaudeCodeError(RuntimeError):
    def __init__(self, reason: str, kind: str = "unknown"):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


def invoke(
    messages: list[dict],
    *,
    model: str,
    effort: str | None = None,
    timeout: float | None = None,
    max_budget_usd: float | None = None,
    binary: str | None = None,
    system: str | None = None,
    bare: bool | None = None,
    deny: str = TOOL_NAMES,
) -> dict:
    """One call. Returns {content, finish_reason, cost_usd, usage, input_tokens, raw, version}.

    Raises ClaudeCodeError with `kind` set, so a caller can retry the transient class and fail
    fast on the permanent one."""
    exe = binary or find_binary()
    if not exe:
        raise ClaudeCodeError(
            "the Claude Code CLI was not found: put `claude` on PATH, or set CLAUDE_CODE_BINARY", "permanent"
        )
    rendered_system, prompt = render(messages)
    if system:
        rendered_system = f"{system}\n\n{rendered_system}".strip() if rendered_system else system
    if bare is None:
        bare = bool(os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get("CLAUDE_CODE_BARE") == "1"
    argv = build_argv(
        exe,
        model=model,
        system=rendered_system,
        effort=effort or DEFAULT_EFFORT,
        max_budget_usd=max_budget_usd,
        deny=deny,
        bare=bare,
    )
    with _SEMAPHORE:
        try:
            # errors="replace": the answer need not be valid UTF-8, and a strict decode
            # would raise inside the call instead of returning what came back.
            p = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=float(timeout or DEFAULT_TIMEOUT),
                cwd=workdir(),
                env=environment(),
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCodeError(f"the CLI did not answer within {timeout or DEFAULT_TIMEOUT} s", "transient") from e
    out, err = p.stdout or "", p.stderr or ""
    text = out.strip()
    if not text.startswith("{"):
        # An older CLI without --output-format prints prose; pass it through unchanged.
        if p.returncode == 0 and text:
            return {
                "content": text,
                "finish_reason": "stop",
                "cost_usd": 0.0,
                "usage": {},
                "input_tokens": 0,
                "raw": {"stdout": text},
                "version": "",
            }
        reason = failure_reason(out, err)
        raise ClaudeCodeError(reason or "the CLI produced no output", classify(reason))
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ClaudeCodeError(f"the CLI's output is not JSON: {e}", "transient") from e
    if obj.get("is_error") or p.returncode != 0:
        reason = failure_reason(out, err)
        raise ClaudeCodeError(reason or "the CLI reported an error", classify(reason))
    usage = obj.get("usage") or {}
    return {
        "content": str(obj.get("result") or ""),
        "finish_reason": "stop",
        "cost_usd": float(obj.get("total_cost_usd") or 0.0),
        "usage": usage,
        "input_tokens": summed_input_tokens(usage),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "models_billed": sorted((obj.get("modelUsage") or {}).keys()),
        "duration_ms": obj.get("duration_ms"),
        "raw": obj,
    }


def as_completion(result: dict, model: str) -> dict:
    """The result in the shape a chat-completions caller reads."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": result.get("content") or ""},
                "finish_reason": result.get("finish_reason") or "stop",
                "index": 0,
            }
        ],
        "model": model_id(model),
        "usage": {
            "prompt_tokens": result.get("input_tokens") or 0,
            "completion_tokens": result.get("output_tokens") or 0,
            "total_tokens": (result.get("input_tokens") or 0) + (result.get("output_tokens") or 0),
        },
        "_claude_code": {k: result.get(k) for k in ("cost_usd", "usage", "models_billed", "duration_ms")},
    }


# ── the agentcore model ─────────────────────────────────────────────────────────────────


class ClaudeCodeModel:
    """A model backed by the CLI. Matches the agentcore.protocols.Model protocol.

    Configured as `class: claude-code`, `name: <model-id>`, with `kwargs.effort`. There is
    no temperature: a configured one is ignored and noted on the response, because silently
    honouring a knob the endpoint does not have would misreport what was run."""

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
        self.cache_control = cache_control  # the CLI caches on its own; nothing to mark
        self.extra_kwargs = dict(kwargs or {})
        self.effort = str(self.extra_kwargs.pop("effort", None) or DEFAULT_EFFORT)
        self.timeout = float(self.extra_kwargs.pop("timeout", None) or DEFAULT_TIMEOUT)
        self.max_budget_usd = self.extra_kwargs.pop("max_budget_usd", None)
        self.system = self.extra_kwargs.pop("system", None)
        self.binary = self.extra_kwargs.pop("binary", None) or find_binary()
        self._cost = 0.0
        self.n_calls = 0
        self.on_retry = on_retry
        self.n_transient_retries = 0
        # Seconds the run has left (the loop sets it from its wallclock meter); a session
        # limit is waited out only when its reset comes before that. None: no run around it.
        self.wait_budget_s = None

    def _invoke_patiently(self, messages: list[dict], **call) -> dict:
        """invoke(), with transient failures retried on a backoff and a session limit waited
        out when the run has the time; permanent and unknown failures raise at once.

        Instant session-limit errors are not counted as defects."""
        deadline = time.monotonic() + TRANSIENT_RETRY_SECONDS
        attempt = 0
        while True:
            try:
                return invoke(messages, **call)
            except ClaudeCodeError as e:
                if e.kind == "session_limit":
                    reset_in = seconds_until_reset(e.reason)
                    left = self.wait_budget_s() if callable(self.wait_budget_s) else None
                    if reset_in is None or left is None or reset_in + RESET_MARGIN_SECONDS > left:
                        when = (
                            "no reset time could be read from the message"
                            if reset_in is None
                            else f"the reset is {reset_in / 60:.0f} min away"
                            + (
                                f", the run has {left / 60:.0f} min left"
                                if left is not None
                                else ", and no run budget to wait within"
                            )
                        )
                        raise ClaudeCodeError(f"{e.reason} -- not waited for: {when}", "session_limit") from e
                    wait = reset_in + RESET_MARGIN_SECONDS
                    remaining = left
                elif e.kind == "transient":
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    attempt += 1
                    wait = min(min(2**attempt, TRANSIENT_RETRY_MAX_WAIT), remaining)
                else:
                    raise
                attempt = max(attempt, 1)
                self.n_transient_retries += 1
                if self.on_retry:
                    try:
                        self.on_retry(attempt=attempt, wait=wait, remaining=remaining, error=e)
                    except Exception:
                        pass  # observability must never break the call path
                _sleep(wait)

    def query(self, messages: list[dict], **kwargs) -> Any:
        from agentcore.protocols import ModelResponse

        model = kwargs.pop("model", None) or self.name
        dropped = [k for k in ("temperature", "top_p", "seed", "max_tokens") if k in kwargs]
        for k in dropped:
            kwargs.pop(k)
        result = self._invoke_patiently(
            messages,
            model=model,
            effort=kwargs.pop("effort", None) or self.effort,
            timeout=kwargs.pop("timeout", None) or self.timeout,
            max_budget_usd=self.max_budget_usd,
            binary=self.binary,
            system=self.system,
        )
        self.n_calls += 1
        self._cost += float(result.get("cost_usd") or 0.0)
        raw = as_completion(result, model)
        if dropped:
            raw["_claude_code"]["dropped_params"] = dropped  # the CLI has only --effort
        return ModelResponse(
            content=result.get("content") or "",
            raw=raw,
            cost=float(result.get("cost_usd") or 0.0),
            finish_reason=result.get("finish_reason") or "stop",
        )

    def cost(self) -> float:
        return self._cost

    def get_template_vars(self) -> dict:
        return {"model_name": self.name, "model_cost": round(self._cost, 4), "n_model_calls": self.n_calls}
