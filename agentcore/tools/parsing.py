"""Action parsing + output truncation/reducers + a structurer fallback.

Lifted and unified from the two real agents:
  - JSON-from-text extraction (fenced block / brace matching) + double-escape repair
    (execution_agent.agent._extract_json_object / _fix_escaped_newlines)
  - failure-region extraction across ecosystems (analysis_agent.agent)
  - head+middle+tail truncation (execution_agent.agent._smart_truncate_output)
  - a "structurer" LLM pass with model rotation for stubborn outputs
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── JSON action extraction ───────────────────────────────────────────────────


def _extract_json_substring(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _fix_escaped_newlines(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _fix_escaped_newlines(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_escaped_newlines(v) for v in obj]
    if isinstance(obj, str) and ("\\n" in obj or "\\t" in obj or "\\r" in obj):
        return obj.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    return obj


def _candidates(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    cands = [raw]
    for blk in re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE):
        if blk.strip():
            cands.append(blk.strip())
    sub = _extract_json_substring(raw)
    if sub:
        cands.append(sub)
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


_BAD_ESCAPE = re.compile(r'\\(?![\\/"bfnrtu])')


#: JSON's own escape set. Everything else after a backslash, inside a string, is a
#: backslash the model meant literally -- a regex `\d`, a Windows path, a LaTeX macro.
_JSON_ESCAPES = set('"\\/bfnrtu')


def repair_json(text: str) -> str:
    """A reply that is JSON except for how it spelled a backslash or a newline INSIDE a string.

    ONE resolver for both parsers, so the action parser and the formalizer's caller share
    one string-aware scanner. Typical accidents are "Invalid \\escape" and
    "Invalid control character", neither of which is about the answer: a
    canonical spec is full of regexes, and a model writing one into a JSON string emits the
    single backslash it means. Inside a string both have exactly one legal spelling, so the
    repair is a resolution, not a guess. Only string interiors are touched; structure,
    whitespace between fields and valid escapes are left exactly as they are, and the caller
    accepts the result only if it parses.
    """
    out, in_string, i, n = [], False, 0, len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt in _JSON_ESCAPES:
                out.append(ch)
                out.append(nxt)
                i += 2
            else:  # a literal backslash the model did not escape
                out.append("\\\\")
                i += 1
            continue
        if ch == '"':
            in_string = False
            out.append(ch)
            i += 1
            continue
        if ord(ch) < 0x20:  # a raw control character inside a string
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(ch, f"\\u{ord(ch):04x}"))
            i += 1
            continue
        out.append(ch)
        i += 1
    return _close_open_structures("".join(out))


def _close_open_structures(text: str) -> str:
    """Append the closers a reply is missing at its END, in the only order JSON allows.

    A reply can arrive as `...}]}]` instead of `...}]}]}`, the root object's final brace
    missing, with finish_reason "stop". The intent is unambiguous when the text ends outside a string with structures still open; the parser
    then accepts the result only if it parses, so a reply broken anywhere else still goes to
    the retry unchanged. Brackets inside strings are text and are not counted.
    """
    stack: list[str] = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return text  # mismatched closer: not a truncation, leave it
        i += 1
    if in_string or not stack:
        return text
    closed = text.rstrip() + "".join(reversed(stack))
    try:
        json.loads(closed)
    except ValueError:
        return text  # broken somewhere else too: nothing to resolve here
    return closed


def _repaired(cand: str, err: json.JSONDecodeError) -> dict | None:
    """The two unambiguous JSON faults a model's reply carries, resolved instead of refused.

    A backslash before a character JSON does not escape (`\\|` in a grep alternation, `\\.`
    in a regex) can only have been meant literally: it is
    doubled and the reply re-read. A reply that ran out of input a few braces early is closed
    (`_closed_short`). Anything else stays a format error.
    """
    if err.msg.startswith("Invalid \\escape") or err.msg.startswith("Invalid control character"):
        fixed = repair_json(cand)
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError as e2:
            return _closed_short(fixed, e2)
        return obj if isinstance(obj, dict) else None
    return _closed_short(cand, err)


def _closed_short(cand: str, err: json.JSONDecodeError) -> dict | None:
    """A reply whose JSON ran out of input one to three closing braces early, closed.

    Typical shape: {"thoughts": ..., "command": {"name": ..., "args": {..., "command": "...heredoc..."}} --
    three opens, two closes. Only when the decoder ran OUT of input:
    an error anywhere before the end is a real one and stays refused.
    """
    text = cand.rstrip()
    if err.pos < len(text):
        return None
    for k in (1, 2, 3):
        try:
            obj = json.loads(text + "}" * k)
        except json.JSONDecodeError:
            continue
        return obj if isinstance(obj, dict) else None
    return None


def extract_json_action(text: str) -> dict | None:
    """Return a dict like {"thoughts":..,"command":{"name":..,"args":{...}}} or None.

    Accepts raw JSON, fenced ```json blocks, or JSON embedded in prose. Normalizes
    a couple of shapes the models emit (top-level name/args without a command key).
    """
    for cand in _candidates(text):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            obj = _repaired(cand, e)
            if obj is None:
                continue
        if not isinstance(obj, dict):
            continue
        # NOTHING is "repaired" here. `json.loads` above has already decoded the payload
        # correctly, and running _fix_escaped_newlines over the result rewrote every `\n`,
        # `\t` and `\r` in every string into a real newline/tab/return -- which in a source
        # file are exactly the escapes a script needs. `content.split('\n')` arrived as a
        # string literal split across two lines and would not compile.
        #
        # Some models genuinely do double-escape, but repairing that is the wrong trade: corrupting correct input is silent, unrecoverable and
        # invisible to the model, while leaving a real double-escape alone shows up as a
        # visible syntax error the model can see and fix. Prefer the recoverable failure.
        # (_fix_escaped_newlines survives for sentinel matching in loop.py, where collapsing an
        # escaped newline before searching for a token IS what is wanted.)
        if "command" in obj and isinstance(obj["command"], dict):
            cmd = obj["command"]
            if cmd.get("name"):
                cmd["args"] = _args_of(cmd)
                return obj
        if obj.get("name") and ("args" in obj or len(obj) > 1):  # flattened shape
            return {"thoughts": obj.get("thoughts", ""), "command": {"name": obj["name"], "args": _args_of(obj)}}
    return None


# The keys of a call envelope that are not arguments, in either shape the parser accepts.
_ENVELOPE_KEYS = ("name", "args", "thoughts")


def _args_of(cmd: dict) -> dict:
    """The arguments of one call envelope.

    ARGUMENTS WRITTEN AS SIBLINGS OF `name` ARE STILL ARGUMENTS. Models routinely
    emit {"name": "record", "key": ..., "value": ...} with no `args` at all; handing the tool an
    empty dict would throw the values away and produce a misleading "missing required
    positional arguments" error. The siblings are folded in ONLY when `args` is absent or empty: a
    call that did nest its arguments has said where they are, and a sibling beside a filled
    `args` is envelope prose (a `reasoning` field), which the fuzzy resolver would otherwise
    bind to the nearest parameter -- `reason`, on a worklist_drop.
    """
    args = cmd.get("args")
    if isinstance(args, dict) and args:
        return args
    if args is not None and not isinstance(args, dict):
        return args
    return {k: v for k, v in cmd.items() if k not in _ENVELOPE_KEYS}


def extract_bash_action(text: str, regex: str = r"```bash\s*\n(.*?)\n```") -> dict | None:
    """bash_block protocol: a single fenced bash command -> linux_terminal call."""
    actions = re.findall(regex, text or "", re.DOTALL)
    if len(actions) == 1:
        return {"thoughts": "", "command": {"name": "linux_terminal", "args": {"command": actions[0].strip()}}}
    return None


# ── failure-region extraction (for the first_error reducer) ──────────────────

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_FAILURE_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bfatal(?: error)?:",
        r"\berror\b(?!\s*[:=]\s*0\b)",
        r"\bfailed\b",
        r"\bfailure\b",
        r"\bexception\b",
        r"\btraceback \(most recent call last\):",
        r"\bsegmentation fault\b",
        r"\bassert(ion)? failed\b",
        r"\bpermission denied\b",
        r"\bcommand not found\b",
        r"\bno such file or directory\b",
        r"\bmodulenotfounderror",
        r"\bimporterror",
        r"\breturned non[- ]zero exit status\b",
        r"\bexit (status|code)\s*[:=]?\s*[1-9]\d*\b",
        r"^make(?:\[\d+\])?: \*\*\*",
        r"^npm err!",
        r"^build failed",
        r"^fail(ed)?:",
        # static-analyzer / compiler diagnostics (clang/gcc style + scan-build summary)
        r":\d+:\d+:\s+(error|warning):",
        r"\bbugs?\s+found\b",
        r"\[-W[a-z-]+\]",
        r"\[core\.[a-zA-Z.]+\]",
    )
]


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s or "")


def find_failure_regions(text: str, context: int = 2) -> list[str]:
    lines = strip_ansi(text).splitlines()
    hits = [i for i, ln in enumerate(lines) if any(p.search(ln) for p in _FAILURE_PATTERNS)]
    if not hits:
        return []
    ranges: list[tuple[int, int]] = []
    for i in hits:
        ranges.append((max(0, i - context), min(len(lines) - 1, i + context)))
    # merge overlapping
    ranges.sort()
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le + 1:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return ["\n".join(lines[s : e + 1]) for s, e in merged]


# ── truncation / reducers ────────────────────────────────────────────────────


def clip(text: str, limit: int, label: str = "") -> str:
    """One string cut to `limit`, keeping the head (2/3) and the TAIL (1/3): a traceback's
    verdict and a refusal's instruction are both on the last line. The marker announces the
    cut and the true length, so a clipped record never reads as a short one. `label` prefixes
    the marker for writers that want a greppable token of their own.

    THE one head+tail cut: the action-log replay, the parked candidate's last refusal and the
    trajectory writer each had their own arithmetic and their own marker wording."""
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return text[:head] + f"\n... [{label}{len(text) - limit} of {len(text)} chars omitted] ...\n" + text[-tail:]


def truncate(
    text: str,
    policy: str = "head_tail",
    *,
    head: int = 8000,
    middle: int = 2000,
    tail: int = 8000,
    threshold: int = 16000,
    n_lines: int = 60,
) -> str:
    s = text or ""
    if policy == "none":
        return s
    if len(s) <= threshold:
        return s
    if policy == "latest_n_lines":
        lines = s.splitlines()
        kept = lines[-n_lines:]
        if len(lines) <= len(kept):
            return s
        dropped = len(lines) - len(kept)
        dropped_chars = len(s) - len("\n".join(kept))
        # SILENT truncation is the harness lying by omission:
        # an agent that copies what it sees records a fragment believing it is whole. Every
        # truncation names what was cut and how much, so "record this output" is either
        # complete or visibly incomplete.
        return (
            f"[output truncated: showing the last {len(kept)} of {len(lines)} lines; "
            f"{dropped} lines / ~{dropped_chars} chars omitted above]\n" + "\n".join(kept)
        )
    if policy == "first_error":
        regions = find_failure_regions(s)
        if regions:
            joined = "\n...\n".join(regions)
            return f"{s[:head]}\n... [truncated; failure regions extracted] ...\n{joined}"[: head + tail + 4000]
        # fall through to head_tail if no error found
    # head_tail (default): head + sampled middle + tail
    if len(s) <= head + middle + tail + 200:
        return s
    mid_start = head + (len(s) - head - tail - middle) // 2
    parts = [
        s[:head],
        f"\n\n... [output truncated: {mid_start - head} chars omitted] ...\n\n",
        s[mid_start : mid_start + middle],
        f"\n\n... [output truncated: {len(s) - tail - (mid_start + middle)} chars omitted] ...\n\n",
        s[-tail:],
    ]
    return "".join(parts)


# ── structurer fallback ──────────────────────────────────────────────────────


_PLACEHOLDER = {"string": "...", "integer": 0, "number": 0, "boolean": False, "object": {}, "array": []}


def _example_for(spec: dict, message: str) -> str:
    args = {k: _PLACEHOLDER.get(v.get("type", "string"), "...") for k, v in spec.get("properties", {}).items()}
    payload = {"thoughts": "...", "command": {"name": spec["name"], "args": args}}
    return f'Message: "{message}"\n{json.dumps(payload)}'


def build_structurer_system(tool_specs: list[dict], extra_instructions: str = "") -> str:
    """The structurer's instructions, DERIVED from the active tools: the available
    commands + when-to-use, a completion mapping naming the actual finishing tools, and
    worked examples built from real tools (nothing hardcoded to a specific toolset)."""
    listing = "\n".join(
        f"- {s['name']}({s['signature']})" + (f" — {s['doc'].splitlines()[0]}" if s.get("doc") else "")
        for s in tool_specs
    )
    finishers = [s for s in tool_specs if s.get("finishes")]
    non_finishers = [s for s in tool_specs if not s.get("finishes")]

    finish_line = ""
    if finishers:
        names = " or ".join(s["name"] for s in finishers)
        finish_line = (
            f"If the message says the task is finished / accomplished / complete / done, emit "
            f"the completion tool ({names}) — do NOT invent another command to finish.\n"
        )

    examples = []
    if non_finishers:
        s = non_finishers[0]
        examples.append(_example_for(s, f"Next I will use {s['name']} to make progress on the task."))
    if finishers:
        examples.append(_example_for(finishers[0], "The task is complete and successful."))
    examples_text = ("\n\nExamples:\n" + "\n".join(examples)) if examples else ""

    return (
        "You translate an assistant's free-form message into EXACTLY ONE tool call.\n"
        "Output STRICT JSON only — no prose, no markdown fences — in this shape:\n"
        '{"thoughts": "<short>", "command": {"name": "<tool>", "args": { ... }}}\n\n'
        "Pick the SINGLE tool from the list below that best matches what the message says to do next, "
        "using each tool's description, and fill its args.\n"
        + finish_line
        + "\nAvailable tools (name(args) — when to use):\n"
        + listing
        + "\n\n"
        + (extra_instructions.strip() + "\n\n" if extra_instructions.strip() else "")
        + "ESCAPING: produce valid JSON; use single-backslash escapes (\\n, \\t) inside strings; never double-escape."
        + examples_text
    )


def structure_action(
    models: dict,
    freeform: str,
    tool_specs: list[dict],
    rotation: list[str] | None = None,
    max_attempts: int = 6,
    extra_instructions: str = "",
    cost_sink=None,
) -> dict | None:
    """Coerce a free-form message into a JSON tool call using a model.

    `models` maps role->Model; prefers a 'structurer' model, falls back to 'main'.
    `tool_specs` is the active tools' structured specs (registry.specs(allow)); the
    structurer prompt (listing, completion mapping, examples) is built from them.
    Rotates through `rotation` on parse failure. `cost_sink(delta)` (e.g. budget.note_cost)
    is called with each call's cost so the budget meter accounts for structurer spend.
    """
    model = models.get("structurer") or models.get("main")
    if model is None:
        return None
    system = build_structurer_system(tool_specs, extra_instructions)
    rotation = rotation or []
    for attempt in range(max_attempts):
        kwargs = {}
        if rotation:
            kwargs["model"] = rotation[min(attempt // 2, len(rotation) - 1)]
        nudge = "" if attempt == 0 else "\n\nReturn ONLY one valid JSON object, no other text."
        try:
            resp = model.query(
                [{"role": "system", "content": system + nudge}, {"role": "user", "content": freeform}], **kwargs
            )
        except Exception:
            continue
        if cost_sink is not None:
            cost_sink(getattr(resp, "cost", 0.0))
        action = extract_json_action(getattr(resp, "content", "") or "")
        if action and action.get("command", {}).get("name"):
            return action
    return None
