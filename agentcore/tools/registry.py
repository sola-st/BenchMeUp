"""ToolRegistry + the @tool decorator.

A tool is `def f(ctx: ToolContext, **args) -> ToolResult`. The @tool decorator
derives a JSON schema from the function signature (so there is no second
`commands_schema` to keep in sync — the desync pain from the legacy agents).
Tools are enabled per agent (and optionally per stage) by name in agent.yaml.
"""

from __future__ import annotations

import inspect
import json
import typing
from typing import Any, Callable

from agentcore.tools.context import ToolContext
from agentcore.tools.result import ToolResult

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}


# Every tool module starts with `from __future__ import annotations`, so by the time this sees
# a signature the annotations are STRINGS -- "int", "bool", "str | None". `_PY_TO_JSON.get("int")`
# missed every one of them and returned "string", so the schema handed to the model declared
# `linux_terminal(timeout: string)`, and the model would send "120". Resolve the name as well
# as the type object.
_NAME_TO_JSON = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}


def _json_type(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = typing.get_origin(annotation)
    if origin is not None:
        # Optional[X] / X | None -> type of X
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if args:
            return _json_type(args[0])
        return "string"
    if isinstance(annotation, str):
        # A postponed annotation. Strip Optional/union spelling and look the base name up.
        base = annotation.split("|")[0].strip()
        if base.startswith("Optional[") and base.endswith("]"):
            base = base[len("Optional[") : -1].strip()
        return _NAME_TO_JSON.get(base, "string")
    return _PY_TO_JSON.get(annotation, "string")


def _schema_from_signature(fn: Callable) -> dict:
    sig = inspect.signature(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    params = list(sig.parameters.values())
    for p in params[1:]:  # skip ctx
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        props[p.name] = {"type": _json_type(p.annotation)}
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    return {"type": "object", "properties": props, "required": required}


def tool(
    name: str | None = None,
    *,
    stages: list[str] | None = None,
    truncation: str = "head_tail",
    idempotent_ok: bool = False,
    finishes: bool = False,
    description: str | None = None,
    params: dict[str, str] | None = None,
    aliases: dict[str, list[str]] | None = None,
    if_granted: dict[str, str] | None = None,
):
    """Register a function as an agent tool. Schema is derived from the signature.

    THE DOCSTRING IS WHAT THE AGENT READS, whole, in its tool listing: write it for
    the agent -- what the tool does, what it returns, what to do with the reply -- and keep
    history and rationale in comments. `if_granted={"container_exec": "..."}` adds a paragraph
    only for an agent that also holds that tool, so the text never names a tool it lacks.

    `aliases` declares the OTHER NAMES a parameter has been called by, so a call under one
    of them is resolved rather than refused: `aliases={"command": ["cmd", "break"]}`. Refusing a
    synonym (`cmd`, `variant`, `build`, `setup`, `materialise`, `break`, `undo`) names one
    missing parameter per call, so the model would learn the signature one wasted cycle at a
    time. See `ToolRegistry.resolve_args`.

    finishes=True marks a completion tool (goals_accomplished/submit) so prompts and the
    structurer can reference the *actual* finishing tools available, not a hardcoded name.

    `params` documents individual arguments IN THE PROMPT. The tool listing renders one line
    per tool from the docstring's first line, so anything explained in the body is invisible to
    the model -- and for a parameter that decides what gets recorded, invisible means guessed.
    For example `worklist_drop(reason, blocked)` distinguishes `blocked=true` (a pipeline fault,
    not counted against the sample) from `blocked=false` (the candidate is genuinely
    inadmissible); that distinction belongs in `params`, not only in the docstring body.

    Only document parameters whose value changes MEANING; a gloss on every argument would
    triple a tool block in an already long prompt.

    A GLOSS THAT CROSS-REFERENCES ANOTHER TOOL must say which, so the listing can drop it when
    that tool is not granted. Pass a dict instead of a string:

        params={"command": {"text": "... belongs in that candidate's sandbox via "
                                    "container_exec, never here",
                            "needs": "container_exec",
                            "without": "... the shared one, which is not any sandbox"}}

    Without this, an agent that grants no container tools would be told that candidate evidence
    "belongs in that candidate's sandbox via container_exec, never here". Prompt templates
    follow the same rule (`AgentLoop._template_vars`, "THE NAMES, separately from the prose").
    An instruction to use a tool the model cannot use sets the agent up to fail.
    """

    def deco(fn: Callable) -> Callable:
        schema = _schema_from_signature(fn)
        for pname, gloss in (params or {}).items():
            if isinstance(gloss, dict) and not (gloss.get("text") and gloss.get("needs")):
                raise ValueError(
                    f"@tool({name or fn.__name__!r}, params=...): {pname!r} is a conditional "
                    f"gloss and needs both `text` and `needs` (and normally `without`). "
                    f"A half-declared condition renders nothing or renders always."
                )
            if pname not in schema["properties"]:
                raise ValueError(
                    f"@tool({name or fn.__name__!r}, params=...): {pname!r} is not a parameter "
                    f"of {fn.__name__}; it has {sorted(schema['properties'])}. A typo here "
                    f"documents nothing, which is the failure this argument exists to fix."
                )
        for pname in aliases or {}:
            if pname not in schema["properties"]:
                raise ValueError(
                    f"@tool({name or fn.__name__!r}, aliases=...): {pname!r} is not a parameter "
                    f"of {fn.__name__}; it has {sorted(schema['properties'])}."
                )
        fn._tool_meta = {  # type: ignore[attr-defined]
            "name": name or fn.__name__,
            "stages": stages,
            "truncation": truncation,
            "idempotent_ok": idempotent_ok,
            "finishes": finishes,
            "description": (description or (fn.__doc__ or "")).strip(),
            "params": dict(params or {}),
            "if_granted": dict(if_granted or {}),
            "aliases": {p: [str(a).lower() for a in names] for p, names in (aliases or {}).items()},
            "schema": schema,
        }
        return fn

    return deco


def is_tool(fn: Any) -> bool:
    return callable(fn) and hasattr(fn, "_tool_meta")


# Models emit JSON-shaped TEXT, so a number arrives quoted more often than not. `timeout: "120"`
# is not a mistake worth refusing -- it is the normal spelling -- so it is coerced; otherwise
# the string would travel into the tool until something compared it with an int and the model
# was told `'>' not supported between instances of 'int' and 'str'`, a message that names
# nothing it can change.
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "off", "none", "null", "f"})
_TRUE_WORDS = frozenset({"true", "yes", "y", "1", "on", "t"})


class ArgTypeError(TypeError):
    """A value that cannot be coerced to the declared type, phrased for the model."""


class _Unset(Exception):
    """Sentinel: drop the argument entirely so the tool's own default applies."""


class ResolvedArgs(dict):
    """The arguments `resolve_args` bound, plus the keys it could not bind.

    A key that matched nothing must not vanish without a word; otherwise the call runs and the model
    never learns that `shell="bash"` had done nothing. `call()` appends them to the result.
    """

    ignored: list[str]

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.ignored = []


# The one convention for a signature, wherever it is shown: the agent-facing listing, the
# structurer's specs, and the missing-parameter refusal. Required parameters first, then the
# optional ones marked `?` -- the mark the structurer already used, which the listing the
# MODEL reads never did: it rendered `container_create(name, image, dockerfile, workdir,
# lifetime, setup)` flat, and "mutually exclusive with dockerfile" lived in prose only.
LISTING_HEADER = "Parameters: the required ones come first, then the optional ones, marked `?`."


def _signature(schema: dict, typed: bool = False) -> str:
    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])
    ordered = [k for k in props if k in req] + [k for k in props if k not in req]
    return ", ".join(
        f"{k}{(': ' + str(props[k].get('type', 'string'))) if typed else ''}{'' if k in req else '?'}" for k in ordered
    )


def _coerce(name: str, value, want: str):
    """Best-effort cast of one argument to its declared JSON type.

    Only strings are converted, and only towards the declared type. A value that is already
    right is returned untouched; one that cannot be converted raises with the argument named,
    because "timeout must be an integer; got 'soon'" is actionable and a Python comparison
    error is not.
    """
    if want not in ("integer", "number", "boolean", "array", "object") or not isinstance(value, str):
        return value
    text = value.strip()
    if want in ("array", "object"):
        # A list or dict arrives as its JSON TEXT at least as often as it arrives decoded:
        # the model wrote a string because the whole message is a string. Read it when it
        # parses to the declared shape; otherwise leave the value for the tool, whose own
        # refusal says what shape it wanted.
        if text[:1] == ("[" if want == "array" else "{"):
            try:
                decoded = json.loads(text)
            except ValueError:
                return value
            if isinstance(decoded, list if want == "array" else dict):
                return decoded
        return value
    if want == "boolean":
        low = text.lower()
        if low in _TRUE_WORDS:
            return True
        if low in _FALSE_WORDS or low == "":
            return False
        raise ArgTypeError(f"{name} must be true or false; got {value!r}")
    if text == "":
        raise _Unset()  # "I am not setting this" -- let the default stand
    try:
        return int(text) if want == "integer" else float(text)
    except ValueError:
        raise ArgTypeError(f"{name} must be an {'integer' if want == 'integer' else 'number'}; got {value!r}") from None


def _gloss_for(gloss, allow: list[str] | None) -> str:
    """One parameter's gloss, for an agent holding `allow`.

    A plain string renders as-is. A dict declares `needs`: the tool the gloss talks about. When
    that tool is not granted, `without` is rendered instead (or nothing, if there is none) --
    because a gloss pointing at a tool the agent does not have is worse than no gloss. `allow`
    of None means "every tool", the shape `describe()` uses for listings not tied to a stage.
    """
    if not isinstance(gloss, dict):
        return str(gloss)
    if allow is None or gloss.get("needs") in set(allow):
        return str(gloss.get("text") or "")
    return str(gloss.get("without") or "")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}

    def register(self, fn: Callable) -> None:
        if not is_tool(fn):
            raise TypeError(f"{fn!r} is not decorated with @tool")
        self._tools[fn._tool_meta["name"]] = fn

    def register_module(self, module) -> int:
        """Register every @tool-decorated callable found in a module. Returns count."""
        n = 0
        for obj in vars(module).values():
            if is_tool(obj):
                self.register(obj)
                n += 1
        return n

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Callable:
        return self._tools[name]

    def meta(self, name: str) -> dict:
        return self._tools[name]._tool_meta

    def enabled_for(self, stage: str | None, allow: list[str]) -> dict[str, Callable]:
        out = {}
        for n, t in self._tools.items():
            if n not in allow:
                continue
            st = t._tool_meta["stages"]
            if stage is None or st is None or stage in st:
                out[n] = t
        return out

    def resolve_args(self, name: str, raw: dict, fuzzy: bool = False) -> dict:
        """Map raw args onto the tool's declared params; optional fuzzy/substring remap.

        RESOLUTION ORDER, each pass only over what the passes before it left unmatched:
          1. the declared name itself;
          2. a synonym the tool DECLARED (`@tool(aliases=...)`), exactly: `cmd` -> `command`,
             `variant` -> `name`;
          3. (fuzzy) a declared name inside the key, or the key inside it:
             `check_cmd` -> `check`, `base_image` -> `image`, `variant` -> `target_variant`;
          4. (fuzzy) a declared synonym inside the key: `break_cmd` -> `command` via `break`,
             `setup_command` -> `rebuild` via `setup`, `undo_break_command` -> `restore`.
        Declared names get their turn (3) before synonym stems (4), so `restore_cmd` reaches
        `restore` and never `command` through `cmd`.
        """
        meta = self._tools[name]._tool_meta
        schema = meta["schema"]
        props = list(schema["properties"].keys())
        if not props:
            return ResolvedArgs(raw or {})
        aliases = {p: list(a) for p, a in (meta.get("aliases") or {}).items()}
        out = ResolvedArgs()

        def _take(pk: str, value) -> None:
            try:
                out[pk] = _coerce(pk, value, str(schema["properties"][pk].get("type", "string")))
            except _Unset:
                pass  # let the declared default stand
            except ArgTypeError:
                out[pk] = value  # call() phrases the refusal, in one place

        for k, v in (raw or {}).items():
            if k in props:
                _take(k, v)
        pending = [k for k in (raw or {}) if k not in props]

        def _match(test) -> None:
            # A key binds only when it matches EXACTLY ONE unbound parameter. `image_name`
            # against container_create(name, image, ...) matched both, and the old loop bound
            # it to whichever came first in the signature -- a wrong value in the right
            # place, which no refusal ever surfaces. An ambiguous key is left unbound: it is
            # then named in the result (or in the missing-parameter refusal, with the
            # signature), which costs one honest cycle instead of a silent mis-binding.
            # TO A FIXPOINT, so the answer does not depend on the order the keys arrived in:
            # `restore_command` matches both `restore` and `command` until `break_command`
            # has taken `command`, after which it is unambiguous -- whichever came first.
            progressed = True
            while progressed and pending:
                progressed = False
                for rk in list(pending):
                    hits = [pk for pk in props if pk not in out and test(rk.lower(), pk)]
                    if len(hits) == 1:
                        _take(hits[0], raw[rk])
                        pending.remove(rk)
                        progressed = True

        _match(lambda rk, pk: rk in aliases.get(pk, ()))
        if fuzzy:
            _match(lambda rk, pk: rk in pk.lower() or pk.lower() in rk)
            _match(lambda rk, pk: any(a in rk for a in aliases.get(pk, ())))
        out.ignored = list(pending)
        return out

    def specs(self, allow: list[str] | None = None) -> list[dict]:
        """Structured tool specs (name, signature, doc, finishes, schema) for prompt assembly."""
        out = []
        for n, t in self._tools.items():
            if allow is not None and n not in allow:
                continue
            m = t._tool_meta
            sch = m["schema"]
            sig = _signature(sch, typed=True)
            out.append(
                {
                    "name": n,
                    "signature": sig,
                    "doc": (m["description"] or "").strip(),
                    "finishes": m.get("finishes", False),
                    "properties": sch["properties"],
                }
            )
        return out

    def call(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        fn = self._tools.get(name)
        if fn is None:
            return ToolResult.fail(f"Unknown tool '{name}'")
        schema = fn._tool_meta["schema"]
        props = schema["properties"]
        sig = f"{name}({_signature(schema)})"
        ignored = list(getattr(args, "ignored", None) or [])
        coerced = {}
        for k, v in (args or {}).items():
            if k not in props:
                coerced[k] = v
                continue
            try:
                coerced[k] = _coerce(k, v, str(props[k].get("type", "string")))
            except _Unset:
                continue
            except ArgTypeError as e:
                return ToolResult.fail(str(e))
        args = coerced
        try:
            res = fn(ctx, **(args or {}))
        except ArgTypeError as e:
            return ToolResult.fail(str(e))  # already phrased for the model
        except TypeError as e:
            # A MISSING PARAMETER IS REFUSED WITH THE WHOLE SIGNATURE, never with Python's
            # "missing 1 required positional argument: 'name'". That names one parameter per
            # refusal, so a model holding the values under other names would learn the signature
            # one cycle at a time.
            # The signature is rendered by the same helper as the listing, so the refusal
            # and the prompt can never show two different shapes of the same tool.
            missing = [p for p in schema.get("required", []) if p not in (args or {})]
            unknown = [k for k in (args or {}) if k not in props]
            if missing:
                return ToolResult.fail(
                    f"{sig} is the signature; this call "
                    f"sent ({', '.join(args or {}) or 'nothing'}) and is missing: "
                    f"{', '.join(missing)}. Send the same values under exactly those "
                    f"parameter names."
                )
            if unknown:
                # Only reachable when the caller skipped resolve_args; the loop never does.
                return ToolResult.fail(
                    f"{sig} is the signature; this call sent ({', '.join(args)}), and "
                    f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not among "
                    f"its parameters. Send the same values under the names above."
                )
            return ToolResult.fail(f"Invalid args for '{name}': {e}")
        except Exception as e:  # noqa: BLE001 - surface tool errors as observations
            return ToolResult.fail(f"Tool '{name}' failed: {type(e).__name__}: {e}")
        if isinstance(res, ToolResult):
            pass
        elif isinstance(res, str):
            res = ToolResult.text(res)
        elif isinstance(res, dict):
            res = ToolResult(display_text=str(res.get("output", res)), structured=res)
        else:
            res = ToolResult.text(str(res))
        if ignored and not res.error:
            # The call ran without them; say so, or the model keeps sending them.
            res = res.model_copy(
                update={
                    "display_text": (
                        (res.display_text or "")
                        + f"\n[ignored: {', '.join(ignored)} -- not parameters of {name}; the signature "
                        f"is {sig}]"
                    )
                }
            )
        return res

    def names(self, allow: list[str] | None = None) -> list[str]:
        """The tools a stage can actually call, as names.

        `describe()` renders their prose, and prompt templates were guarding capability
        blocks must not be guarded on substrings of that -- a block would fire whenever some
        other tool's help text mentioned the name. A guard needs the list, not the paragraph.
        """
        return sorted(n for n in self._tools if allow is None or n in allow)

    def describe(self, allow: list[str] | None = None, detailed: bool = False) -> str:
        """Human/LLM-readable tool listing for prompts.

        detailed=True emits args with types + required markers and the FULL docstring
        ("when to use" guidance) — used to instruct the structurer model.
        """
        lines = []
        for n, t in self._tools.items():
            if allow is not None and n not in allow:
                continue
            m = t._tool_meta
            schema = m["schema"]
            if detailed:
                params = _signature(schema, typed=True)
                doc = (m["description"] or "").strip()
                lines.append(f"- {n}({params})\n    {doc}" if doc else f"- {n}({params})")
            else:
                # THE WHOLE DESCRIPTION, NOT ITS FIRST LINE: a wrapped first sentence would reach
                # the agent cut in half, and everything after it -- what a duplicate does,
                # what a failing check shows -- would never reach it at all.
                params = _signature(schema)
                if lines:
                    lines.append("")  # one blank line between tools, none inside
                lines.append(f"- {n}({params})")
                for text in [m["description"]] + [
                    extra for needs, extra in (m.get("if_granted") or {}).items() if allow is None or needs in allow
                ]:
                    lines += [f"    {ln}" for ln in inspect.cleandoc(text or "").splitlines() if ln.strip()]
                # Per-parameter glosses, for the arguments whose value decides what is recorded.
                # See the `params` note on @tool: without these the model guesses.
                for pname, gloss in (m.get("params") or {}).items():
                    text = _gloss_for(gloss, allow)
                    if text:
                        lines.append(f"    {pname}: {text}")
        if lines:
            lines.insert(0, LISTING_HEADER)  # stated once, before the first tool
        return "\n".join(lines)
