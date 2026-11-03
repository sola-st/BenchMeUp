"""How does a recorded field become a command a shell can run? Asked once, here.

MIRROR of `benchcompiler/render.py` -- the two packages deliberately do not
import each other. `benchcompiler/render_cases.py` is the rule and
`benchcompiler/tests/test_the_rendering_rule_is_one_rule.py` runs both implementations
against every case, so a divergence is a failing test rather than a silently different command.

WHY THIS EXISTS. The rule must have one answer. Divergent implementations differ in: one pass
versus a fixed point (a nested `{fix_patch}` left literal), argv versus a file (a diff
spread across a command line), how that file is written (a heredoc a patch can terminate),
a missing field left as braces for the shell versus named, and an unset working directory.

ORDER OF OPERATIONS. Reachable
field names are resolved WITHOUT substituting any values; the multi-line ones among them are
written into the container and their values replaced by paths; only then does expansion run.
Materialising first would miss fields reached through a nested template. Expanding first
would put file CONTENT into the command string, where a
`{name}` inside a patch would have been read as a placeholder.

`write_file(name, content) -> path` is injected because the two packages reach a container
differently -- one pipes through the runner's stdin, one base64s through its exec seam --
and that is the only thing about this operation that legitimately differs.
"""

from __future__ import annotations

import re

#: `{name}` but never `${name}`, which is a shell variable.
PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: How many expansion rounds before a value is treated as self-referential. A cycle must end
#: the gate, not hang it.
MAX_PASSES = 8


def referenced(template: str, fields: dict) -> tuple[list[str], list[str]]:
    """(reachable field names, unrecorded names), transitively, substituting nothing.

    Walking the graph rather than the rendered text is what lets a multi-line value be
    written to a file BEFORE its content could reach the command string.
    """
    seen: list[str] = []
    missing: list[str] = []
    frontier = list(PLACEHOLDER.findall(template or ""))
    while frontier:
        name = frontier.pop(0)
        if name in seen:
            continue
        if name not in fields:
            if name not in missing:
                missing.append(name)
            continue
        seen.append(name)
        val = fields[name]
        # A MULTI-LINE VALUE IS DATA, NOT A TEMPLATE. It is about to become a file, so its
        # content never reaches the command line and must not be scanned for names: a patch
        # containing `if (x) {cond}` would otherwise demand a field called `cond`. Only
        # single-line values are templates, which is what makes a nested
        # `{fix_patch}` expand while a diff's braces stay literal.
        if isinstance(val, str) and "\n" not in val.strip():
            frontier.extend(PLACEHOLDER.findall(val))
    return seen, missing


def substitute(template: str, fields: dict, strict: bool = True) -> str:
    """Expand `{name}` from ALREADY-RESOLVED fields, to a fixed point.

    The last step of `render_command`, exposed for the callers whose fields are already
    materialised (a verb's steps, a workdir, a measurement command). `strict` raises on a
    name the fields do not carry -- a brace reaching the shell is a silently wrong command,
    and the callers that want a reason instead of an exception call `render_command`.
    """
    out = "" if template is None else str(template)
    for _ in range(MAX_PASSES):
        missing: list[str] = []

        def _one(m):
            nm = m.group(1)
            if nm in fields:
                return str(fields[nm])
            missing.append(nm)
            return m.group(0)

        nxt = PLACEHOLDER.sub(_one, out)
        if nxt == out:
            if strict and missing:
                raise KeyError(sorted(set(missing)))
            return out
        out = nxt
    if strict:
        raise KeyError(f"expansion did not settle after {MAX_PASSES} passes (a cycle)")
    return out


def render_command(template: str, fields: dict, write_file=None) -> tuple[str | None, str]:
    """(command, error). A non-empty error means nothing may be run.

    `write_file(name, content) -> path` is called for every reachable field whose value spans
    lines. Omit it to refuse rather than materialise -- a caller with no way to put a file in
    the container must not silently spray one across argv instead.
    """
    template = "" if template is None else str(template)
    reach, missing = referenced(template, fields)
    if missing:
        return None, (
            f"references {sorted(set(missing))}, which this candidate has not "
            f"recorded. Every name a command templates must be a recorded field"
        )

    resolved = dict(fields)
    for name in reach:
        val = resolved.get(name)
        if isinstance(val, str) and "\n" in val.strip():
            if write_file is None:
                return None, (
                    f"{name!r} spans lines, and a value that spans lines can never "
                    f"be a shell argument. This caller has no way to write it into "
                    f"the container, so the command was not built"
                )
            try:
                resolved[name] = write_file(name, val)
            except Exception as e:  # noqa: BLE001 - the caller turns this into a refusal
                return None, (
                    f"could not write {name!r} into the container as a file "
                    f"({type(e).__name__}: {e}); the command was not built"
                )

    out = template
    for _ in range(MAX_PASSES):
        nxt = PLACEHOLDER.sub(lambda m: str(resolved.get(m.group(1), m.group(0))), out)
        if nxt == out:
            return out, ""
        out = nxt
    return None, (
        f"expanding this command did not settle after {MAX_PASSES} passes -- a "
        f"recorded value refers to itself (a cycle), so no command exists to run"
    )
