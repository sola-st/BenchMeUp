"""Built-in, environment-agnostic tools.

Mutating tools (write/apply/insert/set_environment/checkpoint/rollback) check
ctx.env.spec.read_only and refuse when the agent is in read-only mode (PR review).
File ops prefer a host workspace path; if the env exposes put_file/get_file
(DockerEnvironment), those are used instead.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any

from agentcore.tools.context import ToolContext
from agentcore.tools.registry import tool
from agentcore.tools.result import ToolResult


def _read_only(ctx: ToolContext) -> bool:
    return bool(getattr(getattr(ctx.env, "spec", None), "read_only", False))


def _host_root(ctx: ToolContext) -> str | None:
    ws = getattr(ctx.env, "workspace", None)
    return ws if ws and os.path.isdir(ws) else None


def _resolve_host(ctx: ToolContext, path: str) -> str:
    root = _host_root(ctx)
    if os.path.isabs(path) or root is None:
        return path
    return os.path.join(root, path)


# ── shell ─────────────────────────────────────────────────────────────────────


_STDIN_MARKERS = ('file "<stdin>"', "file '<stdin>'", "<stdin>, line")
_SYNTAX_MARKERS = ("syntaxerror", "indentationerror", "taberror")


def _piped_script_hint(command: str, returncode: int, body: str) -> str:
    """When a PIPED script fails to parse, say why it cannot be patched.

    `write_to_file` already answers this well for a script on disk: it names str_replace and
    hands over the exact bytes of the bad line. A heredoc has no such answer, because nothing
    was ever written -- `File "<stdin>"` means the source existed only in the pipe. The only
    repair the model can see from there is to regenerate the whole script and re-pipe it, which
    re-rolls the dice on every other line to fix one.

    Deliberately narrow: it fires only when the interpreter itself reported a PARSE error
    against stdin. A runtime error is not fixed by moving the script to a file, and a script
    that already lives on disk has a file to patch -- pointing at write_to_file there would be
    advice to create something the model already has.
    """
    low = (body or "").lower()
    if returncode == 0:
        return ""
    if not any(m in low for m in _SYNTAX_MARKERS):
        return ""
    if not any(m in low for m in _STDIN_MARKERS):
        return ""
    return (
        "\n\n--- HOW TO FIX THIS ---\n"
        "You piped this script in, so it was never written anywhere: `<stdin>` is the pipe, "
        "and there is no file on disk holding those lines. That is why there is nothing to "
        "patch, and it is why rewriting the whole script from scratch looks like the only "
        "option -- it is not, and it will re-roll every other line to fix this one.\n"
        "Write the script to a file with write_to_file, then run that file. The syntax is "
        "checked as you write, and after that a single bad line is one str_replace away."
    )


@tool(
    "linux_terminal",
    params={
        # Conditional: the three lead prospectors grant no container tools at all, by design, and
        # their DELIVERED validator reads a clone on THIS machine -- so telling them the evidence
        # belongs elsewhere contradicted the gate they are held to. See `_gloss_for` in registry.py.
        "command": {
            "text": "shell command; runs on YOUR machine -- candidate evidence belongs in "
            "that candidate's sandbox via container_exec, never here",
            "needs": "container_exec",
            "without": "the shell command to run on your own machine",
        },
        "cwd": "directory to run in (optional)",
        "timeout": "seconds before the command is killed (optional)",
    },
    aliases={"command": ["cmd", "shell_command", "script", "bash"]},
    # first_error, not head_tail: this runs builds and tests, and head_tail keeps the
    # prologue and the epilogue while discarding the middle -- which is where the
    # compiler error is.
    truncation="first_error",
    # The sandbox paragraphs only for an agent that has sandboxes: the search agents and the
    # construction workers have none, and "THIS IS NOT THE SAME MACHINE AS container_exec"
    # would be misleading to them.
    if_granted={
        "launch_background": """
    A job longer than you want to wait for in one command goes to launch_background.""",
        "container_exec": """
    This is not the same machine as container_exec. Each sandbox you create with
    container_create is a separate container with its own filesystem, and nothing is shared
    between them or with here. `/workspace/foo` here and `/workspace/foo` in a sandbox are two
    different directories with the same name: a file written in one is not in the other, and
    neither command tells you so.

    Use linux_terminal for work that is not tied to one candidate: searching, fetching metadata,
    writing your scripts. Clone, build and test a candidate in that candidate's sandbox with
    container_exec: its evidence is checked inside the sandbox you recorded, and work done here
    is not found there.""",
    },
)
def linux_terminal(ctx: ToolContext, command: str, cwd: str = "", timeout: int = 0) -> ToolResult:
    """Run a shell command on your own machine and return its exit code and its output
    (standard output, then standard error).

    The command is killed when its timeout runs out; pass a timeout in seconds for a long
    command. Long output is shortened before you see it, and the reply says what was cut: send
    long output to a file and read the part you need (grep, head, tail)."""
    if isinstance(command, list):
        # The argv spelling (['bash','-lc','...']) -- unambiguous, so resolved rather
        # than mangled: un-normalized it leaves bash waiting on stdin until the timeout
        # kills it.
        # Same class as flat key dicts and dict-valued digests.
        parts = [str(x) for x in command]
        if len(parts) >= 3 and parts[0] in ("bash", "sh") and parts[1] in ("-c", "-lc"):
            command = " ".join(parts[2:])
        else:
            import shlex

            command = " ".join(shlex.quote(x) for x in parts)
    try:
        res = ctx.env.execute(command, cwd=cwd, timeout=timeout or None)
    except Exception as e:  # noqa: BLE001 - an unreachable environment is a refusal, not a crash
        # `container_exec` catches exactly this too (containers.py: "Exec in sandbox
        # {name!r} failed"); otherwise anything DockerEnvironment.execute raised would go out
        # through registry.py's last-resort net as `Tool 'linux_terminal' failed: <ExcName>:
        # <docker URL>`. Name the machine that is unreachable and say what recovers it.
        from agentcore.environments.exec_supervisor import unreachable_reason

        return ToolResult.fail(
            f"Your own environment could not run that command "
            f"({unreachable_reason(e)}). Nothing ran. This is the MAIN "
            f"MACHINE being unreachable, not your command failing -- retrying the same "
            f"command will not fix it. Candidate work belongs in that candidate's "
            f"sandbox via container_exec, which has its own container."
        )
    body = res.output
    if res.stderr:
        body += "\n[stderr]\n" + res.stderr
    body += _piped_script_hint(command, res.returncode, body)
    # Say WHERE this ran, on every single output. The model's only other clue was the absence of
    # a `[name]` prefix, which is not a signal anyone notices: runs cloned the same repo on this
    # machine and in a sandbox, at the same path, and could not tell the two apart afterwards.
    # A banner costs one line and is present at the exact moment the mistake is being made.
    return ToolResult(
        display_text=(
            f"=== RAN ON: MAIN MACHINE (your own environment -- NOT any candidate "
            f"sandbox) ===\n(exit {res.returncode}) {body}"
        ),
        structured={"returncode": res.returncode, "timed_out": res.timed_out, "ran_on": "main"},
    )


# ── files ──────────────────────────────────────────────────────────────────────


@tool(
    "read_file",
    truncation="none",
    idempotent_ok=True,
    params={"file_path": "the file to read, absolute or relative to your working directory"},
    aliases={"file_path": ["path", "file", "filename"]},
    if_granted={
        "container_exec": """
    It reads your own machine only; a file inside a sandbox is read with container_exec."""
    },
)
def read_file(ctx: ToolContext, file_path: str) -> ToolResult:
    """Return the whole text of a file on your own machine.

    Nothing is cut, however long the file is: for a large file, read the part you need with a
    command instead (grep, head, sed -n)."""
    if hasattr(ctx.env, "get_file"):
        try:
            return ToolResult.text(ctx.env.get_file(file_path))  # type: ignore[attr-defined]
        except Exception:
            pass
    host = _resolve_host(ctx, file_path)
    if os.path.exists(host):
        with open(host, errors="replace") as f:
            return ToolResult.text(f.read())
    try:
        res = ctx.env.execute(f"cat -- {file_path}")
    except Exception as e:  # noqa: BLE001 - same boundary rule as linux_terminal
        from agentcore.environments.exec_supervisor import unreachable_reason

        return ToolResult.fail(
            f"Could not read {file_path}: your own environment is unreachable "
            f"({unreachable_reason(e)}), so nothing was read. A file inside a "
            f"candidate's sandbox is read with container_exec, not from here."
        )
    if res.returncode == 0:
        return ToolResult.text(res.output)
    return ToolResult.fail(f"Could not read {file_path}: {res.stderr or res.output}")


@tool(
    "write_to_file",
    params={
        "path": "the file to write, absolute or relative to your working directory",
        "content": "the complete new text of the file -- it is overwritten, not merged",
    },
    aliases={"path": ["file_path", "file", "filename"], "content": ["text", "contents", "data", "body"]},
    if_granted={
        "str_replace": """
    To change part of a file, use str_replace.""",
        "container_exec": """
    It writes your own machine only; a file inside a sandbox is not reachable from here.""",
    },
)
def write_to_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    """Write a text file on your own machine, creating missing directories.

    The file is replaced as a whole: whatever it held before is gone. After a .py file is
    written it is compiled, and a syntax error is reported in the reply."""
    # If the filename is configured in env.build_on_write (e.g. "Dockerfile"), writing it
    # auto-triggers an image build and (re)enters the resulting container -- the build log is
    # returned so the model can retry on failure (LLM-authored-Dockerfile flow). The agents with
    # such a file are told so by their brief, where the file is named.
    if _read_only(ctx):
        return ToolResult.fail("Environment is read-only; write_to_file is disabled.")
    if hasattr(ctx.env, "put_file"):
        try:
            ctx.env.put_file(path, content)  # type: ignore[attr-defined]
        except Exception as e:
            return ToolResult.fail(f"write failed: {e}")
    else:
        host = _resolve_host(ctx, path)
        os.makedirs(os.path.dirname(host) or ".", exist_ok=True)
        with open(host, "w") as f:
            f.write(content)

    build_msg = _maybe_build_on_write(ctx, path, content)
    if build_msg is not None:
        return build_msg
    text = f"Wrote {len(content)} bytes to {path}"
    syntax_note = _maybe_check_python_syntax(ctx, path, content)
    if syntax_note:
        text += f"\n{syntax_note}"
    return ToolResult.text(text)


def _maybe_check_python_syntax(ctx: ToolContext, path: str, content: str = "") -> str | None:
    """Free, automatic syntax check on every .py write. A real, recurring failure mode
    (a literal '\\n' inside a JSON string decodes to an actual newline, splitting a
    Python string literal across two lines) kept recurring across multiple collectors
    THIS SESSION despite an explicit prompt warning about it -- prose alone isn't
    reliable for a weak model to remember every time. Catching it here means the model
    sees the real compiler error immediately after writing, not several turns later
    after wasting a full run attempt on it."""
    if not path.endswith(".py") or not hasattr(ctx.env, "execute"):
        return None
    try:
        res = ctx.env.execute(f"python3 -m py_compile {shlex.quote(path)}")
    except Exception:
        return None
    if res.returncode == 0:
        return None
    detail = ((res.output or "") + ("\n" + res.stderr if res.stderr else ""))[:2000]

    # Catching the error is only half of it: told only the compiler's message, a model tends to
    # rewrite the ENTIRE file under a new name. So say the two
    # things that make the cheap move visible: the file IS on disk, and one line is what needs
    # changing.
    guidance = (
        "\nThe file WAS written and is on disk -- only its syntax is wrong. Fix THAT "
        "LINE with str_replace; do not rewrite the whole file, which regenerates every "
        "other line and tends to reintroduce the same mistake somewhere else."
    )

    # Hand over the EXACT bytes of the offending line. The compiler echoes the line too, but
    # re-indented to 4 spaces -- a line that really has 8 comes back with 4, and `str_replace`
    # matches exact text, so copying the echo yields "not found" and the model rewrites the
    # file after all. The one place that knows the true bytes is right here: we just wrote them.
    lineno = _syntax_error_line(detail)
    lines = (content or "").splitlines()
    if lineno and 1 <= lineno <= len(lines):
        exact = lines[lineno - 1]
        guidance += (
            f"\nLine {lineno} reads EXACTLY this (copy it between the markers as "
            f"str_replace's `old` -- note the leading whitespace, which the compiler's "
            f"echo above does not preserve):\n<<<{exact}>>>"
        )
    if "unterminated" in detail.lower() or "was never closed" in detail.lower():
        # The recurring cause, named at the moment it happens rather than in a prompt read once.
        guidance += (
            "\nAn unterminated literal here is almost always a newline that ended up "
            "INSIDE a string: a literal \\n written in the JSON you sent decodes to a "
            "real line break. Use \\\\n for a newline inside the code, or build the "
            "string with a triple-quoted literal."
        )
    return f"[auto syntax check] python3 -m py_compile {path} FAILED:\n{detail}{guidance}"


def _syntax_error_line(detail: str) -> int | None:
    """The line number py_compile blamed, or None. Prefers the LAST match: the message ends
    with `(detected at line N)`, which for an unterminated literal is where the error was
    noticed, while the earlier `File "...", line N` is where the construct began -- and the
    beginning is the line that needs editing."""
    nums = re.findall(r"line (\d+)", detail)
    return int(nums[0]) if nums else None


def _maybe_build_on_write(ctx: ToolContext, path: str, content: str) -> ToolResult | None:
    """If `path` is a configured build trigger and the env can build, build + enter."""
    triggers = list(getattr(getattr(ctx.env, "spec", None), "build_on_write", None) or [])
    if os.path.basename(path) not in triggers or not hasattr(ctx.env, "build_and_start"):
        return None
    try:
        tag = ctx.env.build_and_start(content)  # type: ignore[attr-defined]
    except Exception as e:
        # Build failed — hand the log back so the model can fix the Dockerfile and retry.
        return ToolResult(
            display_text=f"Wrote {path}, but the image build FAILED. Fix the Dockerfile and write it again.\n\n{e}",
            error="docker build failed",
            structured={"build_ok": False},
        )
    return ToolResult(
        display_text=(
            f"Wrote {path}, built image '{tag}', and entered the container. "
            "You are now operating INSIDE the container; subsequent commands run there."
        ),
        structured={"build_ok": True, "image": tag},
    )


@tool(
    "insert_into_file",
    params={
        "path": "the file to append to, on YOUR OWN machine (the agent's environment) -- a file inside a sandbox is not reachable from here",
        "content": "the text to add at the end, verbatim",
    },
    aliases={"path": ["file_path", "file", "filename"], "content": ["text", "contents", "data", "body"]},
)
def insert_into_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    """Append content to a file without clobbering it (e.g. add tests)."""
    if _read_only(ctx):
        return ToolResult.fail("Environment is read-only; insert_into_file is disabled.")
    host = _resolve_host(ctx, path)
    os.makedirs(os.path.dirname(host) or ".", exist_ok=True)
    with open(host, "a") as f:
        f.write(("\n" if os.path.exists(host) else "") + content)
    return ToolResult.text(f"Appended {len(content)} bytes to {path}")


def _env_read(ctx: ToolContext, path: str) -> str | None:
    if hasattr(ctx.env, "get_file"):
        try:
            return ctx.env.get_file(path)  # type: ignore[attr-defined]
        except Exception:
            return None
    host = _resolve_host(ctx, path)
    if os.path.exists(host):
        with open(host, errors="replace") as f:
            return f.read()
    return None


def _env_write(ctx: ToolContext, path: str, content: str) -> None:
    if hasattr(ctx.env, "put_file"):
        ctx.env.put_file(path, content)  # type: ignore[attr-defined]
    else:
        host = _resolve_host(ctx, path)
        os.makedirs(os.path.dirname(host) or ".", exist_ok=True)
        with open(host, "w") as f:
            f.write(content)


def _env_delete(ctx: ToolContext, path: str) -> None:
    if hasattr(ctx.env, "put_file"):
        ctx.env.execute(f"rm -f -- {path}")
    else:
        try:
            os.remove(_resolve_host(ctx, path))
        except OSError:
            pass


@tool(
    "apply_patch",
    params={
        "patch": "the patch text itself, a unified diff; it is applied on YOUR OWN machine (the agent's environment) -- a file inside a sandbox is not reachable from here"
    },
    aliases={"patch": ["diff", "patch_text", "content"]},
)
def apply_patch(ctx: ToolContext, patch: str) -> ToolResult:
    """Apply a code patch to files in the environment (unified diff, or the replace-a-block format described when you call it wrongly).

    1. a unified diff (`git diff` / `diff -u`: `--- a/f` / `+++ b/f` headers + `@@` hunks), or
    2. an OpenAI `*** Begin Patch` envelope (`*** Update File:` / `*** Add File:` / `*** Delete File:`).

    Edit the SOURCE, not the tests. For a whole-file rewrite, use write_to_file instead.
    """
    if _read_only(ctx):
        return ToolResult.fail("Environment is read-only; apply_patch is disabled.")

    # Format 2: OpenAI '*** Begin Patch' envelope — apply natively (no git required).
    from agentcore.tools import patch as _patch

    if _patch.is_v4a(patch):
        ok, msg = _patch.parse_and_apply_v4a(
            patch,
            lambda p: _env_read(ctx, p),
            lambda p, c: _env_write(ctx, p, c),
            lambda p: _env_delete(ctx, p),
        )
        return ToolResult(
            display_text=("patch applied: " + msg) if ok else ("patch failed: " + msg),
            structured={"format": "v4a", "ok": ok},
            error=None if ok else "apply_patch failed",
        )

    # Format 1: unified diff — write it and apply in-env via git/patch.
    tmp = ".agentcore.patch"
    _env_write(ctx, tmp, patch)
    res = ctx.env.execute(f"git apply --whitespace=nowarn {tmp} || patch -p1 < {tmp}")
    if res.returncode == 0:
        return ToolResult(display_text="patch applied", structured={"format": "unified", "returncode": 0})
    return ToolResult(
        display_text=f"patch did not apply: {res.stderr or res.output}",
        structured={"returncode": res.returncode},
        error="apply_patch failed",
    )


@tool(
    "str_replace",
    params={
        "path": "the file to edit on your own machine, absolute or relative to your working directory",
        "old": "the exact text to find, verbatim with its indentation; it must occur in the file",
        "new": "the text that replaces it, verbatim",
    },
    aliases={
        "path": ["file_path", "file", "filename"],
        "old": ["old_str", "old_string", "search", "find", "before"],
        "new": ["new_str", "new_string", "replacement", "replace", "after"],
    },
    if_granted={
        "apply_patch": """
    For changes in several places at once, apply_patch takes a unified diff."""
    },
)
def str_replace(
    ctx: ToolContext, path: str, old: str, new: str, replace_all: bool = False, occurrence: int = 0
) -> ToolResult:
    """Replace one exact piece of text in a file with another; the rest of the file is unchanged.

    `old` must match the file exactly, spaces and indentation included. By default it must occur
    once: when it occurs more often, include enough surrounding text to make it unique, or pass
    replace_all=true to replace every occurrence, or occurrence=N to replace the Nth (counting
    from 1). After a .py file is edited it is compiled, and a syntax error is reported in the
    reply."""
    if _read_only(ctx):
        return ToolResult.fail("Environment is read-only; str_replace is disabled.")
    if not old:
        return ToolResult.fail("`old` must be a non-empty string.")
    content = _env_read(ctx, path)
    if content is None:
        return ToolResult.fail(f"Could not read {path}")
    count = content.count(old)
    if count == 0:
        return ToolResult.fail(
            f"No exact match for `old` in {path}. Check whitespace/indentation and copy the text verbatim."
        )

    if replace_all:
        updated, n = content.replace(old, new), count
    elif occurrence:
        if not 1 <= occurrence <= count:
            return ToolResult.fail(f"occurrence {occurrence} out of range (found {count} match(es) in {path})")
        idx, start = -1, 0
        for _ in range(occurrence):
            idx = content.find(old, start)
            start = idx + len(old)
        updated, n = content[:idx] + new + content[idx + len(old) :], 1
    elif count > 1:
        return ToolResult.fail(
            f"`old` occurs {count} times in {path}; add surrounding context to make it unique, "
            f"or pass replace_all=true / occurrence=N."
        )
    else:
        updated, n = content.replace(old, new, 1), 1

    _env_write(ctx, path, updated)
    text = f"Replaced {n} occurrence(s) in {path}"
    syntax_note = _maybe_check_python_syntax(ctx, path)
    if syntax_note:
        text += f"\n{syntax_note}"
    return ToolResult(display_text=text, structured={"replaced": n, "path": path})


# ── environment control ─────────────────────────────────────────────────────


@tool(
    "set_environment",
    params={
        "image": {
            "text": "docker image to switch YOUR OWN environment to (not a candidate "
            "sandbox; those use container_create)",
            "needs": "container_create",
            "without": "docker image to switch YOUR OWN environment to",
        }
    },
)
def set_environment(ctx: ToolContext, image: str) -> ToolResult:
    """Switch/(re)build the execution environment to a new base image (mid-run)."""
    if _read_only(ctx):
        return ToolResult.fail("Environment is read-only; set_environment is disabled.")
    spec = getattr(ctx.env, "spec", None)
    if spec is None:
        return ToolResult.fail("Environment does not support reprovisioning.")
    new = spec.model_copy(update={"image": image, "type": "docker"})
    ctx.env.reprovision(new)
    return ToolResult.text(f"Environment reprovisioned to image {image}")


@tool("checkpoint")
def checkpoint(ctx: ToolContext, label: str = "ckpt") -> ToolResult:
    """Snapshot the environment; store the handle in the working set scratch."""
    handle = ctx.env.checkpoint()
    ctx.scratch.setdefault("checkpoints", {})[label] = handle
    return ToolResult.text(f"checkpoint '{label}' -> {handle or '(noop)'}")


@tool("rollback")
def rollback(ctx: ToolContext, label: str = "ckpt") -> ToolResult:
    """Restore a previously taken checkpoint by label."""
    handle = ctx.scratch.get("checkpoints", {}).get(label)
    if not handle:
        return ToolResult.fail(f"No checkpoint labeled {label!r}")
    ctx.env.restore(handle)
    return ToolResult.text(f"rolled back to '{label}'")


# ── long-running jobs ───────────────────────────────────────────────────────


@tool(
    "launch_background",
    params={"command": "the shell command to start in the background; its output goes to logfile"},
    aliases={"command": ["cmd", "script"]},
)
def launch_background(ctx: ToolContext, command: str, logfile: str = "bg.log") -> ToolResult:
    """Start a shell command in the background on your own machine and return at once.

    Its output (standard output and standard error) goes to logfile (bg.log unless you name
    another), and its process id to <logfile>.pid. Follow the job with poll_process, giving it
    the same logfile. For several commands in a row, put them in a script file and launch the
    script."""
    ctx.env.execute(f"nohup {command} > {logfile} 2>&1 & echo $! > {logfile}.pid")
    ctx.scratch.setdefault("bg", {})[logfile] = True
    return ToolResult.text(f"launched in background; log -> {logfile}")


@tool(
    "poll_process",
    idempotent_ok=True,
    params={
        "wait": "seconds to wait for the job to finish before reporting (0: report at once); a long job then costs one turn instead of one per poll"
    },
)
def poll_process(ctx: ToolContext, logfile: str = "bg.log", tail: int = 40, wait: int = 0) -> ToolResult:
    """Report on a job started with launch_background: RUNNING or DONE, then the last lines of
    its log (tail lines, 40 unless you say otherwise).

    Give it the same logfile you gave launch_background. With wait=N it first waits up to N
    seconds for the job to end, so a long job costs one call instead of one per check."""
    wait = max(0, int(wait or 0))
    if wait:
        # One command, inside the job's own machine, that returns as soon as the job ends
        # or the wait elapses, so polling does not cost one model call per check.
        ctx.env.execute(
            f"i=0; while [ $i -lt {wait} ] && kill -0 $(cat {logfile}.pid 2>/dev/null) 2>/dev/null; "
            f"do sleep 5; i=$((i+5)); done",
            timeout=wait + 30,
        )
    alive = ctx.env.execute(f"kill -0 $(cat {logfile}.pid 2>/dev/null) 2>/dev/null && echo RUNNING || echo DONE")
    log = ctx.env.execute(f"tail -n {tail} {logfile} 2>/dev/null")
    state = alive.output.strip()
    return ToolResult(display_text=f"[{state}]\n{log.output}", structured={"running": state == "RUNNING"})


# ── environment introspection ───────────────────────────────────────────────────


@tool("env_info", truncation="head_tail", idempotent_ok=True)
def env_info(ctx: ToolContext, packages: bool = True) -> ToolResult:
    """Report the execution environment: OS, Python version, installed packages, workdir.
    and a top-level workspace map (per directory: recursive subdir/file counts). Use it to ground
    decisions in the actual environment instead of assuming."""
    from agentcore.context.env_fingerprint import fingerprint, summarize

    fp = fingerprint(ctx, packages=packages, files=True, system=True)
    return ToolResult(display_text=summarize(fp), structured=fp)


# ── result capture ────────────────────────────────────────────────────────────


def _preview(value: Any, n: int = 120) -> str:
    """Short display form of a recorded value -- value can be a string OR a structured
    dict/list (findings/decisions are often naturally structured), so don't assume str."""
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return s[:n]


# Working-set keys the HARNESS writes, which `record` must refuse.
#
# Several gates now read working-set state to decide whether a run may proceed: the calibrate
# exit reads `collector_tested`, the worklist holds per-candidate states, sandbox recipes drive
# environment rebuilds. Each is only meaningful because a harness wrote it from a real
# observation. `record` accepts an arbitrary key, so without this guard a model could satisfy any
# of those gates by writing the expected value directly -- exactly the "model declares its own
# state" failure the worklist exists to prevent, reintroduced through a side door.
HARNESS_OWNED_KEYS = {
    "collector_tested": "It is written by collector_test from the real pytest exit status.",
    "worklist": "Use worklist_add / worklist_record / worklist_drop instead.",
    "sandbox_recipes": "It is written by container_create.",
}


def _owned_keys(ctx) -> dict:
    """Harness-owned keys for THIS agent: the always-owned core plus anything its config derives.

    `calibration_result` is only owned when a worklist derives it (`derive_calibration_to`).
    Blocking it globally would break every agent that has no worklist, where `record` is the
    legitimate way to report it -- the fix has to be as narrow as the claim it protects.
    """
    owned = dict(HARNESS_OWNED_KEYS)
    # The source list. `record` overwrites, so a second pass at surveying could shrink the pool;
    # it also skips the deduplication and the admissibility clauses that `sources_add` applies,
    # which made it a way to put unvetted places on the list after the survey gate had run.
    ss = getattr(getattr(getattr(ctx, "agent", None), "spec", None), "sources", None)
    if ss is not None and getattr(ss, "enabled", False) and getattr(ss, "key", ""):
        owned[ss.key] = (
            "A list value is forwarded to `sources_add` when this stage grants it "
            "(it keeps what you already have, tells you which entries you already "
            "had, and checks each one against this benchmark's requirements); "
            "this stage does not, so the list is closed here."
        )
    w = getattr(getattr(getattr(ctx, "agent", None), "spec", None), "worklist", None)
    if w is not None and getattr(w, "enabled", False):
        if getattr(w, "derive_calibration_to", ""):
            owned[w.derive_calibration_to] = (
                "The harness writes it when a candidate completes every verified step of the "
                "worklist. Advance a real candidate instead of reporting one."
            )
        if getattr(w, "publish_to", ""):
            # Observed live: the harness published one verified candidate on entering wrap_up,
            # and 19 cycles later the model wrote final_sample=[] -- accepted -- so the run
            # finished with an empty deliverable while still holding a DELIVERED candidate.
            # Publishing is worthless if it can then be overwritten.
            owned[w.publish_to] = (
                "The harness writes it from the candidates that passed every verified step. "
                "To change what ships, advance or drop candidates in the worklist."
            )
    return owned


def _forward_source_list(ctx: ToolContext, key: str, value: Any) -> ToolResult | None:
    """`record` of the declared sources key, with a list value, IS a `sources_add` call --
    when the current stage grants sources_add. None when this is not that case."""
    spec = getattr(getattr(ctx, "agent", None), "spec", None)
    ss = getattr(spec, "sources", None)
    if ss is None or not getattr(ss, "enabled", False) or key != getattr(ss, "key", ""):
        return None
    # The allowlist that binds right now: the stage's, or the agent's global one when the
    # agent is unstaged. Only an agent that declares NO allowlist at all forwards unchecked.
    global_allow = list(getattr(spec, "tools", None) or [])
    stage = getattr(ctx, "stage", None)
    try:
        allow = stage.tools_allow(global_allow) if stage is not None else (global_allow or None)
    except Exception:  # noqa: BLE001 - an unreadable allowlist must not widen it
        allow = global_allow or None
    if allow is not None and "sources_add" not in allow:
        return None
    # `value` is typed Any, so a list arrives as its JSON text at least as often as decoded
    # (registry._coerce only decodes for parameters declared list/dict). Read it here.
    if isinstance(value, str) and value.strip()[:1] in ("[", "{"):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    entries = value if isinstance(value, list) else ([value] if isinstance(value, dict) else None)
    if entries is None:
        # This stage grants sources_add and this IS its list, so the refusal must not send the
        # model elsewhere: it is the value's shape that is wrong.
        return ToolResult.fail(
            f"{key!r} is the source list, and a source list is a JSON list of entry objects "
            f"-- this value is a {type(value).__name__}. Send the entries as a list (it is "
            f"forwarded to `sources_add`, which grows the list), or call `sources_add` "
            f"directly."
        )
    # AN EMPTY LIST BANKS THE STATE. The survey rhythm tells the model to record "the
    # entries so far, or the empty list [] if you have none yet" so its search window
    # reopens; sources_add refuses an empty list ("must be a non-empty list"), and 15 of the
    # 40 `record source_pool` calls in the current era were exactly that. Nothing to add is a
    # fact, not a mistake: say so and let the record stand as the banking action it was.
    if not entries:
        return ToolResult(
            display_text=(
                f"{key!r}: nothing to add -- the list is unchanged "
                f"({len(getattr(ctx.working_set, key, None) or [])} entries). "
                f"That is banked; keep searching and add entries with `sources_add` "
                f"as you find them."
            ),
            structured={key: list(getattr(ctx.working_set, key, None) or [])},
        )
    from agentcore.tools.source_tools import sources_add

    res = sources_add(ctx, entries)
    return ToolResult(
        display_text=(
            f"`record` of {key!r} is `sources_add` -- forwarded there, since the "
            f"list grows rather than being overwritten. {res.display_text}"
        ),
        structured=res.structured,
        error=res.error,
    )


@tool(
    "record",
    idempotent_ok=True,
    params={
        "key": "name of the finding (snake_case); becomes a key in result.json",
        "value": "the finding itself -- strings, numbers, lists and dicts all keep their type",
        "append": "true to append to a list under this key instead of overwriting",
    },
    aliases={"key": ["name", "field"], "value": ["val", "data", "content"]},
)
def record(ctx: ToolContext, key: str, value: Any, append: bool = False) -> ToolResult:
    """Capture a result/finding into the working set (persists to result.json + the report).

    `value` may be a string OR a structured JSON value (dict/list/number/bool) -- use whichever
    fits the finding. Use this to record what the run produced — a metric, a finding, a
    decision — instead of leaving it only in prose. append=true accumulates into a list under
    `key` (e.g. one entry per finding); otherwise it sets/overwrites `key`.
    """
    # THE SOURCE LIST: RESOLVE, DO NOT REFUSE. A model may write
    # `record key="source_pool" value=[...]` with a correct list of vetted entries (a generated
    # survey prompt can tell it to), and answering "use sources_add instead" would be a
    # refusal of a correct value. The value is exactly what sources_add
    # takes, so it is handed over: the stricter tool (grow + dedupe + admissibility) runs on
    # it, which weakens nothing. Guarded by the stage's allowlist so a stage that does not
    # grant sources_add cannot reach it through record's side door.
    forwarded = _forward_source_list(ctx, key, value)
    if forwarded is not None:
        return forwarded
    owned = _owned_keys(ctx)
    if key in owned:
        return ToolResult.fail(
            f"{key!r} is maintained by the harness and cannot be written with `record`. {owned[key]}"
        )
    ws = ctx.working_set
    if append:
        cur = getattr(ws, key, None)
        lst = list(cur) if isinstance(cur, list) else ([] if cur is None else [cur])
        lst.append(value)
        ws.merge({key: lst})
        return ToolResult(display_text=f"recorded {key}[{len(lst) - 1}] = {_preview(value)}", structured={key: lst})
    ws.merge({key: value})
    return ToolResult(display_text=f"recorded {key} = {_preview(value)}", structured={key: value})


# ── completion ──────────────────────────────────────────────────────────────


@tool("goals_accomplished", finishes=True)
def goals_accomplished(ctx: ToolContext, reason: str = "") -> ToolResult:
    """Declare the task complete and successful (ends the run)."""
    return ToolResult(
        display_text=reason or "goals accomplished",
        finished=True,
        structured={"success_declared": True, "reason": reason},
    )


@tool("submit", finishes=True)
def submit(ctx: ToolContext, summary: str = "") -> ToolResult:
    """Submit the produced artifact (working_set) and end the run."""
    return ToolResult(
        display_text=summary or "submitted",
        finished=True,
        structured={"submitted": True, "working_set": ctx.working_set.snapshot()},
    )


def builtin_module():
    """Return this module so a ToolRegistry can register_module(builtin_module())."""
    import sys

    return sys.modules[__name__]
