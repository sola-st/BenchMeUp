"""Apply an OpenAI-style '*** Begin Patch' (V4A) envelope.

Many models (incl. GPT-style) emit patches in this format rather than a unified diff:

    *** Begin Patch
    *** Update File: path/to/file.py
    @@ [optional context anchor]
     context line
    -removed line
    +added line
    *** Add File: new.py
    +line 1
    *** Delete File: old.py
    *** End Patch

We parse it and apply hunks by locating the (context+removed) block in the file and
replacing it with the (context+added) block — no git needed. Returns (ok, message).
"""

from __future__ import annotations

from typing import Callable


def is_v4a(text: str) -> bool:
    return (
        "*** Begin Patch" in text or "*** Update File:" in text or "*** Add File:" in text or "*** Delete File:" in text
    )


def parse_v4a(text: str) -> list[dict]:
    ops: list[dict] = []
    cur: dict | None = None
    hunk: list[str] | None = None
    for ln in text.splitlines():
        if ln.startswith("*** Begin Patch") or ln.startswith("*** End Patch"):
            continue
        if ln.startswith("*** Update File:"):
            cur = {"op": "update", "path": ln.split(":", 1)[1].strip(), "hunks": []}
            ops.append(cur)
            hunk = None
        elif ln.startswith("*** Add File:"):
            cur = {"op": "add", "path": ln.split(":", 1)[1].strip(), "content": []}
            ops.append(cur)
            hunk = None
        elif ln.startswith("*** Delete File:"):
            ops.append({"op": "delete", "path": ln.split(":", 1)[1].strip()})
            cur = None
            hunk = None
        elif ln.startswith("@@"):
            if cur and cur["op"] == "update":
                hunk = []
                cur["hunks"].append(hunk)
        elif cur is None:
            continue
        elif cur["op"] == "add":
            cur["content"].append(ln[1:] if ln.startswith("+") else ln)
        elif cur["op"] == "update":
            if hunk is None:
                hunk = []
                cur["hunks"].append(hunk)
            hunk.append(ln)
    return ops


def _apply_hunk(file_lines: list[str], hunk: list[str]) -> list[str] | None:
    before = [ln[1:] for ln in hunk if ln[:1] in (" ", "-")]
    after = [ln[1:] for ln in hunk if ln[:1] in (" ", "+")]
    if not before:  # pure insertion with no anchor — can't place safely
        return None
    for start in range(0, len(file_lines) - len(before) + 1):
        if file_lines[start : start + len(before)] == before:
            return file_lines[:start] + after + file_lines[start + len(before) :]
    return None


def parse_and_apply_v4a(
    text: str,
    read_fn: Callable[[str], str | None],
    write_fn: Callable[[str, str], None],
    delete_fn: Callable[[str], None],
) -> tuple[bool, str]:
    ops = parse_v4a(text)
    if not ops:
        return False, "no file operations found in the patch"
    done = []
    for op in ops:
        path = op["path"]
        if op["op"] == "delete":
            delete_fn(path)
            done.append(f"deleted {path}")
            continue
        if op["op"] == "add":
            write_fn(path, "\n".join(op["content"]) + "\n")
            done.append(f"added {path}")
            continue
        content = read_fn(path)
        if content is None:
            return False, f"file not found: {path}"
        had_trailing_nl = content.endswith("\n")
        lines = content.split("\n")
        if had_trailing_nl and lines and lines[-1] == "":
            lines = lines[:-1]
        for hunk in op["hunks"]:
            new = _apply_hunk(lines, hunk)
            if new is None:
                return False, (
                    f"could not locate the patched context in {path} "
                    "(the removed/context lines did not match the file exactly)"
                )
            lines = new
        write_fn(path, "\n".join(lines) + ("\n" if had_trailing_nl else ""))
        done.append(f"updated {path}")
    return True, "; ".join(done)
