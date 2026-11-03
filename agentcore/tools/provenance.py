"""What an instance's environment actually was, recorded by the harness as it happens.

WHY. An agent-written `env_recipe: "ubuntu:22.04, OpenJDK 11, Maven 3.6.3"` is well-formed and
not reproducible -- the base image has no Maven; the agent apt-installed it and recorded where
it STARTED rather than how it got there. The harness therefore records the environment itself.

Four pieces, deliberately layered so the weakest one cannot undermine the strongest:

    exec trace     every (command, output) run in a sandbox, in order. Complete by
                   construction -- no heuristic decides what counts as "provisioning", because
                   every heuristic misses the `curl | tar` case.
    image digest   the tag resolved at create time. `gcc:latest` is a moving pointer; the digest
                   is not.
    snapshot       `docker commit` of the container an instance was validated in. This is the
                   AUTHORITATIVE artifact: it captures the ad-hoc provisioning that no recipe
                   records, and needs nothing from the agent.
    Dockerfile     DOCUMENTARY, and labelled as such in the file itself. Reconstructed from the
                   digest plus the trace, it is for a human to read and adapt. It is not a
                   guarantee that `docker build` reproduces anything, and saying otherwise would
                   be the same overstatement this module exists to fix.

The ordering matters: if the Dockerfile is wrong, the snapshot still runs.
"""

from __future__ import annotations

import re
import shlex

TRACE_KEY = "sandbox_exec_trace"
SNAPSHOT_KEY = "sandbox_snapshots"
SNAPSHOT_FAIL_KEY = "sandbox_snapshot_failures"
# Sandboxes whose trace hit MAX_TRACE. The cap is fine; pretending the trace is complete is not
# -- a delivered record's Dockerfile is reconstructed from this trace, and one mining sandbox
# shared by ~20 boca candidates fills it long before the last of them is delivered.
TRACE_FULL_KEY = "sandbox_trace_truncated"

# One command's output can be a whole Maven build. The trace is for reconstructing an
# environment, so what matters is which commands ran and roughly what they said -- keeping every
# byte would push megabytes of build spam into the working set and into every later prompt.
OUTPUT_KEEP = 2000
MAX_TRACE = 400


def _bag(ws, key: str) -> dict:
    bag = getattr(ws, key, None)
    if not isinstance(bag, dict):
        bag = {}
        setattr(ws, key, bag)
    return bag


def record_exec(ws, sandbox: str, command: str, output: str, code: int) -> None:
    """Append one (command, output) pair to this sandbox's trace.

    EVERY exec is recorded, not just the ones that look like provisioning. Detecting "env-mutating"
    commands by pattern (`apt|pip|make install`) is cheap and wrong: it misses `curl | tar`, a
    wget of a prebuilt binary, a `./configure` that writes a Makefile, an `update-alternatives`.
    The trace is the raw material; deciding what mattered can happen later, with the whole
    sequence in hand.
    """
    if not sandbox:
        return
    trace = _bag(ws, TRACE_KEY).setdefault(sandbox, [])
    if len(trace) >= MAX_TRACE:
        # Record THAT the trace is incomplete. Dropping commands silently made every later
        # candidate's provenance block a verbatim copy of the first candidates' provisioning,
        # presented as its own -- the exact inheritance trace_mark exists to prevent.
        _bag(ws, TRACE_FULL_KEY)[sandbox] = int(_bag(ws, TRACE_FULL_KEY).get(sandbox, 0)) + 1
        return
    text = str(output or "")
    trace.append(
        {
            "n": len(trace),
            "command": str(command or ""),
            "exit": int(code) if isinstance(code, int) else None,
            "output": text[:OUTPUT_KEEP],
            "output_truncated": len(text) > OUTPUT_KEEP,
        }
    )


def trace_for(ws, sandbox: str) -> list:
    return list(_bag(ws, TRACE_KEY).get((sandbox or "").strip()) or [])


def record_snapshot(
    ws, key: str, image_id: str, note: str = "", *, sandbox: str = "", trace_mark: int | None = None
) -> None:
    """Keyed by CANDIDATE, not by sandbox.

    Two delivered candidates can end up in one container despite the one-candidate-one-sandbox
    rule the prompt states. Keying snapshots by sandbox would capture only the first, and the
    image captured would not describe the second: the container goes on mutating between the
    two deliveries.

    `trace_mark` is how many commands had run when this candidate was delivered, so its slice of
    a shared trace can be recovered instead of inheriting the earlier candidate's work.
    """
    if key and image_id:
        _bag(ws, SNAPSHOT_KEY)[key] = {
            "image": image_id,
            "note": note,
            "sandbox": sandbox or "",
            "trace_mark": trace_mark,
        }


def snapshot_for(ws, key: str) -> dict | None:
    return _bag(ws, SNAPSHOT_KEY).get((key or "").strip())


def record_snapshot_failure(ws, key: str, reason: str, *, sandbox: str = "") -> None:
    """Why this candidate has no image.

    record_snapshot() stores nothing when the image id is empty -- an empty id is not a
    snapshot -- so without this a failed commit would leave no trace in the record, the state
    or the log. A failure is kept in its own bag so it can never be mistaken for a
    snapshot, and so a later retry is not blocked by it.
    """
    if key and reason:
        _bag(ws, SNAPSHOT_FAIL_KEY)[key] = {"reason": str(reason)[:2000], "sandbox": sandbox or ""}


def snapshot_failure_for(ws, key: str) -> dict | None:
    return _bag(ws, SNAPSHOT_FAIL_KEY).get((key or "").strip())


# ── the image, pinned ───────────────────────────────────────────────────────────────────────

_DIGEST = re.compile(r"^[\w./-]+@sha256:[0-9a-f]{64}$")


def resolve_digest(runner, image: str) -> str:
    """`gcc:latest` -> `gcc@sha256:cde79a…`, or "" if it cannot be resolved.

    `runner` is a callable (cmd) -> (code, output), so this works against the host, a container
    helper, or a stub in tests.

    Returns "" rather than raising for a locally-built image: those have no RepoDigests at all
    (nothing published them), and that is precisely the case the snapshot covers instead.
    """
    img = (image or "").strip()
    if not img or _DIGEST.match(img):
        return img if _DIGEST.match(img) else ""
    code, out = runner("docker inspect --format '{{index .RepoDigests 0}}' " + shlex.quote(img))
    if code != 0:
        return ""
    for line in (out or "").splitlines():
        cand = line.strip().strip("'\"")
        if _DIGEST.match(cand):
            return cand
    return ""


# ── the documentary Dockerfile ──────────────────────────────────────────────────────────────

# Commands that only LOOK at the environment. Keeping them would bury the two or three lines
# that actually changed it, and a documentary file nobody reads is worth nothing.
_READ_ONLY = re.compile(
    r"^\s*(ls|cat|head|tail|grep|find|wc|stat|file|echo|pwd|which|test|\[|"
    r"git\s+(log|show|diff|status|rev-parse|cat-file|ls-files|branch)|"
    r"mvn\s+(-v|--version)|java\s+-version|python3?\s+--version|pip\s+(freeze|list)|"
    r"dpkg\s+-l|docker\s+inspect)\b"
)


# A redirection makes any command a write, whatever verb it starts with. Caught by building the
# emitted Dockerfile for real: `echo provisioned > /opt/marker.txt` was filtered as "read-only"
# because it starts with echo, the file was silently absent from the built image, and the
# Dockerfile still built cleanly -- documentation that omits the step you needed, with nothing
# to signal the omission.
_WRITES = re.compile(r"(?<![0-9<>])>{1,2}(?![&>])|\|\s*(tee|dd)\b|\btee\s+-a?\s*/")


def _changed_the_environment(command: str) -> bool:
    c = (command or "").strip()
    if not c:
        return False
    if _WRITES.search(c):
        return True
    return not _READ_ONLY.match(c)


def emit_dockerfile(recipe: dict, trace: list, *, image_digest: str = "", snapshot: str = "") -> str:
    """A readable reconstruction of how this environment came to be. NOT a guarantee.

    The header says so in the file itself, because a Dockerfile that silently fails to rebuild is
    worse than no Dockerfile: it invites someone to trust a build that never happened. The
    authoritative artifact is the snapshot image, named here so a reader knows where to go.
    """
    base = image_digest or recipe.get("built_image") or recipe.get("image") or "<unknown>"
    lines = [
        "# DOCUMENTARY -- reconstructed from the recorded exec trace, not a verified build.",
        "# `docker build` on this file may not reproduce the environment: read-only commands are",
        "# filtered out, output is truncated, and anything done outside a recorded exec is",
        "# invisible here. It is written to be READ.",
        "#",
        f"# Authoritative artifact: {snapshot or '(no snapshot recorded)'}",
        f"# Reconstructed from {len(trace)} recorded command(s).",
        "",
        f"FROM {base}",
    ]
    workdir = str(recipe.get("workdir") or "").strip()
    if workdir:
        lines.append(f"WORKDIR {workdir}")
    for cmd in recipe.get("setup") or []:
        if str(cmd).strip():
            lines.append(f"RUN {str(cmd).strip()}")
    if recipe.get("setup"):
        lines.append("")
        lines.append("# --- from the exec trace ---")
    kept = 0
    for step in trace:
        cmd = str(step.get("command") or "").strip()
        if not _changed_the_environment(cmd):
            continue
        kept += 1
        if step.get("exit") not in (0, None):
            lines.append(f"# exited {step.get('exit')} when it was run:")
            lines.append(f"# RUN {cmd}")
            continue
        lines.append(f"RUN {cmd}")
    if not kept:
        lines.append("# (no environment-changing commands were recorded)")
    return "\n".join(lines) + "\n"


def provenance_block(ws, sandbox: str, recipe: dict | None = None, *, candidate: str = "") -> dict:
    """Everything known about ONE CANDIDATE's environment, for embedding in its record.

    When candidates share a sandbox the trace is shared too, so a later candidate would otherwise
    inherit an earlier one's commands. The trace_mark recorded at delivery bounds the slice.
    """
    recipe = recipe or {}
    snap = snapshot_for(ws, candidate or sandbox) or {}
    trace = trace_for(ws, sandbox)
    mark = snap.get("trace_mark")
    if isinstance(mark, int):
        trace = trace[:mark]
    dropped = int(_bag(ws, TRACE_FULL_KEY).get(sandbox, 0) or 0)
    return {
        # Stated, not implied: with this set, the commands below are the FIRST MAX_TRACE run in
        # this sandbox and the rest were never recorded, so the Dockerfile is a partial replay.
        "trace_truncated": bool(dropped),
        "trace_dropped": dropped,
        "sandbox": sandbox,
        "candidate": candidate or "",
        "image": recipe.get("built_image") or recipe.get("image") or "",
        "image_digest": recipe.get("image_digest") or "",
        "snapshot_image": snap.get("image") or "",
        "setup": list(recipe.get("setup") or []),
        "exec_count": len(trace),
        "dockerfile": emit_dockerfile(
            recipe, trace, image_digest=recipe.get("image_digest") or "", snapshot=snap.get("image") or ""
        )
        + (
            f"# WARNING: this sandbox's command trace was TRUNCATED at {MAX_TRACE} commands; "
            f"{dropped} later command(s) were never recorded, so the steps above are a partial "
            f"replay of how this instance was built.\n"
            if dropped
            else ""
        ),
    }


def summarise(ws, sandbox: str) -> str:
    trace = trace_for(ws, sandbox)
    changed = sum(1 for s in trace if _changed_the_environment(str(s.get("command") or "")))
    snap = snapshot_for(ws, sandbox) or {}
    return f"{len(trace)} command(s) recorded, {changed} changed the environment" + (
        f", snapshot {snap['image'][:19]}" if snap.get("image") else ", no snapshot"
    )


def audit_delivered(ws, candidates: dict, terminal: str = "DELIVERED") -> dict:
    """End-of-run sweep: does every delivered instance actually have its own saved container?

    The per-candidate checks refuse a bad container when a candidate is validated, which is the
    right place to catch it. This is the belt-and-braces pass over the finished output, because
    the two failures seen in a real run were both invisible until someone went looking:

        5 instances recorded sandbox="pending" -- a placeholder, so nothing was ever saved
        3 instances shared one container, so 2 cite an image that never held their project

    Returns what is wrong rather than raising: a run that has already produced good instances
    must not lose them because one of them cannot be rebuilt. The caller reports it.

    `terminal` is the task's success-state name (WorklistSpec.terminal[0]); pass it when the
    caller has the spec in hand, or the audit silently skips every instance of a collector
    that renamed its success state.
    """
    ok, problems = [], []
    for key, cand in (candidates or {}).items():
        if cand.get("state") != terminal:
            continue
        fields = cand.get("fields") or {}
        sandbox = str(fields.get("sandbox") or "").strip()
        snap = snapshot_for(ws, key) or snapshot_for(ws, sandbox) or {}
        recipes = getattr(ws, "sandbox_recipes", None) or {}
        # The SNAPSHOT is the artifact; the recipe is bookkeeping. Checking the recipe first
        # reported 2 of 18 rebuildable on a run whose images I had personally opened and
        # verified -- containers are removed when a candidate finishes (there is a cap on how
        # many can be live), which drops the recipe while leaving the image perfectly good.
        if not sandbox:
            problems.append({"candidate": key, "why": "no sandbox recorded"})
        elif not snap.get("image"):
            why = f"no image saved for sandbox {sandbox!r}"
            if recipes and sandbox not in recipes:
                why += " -- and no such sandbox was ever created"
            problems.append({"candidate": key, "why": why})
        elif snap.get("sandbox") and snap["sandbox"] != sandbox:
            problems.append(
                {
                    "candidate": key,
                    "why": (f"the saved image came from sandbox {snap['sandbox']!r}, not this candidate's {sandbox!r}"),
                }
            )
        else:
            ok.append(key)
    return {"rebuildable": len(ok), "problems": problems, "total": len(ok) + len(problems)}
