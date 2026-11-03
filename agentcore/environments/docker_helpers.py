"""Low-level docker helpers for the docker/deferred environments.

Lifted + unified from ExecutionAgent (docker_helpers_static.py, shared_utils.py)
and analysis_agent (legacy_tools.py). Everything that touches the docker SDK or
the daemon is connected LAZILY inside the functions, so importing this module
never requires the docker SDK to be importable nor a running daemon.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import tempfile
import threading
import time
import uuid
from typing import Any, Optional

# ── lazy docker import ────────────────────────────────────────────────────────
# We deliberately DO NOT import docker at module top-level in a way that would
# fail the import. The SDK is optional; the daemon is connected only inside
# get_docker_client(). DockerContainer is only a type alias for annotations.
try:  # pragma: no cover - import-time best effort
    from docker.models.containers import Container as DockerContainer  # type: ignore
except Exception:  # pragma: no cover
    DockerContainer = Any  # type: ignore


# ── constants ─────────────────────────────────────────────────────────────────

SCREEN_SESSION = os.environ.get("AGENTCORE_SCREEN_SESSION", "agentcore_screen")
RUN_DIR = os.environ.get("AGENTCORE_RUN_DIR", "/tmp/screen_runs")
LOG_DIR = os.environ.get("AGENTCORE_LOG_DIR", "/tmp")
STUCK_TIMEOUT_SECONDS = int(os.environ.get("AGENTCORE_STUCK_TIMEOUT", "300"))
POLL_INTERVAL_SECONDS = float(os.environ.get("AGENTCORE_POLL_INTERVAL", "0.5"))
MAX_TAIL_BYTES = int(os.environ.get("AGENTCORE_MAX_TAIL_BYTES", "2000000"))
DEFAULT_EXEC_TIMEOUT = int(os.environ.get("AGENTCORE_EXEC_TIMEOUT", "60"))
# Wall-clock ceiling of one image build (see build_image_from_dockerfile).
BUILD_TIMEOUT = int(os.environ.get("AGENTCORE_BUILD_TIMEOUT", "1800"))
# Above this many base64 characters a file is written as a tar archive, not on a command
# line (the kernel's argument limit is about 128 KB; see put_file).
PUT_FILE_INLINE_LIMIT = 64_000

# 25 GiB writable-layer soft limit per container (probe only; not enforced).
CONTAINER_DISK_LIMIT_BYTES = int(os.environ.get("AGENTCORE_DISK_LIMIT_BYTES", str(25 * 1024 * 1024 * 1024)))

# Docker containers do NOT inherit the host process's environment by default (unlike
# DeferredEnvironment's pre-container host-execution path, which does via
# `{**os.environ, **spec.env}`) -- a tool/script running inside a container that wants
# to call an LLM provider directly (e.g. a collector script doing its own cheap
# semantic pre-filtering before an expensive build+test) otherwise has no key at all.
# This is a narrow, explicit WHITELIST passthrough -- not the full host environment --
# so a generated script running with network access + shell access inside the
# container can't also exfiltrate unrelated host secrets (AWS creds, other tokens,
# etc.) it was never meant to see.
LLM_KEY_PASSTHROUGH_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    # Not an LLM key -- a fine-grained GitHub PAT. Passed through the same whitelist since a
    # collector's own script mining GitHub (issues/PRs/commits) hits the unauthenticated 60/hr
    # rate limit fast. Authenticated requests get 5000/hr.
    "GITHUB_TOKEN",
)


#: Every exec in a sandbox runs without a terminal, and the environment says so. Package
#: tooling that would otherwise prompt (debconf's tzdata "geographic area") sat in that
#: prompt for the whole exec ceiling with zero output and zero CPU. The harness's own
#: provisioning installs carry it (_install_cmds); this applies it to the model's commands and
#: the sandbox setup steps too. A sandbox fact, applied to every exec and to the container.
NO_TERMINAL_ENV = ["DEBIAN_FRONTEND=noninteractive"]


def llm_key_passthrough_env(explicit_env: dict[str, str]) -> dict[str, str]:
    """Merge whitelisted LLM-provider keys from the host's own environment underneath
    an agent's explicit `env.env` config (explicit config always wins on conflict)."""
    passthrough = {k: v for k in LLM_KEY_PASSTHROUGH_VARS if (v := os.environ.get(k))}
    return {**passthrough, **explicit_env}


ANSI_ESCAPE_RE = re.compile(r"(?:\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[@-Z\\-_])")
RC_MARKER_RE = re.compile(r"<<RC:[^:]+:(-?\d+)>>")


# ── docker client (lazy) ──────────────────────────────────────────────────────


def get_docker_client(timeout: Optional[int] = None):
    """Connect to the docker daemon. Imports the SDK lazily so this module can be
    imported without docker installed; raises a clear error only when actually used.

    `timeout` overrides docker-py's 60-second default for one client. Sixty seconds is
    fine for starting and inspecting containers and far too short for committing one;
    see COMMIT_TIMEOUT.
    """
    try:
        import docker  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Docker SDK not available. Install 'docker' (pip) and ensure the "
            f"Docker daemon is reachable. Original error: {e}"
        ) from e
    return docker.from_env() if timeout is None else docker.from_env(timeout=timeout)


# `docker commit` tars the container's whole writable layer, and docker-py's DEFAULT client
# timeout is 60 seconds, so a large layer (several GB) raises ReadTimeout. The commit is the
# artifact; it may take as long as it takes.
COMMIT_TIMEOUT = 1800


def commit_container(container, *, repository: str = "", tag: str = ""):
    """`docker commit` a container on a client that will wait for a large layer.

    Rebinds the container to a long-timeout client rather than raising the shared one's
    timeout: every other call stays snappy, and a hung commit still cannot wedge a run
    forever. Falls back to committing the object as given when it cannot be rebound (a
    stub container in tests, or a daemon that no longer knows the id).
    """
    target = container
    cid = getattr(container, "id", None)
    if isinstance(cid, str) and cid:
        try:
            target = get_docker_client(timeout=COMMIT_TIMEOUT).containers.get(cid)
        except Exception:
            target = container
    kwargs = {}
    if repository:
        kwargs["repository"] = repository
    if tag:
        kwargs["tag"] = tag
    try:
        return target.commit(**kwargs)
    except TypeError:  # a stub/fake container in tests
        return target.commit()


# ── text utilities ────────────────────────────────────────────────────────────


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE_RE.sub("", text or "")


def has_return_code_marker(text: str) -> bool:
    return bool(RC_MARKER_RE.search(text or ""))


def extract_return_code(text: str) -> Optional[int]:
    m = RC_MARKER_RE.search(text or "")
    return int(m.group(1)) if m else None


# ── the stall judge: env-armed, off by default ────────────────────────────────

DEFAULT_JUDGE_STALL_WINDOW = 60.0


def stall_judge_config() -> Optional[tuple[str, float]]:
    """(model name, stall window seconds) from the operator env, or None when unarmed.

    AGENT_JUDGE_MODEL is operator content the same way AGENT_MODEL is: absent means
    the judge is off and supervision uses the plain ceiling. The window defaults to a
    minute -- the point of the judge is that a hang dies there, not at the
    900-1800s ceiling. run_meta.json records an armed judge next to the run's
    models, because a config that can kill commands is provenance."""
    name = (os.environ.get("AGENT_JUDGE_MODEL") or "").strip()
    if not name:
        return None
    window = float(os.environ.get("AGENT_JUDGE_STALL_WINDOW") or DEFAULT_JUDGE_STALL_WINDOW)
    return name, window


def _judge_model(name: str):
    """The model behind the stall judge; the seam tests replace with a scripted one."""
    from agentcore.models.litellm_model import LitellmModel

    return LitellmModel(name, temperature=0.0)


def _stall_judge():
    """(judge, stall_window) for supervised_exec, or (None, None) when unarmed."""
    cfg = stall_judge_config()
    if cfg is None:
        return None, None
    name, window = cfg
    from agentcore.environments.exec_supervisor import make_model_judge

    return make_model_judge(_judge_model(name)), window


# ── container exec primitives ─────────────────────────────────────────────────


def container_user(container: "DockerContainer") -> str:
    """The user the container runs commands as (its image's USER), or "" for root/unset."""
    try:
        container.reload()
        user = str(((container.attrs or {}).get("Config") or {}).get("User") or "").strip()
    except Exception:  # noqa: BLE001 - daemon dependent
        return ""
    return "" if user in ("", "root", "0", "0:0", "root:root") else user


def exec_in_container(
    container: "DockerContainer",
    cmd: str,
    tty: bool = False,
    timeout: int = DEFAULT_EXEC_TIMEOUT,
    workdir: Optional[str] = None,
    capture_limit: Optional[int] = None,
) -> tuple[int, str]:
    """Run a command inside the container via an explicit ``/bin/sh -lc`` shell.

    ``capture_limit`` bounds how much of the command's output is RETAINED (None: the
    supervisor's default window; 0: everything). The harness's own reads of a file or a
    report pass 0 -- a window cut into the middle of a JUnit XML or a results JSON is a
    corrupted document, not a long one.

    Uses a worker thread to enforce a wall-clock timeout (docker-py's exec_run is
    blocking). Returns ``(exit_code, output)``; ``(124, msg)`` on timeout.

    With AGENTCORE_EXEC_SUPERVISED=1 the call is delegated to the exec supervisor:
    same signature and semantics on the happy path, but stalls are observed (output
    and CPU growth over the command's own processes), timeouts kill the command's
    footprint instead of abandoning it, and leftover processes are reported. The
    supervision story is appended to the output where there is one.

    With AGENT_JUDGE_MODEL also set, a zero-output zero-CPU stall past
    AGENT_JUDGE_STALL_WINDOW seconds (default 60) is put to that model, and a
    double-confirmed 'stuck' verdict kills the command early instead of burning
    the ceiling; verdict and reason are appended to the output with the story.
    """
    # SUPERVISED BY DEFAULT: the blind ceiling throws away the partial output and abandons
    # the process, which is what the supervisor exists to fix. Set
    # AGENTCORE_EXEC_SUPERVISED=0 for the blind ceiling.
    if os.environ.get("AGENTCORE_EXEC_SUPERVISED", "1") != "0":
        from agentcore.environments.exec_supervisor import render_report, supervised_exec

        judge, stall_window = _stall_judge()
        extra = {"judge": judge, "stall_window": stall_window} if judge is not None else {}
        code, out, report = supervised_exec(
            container, cmd, timeout=timeout, workdir=workdir, tty=tty, capture_limit=capture_limit, **extra
        )
        story = render_report(report)
        return code, (out + ("\n" + story if story else ""))

    result: dict[str, Any] = {"exit_code": None, "output": None, "error": None}

    def _run() -> None:
        try:
            kwargs: dict[str, Any] = {"tty": tty, "environment": list(NO_TERMINAL_ENV)}
            if workdir:
                kwargs["workdir"] = workdir
            res = container.exec_run(["/bin/sh", "-lc", cmd], **kwargs)
            out = res.output
            if isinstance(out, (bytes, bytearray)):
                out = out.decode("utf-8", errors="replace")
            else:
                out = str(out)
            result["exit_code"] = int(res.exit_code)
            result["output"] = out
        except Exception as e:  # pragma: no cover - daemon dependent
            result["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # THE FALLBACK, and it says what it actually did. docker-py's exec_run blocks, so
        # nothing here has the partial output, and the worker is a daemon thread -- the
        # command is still running inside the container after this returns. An agent whose
        # next move is "retry with a longer timeout" needs to know it is racing a process
        # that never stopped.
        preview = cmd[:100] + "..." if len(cmd) > 100 else cmd
        return 124, (
            f"Command timed out after {timeout} seconds: {preview}\n"
            f"(unsupervised exec: whatever it had printed is lost, and the command "
            f"was ABANDONED rather than stopped -- it is still running in this "
            f"container and still consuming its CPU. Unset "
            f"AGENTCORE_EXEC_SUPERVISED=0 to get partial output and a real kill.)"
        )
    if result["error"]:
        return 1, f"Error executing command: {result['error']}"
    return int(result["exit_code"] or 0), result["output"] or ""


def read_file_tail(container: "DockerContainer", path: str, max_bytes: int = MAX_TAIL_BYTES) -> str:
    q = shlex.quote(path)
    code, out = exec_in_container(container, f"if [ -f {q} ]; then tail -c {max_bytes} {q}; fi")
    return out if code == 0 else ""


# ── file transfer (tar / cp semantics, base64 fallback) ───────────────────────


def put_file(container: "DockerContainer", path: str, content: str) -> Optional[str]:
    """Write ``content`` to ``path`` inside the container.

    Uses base64 over a shell pipe (robust to arbitrary text, avoids tar plumbing
    and shell-escaping pitfalls). Creates parent dirs and verifies the write.
    Returns ``None`` on success or an error string on failure.
    """
    parent = os.path.dirname(path)
    if parent and parent != "/":
        code, out = exec_in_container(container, f"mkdir -p {shlex.quote(parent)}")
        if code != 0:
            return f"Failed to create parent directory {parent}: {out}"

    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    if len(encoded) > PUT_FILE_INLINE_LIMIT:
        # A shell command line has a length limit (about 128 KB of arguments), so a large
        # file cannot be written inline. Large content travels as a tar archive.
        import io
        import tarfile

        data = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=os.path.basename(path))
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        try:
            if not container.put_archive(parent or "/", buf.getvalue()):
                return f"Write (archive) failed at {path}"
        except Exception as e:  # noqa: BLE001 - reported like the inline path's failure
            return f"Write (archive) failed at {path}: {e}"
    else:
        payload = f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
        code, out = exec_in_container(container, payload)
        if code != 0:
            return f"Write command failed: {out}"

    vcode, vout = exec_in_container(container, f"test -f {shlex.quote(path)} && wc -c < {shlex.quote(path)}")
    if vcode != 0:
        return f"File verification failed at {path}: {vout}"
    try:
        if content and int(vout.strip()) == 0:
            return "File exists but appears empty (0 bytes); write may have failed."
    except ValueError:
        pass
    return None


def get_file(container: "DockerContainer", path: str) -> str:
    """Read a file from inside the container; returns its contents (or an error string)."""
    code, out = exec_in_container(container, f"cat {shlex.quote(path)}", tty=False, capture_limit=0)
    if code != 0:
        return f"Failed to read {path} in the container.\nOutput:\n{out}"
    return out


# ── image build ───────────────────────────────────────────────────────────────


def _sanitize_tag(raw: str) -> str:
    raw = raw.lower()
    return "".join(ch if (ch.isalnum() or ch in "._-:/") else "-" for ch in raw)


#: Build arguments every image build passes, as a JSON object in the environment. A
#: Dockerfile's `ARG` with a vendor default can then be pointed at a local mirror without
#: editing the instance (a vendor download that is blocked or rate-limited fails the build).
BUILD_ARGS_ENV = "BENCH_BUILD_ARGS"


def build_args_from_env() -> dict:
    raw = os.environ.get(BUILD_ARGS_ENV, "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in val.items()} if isinstance(val, dict) else {}


def build_image_from_dockerfile(
    text: str, tag: Optional[str] = None, *, pull: bool = True, nocache: bool = False, buildargs: Optional[dict] = None
) -> str:
    """Build an image from raw Dockerfile ``text`` written to a temp build context.

    Returns the resulting image tag (built fresh if ``tag`` is None). Raises
    RuntimeError with the build log tail on failure. ``buildargs`` default to the
    BENCH_BUILD_ARGS environment variable (a JSON object).
    """
    client = get_docker_client()
    if buildargs is None:
        buildargs = build_args_from_env()
    if not tag:
        tag = _sanitize_tag(f"agentcore_image:{int(time.time())}-{uuid.uuid4().hex[:8]}")
    else:
        tag = _sanitize_tag(tag)

    build_dir = tempfile.mkdtemp(prefix="agentcore_build_")
    with open(os.path.join(build_dir, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(text)

    log_lines: list[str] = [f"Starting build: tag='{tag}'"]
    # A build has a wall-clock ceiling. `apt-get install` on an image without
    # DEBIAN_FRONTEND=noninteractive waits on a tzdata prompt forever. The
    # build streams in a thread; past the ceiling it is reported as a failed build with the
    # log so far, and the daemon-side build is left to the daemon (it cannot be cancelled
    # through this API).
    import threading

    outcome: dict = {}

    def _stream():
        try:
            extra = {"buildargs": buildargs} if buildargs else {}
            for chunk in client.api.build(
                path=build_dir, tag=tag, rm=True, pull=pull, nocache=nocache, decode=True, **extra
            ):
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("stream"):
                    for ln in str(chunk["stream"]).rstrip("\n").splitlines():
                        log_lines.append(ln)
                if chunk.get("errorDetail") or chunk.get("error"):
                    detail = (chunk.get("errorDetail") or {}).get("message") or chunk.get("error") or ""
                    log_lines.append(f"ERROR: {detail}")
                    outcome["error"] = detail
            outcome["done"] = True
        except Exception as e:  # noqa: BLE001 - reported below, with the log
            outcome["exc"] = e

    t = threading.Thread(target=_stream, daemon=True)
    t.start()
    t.join(BUILD_TIMEOUT)
    if t.is_alive():
        tail = "\n".join(log_lines[-40:])
        raise RuntimeError(
            f"Docker build for tag {tag!r} did not finish within {BUILD_TIMEOUT} s and was abandoned. "
            f"A build cannot answer a prompt: a package install waiting for input (set "
            f"DEBIAN_FRONTEND=noninteractive and pass -y) is the usual cause. Log so far:\n{tail}"
        )
    try:
        if "exc" in outcome:
            raise outcome["exc"]
        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        client.images.get(tag)
        return tag
    except Exception as e:  # pragma: no cover - daemon dependent
        tail = "\n".join(log_lines[-60:])
        build_log = ""
        if hasattr(e, "build_log"):
            try:
                parts = []
                for entry in e.build_log:  # type: ignore[attr-defined]
                    if isinstance(entry, dict) and entry.get("stream"):
                        parts.append(str(entry["stream"]).rstrip("\n"))
                build_log = "\n".join(parts[-60:])
            except Exception:
                build_log = ""
        raise RuntimeError(f"Docker build failed for tag {tag!r}: {e}\n{tail}\n{build_log}") from e


def image_exists(tag: str) -> bool:
    try:
        get_docker_client().images.get(tag)
        return True
    except Exception:
        return False


# ── container lifecycle ───────────────────────────────────────────────────────


def parse_lifetime_seconds(lifetime: Any) -> int:
    """Parse a container_lifetime spec like '2h', '90m', '3600', 7200 -> seconds."""
    if lifetime is None:
        return 7200
    if isinstance(lifetime, (int, float)):
        return int(lifetime)
    s = str(lifetime).strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([smhd]?)$", s)
    if not m:
        return 7200
    val = float(m.group(1))
    unit = m.group(2) or "s"
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return int(val * mult)


#: Every container this harness starts carries this. "Remove what this project leaked" has to
#: be a filter rather than an archaeology exercise, and it must never be able to sweep up
#: work that is not ours.
PROJECT_LABEL = "com.benchcompiler.project"
PROJECT_LABEL_VALUE = "benchcompiler"

# WHICH RUN OWNS THIS CONTAINER. Nothing releases a sandbox when its owner goes away, and
# sandboxes share base images with live workers, so matching by image could kill an active
# candidate. The project label is on every container and does not say which run owns it.
# These two make "which containers belong to this run?" a filter.
RUN_LABEL = "com.benchcompiler.run"
SANDBOX_LABEL = "com.benchcompiler.sandbox"


def run_labels(run_dir: str, sandbox: str) -> dict:
    """Labels identifying the owner of a sandbox container.

    A label whose value is empty is worse than an absent one -- it matches the filter that is
    looking for real values -- so a missing piece is omitted. Values are flattened because
    docker rejects labels it cannot parse, and a run directory with a newline in it must not be
    able to stop a container from being created.
    """
    out = {}
    for key, value in ((RUN_LABEL, run_dir), (SANDBOX_LABEL, sandbox)):
        text = " ".join(str(value or "").split())
        if text:
            out[key] = text
    return out


def project_container_ids() -> set:
    """Ids of every container this project labelled, or an empty set if docker is unreachable.

    A daemon that is not there is not an error: most of the test suite does not need one.
    """
    try:
        client = get_docker_client()
        return {
            c.id for c in client.containers.list(all=True, filters={"label": f"{PROJECT_LABEL}={PROJECT_LABEL_VALUE}"})
        }
    except Exception:  # noqa: BLE001 - never a reason to fail a caller
        return set()


def remove_containers(ids) -> int:
    """Remove exactly these containers. Returns how many went. Never raises.

    The precise form: a caller that recorded what IT created can hand the ids back and cannot
    touch anything else -- which `reap_project_containers` below, matching on label + "newer
    than a snapshot", cannot promise while several sessions share a host.
    """
    ids = [i for i in (ids or []) if i]
    if not ids:
        return 0
    try:
        client = get_docker_client()
    except Exception:  # noqa: BLE001
        return 0
    removed = 0
    for cid in ids:
        try:
            client.containers.get(cid).remove(force=True)
            removed += 1
        except Exception:  # noqa: BLE001 - already gone, or never existed
            pass
    return removed


#: A test session marks the containers IT starts with this label (the session id from the
#: environment variable), and its reaper removes only those. A rule of "appeared since the
#: session began" would also remove the containers of live workers running beside the test
#: session.
TEST_SESSION_LABEL = "com.benchcompiler.test_session"
TEST_SESSION_ENV = "AGENTCORE_TEST_SESSION"


def test_session_container_ids(session_id: str) -> set:
    try:
        client = get_docker_client()
        return {c.id for c in client.containers.list(all=True, filters={"label": f"{TEST_SESSION_LABEL}={session_id}"})}
    except Exception:  # noqa: BLE001
        return set()


def reap_project_containers(before: set) -> int:
    """Remove the containers THIS test session started. Returns how many.

    A session is identified by AGENTCORE_TEST_SESSION, set by the suite's conftest before any
    container starts; `start_container` stamps it on every container the process starts. With
    no session id set, nothing is removed: a container that cannot be attributed to this
    session may be a live run's, and removing it destroys work in flight (see
    TEST_SESSION_LABEL). `before` is accepted for compatibility and ignored.

    Shared by both suites' conftest so the two cannot drift.
    """
    sid = os.environ.get(TEST_SESSION_ENV)
    if not sid:
        return 0
    leaked = test_session_container_ids(sid)
    if not leaked:
        return 0
    try:
        client = get_docker_client()
    except Exception:  # noqa: BLE001
        return 0
    removed = 0
    for cid in leaked:
        try:
            client.containers.get(cid).remove(force=True)
            removed += 1
        except Exception:  # noqa: BLE001 - already gone, or someone else's to remove
            pass
    return removed


def start_container(
    image: str,
    lifetime: Any = "2h",
    *,
    resources: Optional[dict] = None,
    env: Optional[dict] = None,
    workdir: Optional[str] = None,
    labels: Optional[dict] = None,
):
    """Start a long-lived container that idles via ``sleep <lifetime>`` (falls back to
    a tail loop). Maps a subset of the EnvSpec.resources dict onto docker run kwargs.

    On a host split for the measurement lane, a spec silent on `cpuset_cpus` starts on the
    sandbox cores (AGENTCORE_SANDBOX_CPUSET, see agentcore.measurement_lane); a timed
    execution moves the container to the measurement cores only while it holds the lane.
    A spec that declares its own cpuset keeps it.
    Returns the started container object."""
    client = get_docker_client()
    seconds = parse_lifetime_seconds(lifetime)
    resources = dict(resources or {})
    if "cpuset_cpus" not in resources:
        from agentcore.measurement_lane import sandbox_cpuset

        pin = sandbox_cpuset()
        if pin:
            resources["cpuset_cpus"] = pin

    run_kwargs: dict[str, Any] = {
        "detach": True,
        "tty": True,
        # keep alive for `seconds`; sleep handles huge values fine
        "command": ["/bin/sh", "-lc", f"sleep {seconds} 2>/dev/null || tail -f /dev/null"],
        # The caller may add its run and sandbox; the project label is not theirs to unset.
        # A test session's id, when the process runs under one, so its reaper can tell its
        # own containers from a live run's (see TEST_SESSION_LABEL).
        "labels": {
            **(labels or {}),
            PROJECT_LABEL: PROJECT_LABEL_VALUE,
            **({TEST_SESSION_LABEL: os.environ[TEST_SESSION_ENV]} if os.environ.get(TEST_SESSION_ENV) else {}),
        },
    }
    # The no-terminal fact underneath the caller's env: an explicit setting always wins.
    run_kwargs["environment"] = {**dict(kv.split("=", 1) for kv in NO_TERMINAL_ENV), **(dict(env) if env else {})}
    if workdir:
        run_kwargs["working_dir"] = workdir

    # selectively map known resource knobs
    for key in (
        "privileged",
        "cap_add",
        "cap_drop",
        "sysctls",
        "cpuset_cpus",
        "ulimits",
        "volumes",
        "mem_limit",
        "nano_cpus",
        "network",
        "network_mode",
        "shm_size",
        "devices",
        "user",
        "pids_limit",
    ):
        if key in resources:
            run_kwargs[key] = resources[key]

    return client.containers.run(image, **run_kwargs)


def disk_usage(container: "DockerContainer") -> Optional[int]:
    """Best-effort writable-layer size (SizeRw) in bytes; None if undeterminable."""
    try:
        client = get_docker_client()
        info = client.api.inspect_container(container.id, size=True)
        size_rw = info.get("SizeRw")
        return int(size_rw) if size_rw is not None else None
    except Exception:
        return None


def format_bytes(size_bytes: int) -> str:
    val = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(val) < 1024.0:
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} PB"


def cleanup(container: Optional["DockerContainer"], image: Optional[str] = None) -> None:
    """AUTHORITATIVE cleanup: stop+remove the container, optionally remove the image.
    All failures are swallowed (best effort) so cleanup never raises."""
    if container is not None:
        try:
            container.stop(timeout=10)
        except Exception:
            pass
        try:
            container.remove(force=True)
        except Exception:
            pass
    if image and str(image).strip():
        try:
            get_docker_client().images.remove(image=image, force=True)
        except Exception:
            pass


# ── screen-based stateful execution (optional; preserves cwd/env across calls) ─


def _detect_pkg_manager(container: "DockerContainer") -> Optional[str]:
    for pm in ("microdnf", "dnf", "yum", "apt-get", "apk"):
        code, _ = exec_in_container(container, f"command -v {pm} >/dev/null 2>&1")
        if code == 0:
            return pm
    return None


def _escalation_prefix(container: "DockerContainer") -> Optional[str]:
    code, out = exec_in_container(container, "id -u")
    if code == 0 and out.strip() == "0":
        return ""
    if (
        exec_in_container(container, "command -v sudo >/dev/null 2>&1")[0] == 0
        and exec_in_container(container, "sudo -n true")[0] == 0
    ):
        return "sudo -n "
    return None


def _install_cmds(pm: str, pkgs: list[str]) -> list[str]:
    joined = " ".join(shlex.quote(p) for p in pkgs)
    if pm == "apt-get":
        return ["apt-get update", f"DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {joined}"]
    if pm in ("dnf", "microdnf"):
        return [f"{pm} -y install {joined}"]
    if pm == "yum":
        return [f"yum -y install {joined}"]
    if pm == "apk":
        return [f"apk add --no-cache {joined}"]
    return []


def _ensure_packages(container: "DockerContainer", pkgs: list[str]) -> Optional[str]:
    pm = _detect_pkg_manager(container)
    prefix = _escalation_prefix(container)
    if pm is None:
        return "no supported package manager detected (microdnf/dnf/yum/apt-get/apk)."
    if prefix is None:
        return "need root privileges (not root; passwordless sudo unavailable)."
    for c in _install_cmds(pm, pkgs):
        code, out = exec_in_container(container, f"{prefix}{c}", timeout=600)
        if code != 0:
            return f"Package installation failed.\nCommand: {prefix}{c}\nExit code: {code}\nOutput:\n{out}"
    return None


def create_screen_session(container: "DockerContainer") -> tuple[bool, str]:
    """Ensure bash+screen exist, then (re)create a detached screen session."""
    needed = ["bash", "screen", "psmisc"]
    pm = _detect_pkg_manager(container)
    if pm == "apt-get":
        needed.append("coreutils")
    if exec_in_container(container, "command -v screen >/dev/null 2>&1")[0] != 0:
        msg = _ensure_packages(container, needed)
        if msg:
            return False, msg
    exec_in_container(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -X quit >/dev/null 2>&1 || true")
    code, out = exec_in_container(container, f"screen -dmS {shlex.quote(SCREEN_SESSION)} bash -l")
    if code != 0:
        return False, f"Failed to start screen session '{SCREEN_SESSION}'. Output:\n{out}"
    code, ls_out = exec_in_container(container, "screen -ls")
    if code != 0 or f".{SCREEN_SESSION}" not in ls_out:
        return False, f"Screen session '{SCREEN_SESSION}' not found after creation.\n{ls_out}"
    exec_in_container(container, f"mkdir -p {shlex.quote(RUN_DIR)} && chmod 1777 {shlex.quote(RUN_DIR)} || true")
    return True, f"Screen session '{SCREEN_SESSION}' created."


def ensure_screen_alive(container: "DockerContainer") -> tuple[bool, str]:
    code, _ = exec_in_container(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -Q windows")
    if code == 0:
        return True, "ok"
    return create_screen_session(container)


def exec_in_screen(
    container: "DockerContainer", cmd: str, stuck_timeout: int = STUCK_TIMEOUT_SECONDS
) -> tuple[int, str, str, bool]:
    """Stateful execution inside a GNU screen session (cwd/env persist across calls).

    Returns ``(rc, cleaned_output, logfile, stuck_flag)``. Stuck == no output change
    for ``stuck_timeout`` seconds (rc 124).
    """
    ok, msg = ensure_screen_alive(container)
    if not ok:
        return 1, f"Error: could not ensure screen session. {msg}", "", False

    run_id = uuid.uuid4().hex
    BEGIN = f"<<BEGIN:{run_id}>>"
    END = f"<<END:{run_id}>>"
    rc_rx = re.compile(rf"<<RC:{re.escape(run_id)}:(-?\d+)>>")
    delim = f"__PAYLOAD_{run_id}__"
    logfile = f"{LOG_DIR}/screen_exec_{run_id}.log"
    script = f"{LOG_DIR}/screen_src_{run_id}.sh"

    def _stuff(text: str) -> None:
        safe = text.replace("'", r"'\''")
        exec_in_container(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -X stuff '{safe}\\r'")

    exec_in_container(
        container,
        f"cat > {shlex.quote(script)} <<'{delim}'\n{cmd}\n{delim}\nchmod +x {shlex.quote(script)}",
    )
    exec_in_container(container, f": > {shlex.quote(logfile)}")

    payload = (
        f'printf "%s\\n" "{BEGIN}" >> {logfile}; '
        f"if . {script} >> {logfile} 2>&1; then __rc=0; else __rc=$?; fi; "
        f'printf "%s\\n" "{END}" >> {logfile}; '
        f'printf "<<RC:{run_id}:%d>>\\n" "$__rc" >> {logfile}'
    )
    _stuff(payload)

    last_buf = ""
    last_change = time.time()
    rc: Optional[int] = None
    while True:
        buf = read_file_tail(container, logfile)
        if buf != last_buf:
            last_buf = buf
            last_change = time.time()
        m = rc_rx.search(buf.replace("\r", ""))
        if m:
            rc = int(m.group(1))
            break
        if time.time() - last_change >= stuck_timeout:
            clean = strip_ansi(buf)
            return (
                124,
                (f"The command appears stuck/idle (no output change within the timeout).\n\nPartial output:\n{clean}"),
                logfile,
                True,
            )
        time.sleep(POLL_INTERVAL_SECONDS)

    final = strip_ansi(read_file_tail(container, logfile))
    bpos = final.rfind(BEGIN)
    epos = final.rfind(END)
    if bpos != -1 and epos != -1 and epos > bpos:
        region = final[bpos + len(BEGIN) : epos]
    else:
        region = "\n".join(
            ln
            for ln in final.splitlines()
            if not (ln.startswith("<<BEGIN:") or ln.startswith("<<END:") or ln.startswith("<<RC:"))
        )
    region = "\n".join(ln for ln in region.splitlines() if not ln.startswith("<<RC:")).strip()
    exec_in_container(container, f"rm -f {shlex.quote(script)}")
    return (rc if rc is not None else 0), region, logfile, False


def tokens_manifest_cmd(path: str = "/root/.tokens.env") -> str:
    """Shell command that writes the credentials manifest inside a fresh container.

    Only the whitelisted passthrough vars land in the file, read from the container's
    OWN environment (never interpolated from the host side), so the file can never
    hold more than the whitelist allows. Format is plain KEY=VALUE -- sourceable with
    `set -a; . /root/.tokens.env; set +a` and readable by any script.
    """
    pattern = "|".join(LLM_KEY_PASSTHROUGH_VARS)
    # AND A CREDENTIAL THAT CANNOT BE MISLAID. The manifest above is readable but the agent
    # still has to get a shell header right, and a call that goes out anonymous hits the
    # IP-based rate limit. Prose does not fix a quoting mistake.
    #
    # BOTH files are needed: curl ignores .netrc unless told, so .curlrc carries the flag. It is
    # --netrc-OPTIONAL on purpose -- the strict form fails any request to a host with no entry,
    # which would break every non-GitHub fetch the miner makes. Written only when a token is
    # actually present, because an empty password turns a rate-limit into a Bad credentials.
    # ...AND FOR git CLONES: an entry for api.github.com only would leave every
    # `git clone https://github.com/...` anonymous, and anonymous git is rate limited (HTTP
    # 401). git reads ~/.netrc through libcurl (CURL_NETRC_OPTIONAL), so a github.com entry makes
    # clones authenticated without the agent doing anything.
    netrc = (
        'if [ -n \\"$GITHUB_TOKEN\\" ]; then '
        'printf \\"machine api.github.com login x-access-token password %s\\\\n'
        'machine github.com login x-access-token password %s\\\\n\\" '
        '\\"$GITHUB_TOKEN\\" \\"$GITHUB_TOKEN\\" > /root/.netrc; chmod 600 /root/.netrc; '
        'printf -- \\"--netrc-optional\\\\n\\" >> /root/.curlrc; '
        "chmod 600 /root/.curlrc; fi; "
    )
    return (
        f"sh -c \"printenv | grep -E '^({pattern})=' > {path} 2>/dev/null; chmod 600 {path} 2>/dev/null; {netrc}true\""
    )
