"""The host-wide measurement lane: timed executions queue, everything else runs wide.

Why. A verdict for a timing benchmark is a RATIO between two sides measured on this host
against a declared margin. But timed execution is a small fraction of a run's wall-clock --
clones, builds, container setup and model calls are the rest, and none of them needs a quiet
machine. This module is the seam between the two: a timed run holds the lane, a build does
not, and many workers can run around one serial timing queue.

What it is. A file lock. Every process that times anything holds it, including every worker
process, because the lock lives on the host, not in a process.

    AGENTCORE_MEASUREMENT_LANE     path of the lock file (default below); `off` disables
                                   the lane entirely (unit tests, single-process runs)
    AGENTCORE_MEASUREMENT_CPUSET   e.g. `0-3`: while a container holds the lane it is pinned
                                   to these cores (`docker update`) and restored after, so
                                   the timed run also gets cores nobody else is using
    AGENTCORE_SANDBOX_CPUSET       the cores everything else runs on (docker_helpers applies
                                   it at container creation); the value a pinned container
                                   is restored to when it had no cpuset of its own

Holding is re-entrant within a thread: a control that holds around a whole interleaved block
does not deadlock the per-repetition hold inside `measure.run_measurement`. Pinning is a
courtesy to the measurement -- a container that cannot be updated never turns a timing into
an exception; the hold records `pinned=None` and the measurement proceeds.

The lane serializes; it does not make a busy host quiet. Memory bandwidth is shared whatever
the cpuset says, which is why the controls also measure the two sides of a ratio turn about
(`measure.run_measurement_sides`): a steady background load then slows both sides alike.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from dataclasses import dataclass

LANE_ENV = "AGENTCORE_MEASUREMENT_LANE"
CPUSET_ENV = "AGENTCORE_MEASUREMENT_CPUSET"
SANDBOX_CPUSET_ENV = "AGENTCORE_SANDBOX_CPUSET"
DEFAULT_LANE = "/tmp/benchcompiler-measurement.lane"

_local = threading.local()


@dataclass
class Hold:
    """What one hold of the lane was: for the evidence record, and for tests."""

    label: str
    container: str | None = None
    held: bool = False  # False when the lane is off or an outer hold already has it
    waited_s: float = 0.0  # time spent queued behind other timed executions
    pinned: str | None = None  # the cpuset the container ran on while held, if any


def lane_path() -> str | None:
    """The lock file, or None when the lane is switched off."""
    raw = os.environ.get(LANE_ENV, "").strip()
    if raw.lower() == "off":
        return None
    return raw or DEFAULT_LANE


def cpuset() -> str | None:
    raw = os.environ.get(CPUSET_ENV, "").strip()
    return raw or None


def sandbox_cpuset() -> str | None:
    raw = os.environ.get(SANDBOX_CPUSET_ENV, "").strip()
    return raw or None


def _docker(runner, argv: list[str]):
    return (runner or subprocess.run)(argv, capture_output=True, text=True, errors="replace")


def _pin(container: str, cores: str, runner) -> str | None:
    """Pin `container` to `cores`; return the cpuset it had before (possibly ''), or None
    when pinning did not happen."""
    try:
        r = _docker(runner, ["docker", "inspect", "-f", "{{.HostConfig.CpusetCpus}}", container])
        prev = (r.stdout or "").strip() if r.returncode == 0 else ""
        u = _docker(runner, ["docker", "update", f"--cpuset-cpus={cores}", container])
        if u.returncode != 0:
            return None
        return prev
    except Exception:  # noqa: BLE001 - a missing docker, a gone container: not a measurement error
        return None


def _unpin(container: str, prev: str, runner) -> None:
    restore = prev or sandbox_cpuset() or ""
    try:
        _docker(runner, ["docker", "update", f"--cpuset-cpus={restore}", container])
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def hold(label: str, container: str | None = None, runner=None):
    """Hold the lane for one timed execution.

    `label` names what is being timed (it is what a waiting operator sees); `container` is
    the docker container the timed command runs in, pinned for the duration when a cpuset is
    configured; `runner` stands in for subprocess.run in tests.
    """
    h = Hold(label=str(label), container=container)
    path = lane_path()
    depth = int(getattr(_local, "depth", 0) or 0)
    if path is None or depth > 0:
        _local.depth = depth + 1
        try:
            yield h
        finally:
            _local.depth = depth
        return

    import fcntl

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd = open(path, "a+")
    t0 = time.monotonic()
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
    except Exception:
        fd.close()
        raise
    h.waited_s = time.monotonic() - t0
    h.held = True
    _local.depth = depth + 1
    prev: str | None = None
    cores = cpuset()
    try:
        if container and cores:
            prev = _pin(container, cores, runner)
            h.pinned = cores if prev is not None else None
        yield h
    finally:
        if h.pinned is not None and container:
            _unpin(container, prev or "", runner)
        _local.depth = depth
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()
