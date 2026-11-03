"""The command a failing test tells you to run has to run.

The drift guard's refresh command, `python -m benchcompiler.formalizer ... --lock ...`, must import
without `agentcore` already on `sys.path`; the module bootstraps its own path the way `bench_cli`
does.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def test_the_module_runs_standalone():
    """`--help` is enough: it imports the module and reaches argparse, where a missing import would
    fail.
    """
    r = subprocess.run(
        [sys.executable, "-m", "benchcompiler.formalizer", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(f"\n[verbose] exit={r.returncode}\n[verbose] stderr tail: {r.stderr[-400:]}")
    assert r.returncode == 0, r.stderr
    assert "--lock" in r.stdout


def test_the_stage_log_import_is_reachable_from_the_module():
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import benchcompiler.formalizer; from agentcore.observability.stage_log import StageLog; print('ok')",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(f"\n[verbose] exit={r.returncode} out={r.stdout.strip()!r} stderr={r.stderr[-300:]!r}")
    assert r.returncode == 0 and "ok" in r.stdout, (
        "importing the formalizer must make its own stage-log import reachable"
    )
