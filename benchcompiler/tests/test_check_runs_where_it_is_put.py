"""A variant's check must observe the copy the harness rebuilt, not one it navigates to.

The gate runs a check as `cd {fresh} && {check_cmd}` in a directory the setup script has just
rebuilt from scratch. A check that begins with `cd` to another directory (including a path that
resolves outside the fresh copy) leaves the rebuilt copy and proves nothing about reproducibility;
setup belongs to `setup_<name>.sh`, which the certifier records as `stage: rebuild`, distinct from
`stage: check`.
"""

from __future__ import annotations

import os

import pytest

from benchcompiler.spec import check_verifiability

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = os.path.join(HERE, "..", "..", "agents")


def _spec(*cmds):
    return {
        "artifact_variants": [
            {"name": f"v{i}", "check_cmd": c, "expect": "pass" if i == 0 else "fail"} for i, c in enumerate(cmds)
        ]
    }


def _codes(raw):
    return {d.code for d in check_verifiability(raw)}


# ── the rule ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cmd",
    [
        "cd /workspace/repo && make check",
        "cd {source_path}/.. && {build_command} && test x = y",
        "make && cd /tmp && ./run",
        "  cd   /opt/x && ./t",
    ],
)
def test_a_check_that_navigates_away_is_refused(cmd):
    """Whether the escape is a literal path or a placeholder that resolves to one."""
    codes = _codes(_spec(cmd, "true"))
    assert "CHECK_LEAVES_VARIANT" in codes, cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "mvn -q -Dsurefire.failIfNoSpecifiedTests=true -Dtest={triggering_test} test",
        "mvn -q -DskipTests package",
        "python -m pytest -q {triggering_test}",
        "make -C src check",
        "./run.sh && diff -q out expected",
    ],
)
def test_the_checks_that_actually_work_are_untouched(cmd):
    """Commands that stay in the working directory, and the idiom that replaces a subdirectory cd,
    are accepted.
    """
    assert "CHECK_LEAVES_VARIANT" not in _codes(_spec(cmd, "true"))


def test_the_refusal_says_what_to_do_instead():
    d = [x for x in check_verifiability(_spec("cd /w && make", "true")) if x.code == "CHECK_LEAVES_VARIANT"][0]
    low = (d.detail + " " + (d.fix_hint or "")).lower()
    assert "setup_" in low, "the fix is to move setup into the setup script"
    assert "-c" in low or "argument" in low, "and to pass a directory rather than cd to it"


def test_the_code_is_registered():
    from benchcompiler.spec import DIAGNOSTIC_CODES

    assert "CHECK_LEAVES_VARIANT" in DIAGNOSTIC_CODES


# ── it must not name a task ──────────────────────────────────────────────────────────────────


def test_the_rule_names_no_toolchain():
    """The rule concerns the working directory, not which programs a check may invoke (a build-tool
    test command is allowed).
    """
    import inspect
    import re

    from benchcompiler import spec as _spec_mod

    src = inspect.getsource(_spec_mod.check_verifiability).lower()
    for w in ("mvn", "maven", "gcc", "make", "pytest", "npm", "cargo", "configure"):
        assert not re.search(rf"\b{w}\b", src), f"toolchain vocabulary leaked in: {w}"


# ── against the live specs ───────────────────────────────────────────────────────────────────
