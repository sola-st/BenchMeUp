"""Make `agentcore` importable regardless of which tests run.

`benchcompiler` imports `agentcore`, but `agent-scaffold/` is not on the path by default. Adding it
here makes every test work in any order, alone or together.
"""

from __future__ import annotations

import os
import sys

import pytest

_SCAFFOLD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent-scaffold"))

if os.path.isdir(_SCAFFOLD) and _SCAFFOLD not in sys.path:
    sys.path.insert(0, _SCAFFOLD)


@pytest.fixture(scope="session", autouse=True)
def _remove_containers_this_session_started():
    """Remove the containers this process started, at the end of the suite.

    Only containers returned by `start_container` calls made by this process are removed, so
    containers of other concurrent sessions are left alone. Suites killed before teardown are
    not covered.
    """
    try:
        from agentcore.environments import docker_helpers as dh  # _SCAFFOLD is on the path
    except Exception:  # noqa: BLE001 - no runtime, nothing to reap
        yield
        return

    started: list = []
    original = dh.start_container

    def _recording(*a, **k):
        c = original(*a, **k)
        started.append(getattr(c, "id", None) or c)
        return c

    dh.start_container = _recording
    try:
        yield
    finally:
        dh.start_container = original
    removed = dh.remove_containers(started)
    if removed:
        print(f"\nremoved {removed} container(s) this test session started")


# ── scenario-recording erosion is one red line, never a quiet skip pile ──
# llmscenario tests skip on a cache miss when LLM_SCENARIOS is unset. The baseline file
# (tests/_llm_cache/SKIP_BASELINE) pins the expected number of scenario skips; exceeding it fails
# the run. After editing prompts, re-record (LLM_SCENARIOS=1) or update the baseline.


def pytest_runtest_logreport(report):
    if report.skipped and report.when in ("setup", "call"):
        text = str(getattr(report, "longrepr", ""))
        if "no recording" in text or "LLM_SCENARIOS" in text:
            _SCENARIO_SKIPS.append(report.nodeid)


_SCENARIO_SKIPS: list = []


@pytest.fixture
def scenario_skips():
    """Scenario-skip nodeids for the session, shared through a fixture (a re-imported conftest
    module would have its own empty list).
    """
    return _SCENARIO_SKIPS


def pytest_collection_modifyitems(config, items):
    # the baseline check must run LAST so every scenario skip has been counted
    tail = [i for i in items if "scenario_recordings_are_not_eroding" in i.nodeid]
    for t in tail:
        items.remove(t)
        items.append(t)
