"""bench-ui health warnings: a stalled live agent, a provider refusal in a live log, and the
same tool failure across most of a run's agents are each reported once, with what to do.

The live agents are python processes this test starts, whose command line holds their run
directory (that is how the UI tells a live run from a reused pid).
"""

import json
import os
import subprocess
import sys
import time

import pytest

from benchcompiler.ui import data as D
from benchcompiler.ui import health as H


def _w(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _trajectory(path, cycles, ok):
    lines = []
    for c in range(1, cycles + 1):
        lines.append(
            json.dumps(
                {
                    "event": "on_response",
                    "cycle": c,
                    "ts": c,
                    "content": json.dumps(
                        {"thoughts": "t", "command": {"name": "linux_terminal", "args": {"command": "make"}}}
                    ),
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "event": "on_tool_result",
                    "cycle": c,
                    "ts": c,
                    "name": "linux_terminal",
                    "result": {"ok": ok, "display_text": "REFUSED: the check script is missing\nmore"},
                }
            )
        )
    _w(path, "\n".join(lines) + "\n")


@pytest.fixture()
def live(tmp_path):
    procs = []

    def start(run_dir):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", run_dir], start_new_session=True)
        procs.append(p)
        _w(os.path.join(run_dir, "launcher.pid"), str(p.pid))
        return p

    yield start
    for p in procs:
        p.kill()
        p.wait()


def test_stall_provider_refusal_and_run_wide_failures_are_reported(tmp_path, live):
    runs = tmp_path / "runs"
    run = runs / "construct" / "demo" / "epoch_1"
    _w(str(run / "handover.json"), json.dumps({"total_target": 10}))
    for i in range(3):
        w = str(run / f"worker_{i}")
        _w(os.path.join(w, "agent_state.json"), json.dumps({"budget": {"cost_usd": {"value": 1, "limit": 5}}}))
        _trajectory(os.path.join(w, "trajectory.jsonl"), 10, ok=(i == 2))  # workers 0, 1 fail every call
        live(w)
    old = time.time() - 30 * 60
    os.utime(str(run / "worker_2" / "trajectory.jsonl"), (old, old))  # worker 2 stalled
    _w(str(run / "worker_1" / "construct.log"), "step\nlitellm.APIError: 402 out of credits\n")

    ws = H.warnings(D.Tree(str(tmp_path / "repo"), str(runs)))
    for w in ws:
        print(w["level"], w["kind"], "|", w["where"], "|", w["what"])
    kinds = [w["kind"] for w in ws]
    assert kinds.count("stalled") == 1 and "worker_2" in next(w for w in ws if w["kind"] == "stalled")["where"]
    prov = next(w for w in ws if w["kind"] == "log errors")
    assert prov["level"] == "bad" and "402" in prov["what"]
    fail = [w for w in ws if w["kind"] == "failing tools"]
    assert len(fail) == 1 and fail[0]["level"] == "bad"  # once for the run
    assert "2 of 3" in fail[0]["what"] and "REFUSED: the check script is missing" in fail[0]["what"]
    assert all(w["todo"] for w in ws)


def test_a_quiet_healthy_run_reports_nothing(tmp_path, live):
    runs = tmp_path / "runs"
    w = str(runs / "construct" / "demo" / "epoch_1" / "worker_0")
    _w(os.path.join(w, "agent_state.json"), json.dumps({"budget": {"cost_usd": {"value": 1, "limit": 5}}}))
    _trajectory(os.path.join(w, "trajectory.jsonl"), 10, ok=True)
    live(w)
    ws = [x for x in H.warnings(D.Tree(str(tmp_path / "repo"), str(runs))) if x["kind"] != "disk"]
    print(ws)
    assert ws == []
