"""bench-ui actions: nothing runs unless actions are on, the form names every model and budget,
a passing check of the same settings came first, and the confirmation word is typed. Stop ends
the process groups it lists and says what is left.

No test here starts a model call or touches Docker: the commands are replaced by small python
ones, the stop target is a `sleep` this test starts, and the container listing is stubbed.
"""

import json
import os
import re
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from benchcompiler.ui import actions as A  # noqa: E402
from benchcompiler.ui.app import build_app  # noqa: E402


def _w(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content if isinstance(content, str) else json.dumps(content))


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    _w(str(r / "specs/demo/spec.yaml"), "task: |\n  Repair a real bug.\n")
    _w(str(r / "specs/demo/canonical.json"), {"outcome": "ACCEPT", "canonical_spec": {"task": "x"}})
    _w(str(r / "agents/demo_worker/contract.json"), {"_stages": [{"name": "set up"}]})
    _w(str(r / "stock/demo/candidates.jsonl"), json.dumps({"repo_url": "https://github.com/o/p"}) + "\n")
    _w(str(r / "runs/search/demo/epoch_1/attempt_0/stock/candidates.jsonl"), json.dumps({"repo_url": "u"}) + "\n")
    monkeypatch.setattr(A, "run_containers", lambda dirs: [])  # never ask Docker here
    return str(r)


def fake(tree, task):
    """An action with the real gate but harmless commands; it writes a marker when it runs."""
    marker = os.path.join(tree.root, "ran.txt")
    return {
        "id": "fake",
        "label": "Fake",
        "task": task,
        "paid": True,
        "doc": "test",
        "fields": [A.field("model", "model", "model", True), A.field("cost", "cost", "float", True)],
        "argv": lambda v: [sys.executable, "-c", f"open({marker!r}, 'w').write('ran {v['model']}')"],
        "check": lambda v: [sys.executable, "-c", "print('dry run ok')"],
        "detached": False,
        "confirm": lambda v: f"{task} go",
    }


def _csrf(html):
    return re.search(r'name="csrf" value="([^"]+)"', html).group(1)


def test_read_only_by_default_and_actions_need_the_repositorys_runs(root, tmp_path):
    c = TestClient(build_app(root, None))
    r = c.get("/task/demo?tab=actions")
    print(r.status_code, "Actions are off" in r.text)
    assert r.status_code == 200 and "Actions are off" in r.text
    r = c.post("/action/search/demo/check", data={"csrf": "x", "model": "m"})
    print(r.status_code, r.text[:120])
    assert r.status_code == 403
    with pytest.raises(SystemExit) as e:
        build_app(root, str(tmp_path / "elsewhere"), allow_actions=True)
    print(e.value)
    assert "--allow-actions needs the runs folder" in str(e.value)


def test_a_command_runs_only_after_its_check_and_the_typed_word(root, monkeypatch):
    monkeypatch.setitem(A.BUILDERS, "fake", fake)
    c = TestClient(build_app(root, None, allow_actions=True))
    form = c.get("/action/fake/demo").text
    token = _csrf(form)

    r = c.post("/action/fake/demo/check", data={"csrf": token, "model": "", "cost": "5"})
    print("empty model ->", re.search(r"Not run:</b> ([^<]+)", r.text).group(1))
    assert "does not choose it for you" in r.text  # no default model

    r = c.post("/action/fake/demo/check", data={"csrf": "stale", "model": "p/m", "cost": "5"})
    assert r.status_code == 403  # a form from another start

    run = {"csrf": token, "model": "p/m", "cost": "5", "confirm": "demo go"}
    r = c.post("/action/fake/demo/run", data=run, follow_redirects=False)
    print("run before check ->", r.status_code, r.json()["detail"][:60])
    assert r.status_code == 409 and not os.path.exists(os.path.join(root, "ran.txt"))

    r = c.post("/action/fake/demo/check", data={"csrf": token, "model": "p/m", "cost": "5"})
    print("check ->", "dry run ok" in r.text, "type <code>demo go</code>" in r.text)
    assert "dry run ok" in r.text and "demo go" in r.text

    r = c.post("/action/fake/demo/run", data={**run, "cost": "6"}, follow_redirects=False)
    print("other settings ->", r.status_code)
    assert r.status_code == 409  # the check was for cost 5

    r = c.post("/action/fake/demo/run", data={**run, "confirm": "demo"}, follow_redirects=False)
    print("wrong word ->", r.status_code)
    assert r.status_code == 409 and not os.path.exists(os.path.join(root, "ran.txt"))

    r = c.post("/action/fake/demo/run", data=run, follow_redirects=False)
    print("run ->", r.status_code, r.headers.get("location"))
    assert r.status_code == 303
    assert open(os.path.join(root, "ran.txt")).read() == "ran p/m"
    log = c.get("/jobs").text
    assert log.count("fake") >= 2  # the check and the run are logged


def test_the_real_actions_build_the_documented_commands(root):
    from benchcompiler.ui import data as D

    tree = D.Tree(root, None)
    s = A.search_action(tree, "demo")
    v = A.values(
        s,
        {
            "epoch": "2",
            "model": "openrouter/x/y",
            "agents": "4",
            "target": "60",
            "cost": "5",
            "max_search": "8",
            "territory_mode": "none",
            "start_delay": "1200",
            "directions": "off",
            "lessons": "off",
        },
    )
    argv = s["check"](v)
    print(" ".join(argv[1:]))
    assert argv[1:5] == ["scripts/search_campaign.py", "demo", "2", "--agents"] and argv[-1] == "--dry-run"
    assert "--wall" not in argv  # empty optional: script default

    k = A.construct_action(tree, "demo")
    with pytest.raises(A.Refused) as e:
        A.values(
            k,
            {
                "stock": "demo",
                "epoch": "epoch_9",
                "target": "5",
                "arm": "a",
                "workers": "2",
                "cost_usd": "5",
                "wallclock": "3600",
            },
        )
    print(e.value)
    assert "model" in str(e.value)
    v = A.values(
        k,
        {
            "stock": "demo",
            "epoch": "epoch_9",
            "target": "5",
            "arm": "a",
            "model": "openrouter/x/y",
            "workers": "2",
            "cost_usd": "5",
            "wallclock": "3600",
            "order": "shuffle",
            "seed": "7",
        },
    )
    k["before"](v)
    print(" ".join(k["check"](v)[1:]))
    assert json.load(open(os.path.join(root, "runs/construct/demo/epoch_9/arms.json")))[0]["model"] == "openrouter/x/y"
    assert k["check"](v)[-3:] == ["--shuffle", "7", "--dry-run"]

    f = A.freeze_action(tree, "demo")
    ok, text = f["preview"](A.values(f, {"epochs": "epoch_1", "name": "demo"}))
    print(text)
    assert not ok and "EXISTS" in text  # stock/demo is already there


def test_stop_ends_the_process_group_and_checks_nothing_is_left(root):
    wdir = os.path.join(root, "runs/construct/demo/epoch_1/worker_0")
    os.makedirs(wdir)
    p = subprocess.Popen(["sleep", "300"], start_new_session=True)
    try:
        open(os.path.join(wdir, "launcher.pid"), "w").write(str(p.pid))
        c = TestClient(build_app(root, None, allow_actions=True))
        page = c.get("/stop?kind=construction&name=construct~demo~epoch_1").text
        print("listed:", str(p.pid) in page)
        assert str(p.pid) in page and "type <code>epoch_1</code>" in page
        token = _csrf(page)
        q = {"csrf": token, "kind": "construction", "task": "", "name": "construct~demo~epoch_1", "agent": ""}
        r = c.post("/stop", data={**q, "confirm": "epoch"})
        assert r.status_code == 409 and p.poll() is None  # wrong word: still running
        r = c.post("/stop", data={**q, "confirm": "epoch_1"})
        print(re.search(r'class="note ok-note">([^<]+)', r.text).group(1).strip())
        p.wait(timeout=10)
        assert "No process of those groups is left" in r.text
    finally:
        if p.poll() is None:
            p.kill()
