"""The UI (bench-ui) renders every page from a repository and a runs root, and writes nothing.

A synthetic tree holds one task with its spec, rules, plan, a frozen stock, one packaged benchmark,
one search campaign and one construction run. Every page must answer 200, the cycle readers must
find the right cycle in a long trajectory without reading it whole, and no file may change.
"""

import hashlib
import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from benchcompiler.ui import data as D  # noqa: E402
from benchcompiler.ui.app import build_app  # noqa: E402


def _w(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content if isinstance(content, str) else json.dumps(content))


def _trajectory(path, cycles):
    with open(path, "w", encoding="utf-8") as fh:
        for c in range(1, cycles + 1):
            fh.write(json.dumps({"event": "on_cycle_start", "cycle": c, "ts": 1.0 * c}) + "\n")
            fh.write(
                json.dumps(
                    {
                        "event": "on_response",
                        "cycle": c,
                        "ts": 1.0 * c,
                        "cost": 0.001,
                        "content": json.dumps(
                            {
                                "thoughts": f"thinking in cycle {c}",
                                "command": {"name": "linux_terminal", "args": {"command": f"echo {c}"}},
                            }
                        ),
                    }
                )
                + "\n"
            )
            fh.write(
                json.dumps(
                    {
                        "event": "on_tool_result",
                        "cycle": c,
                        "ts": 1.0 * c,
                        "name": "linux_terminal",
                        "args": {"command": f"echo {c}"},
                        "result": {"ok": True, "display_text": f"out {c}"},
                    }
                )
                + "\n"
            )


@pytest.fixture()
def tree(tmp_path):
    root, runs = tmp_path / "repo", tmp_path / "runs"
    _w(str(root / "specs/demo/spec.yaml"), "task: |\n  Repair a real bug.\ndata: |\n  Public Java projects.\n")
    _w(
        str(root / "specs/demo/canonical.json"),
        {
            "outcome": "ACCEPT",
            "canonical_spec": {
                "task": "Repair a real bug.",
                "candidate_definition": "a fixing commit",
                "criteria": ["fails before"],
            },
        },
    )
    _w(
        str(root / "agents/demo_search/instances/instances.jsonl"),
        json.dumps(
            {
                "identity_fields": ["repo_url", "fix_commit"],
                "candidate_definition": "a fixing commit",
                "admissibility": [{"name": "stars", "observe": "GET /repos", "require": ">= 100"}],
                "sample_size": 5,
            }
        )
        + "\n",
    )
    _w(
        str(root / "agents/demo_worker/contract.json"),
        {
            "verbs": {"build": {"args": ["dir"]}},
            "variants": {"buggy": {"setup": [{"verb": "build", "args": {}}]}},
            "checks": [{"verb": "run_tests", "reader": "exit-code"}],
            "oracle": {"kind": "fail-to-pass"},
            "_stages": [{"name": "set up", "files": ["Dockerfile"], "goal": "the image builds"}],
            "_stage_sections": [{"where_you_are": "nothing exists yet", "goal": "build it", "traps": "none"}],
            "_does": {"build": "compile"},
        },
    )
    _w(
        str(root / "stock/demo/candidates.jsonl"),
        json.dumps({"repo_url": "https://github.com/o/p", "fix_commit": "a" * 40, "evidence": "fixes NPE"}) + "\n",
    )
    _w(str(root / "stock/demo/manifest.json"), {"identity": ["repo_url", "fix_commit"], "count": 1})
    bdir = root / "benchmarks/model/demo"
    _w(
        str(bdir / "manifest.json"),
        {
            "name": "demo",
            "count": 1,
            "instances": [
                {
                    "id": "inst-1",
                    "fields": {"repo_url": "https://github.com/o/p", "fix_commit": "a" * 40},
                    "rows": {"oracle": True},
                    "harness_verdict_ok": True,
                    "from_run": "runs/construct/demo",
                }
            ],
        },
    )
    _w(str(bdir / "instances/inst-1/instance.json"), {"repo_url": "https://github.com/o/p"})
    _w(str(root / "benchmarks/model/acceptance.json"), {"demo": {"inst-1": {"accepted": True, "run": "replay"}}})
    _w(str(bdir / "selection_50.json"), {"rule": "first 50", "ids": ["inst-1"]})

    att = runs / "search/demo/epoch_1/attempt_0"
    _w(
        str(att / "agent_state.json"),
        {"cycle": 5000, "budget": {"cost_usd": {"value": 1.2, "limit": 5}, "wallclock": {"value": 600, "limit": 3600}}},
    )
    _w(str(att / "run_meta.json"), {"models": {"main": {"name": "openrouter/m"}}})
    _w(str(att / "stock/candidates.jsonl"), json.dumps({"repo_url": "https://github.com/o/q", "evidence": "x"}) + "\n")
    _w(
        str(att / "cycles/cycle_4321.json"),
        [{"role": "system", "content": "the brief"}, {"role": "user", "content": "go"}],
    )
    _trajectory(str(att / "trajectory.jsonl"), 5000)
    _w(str(runs / "search/demo/epoch_1/campaign.json"), {"agents": 1, "events": [[0, "launch attempt_0"]]})

    wk = runs / "construct/demo/epoch_1/worker_0"
    key = "https://github.com/o/p|" + "a" * 40
    _w(
        str(wk / "agent_state.json"),
        {
            "cycle": 3,
            "budget": {"cost_usd": {"value": 0.3, "limit": 5}},
            "working_set": {
                "worklist": {
                    "candidates": {
                        key: {"state": "STAGE_2_BUILD", "fields": {}, "history": []},
                        "https://github.com/o/r|" + "b" * 40: {"state": "REJECTED"},
                    },
                    "order": [key, "https://github.com/o/r|" + "b" * 40],
                    "terminal": ["STAGE_3_DONE"],
                },
                "rejected_candidates": [
                    {"key": "https://github.com/o/r|" + "b" * 40, "state": "REJECTED", "reason": "does not build"}
                ],
            },
        },
    )
    _w(str(wk / "run_meta.json"), {"models": {"main": {"name": "openrouter/m"}}})
    _trajectory(str(wk / "trajectory.jsonl"), 3)
    slug = "https-github.com-o-p-" + "a" * 40 + "-12345678"
    _w(
        str(wk / f"recipes/{slug}/check_001.json"),
        {
            "ok": False,
            "rows": [
                {"name": "replays", "ok": True, "detail": "ran"},
                {"name": "variants_build", "ok": False, "detail": "mvn failed"},
            ],
        },
    )
    _w(str(wk / f"recipes/{slug}/bench_last/Dockerfile"), "FROM debian\n")
    return str(root), str(runs), slug


def _digest(*dirs):
    h = hashlib.sha256()
    for d in dirs:
        for base, _, files in sorted(os.walk(d)):
            for f in sorted(files):
                p = os.path.join(base, f)
                h.update(p.encode())
                with open(p, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def test_every_page_renders_and_nothing_is_written(tree):
    root, runs, slug = tree
    before = _digest(root, runs)
    c = TestClient(build_app(root, runs))
    key = "construct~demo~epoch_1"
    urls = [
        "/",
        "/running",
        "/benchmarks",
        "/benchmarks/model/demo",
        "/benchmarks/model/demo/inst-1",
        "/benchmarks/model/demo/inst-1/file/instance.json",
        "/task/demo",
        "/task/demo/spec",
        "/task/demo/rules",
        "/task/demo/plan",
        "/candidates/stock/demo",
        "/search/demo/epoch_1",
        "/candidates/search/demo/epoch_1",
        "/search/demo/epoch_1/attempt_0",
        "/candidates/attempt/demo/epoch_1/attempt_0",
        "/search/demo/epoch_1/attempt_0/cycle/4321",
        f"/construction/{key}",
        f"/construction/{key}/worker_0",
        f"/construction/{key}/worker_0/cycle/2",
        f"/construction/{key}/worker_0/recipe/{slug}",
        f"/construction/{key}/worker_0/recipe/{slug}/file/Dockerfile",
    ]
    # every tab of the pages that have tabs
    urls += [
        f"/task/demo?tab={t}"
        for t in ("overview", "funnel", "spec", "rules", "search", "plan", "construction", "benchmarks")
    ]
    urls += [
        "/workflow",
        "/workflow?task=demo",
        "/workflow?task=nosuch",
        f"/task/demo?tab=funnel&run={key}",
        "/task/demo?tab=funnel&run=unknown",
    ]
    urls += [f"/search/demo/epoch_1?tab={t}" for t in ("graph", "agents", "directions", "candidates", "settings")]
    urls += [f"/search/demo/epoch_1/attempt_0?tab={t}" for t in ("activity", "stock", "records")]
    urls += [f"/construction/{key}?tab={t}" for t in ("graph", "agents", "delivered", "dropped", "lessons", "settings")]
    urls += [f"/construction/{key}/worker_0?tab={t}" for t in ("candidates", "activity", "records")]
    for u in urls:
        r = c.get(u)
        print(r.status_code, len(r.text), u)
        assert r.status_code == 200, (u, r.text[:400])
    page = c.get(f"/construction/{key}?tab=dropped").text
    print(page.count("does not build"))
    assert "does not build" in page  # the drop and its reason
    assert "o/r @ bbbbbbbb" in page  # the candidate, shortened
    assert "echo 4999" in c.get("/search/demo/epoch_1/attempt_0?tab=activity").text  # a command as typed
    assert "the brief" in c.get("/search/demo/epoch_1/attempt_0/cycle/4321").text
    graph = c.get(f"/construction/{key}").text  # the default tab is the job graph
    assert "chip k-dropped" in graph and "o/r @ bbbbbbbb" in graph and "S3" in graph  # STAGE_2 -> on stage 3
    pipe = c.get("/workflow?task=demo").text
    assert "bench formalize" in pipe and "scripts/package.py" in pipe and "st-done" in pipe
    assert _digest(root, runs) == before, "the UI wrote into the repository or the runs"


def test_names_from_the_url_cannot_leave_the_tree(tree):
    root, runs, _ = tree
    c = TestClient(build_app(root, runs))
    for u in [
        "/task/..%2F..%2Fetc/spec",
        "/search/demo/..%2F..%2F/attempt_0",
        "/construction/construct~demo~..%2F../worker_0",
        "/benchmarks/model/demo/inst-1/file/..%2F..%2Fmanifest.json",
    ]:
        r = c.get(u)
        print(r.status_code, u)
        assert r.status_code in (400, 404), u


def test_the_cycle_readers_find_cycles_in_a_long_trajectory(tree):
    _, runs, _ = tree
    att = os.path.join(runs, "search/demo/epoch_1/attempt_0")
    latest = D.latest_cycles(att, 10)
    print([x["cycle"] for x in latest])
    assert [x["cycle"] for x in latest] == list(range(5000, 4990, -1))
    assert latest[0]["thoughts"] == "thinking in cycle 5000" and latest[0]["tool"] == "linux_terminal"
    ev = D.cycle_events(att, 1234)
    print(len(ev), {e["cycle"] for e in ev})
    assert len(ev) == 3 and {e["cycle"] for e in ev} == {1234}
    assert D.cycle_detail(att, 4321)["messages"][0]["content"] == "the brief"


def test_the_stage_funnel_follows_each_candidate_to_its_furthest_stage(tmp_path):
    """STAGE_<k>_ means stage k passed. Three stages; one candidate delivered, one dropped on
    stage 2 (it passed stage 1), one left on stage 1, one left on stage 3 in a live-less run."""
    w = tmp_path / "worker_0"
    cands = {
        "a": {"state": "STAGE_3_DONE", "history": ["REGISTERED", "STAGE_1_X", "STAGE_2_Y", "STAGE_3_DONE"]},
        "b": {"state": "REJECTED", "history": ["REGISTERED", "STAGE_1_X", "REJECTED"]},
        "c": {"state": "REGISTERED", "history": ["REGISTERED"]},
        "d": {"state": "deferred", "history": ["REGISTERED", "STAGE_1_X", "STAGE_2_Y", "deferred"]},
    }
    _w(
        str(w / "agent_state.json"),
        {"working_set": {"worklist": {"candidates": cands, "order": list(cands), "terminal": ["STAGE_3_DONE"]}}},
    )
    f = D.stage_funnel([str(w)], 3)
    rows = [(s["entered"], s["passed"], s["dropped"], s["open"]) for s in f["stages"]]
    print(rows, f["taken"], f["delivered"], f["dropped"], f["open"])
    assert rows == [(4, 3, 0, 1), (3, 2, 1, 0), (2, 1, 0, 1)]
    assert (f["taken"], f["delivered"], f["dropped"], f["open"], f["now"]) == (4, 1, 1, 2, 0)


def test_a_fenced_model_reply_is_read_as_thoughts_and_command():
    reply = 'Sure.\n```json\n{"thoughts": "look at it", "command": {"name": "read_file", "args": {"path": "a"}}}\n```'
    d = D.reply_object(reply)
    print(d)
    assert d["thoughts"] == "look at it" and d["command"]["name"] == "read_file"
    assert D.reply_object("no json here") is None


def test_workflow_rows_are_split_by_the_model_that_built_them():
    """Two models built the same task: each gets its own row, with only its own runs and the
    packaged set named after it."""
    from benchcompiler.ui.app import _builds

    class Stub:
        def funnel(self, task, keys):
            return {"keys": keys}

    runs = [
        {"key": "construct~t~e1", "model": "claude-code/claude-opus-5", "started": 2, "delivered_keys": ["a", "b"]},
        {"key": "experiment~x~t", "model": "openrouter/vendor/model-a", "started": 1, "delivered_keys": ["a"]},
        {"key": "construct~t~e2", "model": "claude-code/claude-opus-5", "started": 3, "delivered_keys": ["a"]},
    ]
    sets = [{"set": "opus", "task": "t", "count": 2}, {"set": "model", "task": "t", "count": 1}]
    rows = [
        (g["model"], g["funnel"]["keys"], g["funnel"]["distinct"], [b["set"] for b in g["benchmarks"]])
        for g in _builds(Stub(), "t", runs, sets)
    ]
    print(rows)
    assert rows == [
        ("model-a", ["experiment~x~t"], 1, ["model"]),
        ("claude-opus-5", ["construct~t~e1", "construct~t~e2"], 2, ["opus"]),
    ]


def test_drops_are_grouped_and_a_candidate_is_followed_end_to_end(tree):
    """The dropped candidate is grouped as dropped before any replay check; the packaged one is
    found from its commit in the search agent's stock, the frozen stock, the worker and the set."""
    from benchcompiler.ui import why as W

    root, runs, _ = tree
    t = D.Tree(root, runs)
    rows = W.drops(t, "demo", ["construct~demo~epoch_1"])
    groups = [(len(g["items"]), g["stage"], g["row"]) for g in W.drop_groups(rows)]
    print(groups, rows[0]["reason"])
    assert groups == [(1, 1, "no replay check")] and rows[0]["reason"] == "does not build"
    j = W.journey(t, "https://github.com/o/p @ " + "a" * 12)
    print({k: len(v) for k, v in j.items() if isinstance(v, list)})
    assert (len(j["frozen"]), len(j["built"]), len(j["packaged"])) == (1, 1, 1)
    assert j["built"][0]["outcome"] == "open" and j["packaged"][0]["id"] == "inst-1"
    c = TestClient(build_app(root, runs))
    for u in (
        "/task/demo?tab=drops",
        "/task/demo?tab=drops&run=construct~demo~epoch_1",
        "/candidate",
        "/candidate?q=o/p",
        "/health",
        "/compare",
        "/compare?task=demo",
        "/compare?task=nosuch",
    ):
        r = c.get(u)
        print(r.status_code, u)
        assert r.status_code == 200
    assert "does not build" in c.get("/task/demo?tab=drops").text


def test_a_replay_row_that_changed_is_marked(tmp_path):
    w = tmp_path / "worker_0"
    for i, ok in ((1, True), (2, False), (3, True)):
        _w(
            str(w / f"recipes/r/check_00{i}.json"),
            {"ok": ok, "ts": i, "rows": [{"name": "replays", "ok": True}, {"name": "oracle", "ok": ok}]},
        )
    rec = D.recipe_detail(str(w), "r")
    marks = [[row["change"] for row in c["rows"]] for c in rec["checks"]]
    print(marks)
    assert marks == [[None, None], [None, "newly failing"], [None, "now passes"]]


def test_the_json_cache_never_serves_a_changed_file(tmp_path):
    """A worker rewrites agent_state.json in place; the next read must see the new content even
    when the size is unchanged."""
    p = tmp_path / "agent_state.json"
    p.write_text(json.dumps({"cycle": 11}))
    assert D.read_json(str(p))["cycle"] == 11
    p.write_text(json.dumps({"cycle": 12}))
    os.utime(str(p), ns=(os.stat(str(p)).st_atime_ns, os.stat(str(p)).st_mtime_ns + 1_000_000))
    print(D.read_json(str(p)))
    assert D.read_json(str(p))["cycle"] == 12
    p.unlink()
    assert D.read_json(str(p), "gone") == "gone"


def test_the_static_export_links_its_pages_and_refuses_identifying_strings(tree, tmp_path, capsys):
    from benchcompiler.ui import export as X

    root, runs, _ = tree
    site = tmp_path / "site"
    code = X.export(root, runs, str(site), forbid=["Repair a real"])  # from the task description
    err = capsys.readouterr().err
    print(err[:400])
    assert code == 2 and "Repair a real" in err and not site.exists()  # nothing written
    assert X.export(root, runs, str(site), forbid=[]) == 0
    index = (site / "index.html").read_text()
    print(sorted(p.name for p in site.iterdir())[:8])
    assert 'href="task_demo.html"' in index and 'href="static/ui.css"' in index
    assert "/candidate" not in index and "Actions log" not in index and str(tmp_path) not in index
    assert (site / "construction_construct~demo~epoch_1__tab-graph.html").exists()
