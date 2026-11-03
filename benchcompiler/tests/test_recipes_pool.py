"""A pool of recipe instances merges, certifies from Dockerfiles alone, and its `bench` CLI runs the
instance's own verbs.

A synthetic fixture stands in for a delivered instance. Docker tier.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SYNTH = os.path.join(ROOT, "tests", "fixtures", "recipes", "synth")

pytestmark = pytest.mark.docker

# The tests do not queue behind the host-wide measurement lane: the fixtures' ratios are 2x and up,
# so serialized timing adds nothing.
os.environ.setdefault("AGENTCORE_MEASUREMENT_LANE", "off")


def _fake_run(tmp_path) -> str:
    run = tmp_path / "worker_0"
    inst = run / "recipes" / "fixture-synth-mix.c-fixture-deadbeef" / "instance"
    shutil.copytree(SYNTH, inst)
    os.remove(inst / "contract.yaml")
    (inst.parent / "report.json").write_text(json.dumps({"ok": True, "rows": []}))
    return str(run)


def test_merge_then_certify_from_dockerfiles_alone(tmp_path):
    from agentcore import recipes
    from benchcompiler import recipes_pool

    run = _fake_run(tmp_path)
    pool = str(tmp_path / "pool")
    m = recipes_pool.merge([run, run], pool, os.path.join(SYNTH, "contract.yaml"), name="synth-pool")
    print(json.dumps({k: v for k, v in m.items() if k != "verbs"}))
    assert m["instances"] == 1 and len(m["dropped_duplicates"]) == 1
    assert os.access(os.path.join(pool, "bench"), os.X_OK)
    assert os.path.isfile(os.path.join(pool, "README.md")) and os.path.isfile(os.path.join(pool, "contract.yaml"))
    inst = json.load(open(os.path.join(pool, "instances.json")))[0]
    d = os.path.join(pool, inst["id"])
    # forget every cached image so certification must rebuild from the Dockerfile
    r = recipes.Replay(d, recipes.load_contract(os.path.join(pool, "contract.yaml")))
    for tag in (r.snapshot_tag(), r.image_tag()):
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
    s = recipes_pool.certify(pool, log=print)
    print(s)
    assert s == {**s, "certified": 1, "checked": 1, "failed": []}
    cert = json.load(open(os.path.join(pool, "certify.json")))
    assert cert["results"][0]["summary"]["ratio"] >= 1.05


def test_the_pool_cli_runs_the_instances_own_verbs(tmp_path):
    from benchcompiler import recipes_pool

    run = _fake_run(tmp_path)
    pool = str(tmp_path / "pool")
    recipes_pool.merge([run], pool, os.path.join(SYNTH, "contract.yaml"))
    iid = json.load(open(os.path.join(pool, "instances.json")))[0]["id"]
    bench = [sys.executable, os.path.join(pool, "bench")]

    def run_cli(*args, check=True):
        r = subprocess.run([*bench, *args], capture_output=True, text=True)
        print(">", " ".join(args), "->", r.returncode, r.stdout.strip()[:300], r.stderr.strip()[-300:])
        if check:
            assert r.returncode == 0, r.stderr
        return r

    try:
        assert "build <dir> <flags>" in run_cli("verbs").stdout
        assert iid in run_cli("list").stdout
        run_cli("build", iid, "/work/v1", "-O0")
        run_cli("gen_inputs", iid, "/work/inputs")
        out = run_cli("run", iid, "/work/v1", "/work/inputs/in_1k.bin").stdout
        assert len(out.split()) == 2 and out.split()[1] == "1024"
        r = run_cli("run", iid, "/work/v1", check=False)
        assert r.returncode == 2 and "takes" in r.stderr
    finally:
        run_cli("stop", iid, check=False)
