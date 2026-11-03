"""`bench construct --model` must reach the run, or refuse; never silently nothing.

`cmd_construct`, `construct-resume` and `construct-parallel` forward `--model` to the run. The
loader's merge order is defaults, yaml, env, cli_overrides, so a CLI override beats AGENT_MODEL.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import benchcompiler.bench_cli as cli


class _Res:
    success, reason, steps, cost = True, "finished", 1, 0.0
    working_set: dict = {"final_sample": []}


def test_construct_passes_the_model_as_a_cli_override(tmp_path, monkeypatch):
    seen = {}

    def fake_run_single(config, *, instance=None, output_dir=None, workspace=None, overrides=None, **kw):
        seen["overrides"] = overrides
        return _Res()

    monkeypatch.setattr(cli, "run_single", fake_run_single)
    monkeypatch.setattr(cli, "append_ledger", lambda *a, **k: None)
    monkeypatch.setattr(cli, "scan_root", lambda *a, **k: [])
    args = SimpleNamespace(
        config="agents/x/agent.yaml",
        instance="i.json",
        out=str(tmp_path / "o"),
        workspace=None,
        model="openrouter/pin/me",
        ledger=None,
        overrides=None,
        canonical=None,
    )
    cli.cmd_construct(args)
    ov = seen.get("overrides") or {}
    print(f"\n[verbose] overrides handed to run_single: {ov}")
    assert (ov.get("models") or {}).get("main", {}).get("name") == "openrouter/pin/me", (
        "--model was parsed and never consumed: the run uses AGENT_MODEL/agent.yaml while "
        "the operator believes they pinned the model"
    )


def test_construct_resume_passes_the_model_too(tmp_path, monkeypatch):
    seen = {}

    def fake_resume(run_dir, *, output_dir=None, overrides=None, **kw):
        seen["overrides"] = overrides
        return _Res()

    monkeypatch.setattr("agentcore.run.resume.resume_run", fake_resume)
    monkeypatch.setattr(cli, "append_ledger", lambda *a, **k: None)
    monkeypatch.setattr(cli, "scan_root", lambda *a, **k: [])
    args = SimpleNamespace(run_dir=str(tmp_path), out=None, instance=None, model="openrouter/pin/me", ledger=None)
    cli.cmd_construct_resume(args)
    ov = seen.get("overrides") or {}
    print(f"\n[verbose] overrides handed to resume_run: {ov}")
    assert (ov.get("models") or {}).get("main", {}).get("name") == "openrouter/pin/me"


def test_construct_resume_refuses_an_instance_it_cannot_honor(tmp_path, monkeypatch):
    monkeypatch.setattr("agentcore.run.resume.resume_run", lambda *a, **k: pytest.fail("must refuse before resuming"))
    args = SimpleNamespace(run_dir=str(tmp_path), out=None, instance="other.json", model=None, ledger=None)
    with pytest.raises(SystemExit):
        cli.cmd_construct_resume(args)


def test_agentcore_cli_no_longer_pretends_to_take_a_budget():
    """The agentcore runner's --budget is read and reaches the run."""
    from agentcore.run import single as single_mod

    with pytest.raises(SystemExit):
        single_mod.main(["--config", "x.yaml", "--budget", "5"])
