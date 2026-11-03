"""`bench construct-resume` makes an interrupted run recoverable.

`bench construct` runs the agent, then the post-run pipeline (ledger append). `construct-resume`
goes through the same pipeline so a recovered run is part of the study ledger. These tests cover the
CLI contract and the shared post-run pipeline; resume mechanics are covered in
tests/test_resume_end_to_end.py.
"""

from __future__ import annotations

import argparse
import inspect

import pytest

from benchcompiler import bench_cli


def _parser_actions(name):
    """The subparser for `name`, via the real parser this CLI builds."""
    parser = bench_cli.build_parser() if hasattr(bench_cli, "build_parser") else None
    if parser is None:
        pytest.skip("bench_cli does not expose build_parser()")
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    if name not in sub.choices:
        pytest.fail(f"no `{name}` subcommand registered")
    return sub.choices[name]


def test_construct_resume_is_registered_as_a_subcommand():
    sp = _parser_actions("construct-resume")
    opts = {o for a in sp._actions for o in a.option_strings} | {a.dest for a in sp._actions if not a.option_strings}
    assert "run_dir" in opts, "must take the interrupted run's directory"
    assert "--out" in opts, "must be able to continue into a NEW dir, leaving the original intact"


def test_run_dir_is_positional_and_required_but_out_is_optional():
    """Resuming in place is the common case, so --out is optional; without a run_dir there is
    nothing to resume.
    """
    sp = _parser_actions("construct-resume")
    run_dir = next(a for a in sp._actions if a.dest == "run_dir")
    out = next(a for a in sp._actions if "--out" in a.option_strings)
    assert not run_dir.option_strings, "run_dir should be positional"
    assert out.required is False and out.default is None


def test_construct_and_construct_resume_share_one_post_run_pipeline():
    """Both paths delegate to the same post-run helper rather than each carrying its own copy, so
    resumed runs are comparable.
    """
    assert hasattr(bench_cli, "_post_construct")
    for fn in (bench_cli.cmd_construct, bench_cli.cmd_construct_resume):
        assert "_post_construct" in inspect.getsource(fn), (
            f"{fn.__name__} must delegate to the shared post-run pipeline"
        )


def test_post_construct_records_the_run_where_the_resumed_run_actually_lives(tmp_path, monkeypatch):
    """`construct-resume` derives its output dir from the run being resumed, so the shared post-run
    pipeline writes the ledger entry against that directory.
    """
    scanned = {}
    monkeypatch.setattr(bench_cli, "scan_root", lambda p: scanned.setdefault("root", p) or [])
    monkeypatch.setattr(bench_cli, "append_ledger", lambda *a, **k: None)

    result = type("R", (), {"success": False, "reason": "aborted", "steps": 3, "cost": 0.4, "working_set": {}})()
    args = argparse.Namespace(out=None, instance=None, model=None, ledger=str(tmp_path / "l.jsonl"))
    bench_cli._post_construct(args, result, out_dir=str(tmp_path / "resumed_run"))

    assert scanned["root"] == str(tmp_path / "resumed_run")


def test_an_honest_empty_sample_is_still_distinguished_from_a_missing_one(tmp_path, monkeypatch, capsys):
    """The record check tests key presence, not truthiness: a run that recorded zero instances is a
    real result, while one that never called record() violates the convention.
    """
    monkeypatch.setattr(bench_cli, "scan_root", lambda p: [])
    monkeypatch.setattr(bench_cli, "append_ledger", lambda *a, **k: None)
    args = argparse.Namespace(out=None, instance=None, model=None, ledger=str(tmp_path / "l.jsonl"))

    honest_zero = type(
        "R", (), {"success": True, "reason": "finished", "steps": 9, "cost": 1.0, "working_set": {"final_sample": []}}
    )()
    bench_cli._post_construct(args, honest_zero, out_dir=str(tmp_path / "r"))
    said = capsys.readouterr().out
    assert "0 item(s)" in said, said

    never_recorded = type(
        "R", (), {"success": False, "reason": "aborted", "steps": 2, "cost": 0.1, "working_set": {}}
    )()
    bench_cli._post_construct(args, never_recorded, out_dir=str(tmp_path / "r2"))
    said = capsys.readouterr().out
    assert "No final_sample recorded" in said, said


def test_ns_helper_does_not_mutate_the_callers_namespace():
    """`_post_construct` does not mutate the caller's args when it rewrites `out` for the resume
    path.
    """
    original = argparse.Namespace(out="/original", model="m")
    copy = bench_cli._ns(original, out="/derived")
    assert copy.out == "/derived" and copy.model == "m"
    assert original.out == "/original", "caller's namespace must be untouched"
