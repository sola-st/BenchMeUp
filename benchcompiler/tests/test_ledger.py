"""Offline tests for the persistent experiment ledger. Fixtures mirror two run-output shapes: a
`construct-parallel` root (manifest.json + attempt_N/ run_single dirs) and a flow-parallel root
(attempt_N/steps/<id>/ dirs, no manifest).
"""

import json
import os

from benchcompiler.ledger import append_ledger, format_table, load_ledger, scan_root


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def test_scan_construct_parallel_root_reads_manifest_and_gate_verdict(tmp_path):
    root = tmp_path / "boca_flags_construct_v3"
    a1 = root / "attempt_1"
    _write_json(
        root / "manifest.json",
        {
            "config": "agents/boca_flags_collector/agent.yaml",
            "attempts": [{"index": 1, "out_dir": str(a1)}],
        },
    )
    _write_json(
        a1 / "result.json",
        {
            "success": True,
            "reason": "finished",
            "steps": 90,
            "cost": 0.69,
            "working_set": {"quality_gates": ["g1"], "final_sample": [{}, {}, {}]},
        },
    )
    _write_json(
        a1 / "gate_judge_result.json",
        {"verdict": "NEEDS_FIX", "overall_pass_rate": "1/24", "summary": "missing artifacts"},
    )

    rows = scan_root(str(root))
    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == "boca_flags"
    assert row["variant"] == "construct"
    assert row["self_declared_success"] is True
    assert row["judged"] is True
    assert row["gate_verdict"] == "NEEDS_FIX"
    assert row["sample_count"] == 3


def test_scan_construct_parallel_root_flags_unjudged_when_recording_convention_skipped(tmp_path):
    root = tmp_path / "swtbench_repro_construct_v3"
    a1 = root / "attempt_1"
    _write_json(root / "manifest.json", {"config": "x.yaml", "attempts": [{"index": 1, "out_dir": str(a1)}]})
    _write_json(
        a1 / "result.json", {"success": True, "reason": "finished", "steps": 62, "cost": 0.46, "working_set": {}}
    )

    row = scan_root(str(root))[0]
    assert row["recorded_quality_gates"] is False
    assert row["recorded_final_sample"] is False
    assert row["judged"] is False
    assert row["gate_verdict"] is None


def test_scan_prefers_judge_status_json_reason_when_present(tmp_path):
    """judge_status.json makes the reason a run was not judged explicit, instead of the ledger
    inferring it from an absent file.
    """
    root = tmp_path / "some_construct_v4"
    a1 = root / "attempt_1"
    _write_json(root / "manifest.json", {"config": "x.yaml", "attempts": [{"index": 1, "out_dir": str(a1)}]})
    _write_json(
        a1 / "result.json", {"success": True, "reason": "finished", "steps": 10, "cost": 0.1, "working_set": {}}
    )
    _write_json(a1 / "judge_status.json", {"judged": False, "reason": "recording_convention_not_followed"})

    row = scan_root(str(root))[0]
    assert row["judged"] is False
    assert row["unjudged_reason"] == "recording_convention_not_followed"


def test_scan_flow_parallel_root_ignores_sibling_log_files_as_attempts(tmp_path):
    root = tmp_path / "boca_flags_flow_v1"
    for i in (1, 2):
        attempt = root / f"attempt_{i}"
        _write_json(attempt / "steps" / "brainstorm" / "result.json", {"success": True, "reason": "finished"})
        _write_json(attempt / "steps" / "build" / "run_meta.json", {"config_path": "x"})
        (root / f"attempt_{i}.log").parent.mkdir(parents=True, exist_ok=True)
        (root / f"attempt_{i}.log").write_text(
            "Traceback (most recent call last):\n"
            "AttributeError: module 'instances.schema' has no attribute 'BuildTask'\n"
        )

    rows = scan_root(str(root))
    assert len(rows) == 2  # not 4 -- the *.log files must not be picked up as attempt dirs
    for row in rows:
        assert row["variant"] == "flow"
        assert row["status"] == "crashed"
        assert row["crashed_step"] == "build"
        assert "BuildTask" in row["crashed_error"]


def test_append_and_load_ledger_roundtrip(tmp_path):
    ledger_path = str(tmp_path / "study" / "experiment_ledger.jsonl")
    append_ledger([{"task": "a", "variant": "construct"}], ledger_path)
    append_ledger([{"task": "b", "variant": "flow"}], ledger_path)

    rows = load_ledger(ledger_path)
    assert [r["task"] for r in rows] == ["a", "b"]
    assert all("_scanned_at" in r for r in rows)


def test_format_table_does_not_crash_on_missing_fields():
    out = format_table([{"task": "t", "variant": "construct"}])
    assert "t" in out
    assert "variant" in out
