"""Persistent, cross-session experiment ledger.

Distinct from gate_judge.py (produces one verdict) and bulk_gate_review.py (reviews a bulk
harvest): this module never judges anything -- it only reads what earlier stages already
wrote (run_meta.json, result.json, gate_judge_result.json, manifest.json, flow step dirs)
and normalizes them into one row per attempt, appended to a JSONL ledger that lives in the
repo (study/experiment_ledger.jsonl by convention), not /tmp -- so it survives cleanup and
stays comparable across sessions.

Three run-output shapes are recognized (sniffed, not declared):
  - a `construct-parallel` root: manifest.json + attempt_N/ dirs, each a plain run_single
    output (run_meta.json/result.json/gate_judge_result.json directly inside).
  - an ad hoc flow-parallel root: attempt_N/ dirs each containing steps/<step_id>/ (no
    manifest.json -- these were launched by hand, not by a CLI command).
  - a single run_single output dir (no attempt_N/ children at all).
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Any


def _read_json(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _last_error_line(log_path: str) -> str | None:
    if not os.path.exists(log_path):
        return None
    with open(log_path, errors="replace") as f:
        lines = [l for l in f.read().strip().splitlines() if l.strip()]
    return lines[-1] if lines else None


# Why a run carries no judge verdict. Stated once so every unjudged row says the same thing.
_NO_JUDGE_REASON = (
    "no judge verdict recorded: correctness is owned by per-state validators, "
    "so this run was not assessed by a judge rather than assessed and found wanting"
)


def _extract_run(out_dir: str, *, task: str, variant: str, attempt: Any, config: str | None = None) -> dict:
    """A plain run_single-shaped output dir."""
    run_meta = _read_json(os.path.join(out_dir, "run_meta.json")) or {}
    result = _read_json(os.path.join(out_dir, "result.json"))
    gate = _read_json(os.path.join(out_dir, "gate_judge_result.json"))
    # judge_status.json (written by bench_cli's cmd_construct going forward) records WHY a
    # run wasn't judged, not just that it wasn't -- older runs won't have it, so fall back
    # to inferring from gate_judge_result.json's presence alone.
    judge_status = _read_json(os.path.join(out_dir, "judge_status.json"))
    ws = (result or {}).get("working_set") or {}

    if judge_status is not None:
        judged = bool(judge_status.get("judged"))
        unjudged_reason = None if judged else judge_status.get("reason")
    else:
        judged = gate is not None
        # SAY WHY IT IS ABSENT. `judged: False` with no reason reads as "the judge looked and
        # was not satisfied"; here it means "no judge exists", so the reason is recorded.
        #
        # The judge fields stay, so rows that do carry verdicts remain readable.
        unjudged_reason = None if judged else _NO_JUDGE_REASON

    return {
        "out_dir": out_dir,
        "task": task,
        "variant": variant,
        "attempt": attempt,
        "config": config or run_meta.get("config_path"),
        "status": "finished" if result is not None else "in_progress_or_crashed",
        "self_declared_success": (result or {}).get("success"),
        "self_declared_reason": (result or {}).get("reason"),
        "steps": (result or {}).get("steps"),
        "cost": (result or {}).get("cost"),
        # key PRESENCE, not truthiness -- an honestly-recorded empty final_sample=[] (e.g.
        # after an environment failure) must read as "recorded" here, distinct from never
        # calling record() at all. See bench_cli.py's cmd_construct for the full story.
        "recorded_quality_gates": "quality_gates" in ws,
        "recorded_final_sample": "final_sample" in ws,
        "sample_count": len(ws.get("final_sample") or []),
        "judged": judged,
        "unjudged_reason": unjudged_reason,
        "gate_verdict": (gate or {}).get("verdict"),
        "gate_pass_rate": (gate or {}).get("overall_pass_rate"),
        "gate_summary": (gate or {}).get("summary"),
    }


def _extract_flow_attempt(attempt_dir: str, *, task: str, attempt: Any) -> dict:
    flow_json = _read_json(os.path.join(attempt_dir, "flow.json"))
    step_status: dict[str, dict] = {}
    crashed_step = None
    for step_dir in sorted(glob.glob(os.path.join(attempt_dir, "steps", "*"))):
        step_id = os.path.basename(step_dir)
        result = _read_json(os.path.join(step_dir, "result.json"))
        run_meta = _read_json(os.path.join(step_dir, "run_meta.json"))
        if result is not None:
            step_status[step_id] = {"success": result.get("success"), "reason": result.get("reason")}
        elif run_meta is not None:
            step_status[step_id] = {"success": None, "reason": "crashed before result.json"}
            crashed_step = crashed_step or step_id
        else:
            step_status[step_id] = {"success": None, "reason": "no artifacts"}

    crashed_error = None
    if flow_json is None:
        crashed_error = _last_error_line(attempt_dir.rstrip("/") + ".log")

    if flow_json is not None:
        status = "finished"
    elif crashed_step or crashed_error:
        status = "crashed"
    else:
        status = "in_progress"

    return {
        "out_dir": attempt_dir,
        "task": task,
        "variant": "flow",
        "attempt": attempt,
        "status": status,
        "self_declared_success": (flow_json or {}).get("success"),
        "steps_status": step_status,
        "crashed_step": crashed_step,
        "crashed_error": crashed_error,
    }


def _task_name_from_root(root: str) -> str:
    base = os.path.basename(root.rstrip("/"))
    for marker in ("_construct_", "_flow_"):
        if marker in base:
            return base.split(marker)[0]
    return base


def scan_root(root: str, *, task: str | None = None) -> list[dict]:
    """Sniff one run-output root and return a normalized row per attempt found in it."""
    root = root.rstrip("/")
    task = task or _task_name_from_root(root)
    manifest = _read_json(os.path.join(root, "manifest.json"))

    if manifest is not None:
        return [
            _extract_run(
                a["out_dir"], task=task, variant="construct", attempt=a.get("index"), config=manifest.get("config")
            )
            for a in manifest.get("attempts", [])
        ]

    attempt_dirs = sorted(p for p in glob.glob(os.path.join(root, "attempt_*")) if os.path.isdir(p))
    if attempt_dirs:
        rows = []
        for attempt_dir in attempt_dirs:
            idx = attempt_dir.rsplit("_", 1)[-1]
            if os.path.isdir(os.path.join(attempt_dir, "steps")):
                rows.append(_extract_flow_attempt(attempt_dir, task=task, attempt=idx))
            else:
                rows.append(_extract_run(attempt_dir, task=task, variant="construct", attempt=idx))
        return rows

    return [_extract_run(root, task=task, variant="construct", attempt=1)]


def append_ledger(rows: list[dict], ledger_path: str) -> None:
    parent = os.path.dirname(ledger_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(ledger_path, "a") as f:
        for row in rows:
            row = {**row, "_scanned_at": time.time()}
            f.write(json.dumps(row, default=str) + "\n")


def load_ledger(ledger_path: str) -> list[dict]:
    if not os.path.exists(ledger_path):
        return []
    rows = []
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _short(value: Any, width: int) -> str:
    s = "" if value is None else str(value)
    return s if len(s) <= width else s[: width - 1] + "…"


def format_table(rows: list[dict]) -> str:
    cols = [
        ("task", 16),
        ("variant", 9),
        ("attempt", 7),
        ("status", 12),
        ("self_declared_success", 7),
        ("judged", 6),
        ("gate_verdict", 10),
        ("cost", 8),
        ("steps", 6),
    ]
    header = " | ".join(f"{name:<{w}}" for name, w in cols)
    lines = [header, "-" * len(header)]
    for r in rows:
        verdict = r.get("gate_verdict") or r.get("crashed_error")
        if not verdict and r.get("unjudged_reason"):
            verdict = f"UNJUDGED:{r['unjudged_reason']}"
        verdict = verdict or ""
        vals = {
            "task": r.get("task"),
            "variant": r.get("variant"),
            "attempt": r.get("attempt"),
            "status": r.get("status"),
            "self_declared_success": r.get("self_declared_success"),
            "judged": r.get("judged"),
            "gate_verdict": verdict,
            "cost": r.get("cost"),
            "steps": r.get("steps"),
        }
        lines.append(" | ".join(f"{_short(vals[name], w):<{w}}" for name, w in cols))
    return "\n".join(lines)
