"""A field a ratio clause compares must be derived by a `timed` step, the harness's own clock.

A number printed by an agent's command is the agent's number; a `timed` step uses the harness's
clock. Every field a cross-field ratio clause names must be derived by a `timed` step. The
diagnostic is model-authored and its fix hint names the step to write.
"""

from __future__ import annotations

import json

from benchcompiler.spec import check_canonical

CODE = "UNEXECUTED_COMPARISON"


def _canonical(corroboration):
    base = json.load(open("benchcompiler/examples/boca_flags.canonical.json"))["canonical_spec"]
    cs = json.loads(json.dumps(base))
    cs["entry_contract"] = [
        {"field": "repository_url", "require": "is a clone url"},
        {"field": "commit_hash", "require": "is a git sha"},
        {"field": "clone_path", "require": "is non-empty"},
        {"field": "build_cmd", "require": "is non-empty"},
        {"field": "run_scaled_cmd", "require": "is non-empty"},
        {"field": "probe_baseline_ms", "require": "is a number"},
        {"field": "probe_optimized_ms", "require": "is a number"},
        {"field": "probe_baseline_ms", "require": "is at least 1.05 times field probe_optimized_ms"},
    ]
    cs["sampling"]["lead_requires"] = [
        "repository_url",
        "commit_hash",
        "clone_path",
        "build_cmd",
        "run_scaled_cmd",
        "probe_baseline_ms",
        "probe_optimized_ms",
    ]
    cs["corroboration_contract"] = corroboration
    return cs


BASE = [
    {"field": "clone_path", "require": "is non-empty"},
    {"run": "cd {clone_path} && env CFLAGS=-O0 sh -c {build_cmd}", "require": "exit zero", "timeout": 1800},
]

RUN_DERIVED = BASE + [
    {
        "run": "cd {clone_path} && sh -c {run_scaled_cmd}",
        "require": "exit zero and some output line",
        "derive": "probe_baseline_ms",
    },
    {"run": "cd {clone_path} && env CFLAGS=-O3 sh -c {build_cmd}", "require": "exit zero", "timeout": 1800},
    {
        "run": "cd {clone_path} && sh -c {run_scaled_cmd}",
        "require": "exit zero and some output line",
        "derive": "probe_optimized_ms",
    },
]

TIMED = BASE + [
    {
        "timed": "{run_scaled_cmd}",
        "cwd": "{clone_path}",
        "repetitions": 3,
        "derive": "probe_baseline_ms",
        "timeout": 1800,
    },
    {"run": "cd {clone_path} && env CFLAGS=-O3 sh -c {build_cmd}", "require": "exit zero", "timeout": 1800},
    {
        "timed": "{run_scaled_cmd}",
        "cwd": "{clone_path}",
        "repetitions": 3,
        "derive": "probe_optimized_ms",
        "timeout": 1800,
    },
]


def _codes(cs):
    return [(d.code, d.field, d.detail[:200], d.fix_hint[:200]) for d in check_canonical(cs)]


def test_a_compared_field_from_a_run_steps_output_is_refused_and_the_hint_names_timed():
    got = [g for g in _codes(_canonical(RUN_DERIVED)) if g[0] == CODE]
    print("\n[verbose]", got)
    assert got, "a compared number printed by the agent's own command passed"
    assert "timed" in (got[0][2] + got[0][3])


def test_compared_fields_from_timed_steps_pass_every_check():
    got = _codes(_canonical(TIMED))
    print("\n[verbose]", got)
    assert not [g for g in got if g[0] in (CODE, "UNREADABLE_CONTRACT")], got


def test_a_timed_step_is_a_known_kind_with_its_own_shape_checks():
    bad = BASE + [{"timed": "{run_scaled_cmd}", "cwd": "{clone_path}"}]  # no derive
    got = [g for g in _codes(_canonical(bad)) if g[0] == "UNREADABLE_CONTRACT" and "timed" in g[2]]
    print("\n[verbose]", got)
    assert got, "a timed step that derives nothing was accepted"
