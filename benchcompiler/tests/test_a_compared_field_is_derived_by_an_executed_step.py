"""A field a ratio clause compares must be derived by an executed corroboration step.

Every field named by a cross-field ratio clause ("<a> is at least N times field <b>") must be
provided by a corroboration `run` step's `derive` (or `save`), so the compared number comes from an
execution the harness ran, never from an agent-typed value. The diagnostic is model-authored, so it
buys a retry round with feedback naming the fields rather than refusing the spec.
"""

from __future__ import annotations

import json

from benchcompiler.formalizer import MODEL_AUTHORED_CODES
from benchcompiler.spec import check_canonical

CODE = "UNEXECUTED_COMPARISON"


def _canonical(corroboration):
    base = json.load(open("benchcompiler/examples/boca_flags.canonical.json"))["canonical_spec"]
    cs = json.loads(json.dumps(base))
    cs["entry_contract"] = [
        {"field": "repo_url", "require": "is a clone url"},
        {"field": "commit_hash", "require": "is a git sha"},
        {"field": "clone_path", "require": "is non-empty"},
        {"field": "probe_baseline_ms", "require": "is a number"},
        {"field": "probe_optimized_ms", "require": "is a number"},
        {"field": "probe_baseline_ms", "require": "is at least 1.05 times field probe_optimized_ms"},
    ]
    cs["corroboration_contract"] = corroboration
    return cs


TYPED = [
    {"field": "clone_path", "require": "is non-empty"},
    {
        "run": "git -C {clone_path} rev-parse --verify {commit_hash}^{commit}",
        "require": "output is a git sha",
        "derive": "resolved_commit",
    },
]

# Only a declared `timed` step counts as a derivation a ratio clause may compare
# (see test_a_compared_field_is_timed_by_the_harness).
EXECUTED = TYPED + [
    {"run": "cd {clone_path} && env CFLAGS=-O0 sh -c {probe_build_cmd}", "require": "exit zero", "timeout": 1800},
    {
        "timed": "{probe_workload_cmd}",
        "cwd": "{clone_path}",
        "repetitions": 3,
        "derive": "probe_baseline_ms",
        "timeout": 1800,
    },
    {"run": "cd {clone_path} && env CFLAGS=-O3 sh -c {probe_build_cmd}", "require": "exit zero", "timeout": 1800},
    {
        "timed": "{probe_workload_cmd}",
        "cwd": "{clone_path}",
        "repetitions": 3,
        "derive": "probe_optimized_ms",
        "timeout": 1800,
    },
]


def _codes(cs):
    diags = check_canonical(cs)
    return [(d.code, d.field, d.detail[:160]) for d in diags]


def test_typed_compared_fields_are_refused_and_name_the_fields():
    got = _codes(_canonical(TYPED))
    print("\n[verbose]", got)
    mine = [g for g in got if g[0] == CODE]
    assert mine, f"a ratio over typed fields passed: {got}"
    assert "probe_baseline_ms" in mine[0][2] and "probe_optimized_ms" in mine[0][2]


def test_executed_compared_fields_pass():
    got = _codes(_canonical(EXECUTED))
    print("\n[verbose]", got)
    assert not [g for g in got if g[0] == CODE], got


def test_the_code_is_the_models_to_fix():
    """The corroboration contract is the model's projection of the spec, so this
    diagnostic must buy a retry round, never refuse the researcher's spec."""
    assert CODE in MODEL_AUTHORED_CODES


def test_a_canonical_with_no_ratio_clause_is_untouched():
    cs = _canonical(TYPED)
    cs["entry_contract"] = [c for c in cs["entry_contract"] if "times field" not in c["require"]]
    got = _codes(cs)
    assert not [g for g in got if g[0] == CODE], got
