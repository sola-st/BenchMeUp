"""A run-step template never quotes a placeholder and runs a recorded command via `sh -c {field}`.

The engine shell-quotes every substituted value, so (1) a `{placeholder}` inside a single- or
double-quoted region is nested quoting, and (2) a `{placeholder}` standing as a whole command (at
the start, or after `&&`, `||`, `;` or `|`) runs the recorded string as one word. Both are
UNREADABLE_CONTRACT, model-authored, with the correct shape in the fix hint.
"""

from __future__ import annotations

import json

from benchcompiler.spec import check_canonical


def _canonical(steps):
    base = json.load(open("benchcompiler/examples/boca_flags.canonical.json"))["canonical_spec"]
    cs = json.loads(json.dumps(base))
    cs["corroboration_contract"] = [
        {"field": "clone_path", "require": "is non-empty"},
        {"field": "measure_cmd", "require": "is non-empty"},
        *steps,
    ]
    cs["sampling"]["lead_requires"] = sorted(
        set(cs["sampling"].get("lead_requires") or []) | {"clone_path", "measure_cmd"}
    )
    return cs


def _mine(cs):
    return [
        (d.code, d.detail[:220], d.fix_hint[:160])
        for d in check_canonical(cs)
        if d.field == "corroboration_contract" and "quot" in (d.detail + d.fix_hint).lower()
    ]


def test_a_placeholder_inside_single_quotes_is_refused():
    got = _mine(
        _canonical(
            [{"run": "sh -c 'cd {clone_path} && {measure_cmd}'", "require": "output matches ^[0-9]+$", "derive": "ms"}]
        )
    )
    print("\n[verbose]", got)
    assert got and got[0][0] == "UNREADABLE_CONTRACT", got
    assert "sh -c {" in got[0][2] or "sh -c {" in got[0][1]


def test_a_placeholder_inside_double_quotes_is_refused():
    got = _mine(_canonical([{"run": 'bash -c "cd {clone_path} && make"', "require": "exit zero"}]))
    print("\n[verbose]", got)
    assert got and got[0][0] == "UNREADABLE_CONTRACT", got


def test_a_placeholder_standing_as_a_whole_command_is_refused():
    got = _mine(
        _canonical([{"run": "cd {clone_path} && {measure_cmd}", "require": "output matches ^[0-9]+$", "derive": "ms"}])
    )
    print("\n[verbose]", got)
    assert got and got[0][0] == "UNREADABLE_CONTRACT", got
    assert "sh -c {measure_cmd}" in (got[0][1] + got[0][2])


def test_the_correct_shape_passes():
    got = _mine(
        _canonical(
            [
                {"run": "cd {clone_path} && sh -c {measure_cmd}", "require": "output matches ^[0-9]+$", "derive": "ms"},
                {"run": "cd {clone_path} && env CFLAGS=-O0 sh -c {measure_cmd}", "require": "exit zero"},
                {"run": "git -C {clone_path} rev-parse HEAD", "require": "output is a git sha", "derive": "sha"},
            ]
        )
    )
    print("\n[verbose]", got)
    assert got == []


def test_a_git_peel_suffix_is_not_a_quoted_placeholder():
    got = _mine(
        _canonical(
            [
                {
                    "run": "git -C {clone_path} rev-parse --verify {commit_hash}^{commit}",
                    "require": "output is a git sha",
                    "derive": "sha",
                }
            ]
        )
    )
    assert got == []
