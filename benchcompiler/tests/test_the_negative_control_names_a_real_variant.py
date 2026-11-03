"""`negative_control.check` must name a variant that exists, and it is checked at stage 1.

`check_canonical` cross-validates `negative_control.check` against the declared `artifact_variants`;
otherwise the name becomes `run: ["rebuild-<v>", "check-<v>"]` verbs no instance ships, and the
packaging check rejects it only after construction has been paid for.
"""

from __future__ import annotations

import pytest

from benchcompiler.spec import check_canonical

VARIANTS = [
    {
        "name": "prefix",
        "check_cmd": "sh check_prefix.sh .",
        "expect": "pass",
        "results": "junit-xml",
        "expect_failures": 1,
    },
    {
        "name": "postfix",
        "check_cmd": "sh check_postfix.sh .",
        "expect": "pass",
        "results": "junit-xml",
        "expect_failures": 0,
    },
]


def _canon(check):
    return {
        "oracle_strategy": "fail-to-pass-tests",
        "task": "t",
        # Asked of every canonical spec: the formalizer's own output must say what the benchmark is.
        "candidate_definition": "one fix commit",
        "instance_input": "a buggy tree",
        "instance_output": "a patch",
        "data_source": "fix commits",
        "oracle_detail": "the project's own test suite decides",
        "criteria": ["the added test fails before the fix and passes after it"],
        "admissibility": [{"name": "stars", "observe": "GET /repos/{o}/{p} -> stargazers_count", "require": ">= 100"}],
        "report_discovery": {"junit-xml": ["*surefire-reports/TEST-*.xml", "TEST-*.xml"]},
        "artifact_variants": VARIANTS,
        "solver_input": {
            "kind": "workdir",
            "path": "{clone_path}",
            "describes": "the solver edits the checked-out tree",
        },
        "negative_control": {"break_cmd": "corruption_cmd", "check": check, "describes": "revert the fix"},
    }


def _codes(d):
    return {x.code for x in check_canonical(d)}


def test_a_control_naming_a_variant_that_does_not_exist_is_refused():
    codes = _codes(_canon("fixed"))
    print("check: 'fixed' against variants prefix/postfix ->", sorted(codes))
    assert "NO_NEGATIVE_CONTROL" in codes


def test_the_refusal_names_the_variants_that_do_exist():
    """The refusal names the fix: which variant name to use."""
    d = [x for x in check_canonical(_canon("fixed")) if x.code == "NO_NEGATIVE_CONTROL"][0]
    text = d.detail + d.fix_hint
    print(text)
    assert "prefix" in text and "postfix" in text and "fixed" in text


@pytest.mark.parametrize("name", ["prefix", "postfix"])
def test_a_control_naming_a_real_variant_passes(name):
    assert _codes(_canon(name)) == set(), name


def test_a_control_that_names_no_variant_at_all_is_left_alone():
    """`check` is optional (`artifact_plan` falls back to the first variant); only a wrong name is
    refused.
    """
    canonical = _canon("prefix")
    canonical["negative_control"].pop("check")
    assert _codes(canonical) == set()


def _examples_declaring_both():
    """Every shipped spec that declares both blocks names only variants it ships, and the set
    measured over is not empty.
    """
    import glob
    import os

    from benchcompiler.spec import load_raw_spec

    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for path in sorted(glob.glob(os.path.join(here, "..", "examples", "*.yaml"))):
        raw = load_raw_spec(path)
        if raw.get("artifact_variants") and isinstance(raw.get("negative_control"), dict):
            out.append((os.path.basename(path), raw))
    return out


def test_the_sweep_has_specs_in_it():
    """Guards the premise -- an empty population makes the check below vacuous."""
    names = [n for n, _ in _examples_declaring_both()]
    print("specs declaring both blocks:", names)
    assert len(names) >= 3, names


def test_no_shipped_spec_names_a_variant_it_does_not_ship():
    """No shipped spec names a variant it does not ship, read from the files over all of them."""
    wrong = []
    for name, raw in _examples_declaring_both():
        declared = {v["name"] for v in raw["artifact_variants"] if isinstance(v, dict)}
        check = str(raw["negative_control"].get("check") or "").strip()
        print(f"{name}: variants {sorted(declared)} control checks {check!r}")
        if check and check not in declared:
            wrong.append(f"{name}: {check!r} not in {sorted(declared)}")
    assert not wrong, wrong
