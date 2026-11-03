"""The spec declares what each variant's check is judged on; this is refused at formalization, the
earliest gate.

`artifact_variants` pass verbatim from the raw yaml through canonical and design to the collector,
so the evidence declaration is validated from the formalizer onward.
"""

from benchcompiler.spec import check_canonical


def _canon(variants):
    return {"oracle_strategy": "fail_to_pass", "artifact_variants": variants}


def test_a_variant_without_results_is_rejected_at_formalization():
    diags = check_canonical(_canon([{"name": "buggy", "check_cmd": "sh check_buggy.sh .", "expect": "pass"}]))
    codes = [d.code for d in diags]
    print(f"\n[verbose] diagnostics: {[(d.code, d.detail[:80]) for d in diags]}")
    assert "UNDECLARED_CHECK_EVIDENCE" in codes
    hit = next(d for d in diags if d.code == "UNDECLARED_CHECK_EVIDENCE")
    assert "exit code" in hit.detail
    assert "junit-xml" in hit.fix_hint


def test_junit_xml_with_polarity_passes():
    diags = check_canonical(
        _canon([{"name": "buggy", "check_cmd": "x", "expect": "pass", "results": "junit-xml", "expect_failures": 1}])
    )
    assert not [d for d in diags if d.code == "UNDECLARED_CHECK_EVIDENCE"]


def test_none_needs_its_reason_at_formalization_too():
    diags = check_canonical(_canon([{"name": "default", "check_cmd": "x", "expect": "pass", "results": "none"}]))
    assert any(d.code == "UNDECLARED_CHECK_EVIDENCE" and "reason" in d.detail for d in diags)


# ── criteria must state INSTANCE guarantees, not restate source filters ──────


def _canon_full(criteria, admissibility):
    return {
        "oracle_strategy": "fail_to_pass",
        "artifact_variants": [
            {"name": "buggy", "check_cmd": "x", "expect": "pass", "results": "junit-xml", "expect_failures": 1}
        ],
        "criteria": criteria,
        "admissibility": admissibility,
    }


ADMIS = [
    {"name": "language", "observe": "GET /repos -> language", "require": "java"},
    {"name": "stars", "observe": "GET /repos -> stargazers_count", "require": ">= 100"},
    {"name": "is_fork", "observe": "GET /repos -> fork", "require": "false"},
]


def test_criteria_that_only_restate_admissibility_are_refused():
    """Criteria that are all source filters (language, stars, is_fork, ...) and omit the instance-
    defining property are refused: criteria drive which properties a lifecycle must enforce.
    """
    diags = check_canonical(
        _canon_full(
            ["language: must be one of [Java]", "stars: stargazers_count >= 100", "is_fork: fork must be false"], ADMIS
        )
    )
    codes = [d.code for d in diags]
    print(f"\n[verbose] {[(d.code, d.detail[:90]) for d in diags]}")
    assert "CRITERIA_ARE_SOURCE_FILTERS" in codes
    hit = next(d for d in diags if d.code == "CRITERIA_ARE_SOURCE_FILTERS")
    assert "instance" in hit.detail.lower()


def test_instance_level_criteria_pass():
    """A list mixing instance-level criteria with one source filter is judged per criterion: the
    instance-level ones pass, the `language:` one does not.
    """
    instance_level = [
        "The fix commit must add or modify at least one triggering test which fails "
        "on the commit's parent and passes on the commit itself",
        "Both revisions must build successfully",
    ]
    assert "CRITERIA_ARE_SOURCE_FILTERS" not in [d.code for d in check_canonical(_canon_full(instance_level, ADMIS))]

    diags = check_canonical(_canon_full(instance_level + ["language: must be one of [Java]"], ADMIS))
    hits = [d for d in diags if d.code == "CRITERIA_ARE_SOURCE_FILTERS"]
    print(f"\n[verbose] one filter among two good criteria -> {[d.detail[:80] for d in hits]}")
    assert len(hits) == 1, [d.detail for d in hits]
    assert "language" in hits[0].detail


def test_no_admissibility_block_means_no_opinion():
    diags = check_canonical(_canon_full(["language: java", "stars: >= 100"], []))
    assert "CRITERIA_ARE_SOURCE_FILTERS" not in [d.code for d in diags]
