"""A benchmark declares what a run must report, and what a consumer may vary.

The spec carries two task-neutral lists:

    observables:   named values a run must emit    {name, kind, aggregate?, unit?}
    parameters:    named knobs a consumer may set  {name, kind, default, min?, max?}

Transport is a line protocol: `BENCH_OBS <name>=<value>` on stdout, one per sample, and
`BENCH_<NAME>` env vars carrying a parameter into the script. The harness parses, aggregates numbers
by the declared `aggregate`, and reports; it never learns what a particular observable means.
"""

from __future__ import annotations

import pytest

from benchcompiler.observables import (
    ObservableError,
    aggregate_samples,
    check_observables,
    parameter_env,
    parse_observations,
)

# ── the line protocol ────────────────────────────────────────────────────────────────────────


def test_a_single_observation_is_read():
    out = "building...\nBENCH_OBS elapsed_ms=41.2\ndone\n"
    assert parse_observations(out) == {"elapsed_ms": ["41.2"]}


def test_repeated_names_accumulate_as_samples():
    """One line per run is how a repetition loop reports; the harness aggregates."""
    out = "\n".join(f"BENCH_OBS elapsed_ms={v}" for v in (10, 11, 9, 10, 40))
    assert parse_observations(out)["elapsed_ms"] == ["10", "11", "9", "10", "40"]


def test_other_output_is_ignored():
    """A check script prints whatever it likes; only the marked lines are data."""
    out = "gcc -O0 main.c\nBENCH_OBS output_digest=abc\nCHECK_DEFAULT_PASS\n"
    assert parse_observations(out) == {"output_digest": ["abc"]}


def test_a_value_may_contain_equals_signs():
    out = "BENCH_OBS build_flags=-O2 -DNDEBUG=1\n"
    assert parse_observations(out)["build_flags"] == ["-O2 -DNDEBUG=1"]


def test_whitespace_around_the_value_is_trimmed():
    assert parse_observations("BENCH_OBS  elapsed_ms =  12  \n")["elapsed_ms"] == ["12"]


def test_an_empty_value_is_kept_not_dropped():
    """An observable that legitimately has no content still says the run reported it. Dropping
    it would read as "never emitted", which is a different and more serious finding."""
    assert parse_observations("BENCH_OBS note=\n") == {"note": [""]}


def test_a_malformed_line_is_ignored_rather_than_fatal():
    """A check script is written by a model. One bad line must not lose the good ones."""
    out = "BENCH_OBS\nBENCH_OBS =5\nBENCH_OBS elapsed_ms=7\n"
    assert parse_observations(out) == {"elapsed_ms": ["7"]}


def test_the_protocol_is_line_oriented_so_values_cannot_span_lines():
    """A multi-line excerpt would silently swallow the following observations."""
    out = "BENCH_OBS excerpt=first\nsecond\nBENCH_OBS elapsed_ms=3\n"
    got = parse_observations(out)
    assert got["excerpt"] == ["first"]
    assert got["elapsed_ms"] == ["3"], "a stray newline must not eat the next observation"


# ── aggregation ──────────────────────────────────────────────────────────────────────────────


def test_median_of_numeric_samples():
    assert aggregate_samples(["10", "11", "9"], "median") == 10.0


def test_median_of_an_even_count():
    assert aggregate_samples(["10", "20"], "median") == 15.0


@pytest.mark.parametrize("how,expected", [("min", 9.0), ("max", 11.0), ("mean", 10.0), ("count", 3), ("last", "9")])
def test_the_other_aggregations(how, expected):
    assert aggregate_samples(["10", "11", "9"], how) == expected


def test_no_aggregate_keeps_every_sample():
    """A string observable has no meaningful aggregate; the consumer sees what was emitted."""
    assert aggregate_samples(["abc", "abc"], None) == ["abc", "abc"]


def test_a_non_numeric_sample_under_a_numeric_aggregate_is_an_error_naming_the_value():
    with pytest.raises(ObservableError) as e:
        aggregate_samples(["10", "fast"], "median")
    assert "fast" in str(e.value)


def test_an_unknown_aggregate_is_refused():
    with pytest.raises(ObservableError):
        aggregate_samples(["1"], "geometric-mean-of-tuesdays")


# ── declared vs emitted ──────────────────────────────────────────────────────────────────────

DECLARED = [
    {"name": "elapsed_ms", "kind": "number", "aggregate": "median"},
    {"name": "output_digest", "kind": "string"},
]


def test_all_declared_observables_present_is_clean():
    obs = {"elapsed_ms": ["10", "12"], "output_digest": ["abc"]}
    assert check_observables(DECLARED, obs) == []


def test_a_declared_observable_that_never_appeared_is_reported():
    """A spec that declares an observable must have something that produces it; this is caught
    where the check runs.
    """
    problems = check_observables(DECLARED, {"elapsed_ms": ["10"]})
    assert len(problems) == 1
    assert "output_digest" in problems[0]


def test_an_undeclared_observable_is_not_an_error():
    """A script may print extra measurements; the spec states the minimum."""
    obs = {"elapsed_ms": ["1"], "output_digest": ["a"], "peak_rss_kb": ["900"]}
    assert check_observables(DECLARED, obs) == []


def test_declaring_nothing_requires_nothing():
    """Conditional on the declaration, like every other gate here: a benchmark with no
    observables (an llm-judge task, say) is not asked for any."""
    assert check_observables([], {}) == []


def test_a_numeric_observable_emitted_as_text_is_reported():
    problems = check_observables(DECLARED, {"elapsed_ms": ["quick"], "output_digest": ["a"]})
    assert problems and "elapsed_ms" in problems[0]


# ── parameters ───────────────────────────────────────────────────────────────────────────────

PARAMS = [
    {"name": "repetitions", "kind": "int", "default": 5, "min": 1},
    {"name": "timeout_s", "kind": "int", "default": 600},
]


def test_defaults_become_env_vars():
    assert parameter_env(PARAMS, {}) == {"BENCH_REPETITIONS": "5", "BENCH_TIMEOUT_S": "600"}


def test_an_override_wins():
    """The requested repetition count is applied regardless of what the agent did."""
    assert parameter_env(PARAMS, {"repetitions": 50})["BENCH_REPETITIONS"] == "50"


def test_a_value_below_the_declared_minimum_is_refused():
    with pytest.raises(ObservableError) as e:
        parameter_env(PARAMS, {"repetitions": 0})
    assert "repetitions" in str(e.value) and "1" in str(e.value)


def test_an_unknown_parameter_is_refused_and_lists_what_exists():
    """An unknown parameter name is an error rather than silently ignored."""
    with pytest.raises(ObservableError) as e:
        parameter_env(PARAMS, {"reps": 50})
    assert "reps" in str(e.value) and "repetitions" in str(e.value)


def test_a_non_integer_for_an_int_parameter_is_refused():
    with pytest.raises(ObservableError):
        parameter_env(PARAMS, {"repetitions": "lots"})


def test_no_parameters_declared_yields_no_env():
    assert parameter_env([], {}) == {}


# ── it must not name a task ──────────────────────────────────────────────────────────────────


def test_the_module_knows_no_task_vocabulary():
    """The module is a declared contract, not a timing feature: it must not mention milliseconds or
    tests.
    """
    import inspect
    import io
    import re
    import tokenize

    from benchcompiler import observables as mod

    # Task names may legitimately appear in docstring EXAMPLES -- the module explains itself
    # with boca and defects4j. What must not appear is a BRANCH keyed on one. So: strip
    # comments and strings, and strip attribute accesses too, since `re.compile` and
    # `statistics.median` are Python's vocabulary rather than any benchmark's.
    toks = list(tokenize.generate_tokens(io.StringIO(inspect.getsource(mod)).readline))
    code, skip_next = [], False
    for i, tok in enumerate(toks):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        if skip_next:
            skip_next = False
            continue
        if tok.string == ".":
            skip_next = True  # drop the attribute name that follows
            continue
        code.append(tok.string)
    body = " ".join(code).lower()

    for word in (
        "elapsed",
        "runtime",
        "timing",
        "digest",
        "test",
        "compile",
        "build",
        "optimi",
        "speed",
        "flag",
        "commit",
        "millisecond",
    ):
        assert not re.search(rf"\b{word}\b", body), f"task vocabulary leaked into code: {word}"
