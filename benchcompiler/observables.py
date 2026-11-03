"""What a run must REPORT, and what a consumer may VARY -- both declared by the spec.

A pass/fail expectation per variant is enough to certify that something still works and
nothing else: a check that runs a program and hashes its output says nothing about a
benchmark whose task is "beat this time", and a digest comparison never re-derives a number
the model recorded.

So a benchmark declares two more lists, and the harness stays ignorant of what they mean:

    observables:  [{name, kind, aggregate?, unit?}]   values a run must emit
    parameters:   [{name, kind, default, min?, max?}] knobs a consumer may set

A timing benchmark declares `elapsed_ms` with `aggregate: median` and a `repetitions` parameter;
a bug-fixing benchmark declares `tests_run` and `failures` with a `test_timeout`. Identical machinery. Nothing in this
module may branch on a name -- a test tokenises the source and fails on task vocabulary,
because the moment this file knows what a millisecond is, every later benchmark inherits one
benchmark's assumptions.

TRANSPORT. A check script is written by a model, so the protocol has to survive being written
by one: a line on stdout, `BENCH_OBS <name>=<value>`, one line per sample. Parameters travel
the other way as `BENCH_<NAME>` environment variables. Line-oriented on purpose -- a value that
could span lines would silently swallow the observations after it.
"""

from __future__ import annotations

import re
import statistics

# `BENCH_OBS name=value`. The name is restricted so it can become an env var, a JSON key and a
# column header without escaping; the value runs to end of line and may contain '='.
_OBS_LINE = re.compile(r"^\s*BENCH_OBS\s+([A-Za-z][A-Za-z0-9_]{0,63})\s*=(.*)$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

NUMERIC_AGGREGATES = ("median", "mean", "min", "max")
AGGREGATES = NUMERIC_AGGREGATES + ("count", "last", "first")


class ObservableError(ValueError):
    """A declared contract the run did not honour, or a request the contract cannot serve."""


def parse_observations(output: str) -> dict[str, list[str]]:
    """Every `BENCH_OBS name=value` line, in order, grouped by name.

    Repeats accumulate rather than overwrite: one line per run is how a repetition loop
    reports, and the samples are what an aggregate is computed over. A malformed line is
    skipped rather than fatal -- one bad line from a model-written script must not lose the
    good ones.
    """
    found: dict[str, list[str]] = {}
    for line in str(output or "").splitlines():
        m = _OBS_LINE.match(line)
        if not m:
            continue
        found.setdefault(m.group(1), []).append(m.group(2).strip())
    return found


def aggregate_samples(samples: list[str], how: str | None):
    """Reduce samples the way the spec asked, or return them all when it asked for nothing.

    A non-numeric sample under a numeric aggregate NAMES THE VALUE. "could not compute median"
    sends a reader to the wrong place; `-Ofast` in a list of milliseconds sends them straight
    to the script that printed it.
    """
    values = list(samples or [])
    if how in (None, "", "none"):
        return values
    if how not in AGGREGATES:
        raise ObservableError(f"unknown aggregate {how!r}; the contract may ask for one of {', '.join(AGGREGATES)}")
    if how == "count":
        return len(values)
    if not values:
        raise ObservableError(f"cannot compute {how}: nothing was emitted")
    if how == "last":
        return values[-1]
    if how == "first":
        return values[0]
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            raise ObservableError(
                f"cannot compute {how}: {v!r} is not a number. The contract declares this "
                f"observable numeric, so every emitted sample has to be one."
            ) from None
    return {"median": statistics.median, "mean": statistics.fmean, "min": min, "max": max}[how](nums)


def check_observables(declared: list[dict], observed: dict[str, list[str]]) -> list[str]:
    """Which declared observables the run failed to produce, as sentences.

    Conditional on the declaration, exactly like every other gate here: a benchmark declaring
    none is asked for none. EXTRA observables are fine -- the spec states the minimum, and
    refusing a script for being informative would be an over-refusal.

    Declared-but-absent is the failure worth catching. It is the same shape as a spec declaring
    variants nothing materialises, and a check that reports nothing is indistinguishable from
    one that measured nothing.
    """
    problems = []
    for d in declared or []:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "")
        if not _NAME.match(name):
            problems.append(
                f"observable name {name!r} is not usable: letters, digits and underscores, starting with a letter"
            )
            continue
        samples = observed.get(name)
        if not samples:
            problems.append(
                f"the contract declares observable {name!r} and the run emitted none. The "
                f"check script must print `BENCH_OBS {name}=<value>` (one line per sample)."
            )
            continue
        if str(d.get("kind")) == "number":
            bad = [s for s in samples if not _is_number(s)]
            if bad:
                problems.append(f"observable {name!r} is declared numeric but the run emitted {bad[0]!r}")
    return problems


def _is_number(s: str) -> bool:
    try:
        float(s)
    except (TypeError, ValueError):
        return False
    return True


def parameter_env(declared: list[dict], overrides: dict | None = None) -> dict[str, str]:
    """The environment a check runs under: declared defaults, with the consumer's overrides.

    An unknown name is REFUSED and the refusal lists what exists. Accepting it silently would
    let a consumer read an unchanged result as the answer to a question they never asked --
    they typed `--reps 50`, got the default 5, and nothing said so.
    """
    spec = {}
    for d in declared or []:
        if isinstance(d, dict) and _NAME.match(str(d.get("name") or "")):
            spec[str(d["name"])] = d

    env = {}
    for name, d in spec.items():
        env[f"BENCH_{name.upper()}"] = str(d.get("default", ""))

    for name, value in (overrides or {}).items():
        if name not in spec:
            raise ObservableError(
                f"this benchmark declares no parameter {name!r}; it declares {', '.join(sorted(spec)) or 'none'}"
            )
        d = spec[name]
        if str(d.get("kind")) == "int":
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise ObservableError(f"parameter {name!r} is declared an integer; got {value!r}") from None
            lo, hi = d.get("min"), d.get("max")
            if lo is not None and value < int(lo):
                raise ObservableError(f"parameter {name!r} must be at least {lo}; got {value}")
            if hi is not None and value > int(hi):
                raise ObservableError(f"parameter {name!r} must be at most {hi}; got {value}")
        env[f"BENCH_{name.upper()}"] = str(value)
    return env


def report_rows(declared: list[dict], observed: dict[str, list[str]]) -> list[tuple]:
    """(name, value, unit, samples) per declared observable, for a human-readable table.

    The consumer-facing half: the numbers exist in the record either way, but a benchmark whose
    task is a measurement has to SHOW the measurement, not a PASS.
    """
    rows = []
    for d in declared or []:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "")
        samples = observed.get(name) or []
        try:
            value = aggregate_samples(samples, d.get("aggregate"))
        except ObservableError as e:
            value = f"<{e}>"
        rows.append((name, value, str(d.get("unit") or ""), len(samples)))
    return rows


def captures_of(stdout: str, stderr: str, exit_code: int, wall_ms) -> dict:
    """One execution, as the closed set of primitives a `capture:` entry may ask for.

    Shared by the certifier and by the artifact's own CLI, so there is one implementation.
    """
    import hashlib

    out = stdout or ""
    err = stderr or ""
    # `output_*` is BOTH streams. Which one a program writes its payload to is the program's
    # business: a program may send its payload to a file and its progress to stderr, so
    # capturing stdout alone could measure zero bytes.
    # Both forms stay available -- a benchmark that means stdout says stdout.
    both = out + err
    return {
        "wall_ms": wall_ms,
        "exit": exit_code,
        "stdout_sha256": hashlib.sha256(out.encode("utf-8", "replace")).hexdigest(),
        "stdout_head": " ".join(str(out).split())[:HEAD_CHARS],
        "stdout_bytes": len(out.encode("utf-8", "replace")),
        "stderr_head": " ".join(str(err).split())[:HEAD_CHARS],
        "output_sha256": hashlib.sha256(both.encode("utf-8", "replace")).hexdigest(),
        "output_head": " ".join(str(both).split())[:HEAD_CHARS],
        "output_bytes": len(both.encode("utf-8", "replace")),
    }


HEAD_CHARS = 2000
