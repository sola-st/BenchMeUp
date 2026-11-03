"""Quantities the HARNESS computes from observations the agent already has.

Computing a median or a coefficient of variation inside a container shell is fragile
(tool availability, quoting) and forces measurements to be re-run to recover the numbers.
The harness computes such values instead.

The general form is any field the harness could compute from observations the agent already
holds: medians, CVs, ratios, speedups, percentages, pass rates, sizes. A measurement task
hits it immediately; a scoring task hits it the moment it reports a rate.

So a state declares the arithmetic instead of asking for its result:

    computes:
      median_runtime: {from: timings, agg: median}
      cv:             {from: timings, agg: cv}
      speedup:        {from: [baseline_ms, tuned_ms], agg: ratio}

The agent records `timings: [671, 677, 682]` -- an observation, which is the only thing it
is in a position to know -- and the harness writes `median_runtime`. The declared field
may then appear in a later state's `requires` exactly as if it had been typed, because by the
time `requires` is checked it is present (see Worklist.update, which derives before it
recomputes the state).

Two rules the deriving obeys:

  * The agent's own value WINS if it recorded one. The harness's value is recorded next to it
    and a disagreement is flagged, not silently overwritten -- a field the agent computed
    itself may be the one thing that reveals it measured something other than what it said.
  * A source it cannot read is a PROBLEM, never a guess. `timings: ["677ms"]` does not become
    677: the unit might be the interesting part, and a benchmark whose numbers were repaired
    by a regex is worth nothing.

Shapes: a flat list of numbers gives a scalar; an object of lists (per-input-scale
timings, `{small: [...], large: [...]}`) gives an object of scalars with the SAME keys, which
is the shape every consumer of those records already expects. A JSON string holding either is
parsed first -- models send `"[1.2, 1.3]"` routinely and re-escaping it is a turn spent on
notation.
"""

from __future__ import annotations

import json
import statistics
from typing import Any

# Aggregations over ONE list of numbers.
UNARY = ("median", "mean", "min", "max", "sum", "count", "stdev", "cv")
# Aggregations over TWO fields, `from: [numerator, denominator]`. A speedup and a pass rate are
# both ratios; a percentage is a ratio times 100. Nothing here needs a third form.
BINARY = ("ratio", "percent")
AGGS = UNARY + BINARY

# How close the agent's own value has to be to the harness's before the two are called the same
# number. 1%: rounding a median to whole milliseconds moves it by well under that, while the
# disagreements worth seeing -- a mean reported as a median, a statistic over a different sample,
# a unit conversion that did not happen -- are all far larger.
#
# The default depends on the aggregation. A count, a sum, a min or a max has no rounding
# to forgive: a count of 151 against a harness count of 150 is a different sample, and 1%
# would accept it. So the default is exact for those four and
# 1% for the statistics rounding touches -- and a declaration may say `tolerance: 0.05`
# itself, because how much slack a measurement deserves is the benchmark's knowledge, not
# the harness's. MISMATCH_TOLERANCE stays as the fallback for the inexact aggregations.
MISMATCH_TOLERANCE = 0.01
EXACT_AGGS = ("count", "sum", "min", "max")


def tolerance_for(decl: Any) -> float:
    """The relative slack this declaration gets: its own `tolerance:` when it says, else exact
    for the aggregations that cannot round, else MISMATCH_TOLERANCE."""
    if isinstance(decl, dict) and decl.get("tolerance") is not None:
        try:
            return max(0.0, float(decl["tolerance"]))
        except (TypeError, ValueError):
            pass
    agg = str(decl.get("agg") or "").strip().lower() if isinstance(decl, dict) else ""
    return 0.0 if agg in EXACT_AGGS else MISMATCH_TOLERANCE


def known_aggs() -> str:
    return ", ".join(AGGS)


def sources_of(decl: dict) -> list[str]:
    """The field name(s) a declaration reads. Accepts `from: x` and `from: [x, y]`."""
    src = decl.get("from") if isinstance(decl, dict) else None
    if isinstance(src, (list, tuple)):
        return [str(s) for s in src if str(s).strip()]
    return [str(src)] if src not in (None, "") else []


#: A computed quantity may carry a DECLARED bound: the spec's number, projected by the designer
#: from the contract (a separation's min_ratio, a noise ceiling). The state that computes it
#: refuses a candidate outside the bound, naming the value and the bound, so a lead that cannot
#: separate is retried in the sandbox it still has instead of paying a delivery and a packaging
#: pass.
BOUND_KEYS = ("at_least", "at_most")


def bounds_of(decl: Any) -> dict[str, float]:
    """{'at_least': x, 'at_most': y} -- only the keys the declaration carries."""
    out: dict[str, float] = {}
    if isinstance(decl, dict):
        for k in BOUND_KEYS:
            if decl.get(k) is not None:
                out[k] = float(decl[k])
    return out


def bound_problem(name: str, decl: Any, value: Any) -> str:
    """Why `value` of the computed `name` is outside its declared bound; '' when inside or
    when nothing is declared. A non-numeric value under a bound is a problem too: a bound
    over a number that is not there enforces nothing."""
    try:
        bounds = bounds_of(decl)
    except (TypeError, ValueError):
        return ""
    if not bounds:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return (
            f"{name} is bounded ({', '.join(f'{k} {v:g}' for k, v in bounds.items())}) "
            f"but computed to {value!r}, which is not a number"
        )
    if "at_least" in bounds and x < bounds["at_least"]:
        return f"{name} = {_round(x):g}, below the declared at_least {bounds['at_least']:g}"
    if "at_most" in bounds and x > bounds["at_most"]:
        return f"{name} = {_round(x):g}, above the declared at_most {bounds['at_most']:g}"
    return ""


def declaration_problem(name: str, decl: Any) -> str:
    """Why this declaration cannot be honoured, or "". Used by config load and by the audit, so
    a malformed `computes:` is a startup error rather than a field that silently never appears."""
    if not isinstance(decl, dict):
        return f"computes {name!r} must be an object like {{from: <field>, agg: <one of {known_aggs()}>}}"
    agg = str(decl.get("agg") or "").strip().lower()
    src = sources_of(decl)
    if not src:
        return f"computes {name!r} names no source field (`from:`)"
    if agg not in AGGS:
        return f"computes {name!r} asks for agg {decl.get('agg')!r}; the harness implements {known_aggs()}"
    if agg in BINARY and len(src) != 2:
        return f"computes {name!r} uses agg {agg!r}, which reads two fields: `from: [numerator, denominator]`"
    if agg in UNARY and len(src) != 1:
        return f"computes {name!r} uses agg {agg!r}, which reads one field: `from: <field>`"
    if name in src:
        return f"computes {name!r} reads itself"
    tol = decl.get("tolerance")
    if tol is not None:
        try:
            if float(tol) < 0:
                raise ValueError
        except (TypeError, ValueError):
            return (
                f"computes {name!r} declares tolerance {tol!r}; it must be a number >= 0 "
                f"(the relative slack the agent's value gets, e.g. 0.05 for 5%)"
            )
    for k in BOUND_KEYS:
        if decl.get(k) is not None:
            try:
                float(decl[k])
            except (TypeError, ValueError):
                return (
                    f"computes {name!r} declares {k} {decl[k]!r}; a bound must be a "
                    f"number (the spec's own, e.g. at_least: 1.05 for a separation ratio)"
                )
    return ""


def _numbers(value: Any) -> tuple[Any, str]:
    """Coerce a recorded value to numbers, keeping its shape. Returns (value, problem)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None, f"{value[:60]!r} is not a number, a list of numbers, or JSON holding one"
    if isinstance(value, bool):
        return None, "a boolean is not a measurement"
    if isinstance(value, (int, float)):
        return float(value), ""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            n, prob = _numbers(v)
            if prob:
                return None, f"[{k}] {prob}"
            out[str(k)] = n
        return out, ""
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            n, prob = _numbers(v)
            if prob:
                return None, prob
            if not isinstance(n, float):
                return None, "a nested list is not a sample"
            out.append(n)
        return out, ""
    return None, f"{type(value).__name__} is not a measurement"


def _round(x: float) -> float:
    r = round(x, 6)
    return int(r) if isinstance(r, float) and r.is_integer() and abs(r) < 1e15 else r


def _unary(agg: str, sample: Any) -> tuple[Any, str]:
    if isinstance(sample, float):
        sample = [sample]
    if not isinstance(sample, list) or not sample:
        return None, "the sample is empty"
    if agg in ("stdev", "cv") and len(sample) < 2:
        return None, (
            f"{agg} needs at least two observations; one number has no spread. Record the repeats you actually ran."
        )
    if agg == "median":
        return _round(statistics.median(sample)), ""
    if agg == "mean":
        return _round(statistics.fmean(sample)), ""
    if agg == "min":
        return _round(min(sample)), ""
    if agg == "max":
        return _round(max(sample)), ""
    if agg == "sum":
        return _round(sum(sample)), ""
    if agg == "count":
        return len(sample), ""
    if agg == "stdev":
        return _round(statistics.stdev(sample)), ""
    # cv as a FRACTION (0.031, not 3.1%): every spec threshold is written that
    # way, and a quantity whose unit depends on who read it is the ambiguity being removed here.
    mean = statistics.fmean(sample)
    if mean == 0:
        return None, "cv is undefined for a mean of zero"
    return _round(statistics.stdev(sample) / mean), ""


def _binary(agg: str, a: Any, b: Any) -> tuple[Any, str]:
    if isinstance(a, list) or isinstance(b, list):
        return None, f"{agg} reads two single numbers (or two objects of them), not a sample"
    if b == 0:
        return None, "division by zero"
    v = a / b
    return _round(v * 100 if agg == "percent" else v), ""


def _apply_one(agg: str, values: list[Any]) -> tuple[Any, str]:
    """Compute one declaration, mapping over the keys when the sources are keyed objects."""
    keyed = [v for v in values if isinstance(v, dict)]
    if keyed:
        keys = list(keyed[0])
        for v in keyed[1:]:
            if set(v) != set(keys):
                return None, (
                    f"the sources are keyed by different things "
                    f"({sorted(keys)} vs {sorted(v)}); a derived value can only be "
                    f"computed per key when every source has the same keys"
                )
        out = {}
        for k in keys:
            per = [v[k] if isinstance(v, dict) else v for v in values]
            one, prob = _unary(agg, per[0]) if agg in UNARY else _binary(agg, per[0], per[1])
            if prob:
                return None, f"[{k}] {prob}"
            out[k] = one
        return out, ""
    if agg in UNARY:
        return _unary(agg, values[0])
    return _binary(agg, values[0], values[1])


def _differs(mine: Any, theirs: Any, tolerance: float = MISMATCH_TOLERANCE) -> str:
    """Do the agent's value and the harness's disagree? Returns a description, or ""."""
    if isinstance(mine, dict) or isinstance(theirs, dict):
        if not (isinstance(mine, dict) and isinstance(theirs, dict)):
            return f"agent recorded {mine!r}, harness computed {theirs!r} (different shapes)"
        notes = [
            f"{k}: agent {mine.get(k)!r} vs harness {theirs.get(k)!r}"
            for k in sorted(set(mine) | set(theirs))
            if _differs(mine.get(k), theirs.get(k), tolerance)
        ]
        return "; ".join(notes)
    a, prob_a = _numbers(mine)
    b, prob_b = _numbers(theirs)
    if prob_a or prob_b or not isinstance(a, float) or not isinstance(b, float):
        return (
            ""
            if str(mine).strip() == str(theirs).strip()
            else (f"agent recorded {mine!r}, harness computed {theirs!r}")
        )
    scale = max(abs(a), abs(b), 1e-12)
    if abs(a - b) / scale <= tolerance:
        return ""
    return f"agent recorded {mine!r}, harness computed {theirs!r}"


def apply(decls: dict[str, dict], fields: dict) -> tuple[dict, dict, list[str]]:
    """Derive every declared field whose sources are present.

    Returns (updates, provenance, problems). `updates` holds values to merge into the
    candidate's fields (only fields the agent did NOT record itself); `provenance` maps each
    derived name to what was computed and from what; `problems` are messages for the agent --
    a source it cannot read, a sample too small, a disagreement with the agent's own value.
    The caller decides what to do with them; nothing here mutates `fields`.
    """
    updates: dict[str, Any] = {}
    provenance: dict[str, dict] = {}
    problems: list[str] = []
    for name, decl in (decls or {}).items():
        bad = declaration_problem(name, decl)
        if bad:
            problems.append(
                f"HARNESS ERROR: {bad}. This is a fault in the task's lifecycle, "
                f"not something to work around by recording a different value."
            )
            continue
        agg = str(decl["agg"]).strip().lower()
        srcs = sources_of(decl)
        raw = [fields.get(s) for s in srcs]
        if any(v is None or (isinstance(v, (str, list, tuple, dict)) and len(v) == 0) for v in raw):
            continue  # nothing recorded yet; not a problem, just not time
        values = []
        failed = False
        for s, v in zip(srcs, raw):
            n, prob = _numbers(v)
            if prob:
                # DIGESTS ARE NOT NUMBERS: telling the agent to "record numbers" would make
                # it replace its digests with runtimes and destroy the evidence the field
                # exists to hold. When the samples look like
                # digests, the defect is the DECLARATION, and the agent must be told to
                # keep its evidence, not to launder it into something aggregable.
                import re as _re

                sample = v if isinstance(v, (list, tuple)) else [v]
                if any(_re.fullmatch(r"[0-9a-f]{32,64}", str(x).strip().lower()) for x in sample):
                    problems.append(
                        f"cannot compute {name} from {s}: these values are DIGESTS, and "
                        f"a numeric {agg} over digests is meaningless -- this is a defect "
                        f"in the lifecycle's computes declaration, not in your record. "
                        f"KEEP {s} as the real digests (equality between them is what "
                        f"verifies anything); record {name} yourself with a true/false "
                        f"answer and never convert digests to numbers."
                    )
                else:
                    problems.append(
                        f"cannot compute {name} from {s}: {prob}. Record the "
                        f"raw observations as numbers and the harness computes "
                        f"{name} ({agg}) itself."
                    )
                failed = True
                break
            values.append(n)
        if failed:
            continue
        value, prob = _apply_one(agg, values)
        if prob:
            problems.append(f"cannot compute {name} ({agg} of {', '.join(srcs)}): {prob}")
            continue
        entry = {"value": value, "agg": agg, "from": srcs, "derived_by": "harness"}
        recorded = fields.get(name)
        if recorded is None or (isinstance(recorded, (str, list, tuple, dict)) and not recorded):
            updates[name] = value
        else:
            entry["agent_value"] = recorded
            diff = _differs(recorded, value, tolerance_for(decls.get(name)))
            if diff:
                entry["mismatch"] = diff
                problems.append(
                    f"{name} disagrees with the observations it should summarise: {diff} "
                    f"({agg} of {', '.join(srcs)}). Your value stands; both are recorded. "
                    f"If the observations are the right ones, drop {name} and let the harness "
                    f"compute it."
                )
        provenance[name] = entry
    return updates, provenance, problems
