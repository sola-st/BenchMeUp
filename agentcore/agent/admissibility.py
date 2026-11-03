"""Measurable admissibility clauses: a tiny predicate grammar the harness can enforce.

WHAT THIS IS FOR. A benchmark's sources have to be worth sampling from. Shape checks (is this
a URL, is there a build file) say nothing about worth, and a spec phrase such as "Real, public
C/C++ projects with a Git history" has no threshold for any gate to enforce. Clauses state the
measurable part of worth explicitly.

A CLAUSE IS THREE PARTS:

    name     what is measured           -> becomes an evidence field the model must record
    observe  how to obtain it           -> makes the evidence fetched, not phrased
    require  the predicate it must meet -> what the gate enforces

`observe` is the load-bearing one. A free-text justification can be satisfied by pasting text
already present in the search response; a clause naming where the number comes from cannot be
answered that way.

WHAT IS GENERAL AND WHAT IS NOT. The grammar below is general: a value, a threshold, a
comparison. Which dimensions exist and what the numbers are is spec content and differs per
task. This module never learns what any of them mean; a conformance test pins that its code
contains none of those words.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any

# Every spelling of true/false a model actually sends through a text protocol -- the same list
# the worklist needs, for the same reason: the string "false" is non-empty and therefore truthy.
_TRUE = frozenset({"true", "yes", "y", "1", "on", "t"})
_FALSE = frozenset({"false", "no", "n", "0", "off", "f", "none", "null", "nil"})

_UNITS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "year": 365, "years": 365}

_CMP = re.compile(r"^(>=|<=|==|=|>|<)\s*(-?\d+(?:\.\d+)?)$")
# `within 12 months` follows the clock; `within 12 months of <date>` does not. The second
# form lets a frozen population state its own floor: an anchor stated in the CLAUSE travels
# with it into the record, the prompt and the refusal, whereas one passed as an argument
# lives only in a call site and the same pool could be admissible now and refused later.
_WITHIN = re.compile(r"^within\s+(\d+)\s*([a-z]+)(?:\s+of\s+(\d{4}-\d{2}-\d{2}))?$")
_BOOL = re.compile(r"^is\s+(true|false)$")
_ONE_OF = re.compile(r"^one of\s*\[(.*)\]$")

_NAME_OK = re.compile(r"^[a-z][a-z0-9_]*$")

# Phrasings that mean "use your judgment", which is exactly what a clause exists to replace.
_EMPTY_OBSERVE = frozenset(
    {
        "check it",
        "look at it",
        "judgment",
        "use judgment",
        "obvious",
        "inspect it",
        "see for yourself",
        "manual",
        "n/a",
        "none",
        "-",
    }
)


@dataclass(frozen=True)
class Predicate:
    kind: str  # "cmp" | "within" | "bool" | "one_of"
    op: str = ""
    number: float = 0.0
    days: int = 0
    anchor: str = ""  # `within N months OF this date`; empty means "of today"
    flag: bool = False
    options: tuple = ()
    text: str = ""


def parse(require: Any) -> Predicate | None:
    """A predicate, or None when the text states nothing a program could check."""
    raw = str(require or "").strip()
    low = " ".join(raw.lower().split())
    if not low:
        return None

    m = _CMP.match(low.replace(" ", ""))
    if m:
        op = "==" if m.group(1) == "=" else m.group(1)
        return Predicate(kind="cmp", op=op, number=float(m.group(2)), text=raw)

    m = _WITHIN.match(low)
    if m and m.group(2) in _UNITS:
        anchor = m.group(3)
        if anchor is not None and _as_date(anchor) is None:
            # A date a program cannot read must be refused here, not quietly replaced
            # by the wall clock -- that would be a DIFFERENT clause wearing this text.
            return None
        return Predicate(kind="within", days=int(m.group(1)) * _UNITS[m.group(2)], anchor=anchor or "", text=raw)

    m = _BOOL.match(low)
    if m:
        return Predicate(kind="bool", flag=(m.group(1) == "true"), text=raw)

    m = _ONE_OF.match(low)
    if m:
        opts = tuple(o.strip() for o in m.group(1).split(",") if o.strip())
        return Predicate(kind="one_of", options=opts, text=raw) if opts else None

    return None


def _as_number(value: Any) -> float | None:
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _as_date(value: Any) -> _dt.date | None:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value or "").strip())
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def evaluate(pred: Predicate | None, value: Any, *, now: str | None = None) -> tuple[bool, str]:
    """(ok, why). Fails CLOSED on anything it cannot read.

    A check that cannot see its input must refuse rather than wave the entry through, the same
    rule the worklist validators fail closed under. A silently-passing gate is how a survey
    admits a pool of homework.
    """
    if pred is None:
        return False, "this requirement is not something a program can check"
    # A recorded `false` or `0` is an observation, not an absence -- `False or ""` is `""`,
    # which would refuse the very answer a boolean clause is asking for.
    if value is None or (not isinstance(value, (bool, int, float)) and not str(value).strip()):
        return False, f"not recorded -- this must be observed and written down ({pred.text})"

    if pred.kind == "cmp":
        n = _as_number(value)
        if n is None:
            return False, f"{str(value)[:40]!r} is not a number, so {pred.text!r} cannot be checked"
        ok = {
            ">=": n >= pred.number,
            ">": n > pred.number,
            "<=": n <= pred.number,
            "<": n < pred.number,
            "==": n == pred.number,
        }[pred.op]
        want = int(pred.number) if float(pred.number).is_integer() else pred.number
        got = int(n) if float(n).is_integer() else n
        return ok, ("" if ok else f"{got} does not satisfy {pred.op} {want}")

    if pred.kind == "within":
        d = _as_date(value)
        if d is None:
            return False, f"{str(value)[:40]!r} is not a date, so {pred.text!r} cannot be checked"
        # A CLAUSE'S OWN ANCHOR WINS over a caller's: the pin is a fact about the snapshot the
        # spec froze, and a caller passing `now` is saying what today is, which is a different
        # question. Unpinned, `now` still wins over the clock, so nothing that supplies one
        # changes behaviour.
        today = _as_date(pred.anchor) or _as_date(now) or _dt.date.today()
        age = (today - d).days
        if age <= pred.days:
            return True, ""
        return False, (f"{value} is {age} days old as of {today.isoformat()}, older than {pred.text}")

    if pred.kind == "bool":
        s = str(value).strip().lower()
        got = True if (value is True or s in _TRUE) else (False if (value is False or s in _FALSE) else None)
        if got is None:
            return False, f"{str(value)[:40]!r} is not a yes/no value"
        return got is pred.flag, "" if got is pred.flag else f"is {got}, needs {pred.text}"

    if pred.kind == "one_of":
        s = str(value).strip().lower()
        return (s in pred.options, "" if s in pred.options else f"{value!r} is not one of {list(pred.options)}")

    return False, f"unsupported requirement {pred.text!r}"


def clause_problems(clause: Any) -> list[str]:
    """Everything wrong with one clause, so a spec author sees it all at once."""
    if not isinstance(clause, dict):
        return ["a clause must be an object with name, observe and require"]
    out = []

    name = str(clause.get("name") or "").strip()
    if not name:
        out.append("`name` is missing -- it names the value to record")
    elif not _NAME_OK.match(name):
        out.append(
            f"`name` {name!r} must be a plain field name: lowercase, starting with a "
            f"letter, letters/digits/underscore only"
        )

    observe = " ".join(str(clause.get("observe") or "").strip().split())
    if not observe:
        out.append("`observe` is missing -- say HOW to obtain this value, or it will be guessed rather than looked up")
    elif observe.lower().rstrip(".") in _EMPTY_OBSERVE or len(observe) < 12:
        out.append(
            f"`observe` {observe!r} names no procedure. State where the value comes "
            f"from concretely enough that two people would fetch the same number."
        )

    require = clause.get("require")
    if require is None or not str(require).strip():
        out.append("`require` is missing -- state the threshold this value must meet")
    elif parse(require) is None:
        out.append(
            f"`require` {str(require)[:60]!r} is not measurable. Use a comparison "
            f"(>= 100), a recency (within 12 months), a boolean (is false) or a "
            f"membership (one of [a, b])."
        )
    return out


def check(clauses: list, values: dict, *, now: str | None = None) -> list[str]:
    """Every clause the recorded values fail, phrased for whoever has to fix it."""
    out = []
    for clause in clauses or []:
        if not isinstance(clause, dict):
            continue
        name = str(clause.get("name") or "").strip()
        ok, why = evaluate(parse(clause.get("require")), (values or {}).get(name), now=now)
        if not ok:
            out.append(f"{name}: {why}")
    return out


def field_names(clauses: list) -> list[str]:
    """The evidence fields a task's clauses imply -- derived, never declared twice."""
    names = []
    for clause in clauses or []:
        if isinstance(clause, dict):
            n = str(clause.get("name") or "").strip()
            if n and n not in names:
                names.append(n)
    return names
