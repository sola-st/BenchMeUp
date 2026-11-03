"""A correctness verdict may not be read off a truncated excerpt.

A capture field the harness produces by truncating (e.g. `stdout_head`, the first 200 characters)
cannot be the source of an observable whose kind is a verdict (boolean); hash fields such as
`output_sha256` are the right source.
"""

from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: What the harness fills by TRUNCATING. Named in observables.py beside HEAD_CHARS.
EXCERPT_SOURCES = {"stdout_head", "stderr_head", "output_head"}
#: Kinds that decide something rather than describe it.
VERDICT_KINDS = {"boolean", "bool"}


def _contracts():
    for p in sorted(glob.glob(os.path.join(HERE, "agents", "*", "instances", "instances.jsonl"))):
        try:
            d = json.loads(open(p, encoding="utf-8").readline())
        except Exception:
            continue
        if d.get("measure"):
            yield os.path.basename(os.path.dirname(os.path.dirname(p))), d


def test_no_shipped_contract_reads_a_verdict_off_an_excerpt():
    bad = []
    for name, d in _contracts():
        kinds = {str(o.get("name")): str(o.get("kind") or "").lower() for o in (d.get("observables") or [])}
        for cap in (d.get("measure") or {}).get("capture") or []:
            src, nm = str(cap.get("from")), str(cap.get("name"))
            if src in EXCERPT_SOURCES and kinds.get(nm) in VERDICT_KINDS:
                bad.append(f"{name}: observable {nm!r} (kind {kinds.get(nm)}) <- {src!r}")
            if src in EXCERPT_SOURCES:
                print(f"[verbose] {name}: {nm!r} (kind {kinds.get(nm)!r}) <- {src!r}")
    assert not bad, (
        "a truncated excerpt is deciding correctness; the digest of the same stream is in "
        "the same capture vocabulary: " + "; ".join(bad)
    )


def test_the_checker_refuses_such_a_contract():
    """The rule applied to a synthetic contract, independent of the shipped collectors."""
    from benchcompiler.spec import excerpt_sourced_verdicts

    contract = {
        "observables": [{"name": "correct", "kind": "boolean"}, {"name": "excerpt", "kind": "text"}],
        "measure": {
            "capture": [{"name": "correct", "from": "stdout_head"}, {"name": "excerpt", "from": "output_head"}]
        },
    }
    problems = excerpt_sourced_verdicts(contract)
    print(f"\n[verbose] {problems}")
    assert len(problems) == 1, problems
    assert "correct" in problems[0] and "stdout_head" in problems[0]
    assert "sha256" in problems[0], "the refusal must name what to use instead"


def test_an_honestly_named_excerpt_is_left_alone():
    from benchcompiler.spec import excerpt_sourced_verdicts

    contract = {
        "observables": [{"name": "output_excerpt", "kind": "text"}],
        "measure": {"capture": [{"name": "output_excerpt", "from": "output_head"}]},
    }
    assert excerpt_sourced_verdicts(contract) == []
