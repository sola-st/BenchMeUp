"""`spec/benchspec-v0.md` must describe the format the front end actually enforces.

`mechanical_precheck`'s fix-hint points to the document, so every field the front end can refuse a
spec for omitting must be mentioned in it, its `criteria` entry must not teach what
`CRITERIA_ARE_SOURCE_FILTERS` refuses, and its fenced worked examples are extracted and run through
the real front end.
"""

from __future__ import annotations

import os
import re

import pytest
import yaml

from benchcompiler.spec import mechanical_precheck

DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spec", "benchspec-v0.md")

#: Every field the front end can refuse a spec for omitting, and the code it uses. A field that
#: is mandatory in code and absent from the document is the gap this file exists to close.
MANDATORY = {
    "admissibility": "MISSING_ADMISSIBILITY",
    "artifact_variants": "MISSING_VARIANTS",
    "solver_input": "NO_SOLVER_INPUT",
    "negative_control": "NO_NEGATIVE_CONTROL",
    "parameters": "NO_SOLVER_INPUT",
    "results": "UNDECLARED_CHECK_EVIDENCE",
    "expect_failures": "UNDECLARED_CHECK_EVIDENCE",
}


def _text():
    with open(DOC) as fh:
        return fh.read()


def _yaml_blocks():
    """Every fenced yaml block, labelled with its heading and its section (the ancestor heading
    says these are examples; matching the nearest heading alone finds nothing).
    """
    text = _text()
    out, section, heading = [], "", ""
    for chunk in re.split(r"(?m)^(#{2,4} .*)$", text):
        if chunk.startswith("#"):
            level = len(chunk) - len(chunk.lstrip("#"))
            heading = chunk.strip("# ").strip()
            if level <= 2:
                section = heading
            continue
        for m in re.finditer(r"```ya?ml\n(.*?)```", chunk, re.S):
            out.append((f"{section} / {heading}".strip(" /"), m.group(1)))
    return out


def test_the_document_is_where_the_front_end_points():
    """Guards the premise: if the fix-hint stops naming this file, this file is aimed at the wrong
    document.
    """
    from benchcompiler.spec import mechanical_precheck as mp

    hints = " ".join(d.fix_hint for d in mp({}))
    print("  fix hints mention:", "benchspec-v0.md" in hints)
    assert "benchspec-v0.md" in hints


@pytest.mark.parametrize("field,code", sorted(MANDATORY.items()))
def test_a_field_the_front_end_demands_is_documented(field, code):
    text = _text()
    n = text.count(field)
    print(f"  {field:20} mentioned {n:2}x  (refused as {code})")
    assert n, (
        f"a spec is rejected with {code} for omitting `{field}`, and the document the "
        f"rejection points at never mentions it"
    )


def test_criteria_are_not_described_as_admissibility_filters():
    """The document must not teach the confusion `CRITERIA_ARE_SOURCE_FILTERS` refuses."""
    text = _text().lower()
    for phrase in ("admissibility filters, any granularity", "repo/instance-level predicates in prose"):
        assert phrase not in text, (
            f"the document still defines `criteria` as {phrase!r}, which is the source-filter "
            f"confusion the checker punishes"
        )


def test_the_document_says_what_criteria_are_instead():
    text = _text().lower()
    assert "every shipped instance" in text or "per-instance" in text, (
        "nothing in the document says what a criterion IS, only what it is not"
    )


# ── the examples are executed, not read ─────────────────────────────────────────────────────


def test_there_are_worked_examples_to_check():
    blocks = [(h, b) for h, b in _yaml_blocks() if "example" in h.lower()]
    print("  example blocks:", [h for h, _ in blocks])
    assert len(blocks) >= 2, [h for h, _ in _yaml_blocks()]


@pytest.mark.parametrize(
    "heading,block",
    [(h, b) for h, b in _yaml_blocks() if "example" in h.lower()],
    ids=lambda x: x[:40] if isinstance(x, str) else "",
)
def test_a_worked_example_passes_the_front_end(heading, block):
    """A format document whose own examples the format rejects is a promise it does not keep."""
    raw = yaml.safe_load(block)
    assert isinstance(raw, dict), heading
    diagnostics = mechanical_precheck(raw)
    for d in diagnostics:
        print(f"  {heading}: {d.code}({d.field}) {d.detail[:120]}")
    assert not diagnostics, (
        f"the worked example under {heading!r} is rejected by this project's own front end: "
        f"{sorted({d.code for d in diagnostics})}"
    )
