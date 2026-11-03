"""A model diagnostic on a block the raw spec ships structured is not a verdict on the spec.

Blocks such as the solver input, negative control and variants are carried verbatim by
`build_canonical` and validated by the harness's own checks; the model cannot change them. A REJECT
naming only shipped blocks buys a retry round with feedback naming them; a REJECT that also names
the researcher's prose is final on the prose alone, with the shipped-block diagnostics removed.
"""

from __future__ import annotations

from benchcompiler import formalizer
from benchcompiler.formalizer import FormalizeResult, _formalize_once

RAW = {
    "task": "given a C program, find flags that make it faster while preserving output",
    "instance": "a program at a commit with reference inputs",
    "data": "public C repositories the builder chooses under the criteria",
    "correctness": "reference-equivalence: the default build's output digest",
    "criteria": "- builds with no optimization flags",
    "artifact_variants": [
        {
            "name": "default",
            "check_cmd": "sh check_default.sh .",
            "expect": "pass",
            "results": "none",
            "results_none_reason": "digest is the evidence",
        },
        {
            "name": "optimized",
            "check_cmd": "sh check_optimized.sh .",
            "expect": "pass",
            "results": "none",
            "results_none_reason": "digest is the evidence",
        },
    ],
    "parameters": [{"name": "cflags", "kind": "string", "default": "-O0"}],
    "solver_input": {"kind": "parameter", "name": "cflags", "describes": "the flags", "probe_values": ["-O0", "-O3"]},
    "negative_control": {
        "break_cmd": "corruption_cmd",
        "restore_cmd": "rebuild_cmd",
        "check": "default",
        "describes": "corrupt the binary",
    },
}

SHIPPED = [
    {"code": "NO_SOLVER_INPUT", "field": "solver_input", "detail": "a parameter, not a mechanism"},
    {
        "code": "NO_NEGATIVE_CONTROL",
        "field": "negative_control",
        "detail": "break_cmd is 'corruption_cmd', a placeholder",
    },
    {"code": "MISSING_VARIANTS", "field": "artifact_variants", "detail": "add a candidate variant"},
]
PROSE = [{"code": "AMBIGUOUS_TERM", "field": "criteria", "detail": "'reproducible' is vague"}]


def _model_says(monkeypatch, diagnostics):
    calls = []

    def fake(model, prompt, **kw):
        calls.append(prompt)
        return {"outcome": "REJECT", "diagnostics": list(diagnostics)}

    monkeypatch.setattr(formalizer, "call_llm_json", fake)
    return calls


def test_a_reject_naming_only_shipped_blocks_buys_a_round(monkeypatch):
    _model_says(monkeypatch, SHIPPED)
    out = _formalize_once(RAW, "m", "", None)
    print("\n[verbose]", out)
    assert not isinstance(out, FormalizeResult), "a moot rejection was returned as final"
    reply, diags = out
    assert diags, "the round's feedback must name the blocks the model was not asked to judge"
    assert all(d.field in {"solver_input", "negative_control", "artifact_variants"} for d in diags)


def test_a_reject_on_prose_is_final_on_the_prose_alone(monkeypatch):
    _model_says(monkeypatch, SHIPPED + PROSE)
    out = _formalize_once(RAW, "m", "", None)
    print("\n[verbose]", out)
    assert isinstance(out, FormalizeResult) and out.outcome == "REJECT"
    fields = [d.field for d in out.diagnostics]
    assert fields == ["criteria"], (
        f"the researcher was handed diagnostics about blocks the harness carries verbatim: {fields}"
    )


def test_a_block_the_raw_spec_does_not_ship_is_still_the_models_to_reject(monkeypatch):
    raw = {k: v for k, v in RAW.items() if k != "negative_control"}
    _model_says(monkeypatch, [SHIPPED[1]])
    out = _formalize_once(raw, "m", "", None)
    print("\n[verbose]", out)
    assert isinstance(out, FormalizeResult) and out.outcome == "REJECT"
    assert [d.field for d in out.diagnostics] == ["negative_control"]
