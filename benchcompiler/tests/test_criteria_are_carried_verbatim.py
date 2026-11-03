"""The researcher's criteria are carried into the canonical verbatim, never paraphrased.

When the raw spec writes its criteria as a bullet list (a `|` block whose items start with `- `, or
a YAML list), the canonical's `criteria` are those items, verbatim, one per bullet, and the model's
paraphrase is ignored. `_resolved_criteria` still drops a bullet that restates a source filter
already living in constraints/admissibility, on record. A raw spec with no criteria block keeps the
model's list.
"""

from __future__ import annotations

import yaml

from benchcompiler.formalizer import build_canonical

RAW_CRITERIA = """\
- the program builds with the target compiler using no optimization flags (the default
  build), and that build exits zero
- a working setup is established FIRST: some reference input, however small or fast, for
  which the default build produces a reproducible output
- ALL repetition timings (at least 5 runs) are recorded as a list together with their
  mean, median and coefficient of variation
"""


def _raw(criteria):
    raw = yaml.safe_load(open("benchcompiler/examples/boca_flags.yaml"))
    raw["criteria"] = criteria
    return raw


CS = {
    "task": "t",
    "instance_input": "i",
    "instance_output": "o",
    "data_source": "d",
    "candidate_definition": "c",
    "oracle_strategy": "reference-equivalence",
    "oracle_detail": "digest",
    "interface": "cli",
    "scale": {"instance_count": "25", "compute_budget": "x", "human_budget": "0"},
    "criteria": [
        "The program builds with no optimization flags.",
        "A working setup is established with a small input.",
        "Timings are recorded with variance.",
    ],
}


def test_bullets_in_a_prose_block_are_carried_one_per_bullet():
    canon = build_canonical(_raw(RAW_CRITERIA), CS)
    print("\n[verbose]", canon.criteria)
    assert len(canon.criteria) == 3
    assert canon.criteria[1].startswith("a working setup is established FIRST")
    assert "however small or fast" in canon.criteria[1]
    assert "coefficient of variation" in canon.criteria[2]
    assert not any("variance" in c for c in canon.criteria), "the paraphrase leaked through"


def test_a_yaml_list_is_carried_as_is():
    items = ["one criterion, verbatim", "another, with its OWN emphasis"]
    canon = build_canonical(_raw(items), CS)
    assert canon.criteria == items


def test_no_raw_criteria_keeps_the_models_list():
    raw = _raw(None)
    raw.pop("criteria", None)
    canon = build_canonical(raw, CS)
    assert canon.criteria == CS["criteria"]


def test_a_multi_line_bullet_is_one_criterion():
    canon = build_canonical(_raw(RAW_CRITERIA), CS)
    assert all("\n" not in c for c in canon.criteria), canon.criteria
    assert canon.criteria[0].endswith("and that build exits zero")
