"""An unprovided-placeholder diagnostic names the step it is about.

Each diagnostic names the step's position and the head of its `run` template and lists every unknown
placeholder in that step, so a model can tell which steps to repair.
"""

from __future__ import annotations

from benchcompiler.spec import check_canonical


def _canonical(steps):
    """The smallest canonical the placeholder check reads: a corroboration contract plus the
    sampling block whose lead_requires provides names."""
    return {"corroboration_contract": steps, "sampling": {"lead_requires": ["repo_url"]}}


def _placeholder_diags(canonical):
    return [d for d in check_canonical(canonical) if d.field == "corroboration_contract" and "placeholder" in d.detail]


def test_each_offending_step_is_distinguishable():
    diags = _placeholder_diags(
        _canonical(
            [
                {"run": "git clone {repo_url} {clone_path}"},
                {"run": "cd {clone_path} && make"},
                {"run": "cd {clone_path} && ./run {input_path}"},
            ]
        )
    )
    for d in diags:
        print(f"\n[verbose] {d.detail}")
    assert len(diags) == 3, [d.detail for d in diags]
    assert len({d.detail for d in diags}) == 3, (
        "three identical lines read as one mistake repeated; each must name its own step"
    )


def test_a_step_reports_every_unknown_placeholder_not_just_the_first():
    diags = _placeholder_diags(
        _canonical(
            [
                {"run": "cd {clone_path} && ./run {input_path}"},
            ]
        )
    )
    print(f"\n[verbose] {diags[0].detail}")
    assert "clone_path" in diags[0].detail and "input_path" in diags[0].detail, (
        "naming one unknown at a time turns one repair into two rounds"
    )


def test_a_provided_placeholder_is_still_silent():
    diags = _placeholder_diags(
        _canonical(
            [
                {"field": "clone_path", "require": "is non-empty"},
                {"run": "cd {clone_path} && make"},
            ]
        )
    )
    print(f"\n[verbose] diagnostics: {[d.detail for d in diags]}")
    assert diags == []
