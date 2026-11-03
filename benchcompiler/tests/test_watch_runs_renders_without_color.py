"""The dashboard must render when its output is not a terminal.

`c(txt, *codes)` returns a str on both the coloured and the uncoloured path, so a non-string
argument (such as an int) can be concatenated when colour is off, which happens whenever stdout is
not a tty (`| head`, `> log`, CI).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

watch_runs = pytest.importorskip("watch_runs")


@pytest.fixture
def no_color():
    """Exactly what main() does when stdout is a pipe."""
    prev = watch_runs.USE_COLOR
    watch_runs.USE_COLOR = False
    yield
    watch_runs.USE_COLOR = prev


# ── the helper ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [12, 0, 3.5, None, True])
def test_c_returns_a_string_whatever_it_is_given(value, no_color):
    """Both branches, same contract. A caller concatenates the result; it must be a str."""
    assert isinstance(watch_runs.c(value), str)
    assert isinstance(watch_runs.c(value, watch_runs.BOLD), str)


@pytest.mark.parametrize("value", [12, 0, 3.5, None, True])
def test_c_returns_a_string_with_color_too(value):
    prev = watch_runs.USE_COLOR
    watch_runs.USE_COLOR = True
    try:
        assert isinstance(watch_runs.c(value, watch_runs.BOLD), str)
    finally:
        watch_runs.USE_COLOR = prev


def test_the_two_branches_agree_on_the_text():
    """Colour may add escapes; it must not change what the number reads as."""
    prev = watch_runs.USE_COLOR
    try:
        watch_runs.USE_COLOR = False
        plain = watch_runs.c(12, watch_runs.BOLD)
        watch_runs.USE_COLOR = True
        colored = watch_runs.c(12, watch_runs.BOLD)
    finally:
        watch_runs.USE_COLOR = prev
    assert plain == "12"
    assert "12" in colored


# ── the fan-out frame ────────────────────────────────────────────────────────────


def _fanout_dir(tmp_path):
    """A fan-out directory of the shape `bench fanout` writes."""
    d = tmp_path / "fan"
    d.mkdir()
    runs = []
    for i in range(2):
        r = d / f"worker_{i}"
        (r / "cycles").mkdir(parents=True)
        runs.append(str(r))
        (d / f"worker_{i}.instances.jsonl").write_text(
            json.dumps(
                {
                    "benchmark_name": "t",
                    "sample_size": 6,
                    "fanout": {
                        "worker": i,
                        "n_workers": 2,
                        "own_orgs": ["acme"],
                        "excluded_orgs": ["other"],
                        "leads": [f"p{i}|acme|src/a.c"],
                    },
                }
            )
            + "\n"
        )
    (d / "fanout.json").write_text(
        json.dumps(
            {
                "prospect_run": "/somewhere/r23_x_sampler",
                "workers": 2,
                "total_target": 12,
                "runs": runs,
                "commands": [],
            }
        )
    )
    return str(d)


def test_the_fanout_frame_renders_with_color_off(tmp_path, no_color):
    """The fan-out frame renders when piped."""
    lines = watch_runs._fanout_frame(_fanout_dir(tmp_path))
    assert lines and all(isinstance(x, str) for x in lines)
    assert "12" in "\n".join(lines), "the total target should appear"


def test_the_fanout_frame_renders_with_color_on(tmp_path):
    prev = watch_runs.USE_COLOR
    watch_runs.USE_COLOR = True
    try:
        lines = watch_runs._fanout_frame(_fanout_dir(tmp_path))
    finally:
        watch_runs.USE_COLOR = prev
    assert lines and all(isinstance(x, str) for x in lines)


def test_no_helper_call_site_can_reintroduce_this(tmp_path, no_color):
    """Any non-string argument must survive the uncoloured path, not only the one used by the fan-
    out frame.
    """
    for value in (7, 0.5, None):
        assert isinstance(watch_runs.c(value, watch_runs.DIM) + "", str)
