"""The search campaign's timing, with a fake clock: attempt 0 alone, then one attempt every start delay
with the directions refreshed just before it, then a refresh every sync interval while any attempt
runs, and one at the end. Also the territory modes.
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 1.0)


def test_staggered_start_then_hourly_directions_then_a_final_one():
    sc = _load("search_campaign")
    clock, launched, synced = Clock(), [], []
    ends = {0: 4 * 3600, 1: 4.5 * 3600, 2: 5 * 3600, 3: 5.2 * 3600}  # when each attempt ends
    events = sc.schedule(
        4,
        1200,
        3600,
        launch=lambda k: launched.append((k, clock.t)),
        sync=lambda: synced.append(clock.t),
        alive=lambda k: any(x == k for x, _ in launched) and clock.t < ends[k],
        now=clock.now,
        sleep=clock.sleep,
        log=lambda *_: None,
    )
    for e in events:
        print(e)
    assert [k for k, _ in launched] == [0, 1, 2, 3]
    starts = [t for _, t in launched]
    assert starts[0] == 0 and all(abs(starts[k] - 1200 * k) <= 30 for k in (1, 2, 3))
    assert synced[:3] == starts[1:], "the directions are refreshed right before each later start"
    after = synced[3:]
    assert len(after) >= 4 and all(b - a >= 3600 - 30 for a, b in zip(after, after[1:-1]))
    assert after[-1] >= 5.2 * 3600, "one last refresh after the last attempt ended"


def test_without_directions_all_start_at_once_and_nothing_is_summarized():
    sc = _load("search_campaign")
    clock, launched = Clock(), []
    sc.schedule(
        4,
        0,
        None,
        launch=lambda k: launched.append(clock.t),
        sync=None,
        alive=lambda k: True,
        now=clock.now,
        sleep=clock.sleep,
        log=lambda *_: None,
    )
    assert launched == [0, 0, 0, 0]


def test_the_territory_modes():
    sr = _load("search_replicate")
    ref = {"territory": "band A", "sibling_territories": ["band B", "band C", "band D"]}
    assert sr.territories_for("request", ref, 2, 4) == ("band C", ["band A", "band B", "band D"])
    assert sr.territories_for("none", ref, 3, 4) == ("", [])
    drawn = ["compression tools", "hashing tools", "interpreters"]
    assert sr.territories_for("drawn", ref, 1, 3, drawn) == ("hashing tools", ["compression tools", "interpreters"])
    for bad in (
        lambda: sr.territories_for("request", ref, 4, 5),
        lambda: sr.territories_for("drawn", ref, 0, 4, drawn),
        lambda: sr.territories_for("bands", ref, 0, 1),
    ):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError as e:
            print("refused:", e)
