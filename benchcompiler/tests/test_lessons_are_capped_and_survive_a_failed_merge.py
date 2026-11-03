"""The lessons file has a cap and survives a failed merge.

The merge answer may be cut off at its output limit; the fallback keeps the highest-ranked lessons
within the cap and never concatenates all per-agent lists.
"""

from benchcompiler import lessons as L


def _lesson(i, n_cites=6, words=40):
    return {
        "kind": "strategy",
        "lesson": f"Lesson {i}: " + " ".join(["word"] * words),
        "evidence": [
            {"worker": f"epoch_12:epoch_11:attempt_{k}", "candidate": f"org/repo{k}", "cycles": f"{k}-{k + 3}"}
            for k in range(n_cites)
        ],
    }


def test_a_failed_merge_keeps_the_previous_lessons(monkeypatch):
    monkeypatch.setattr(L, "call_llm_json", lambda *a, **k: None)  # cut off -> nothing usable
    previous = [_lesson(0), _lesson(1)]
    per_agent = {f"attempt_{i}": [_lesson(10 + i), _lesson(20 + i)] for i in range(4)}
    got, calls = L.merge_lessons(per_agent, "m", previous=previous, log=print)
    print("after a failed merge:", [l["lesson"][:9] for l in got], "calls", calls)
    assert got == previous


def test_the_rendered_file_stays_under_the_cap():
    many = [_lesson(i) for i in range(98)]
    md = L.render_markdown(many, sources=[f"attempt_{i}" for i in range(14)], model="m")
    print(f"98 lessons rendered -> {len(md)} chars, {md.count(chr(10) + '- ')} lessons kept")
    assert len(md) <= L.DEFAULT_MAX_CHARS
    assert "Lesson 0:" in md  # the highest-ranked survive; the tail is dropped


def test_citations_are_at_most_three_per_lesson():
    md = L.render_markdown([_lesson(0, n_cites=9)], sources=["attempt_0"], model="m")
    line = next(l for l in md.splitlines() if l.startswith("- "))
    print("rendered:", line)
    assert line.count("epoch_12:epoch_11:attempt_") == 3


def test_the_per_agent_read_asks_for_short_lessons_and_few_citations_too(monkeypatch):
    # the per-agent read answers stay within the reader's output limit.
    seen = {}

    def fake(model, prompt, **k):
        seen["prompt"] = prompt
        return {"lessons": [_lesson(0, n_cites=9)] * 30}

    monkeypatch.setattr(L, "call_llm_json", fake)
    merged, kept, _ = L.read_lessons([("attempt_0", "[cycle 1]\nx", 1)], "m", log=print)
    p = seen["prompt"]
    assert f"At most {L.DEFAULT_MAX_LESSONS} lessons" in p and "at most 3 citations" in p
    assert len(kept["attempt_0"]) <= L.DEFAULT_MAX_LESSONS
    assert all(len(l["evidence"]) <= L.MAX_CITES for l in kept["attempt_0"])


def test_the_merge_asks_for_short_lessons_and_few_citations(monkeypatch):
    seen = {}

    def fake(model, prompt, **k):
        seen["prompt"] = prompt
        return {"lessons": [_lesson(0)]}

    monkeypatch.setattr(L, "call_llm_json", fake)
    L.merge_lessons({"a": [_lesson(1)], "b": [_lesson(2)]}, "m", log=print)
    p = seen["prompt"]
    print(p[:1200])
    assert f"At most {L.DEFAULT_MAX_LESSONS} lessons" in p and L.DEFAULT_MAX_LESSONS <= 20
    assert "at most 3 citations" in p
