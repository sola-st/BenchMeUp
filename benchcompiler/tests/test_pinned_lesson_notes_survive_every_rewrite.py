"""A note in <agent>/lessons_pinned.md heads every rewritten <agent>/lessons.md, and the cap still
holds for the whole file. The lessons loop rewrites lessons.md every pass, so notes belong in
lessons_pinned.md.
"""

import os

from benchcompiler import lessons as L

NOTE = "NOTE: fetch the SonarQube server zip through ${MAVEN_MIRROR}, never repo1 directly."


def _lesson(i, words=60):
    return {
        "kind": "strategy",
        "lesson": f"Lesson {i}: " + " ".join(["word"] * words),
        "evidence": [{"worker": "worker_0", "candidate": "org/repo", "cycles": "1-2"}],
    }


def _write(tmp_path, lessons):
    agent = tmp_path / "agent"
    agent.mkdir(exist_ok=True)
    (agent / "lessons_pinned.md").write_text(NOTE + "\n")
    base = tmp_path / "epoch" / "lessons.md"
    L._write_out(str(base), str(agent), lessons, {"worker_0": {}}, "m", 1, 0)
    return (agent / "lessons.md").read_text(), base.read_text()


def test_the_pinned_note_heads_the_rewritten_file(tmp_path):
    agent_md, epoch_md = _write(tmp_path, [_lesson(i) for i in range(3)])
    print(agent_md)
    assert agent_md.startswith(NOTE)
    assert epoch_md == agent_md
    assert "Lesson 0:" in agent_md


def test_the_cap_holds_with_the_note(tmp_path):
    agent_md, _ = _write(tmp_path, [_lesson(i) for i in range(60)])
    print(f"{len(agent_md)} chars, {agent_md.count(chr(10) + '- ')} lessons kept")
    assert agent_md.startswith(NOTE)
    assert len(agent_md) <= L.DEFAULT_MAX_CHARS


def test_the_header_does_not_spend_the_cap_on_agent_names():
    # the header stays short so the merged lessons fit within the cap.
    names = [f"seed:epoch_9:epoch_8:epoch_7:epoch_6:epoch_5:worker_{i}" for i in range(51)]
    lessons = [_lesson(i, words=30) for i in range(20)]
    md = L.render_markdown(lessons, sources=names, model="m")
    head = md.splitlines()[0]
    print(head, f"| {len(md)} chars, {md.count(chr(10) + '- ')} lessons kept")
    assert len(head) < 200 and "51" in head
    assert md.count("\n- ") == 20


def test_no_pinned_file_changes_nothing(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    base = tmp_path / "epoch" / "lessons.md"
    L._write_out(str(base), str(agent), [_lesson(0)], {"worker_0": {}}, "m", 1, 0)
    md = (agent / "lessons.md").read_text()
    assert md.startswith("Read on ") and not os.path.exists(agent / "lessons_pinned.md")
