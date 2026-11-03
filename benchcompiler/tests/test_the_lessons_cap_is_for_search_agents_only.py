"""The lessons cap (20 lessons, 8,000 characters) applies to search agents' lessons only. Construction
workers keep a limit of 60 lessons and no character cap.
"""

from benchcompiler import lessons as L


def _lesson(i, words=60):
    return {
        "kind": "fix",
        "lesson": f"Lesson {i}: " + " ".join(["word"] * words),
        "evidence": [{"worker": "worker_0", "candidate": "org/repo", "cycles": "1-2"}],
    }


def _write(tmp_path, agent_name, n):
    agent = tmp_path / agent_name
    agent.mkdir()
    base = tmp_path / "epoch" / "lessons.md"
    L._write_out(str(base), str(agent), [_lesson(i) for i in range(n)], {"worker_0": {}}, "m", 1, 0)
    return (agent / "lessons.md").read_text()


def test_a_worker_file_is_not_cut_to_the_search_cap(tmp_path):
    md = _write(tmp_path, "sonarqube_worker", 59)
    print(f"worker: {len(md)} chars, {md.count(chr(10) + '- ')} lessons kept")
    assert md.count("\n- ") == 59
    assert len(md) > L.DEFAULT_MAX_CHARS


def test_a_search_file_keeps_the_cap(tmp_path):
    md = _write(tmp_path, "comment_search", 59)
    print(f"search: {len(md)} chars, {md.count(chr(10) + '- ')} lessons kept")
    assert len(md) <= L.DEFAULT_MAX_CHARS


def test_the_merge_ceiling_follows_the_agent(tmp_path):
    assert L.caps_for(str(tmp_path / "sonarqube_worker")) == (L.WORKER_MAX_LESSONS, None)
    assert L.caps_for(str(tmp_path / "boca_search")) == (L.DEFAULT_MAX_LESSONS, L.DEFAULT_MAX_CHARS)
    assert L.WORKER_MAX_LESSONS == 60
