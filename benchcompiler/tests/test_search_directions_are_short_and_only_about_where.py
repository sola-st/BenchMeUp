"""Search directions: a model summarizes where each searching agent looks, from its actions, and the
summaries reach the agents' briefs. They are short, only about direction, and the last one is kept
when a new one cannot be made.
"""

import json

from benchcompiler import directions as D


def _run(tmp_path, name, n=5):
    d = tmp_path / name
    d.mkdir(parents=True)
    with open(d / "trajectory.jsonl", "w") as fh:
        for i in range(n):
            reply = {
                "thoughts": f"search C repos with 100..133 stars, page {i}",
                "command": {
                    "name": "linux_terminal",
                    "args": {
                        "command": f"curl https://api.github.com/search/repositories?q=language:C+stars:100..133&page={i}"
                    },
                },
            }
            fh.write(json.dumps({"event": "on_response", "cycle": i, "content": json.dumps(reply)}) + "\n")
            fh.write(
                json.dumps({"event": "on_message", "cycle": i, "preview": "OUTPUT THE MODEL DOES NOT NEED"}) + "\n"
            )
        fh.write(
            json.dumps({"event": "on_response", "cycle": n, "finish_reason": "length", "content": "'rpmmi', " * 50})
            + "\n"
        )
    return str(d)


def test_the_record_holds_the_actions_not_the_outputs_and_not_cut_off_replies(tmp_path):
    rec = D.action_record(_run(tmp_path, "attempt_0"))
    print(rec[:400])
    assert "stars:100..133" in rec and "search C repos" in rec
    assert "OUTPUT THE MODEL DOES NOT NEED" not in rec and "rpmmi" not in rec


def test_the_prompt_asks_for_where_only_and_the_answer_is_cut(tmp_path, monkeypatch):
    seen = {}

    def fake(model, prompt, **k):
        seen["prompt"] = prompt
        return {"direction": "x" * 5000}

    monkeypatch.setattr(D, "call_llm_json", fake)
    out = D.summarize(_run(tmp_path, "attempt_0"), "attempt_0", "m")
    p = seen["prompt"]
    assert "WHERE this agent is searching" in p and "No advice, no lessons" in p
    assert f"At most {D.MAX_WORDS} words" in p
    assert len(out) <= D.MAX_CHARS_PER_AGENT


def test_the_file_reaches_the_agent_dir_capped_and_keeps_the_last_summary(tmp_path, monkeypatch):
    epoch, agent = tmp_path / "epoch", tmp_path / "agent"
    agent.mkdir()
    _run(epoch, "attempt_0")
    _run(epoch, "attempt_1")
    monkeypatch.setattr(
        D, "call_llm_json", lambda m, p, **k: {"direction": "searches stars 100..133 via the API " * 30}
    )
    D.write_directions(str(epoch), str(agent), "m")
    md = (agent / "directions.md").read_text()
    print(md[:600])
    assert "- attempt_0:" in md and "- attempt_1:" in md and len(md) <= D.MAX_CHARS
    monkeypatch.setattr(D, "call_llm_json", lambda m, p, **k: None)  # a failed pass
    D.write_directions(str(epoch), str(agent), "m")
    assert "- attempt_0: searches stars" in (agent / "directions.md").read_text()


def test_an_agent_with_no_record_yet_is_left_out(tmp_path, monkeypatch):
    epoch, agent = tmp_path / "epoch", tmp_path / "agent"
    agent.mkdir()
    _run(epoch, "attempt_0")
    (epoch / "attempt_1").mkdir()
    monkeypatch.setattr(D, "call_llm_json", lambda m, p, **k: {"direction": "d"})
    D.write_directions(str(epoch), str(agent), "m")
    md = (agent / "directions.md").read_text()
    assert "attempt_0" in md and "- attempt_1" not in md
