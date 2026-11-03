"""Search directions: where each searching agent is looking, summarized by a model, so that the
agents started later (and, once all run, every agent at each refresh) search elsewhere.

Synchronization: after a start delay, an LLM summarizes an agent's search direction (queries,
filters, and repositories explored), and the next agent starts with that summary and the
instruction to explore another part of the search space under the same rules. Once all agents
are running, the summaries are refreshed and redistributed at a fixed interval. Only the
direction is summarized. This is not the lessons loop (benchcompiler.lessons), which reads how
an agent worked and writes advice.

What the model reads: the agent's own actions in order -- each reply's thoughts (cut short) and
the command it ran -- not the tool outputs. What it writes: one short paragraph per agent. The
harness writes `directions.md` into the agent dir, where the search brief reads it every cycle,
and keeps the last summary of an agent when a new one cannot be made.
"""

from __future__ import annotations

import glob
import json
import os
import time

from benchcompiler.llm import call_llm_json

#: The words the model is asked for, per agent, and the hard cut on what is kept of its answer.
MAX_WORDS = int(os.environ.get("DIRECTIONS_MAX_WORDS") or 120)
MAX_CHARS_PER_AGENT = int(os.environ.get("DIRECTIONS_MAX_CHARS_PER_AGENT") or 1000)
#: The file every agent reads every cycle.
MAX_CHARS = int(os.environ.get("DIRECTIONS_MAX_CHARS") or 6000)
#: How much of an agent's action record one call reads: the start and the most recent part.
RECORD_HEAD = 20_000
RECORD_TAIL = 100_000
CALL_TIMEOUT = float(os.environ.get("DIRECTIONS_CALL_TIMEOUT") or 600)

_PROMPT = """You are reading the actions of one agent that searches for candidates for a benchmark:
in order, what it was thinking and each command it ran. Other agents search the same benchmark
beside it. Write WHERE this agent is searching, so that the others can search elsewhere:

- the data sources it queries (which API, which site, clones of which repositories);
- the search queries and filters it uses (keywords, languages, star ranges, dates, topics);
- the repositories or organizations it has explored, by name;
- what part of the space it was working on most recently.

Only the direction of the search. No advice, no lessons, no verdict on whether a repository or
a candidate is good, nothing about tools, errors or how it works. At most {max_words} words,
plain sentences.

Answer with ONE JSON object and nothing else: {{"direction": "<the paragraph>"}}

Agent: {name}
=== ACTIONS ===
{record}
=== END ===
"""


def action_record(run_dir: str) -> str:
    """The agent's replies in order, each as its cycle, its thoughts (cut short) and the command
    it ran; long records keep their start and their most recent part."""
    out = []
    path = os.path.join(run_dir, "trajectory.jsonl")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") != "on_response" or e.get("finish_reason") == "length":
                continue
            c = str(e.get("content") or "")
            try:
                d = json.loads(c[c.find("{") : c.rfind("}") + 1])
                thoughts = str(d.get("thoughts") or "")[:300]
                cmd = d.get("command") or {}
                args = cmd.get("args") or {}
                what = args.get("command") or args.get("path") or args.get("file_path") or ""
                out.append(f"[cycle {e.get('cycle')}] {thoughts}\n  -> {cmd.get('name')}: {str(what)[:400]}")
            except (ValueError, AttributeError):
                continue
    text = "\n".join(out)
    if len(text) > RECORD_HEAD + RECORD_TAIL:
        text = text[:RECORD_HEAD] + "\n[... middle of the record left out ...]\n" + text[-RECORD_TAIL:]
    return text


def summarize(run_dir: str, name: str, model: str) -> str | None:
    """One call: this agent's direction, or None when there is nothing to read or no answer."""
    record = action_record(run_dir)
    if not record:
        return None
    ans = call_llm_json(
        model,
        _PROMPT.format(max_words=MAX_WORDS, name=name, record=record),
        site="directions.read",
        timeout=CALL_TIMEOUT,
        max_tokens=1500,
    )
    text = str((ans or {}).get("direction") or "").strip() if isinstance(ans, dict) else ""
    if not text:
        return None
    return text[:MAX_CHARS_PER_AGENT].rstrip()


def render(directions: dict, *, model: str, max_chars: int = MAX_CHARS) -> str:
    """The file the briefs read; the oldest-started agents first, cut to `max_chars` by
    shortening every entry alike."""
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    names = sorted(directions)
    head = f"Summarized on {when} by {model} from the actions of: {', '.join(names)}.\n"
    per = MAX_CHARS_PER_AGENT
    while True:
        body = "".join(f"- {n}: {directions[n]['direction'][:per].rstrip()}\n" for n in names)
        if len(head) + len(body) <= max_chars or per <= 80:
            return head + body
        per -= 50


def _state_path(epoch_dir: str) -> str:
    return os.path.join(epoch_dir, "directions.json")


def write_directions(epoch_dir: str, agent_dir: str, model: str, *, log=print) -> dict:
    """Summarize every attempt under `epoch_dir` that has a record, keep an agent's last summary
    when a new one cannot be made, and write directions.md beside the runs and into the agent
    dir (where the search brief reads it every cycle). Returns the state."""
    try:
        with open(_state_path(epoch_dir), encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        state = {}
    calls = 0
    for run in sorted(glob.glob(os.path.join(epoch_dir, "attempt_*"))):
        name = os.path.basename(run)
        text = summarize(run, name, model)
        calls += 1
        if text:
            state[name] = {"direction": text, "at": time.time()}
            log(f"{name}: direction summarized ({len(text)} chars)")
        elif name in state:
            log(f"{name}: no new summary; keeping the one of {time.strftime('%H:%M', time.gmtime(state[name]['at']))}")
        else:
            log(f"{name}: nothing to summarize yet")
    with open(_state_path(epoch_dir), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    if state:
        md = render(state, model=model)
        for p in (os.path.join(epoch_dir, "directions.md"), os.path.join(agent_dir, "directions.md")):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(md)
    log(f"{len(state)} direction(s), {calls} call(s) -> {os.path.join(agent_dir, 'directions.md')}")
    return state
