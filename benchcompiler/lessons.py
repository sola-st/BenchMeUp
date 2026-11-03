"""Lessons: what a family's agents figured out, read from their WHOLE trajectories by a model
and written as text into the brief of the agents that run next or are still running.

Several searching agents and workers run on one brief and each would otherwise learn the same
things alone, paying again for facts the others have already found. The spread between agents
and each one's failure-to-fix sequences are evidence that would be thrown away at the freeze.

What this module does and does not do:
- It hands the model the ENTIRE sequence of an agent's thoughts and actions -- every reply
  (reasoning and command, verbatim), every tool result as the agent saw it, every harness
  message (nudges, drop notices) -- in order. Not a digest, not one line per cycle, not the
  deliveries only: dropped candidates are where the dead ends are.
- A transcript longer than the model can hold is split at cycle boundaries and nowhere else;
  each piece is read with the lessons found so far, and every cycle reaches the model.
- The judgment is the model's. The harness carries transcripts in and lessons out, renders
  the lessons as markdown with their citations, and puts the file where the briefs read it
  (`<agent_dir>/lessons.md`; the templates are re-read every cycle, so a running agent sees a
  new file on its next cycle). No lesson is a gate, a filter or a rule in code.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import time

from benchcompiler.llm import call_llm_json

#: Characters of transcript per model call. A worker's full transcript can reach 1-2 M
#: characters, most of it tool output; the reader's model
#: holds a bounded window, so the transcript is cut at cycle heads into pieces of at most this
#: size and read piece by piece with the lessons so far. Override: LESSONS_CHUNK_CHARS.
DEFAULT_CHUNK_CHARS = int(os.environ.get("LESSONS_CHUNK_CHARS") or 350_000)
#: The answer is a list of lessons with citations; a long run yields a few dozen.
ANSWER_MAX_TOKENS = 16_000
#: The merged list rides every agent's prompt every cycle, so it has a ceiling: an unbounded
#: list (hundreds of lessons) crowds out the candidate. The merge ranks and folds under this cap. A knob:
#: --max-lessons, or LESSONS_MAX.
DEFAULT_MAX_LESSONS = int(os.environ.get("LESSONS_MAX") or 20)
#: Short and concise: the file every agent reads every cycle is cut to this many characters
#: (~2,000 tokens); the lowest-ranked lessons are dropped until it fits.
DEFAULT_MAX_CHARS = int(os.environ.get("LESSONS_MAX_CHARS") or 8000)
#: The cap above applies to the SEARCH agents' lessons. Construction workers keep a larger
#: limit: 60 lessons and no character cap.
WORKER_MAX_LESSONS = int(os.environ.get("LESSONS_MAX") or 60)


def caps_for(agent_dir: str) -> tuple[int, int | None]:
    """(merge ceiling, character cap) for the agent whose brief reads the file: a worker
    (<family>_worker) keeps the larger limits, every other agent the search cap."""
    if os.path.basename(os.path.normpath(agent_dir)).endswith("_worker"):
        return WORKER_MAX_LESSONS, (
            int(os.environ["LESSONS_MAX_CHARS"]) if os.environ.get("LESSONS_MAX_CHARS") else None
        )
    return DEFAULT_MAX_LESSONS, DEFAULT_MAX_CHARS


#: Citations per lesson, in the merge's answer and in the file. Capped because citation lists
#: otherwise grow with every seeded epoch until the merge answer no longer fits.
MAX_CITES = 3
#: A 150k-token prompt takes the model longer than the 180 s the LLM helper allows by default,
#: so the timeout is raised. Override: LESSONS_CALL_TIMEOUT.
CALL_TIMEOUT = float(os.environ.get("LESSONS_CALL_TIMEOUT") or 900)

_CYCLE_HEAD = re.compile(r"(?m)^\[cycle ")

KINDS = (
    ("fix", "What worked after repeated failure"),
    ("dead_end", "What never worked"),
    ("early_sign", "What to check first"),
    ("strategy", "What plan paid off"),
    ("other", "Also learned"),
)


# ── the transcript: the whole sequence ────────────────────────────────────────────────


def _blob(run_dir: str, h: str) -> str | None:
    if not h:
        return None
    p = os.path.join(run_dir, "blobs", h)
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def transcript_of(run_dir: str, after_cycle: int = 0) -> tuple[str, int]:
    """The whole sequence of one agent's run: every reply, every result, every harness message,
    verbatim and in order. Each cycle opens with `[cycle N | candidate C]`. With `after_cycle`,
    only the cycles after that one (an hourly pass reads what is new). Returns the text and the
    last cycle it holds."""
    path = os.path.join(run_dir, "trajectory.jsonl")
    out: list[str] = []
    cycle = None
    last = after_cycle
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            ev = r.get("event")
            try:
                rc = int(r.get("cycle") or 0)
            except (TypeError, ValueError):
                rc = 0
            if rc <= after_cycle:
                continue
            if ev in ("on_response", "on_tool_result", "on_message"):
                last = max(last, rc)
            if ev == "on_response":
                c = r.get("cycle")
                if c != cycle:
                    cycle = c
                    out.append(f"[cycle {c} | candidate {r.get('candidate') or '-'}]\n")
                    out.append("MODEL:\n")
                else:
                    out.append("MODEL (again, after a format retry):\n")
                out.append((r.get("content") or "") + "\n\n")
            elif ev == "on_tool_result":
                res = r.get("result")
                text = res.get("display_text", "") if isinstance(res, dict) else str(res or "")
                out.append(f"RESULT of {r.get('name') or 'tool'}:\n{text}\n\n")
            elif ev == "on_message" and r.get("author") == "harness":
                text = _blob(run_dir, str(r.get("hash") or "")) or str(r.get("preview") or "")
                out.append(f"HARNESS ({r.get('site') or r.get('label') or 'message'}):\n{text}\n\n")
    return "".join(out), last


def transcripts(paths: list[str], after: dict[str, int] | None = None) -> list[tuple[str, str, int]]:
    """(name, transcript, last cycle) for every run under `paths`: a run dir (holds
    trajectory.jsonl) or an epoch dir (holds worker_*/ or attempt_*/). `after` maps a run name
    to the last cycle already read; that run's transcript starts after it."""
    found: list[tuple[str, str, int]] = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "trajectory.jsonl")):
            runs = [p]
        else:
            runs = sorted(os.path.dirname(t) for t in glob.glob(os.path.join(p, "*", "trajectory.jsonl")))
        for run in runs:
            name = os.path.basename(os.path.normpath(run))
            text, last = transcript_of(run, (after or {}).get(name, 0))
            if text.strip():
                found.append((name, text, last))
    return found


def split_transcript(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Cut at cycle heads only. Pieces concatenate back to the text exactly; a single cycle
    larger than the window travels whole."""
    starts = [m.start() for m in _CYCLE_HEAD.finditer(text)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    blocks = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    pieces: list[str] = []
    cur = ""
    for b in blocks:
        if cur and len(cur) + len(b) > max_chars:
            pieces.append(cur)
            cur = b
        else:
            cur += b
    if cur:
        pieces.append(cur)
    return pieces


# ── the reading: the model's judgment ─────────────────────────────────────────────────

_READ_PROMPT = """You are reading the complete record of one agent's run: every one of its replies (its
reasoning and the command it chose), what each command returned, and what the system told it.
The agent was a {role} for a benchmark family. Read the whole sequence, in order, as the
experience it was: what it tried, what failed, how many attempts a fix took, what it figured
out after many attempts, what it never got to work, and what would have saved it hours had it
known at the start.

Write lessons for the next agent that does this work on the same family (or one still doing it
beside this one). A lesson is a fact or a step, stated plainly, that the next agent can act on:
what to do, what not to do, what to check first. Each lesson cites where it comes from: the
agent's name, the candidate, and the cycle range of the sequence you read it from. Do not
invent; do not generalize past what the record shows; do not repeat the task's own instructions.

A lesson is general: it holds for every candidate of this family (the tools, the analyzer, the
machine, the build systems, the system's own behaviour, the way to work through a candidate).
Never write a verdict about a named repository or commit: not "skip X", not "X is unbuildable",
not "X is not an instance", not "do not re-attempt X". What one agent finished, the system
already keeps from coming back; and such a line makes a later agent drop a candidate without
looking, on evidence it cannot check (a build that failed offline after a poor warm-up is a
lesson about warming up, not a fact about the repository). A repository's name belongs in the
citation only. Where the record shows something specific to one repository, write the general
lesson it illustrates (what to check, how to tell, what to do), and cite the repository.

At most {max_lessons} lessons, each one or two sentences with at most 3 citations (the ones that
show it best). Keep the list short enough to be read every cycle: fold lessons that say the same thing into
one, and leave out trivia (a syntax slip in a one-liner, a typo) unless it cost many cycles. A
lesson earns its place by the cycles it would have saved.

{sofar}
Answer with ONE JSON object and nothing else:
{{"lessons": [{{"kind": "fix" | "dead_end" | "early_sign" | "strategy" | "other",
              "lesson": "<one or two sentences>",
              "evidence": [{{"worker": "<agent name>", "candidate": "<candidate>", "cycles": "<N-M>"}}]}}, ...]}}

Agent: {name}
{piece_note}
=== RECORD ===
{piece}
=== END OF RECORD ===
"""

_MERGE_PROMPT = """Below are lessons read separately from the complete records of several agents that worked on
the same benchmark family, each lesson with its citations. Merge them into ONE list that the
next agent reads at the top of every cycle, so it must be short and worth the reading:

- Rank by what a lesson would have saved: first the ones that cost the most cycles, then the
  ones that recur across agents (join them, keep every citation), then the ones a worker could
  not know from its brief. Fold lessons that say the same thing into one line.
- Leave out trivia and one-off slips unless they cost many cycles.
- Leave out every lesson whose subject is a named repository or commit ("skip X", "X is
  unbuildable", "X is not an instance", "do not re-attempt X", lists of such names): the system
  already keeps finished candidates from coming back, and such a line makes a later agent drop
  without looking. Keep the general lesson it illustrates, if there is one, with the repository
  in the citation only.
- At most {max_lessons} lessons in total. Each plain, actionable, one or two sentences, with
  at most 3 citations (the ones that show it best). Do not add anything the records did not show.

Answer with ONE JSON object and nothing else, in the same shape:
{{"lessons": [{{"kind": ..., "lesson": ..., "evidence": [...]}}]}}

{lists}
"""


def _sofar(prior: list[dict]) -> str:
    if not prior:
        return ""
    return (
        "Lessons found so far in the earlier part of this record (revise, extend or keep them; "
        "return the full list):\n" + json.dumps({"lessons": prior}, indent=1) + "\n"
    )


def _clean(items) -> list[dict]:
    out = []
    for it in items or []:
        if not isinstance(it, dict) or not str(it.get("lesson") or "").strip():
            continue
        kind = str(it.get("kind") or "other")
        if kind not in dict(KINDS):
            kind = "other"
        ev = [e for e in (it.get("evidence") or []) if isinstance(e, dict)]
        # No scope: a lesson is general or it is not a lesson (an older state's field is dropped).
        out.append({"kind": kind, "lesson": str(it["lesson"]).strip(), "evidence": ev})
    return out


def merge_lessons(
    per_agent: dict[str, list[dict]],
    model: str,
    *,
    max_lessons: int = DEFAULT_MAX_LESSONS,
    previous: list[dict] | None = None,
    log=print,
) -> tuple[list[dict], int]:
    """One call: rank and fold the per-agent lists into the list the briefs carry.
    Returns (lessons, calls made). With one non-empty list there is nothing to merge.
    A merge that returns nothing usable keeps `previous`, the last merged list; the per-agent
    lists are never pasted together."""
    lists = {n: ls for n, ls in per_agent.items() if ls}
    if not lists:
        return [], 0
    if len(lists) == 1 and len(next(iter(lists.values()))) <= max_lessons:
        return next(iter(lists.values())), 0
    lists_text = "\n\n".join(f"=== from {n} ===\n" + json.dumps({"lessons": ls}, indent=1) for n, ls in lists.items())
    ans = call_llm_json(
        model,
        _MERGE_PROMPT.format(lists=lists_text, max_lessons=max_lessons),
        site="lessons.merge",
        timeout=CALL_TIMEOUT,
        max_tokens=ANSWER_MAX_TOKENS,
    )
    if isinstance(ans, dict) and _clean(ans.get("lessons")):
        return _clean(ans.get("lessons"))[:max_lessons], 1
    if previous:
        log(f"merge returned nothing usable; keeping the last {len(previous)} merged lesson(s)")
        return list(previous), 1
    log("merge returned nothing usable and there is no earlier list; keeping the first agent's, capped")
    return next(iter(lists.values()))[:max_lessons], 1


def read_lessons(
    named: list[tuple[str, str, int]],
    model: str,
    *,
    role: str = "worker",
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    prior_by_agent: dict[str, list[dict]] | None = None,
    max_lessons: int = DEFAULT_MAX_LESSONS,
    previous: list[dict] | None = None,
    log=print,
) -> tuple[list[dict], dict[str, list[dict]], int]:
    """Read every transcript whole (piece by piece when long), each with that agent's lessons
    from earlier passes as "so far", then merge across agents. Returns (merged lessons, the
    per-agent lists to keep for the next pass, model calls made)."""
    per_agent: list[list[dict]] = []
    calls = 0
    for name, text, _last in named:
        pieces = split_transcript(text, chunk_chars)
        prior: list[dict] = list((prior_by_agent or {}).get(name) or [])
        for i, piece in enumerate(pieces):
            note = f"This record is in {len(pieces)} parts; this is part {i + 1}." if len(pieces) > 1 else ""
            prompt = _READ_PROMPT.format(
                role=role, sofar=_sofar(prior), name=name, piece_note=note, piece=piece, max_lessons=max_lessons
            )
            calls += 1
            ans = call_llm_json(model, prompt, site="lessons.read", max_tokens=ANSWER_MAX_TOKENS, timeout=CALL_TIMEOUT)
            if not isinstance(ans, dict):
                log(f"{name} part {i + 1}/{len(pieces)}: no answer; keeping {len(prior)} lesson(s) so far")
                continue
            # the same caps as the merge: an agent's list is carried into every later pass,
            # and uncapped it grows until the read answer is cut off
            prior = [{**l, "evidence": l["evidence"][:MAX_CITES]} for l in _clean(ans.get("lessons"))][:max_lessons]
            log(f"{name} part {i + 1}/{len(pieces)}: {len(prior)} lesson(s) ({len(piece):,} chars read)")
        per_agent.append(prior)
    kept = {named[i][0]: ls for i, ls in enumerate(per_agent)}
    merged, n = merge_lessons(kept, model, max_lessons=max_lessons, previous=previous, log=log)
    return merged, kept, calls + n


# ── the file the briefs read ─────────────────────────────────────────────────────────


def _short_candidate(c: str) -> str:
    """`https://github.com/org/repo|<40-char sha>` -> `org/repo@<8 chars>`; anything else as is."""
    c = c.strip()
    if not c:
        return ""
    repo, _, sha = c.partition("|")
    repo = re.sub(r"^https?://(www\.)?github\.com/", "", repo)
    if repo.endswith(".git"):
        repo = repo[:-4]
    repo = repo.rstrip("/")
    return f"{repo}@{sha[:8]}" if sha else repo


def _cite(ev: list[dict]) -> str:
    """One citation per (worker, candidate), cycle ranges joined: the trial's file repeated a
    70-character candidate URL on every line, twice when two ranges cited it."""
    order: list[tuple[str, str]] = []
    ranges: dict[tuple[str, str], list[str]] = {}
    for e in ev:
        key = (str(e.get("worker") or "?"), _short_candidate(str(e.get("candidate") or "")))
        if key not in ranges and len(order) >= MAX_CITES:
            continue
        if key not in ranges:
            order.append(key)
            ranges[key] = []
        cy = str(e.get("cycles") or "").strip()
        if cy and cy not in ranges[key]:
            ranges[key].append(cy)
    parts = []
    for w, c in order:
        s = w + (f" on {c}" if c else "")
        if ranges[(w, c)]:
            s += ", cycles " + ", ".join(ranges[(w, c)])
        parts.append(s)
    return "; ".join(parts)


def render_markdown(lessons: list[dict], *, sources: list[str], model: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The file the briefs read, at most `max_chars`: lessons arrive ranked (the merge ranks
    them), so the lowest-ranked are dropped until it fits."""
    lessons = list(lessons)
    while True:
        md = _render(lessons, sources=sources, model=model)
        if len(md) <= max_chars or not lessons:
            return md
        lessons.pop()


def _render(lessons: list[dict], *, sources: list[str], model: str) -> str:
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    # The count, not the names: a seeded loop carries every earlier agent's name, and the list
    # would take a large share of the character cap. lessons.json keeps the names.
    out = [f"Read on {when} by {model} from the complete records of {len(sources)} agent(s).", ""]
    if lessons:
        out.append("== Lessons that hold for every candidate of this family ==")
    for kind, title in KINDS:
        items = [l for l in lessons if l.get("kind", "other") == kind]
        if not items:
            continue
        out.append(f"{title}:")
        for l in items:
            cite = _cite(l.get("evidence") or [])
            out.append(f"- {l['lesson']}" + (f" ({cite})" if cite else ""))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _state_path(base_md: str) -> str:
    return os.path.splitext(base_md)[0] + ".json"


def _load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            st = json.load(fh)
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def _pinned(agent_dir: str) -> str:
    """<agent>/lessons_pinned.md: notes kept in front of the agents. The loop rewrites lessons.md
    from the records every pass, so a note written there would not last."""
    try:
        with open(os.path.join(agent_dir, "lessons_pinned.md"), encoding="utf-8") as fh:
            text = fh.read().strip()
    except OSError:
        return ""
    return text + "\n\n" if text else ""


def _write_out(
    base: str, agent_dir: str, lessons: list[dict], sources: dict, model: str, calls: int, chars: int
) -> None:
    pinned = _pinned(agent_dir)
    cap = caps_for(agent_dir)[1]
    md = pinned + (
        render_markdown(lessons, sources=sorted(sources), model=model, max_chars=cap - len(pinned))
        if cap
        else _render(lessons, sources=sorted(sources), model=model)
    )
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    with open(base, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(_state_path(base), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": model,
                "sources": sources,
                "lessons": lessons,
                "calls": calls,
                "chars_read": chars,
                "ts": time.time(),
            },
            fh,
            indent=1,
        )
    if lessons:
        shutil.copyfile(base, os.path.join(agent_dir, "lessons.md"))


def seed_state(base_md: str, seed_from: str, log=print) -> bool:
    """Start this epoch's reader from an earlier epoch's per-agent lists, once: the earlier
    lists are kept under `<epoch name>:<agent>` so they take part in every merge as agents
    with no new record, and they cannot collide with this epoch's agent names. Does nothing
    when a state already exists here. Returns whether it seeded."""
    if os.path.exists(_state_path(base_md)):
        return False
    src = os.path.join(seed_from, "lessons.json") if os.path.isdir(seed_from) else seed_from
    st = _load_state(src)
    if not st.get("sources"):
        log(f"nothing to seed from {src}")
        return False
    tag = os.path.basename(os.path.normpath(os.path.dirname(src) if not os.path.isdir(seed_from) else seed_from))
    sources = {
        f"{tag}:{n}": {"cycles_read": int(v.get("cycles_read") or 0), "lessons": list(v.get("lessons") or [])}
        for n, v in st["sources"].items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(base_md)), exist_ok=True)
    with open(_state_path(base_md), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": st.get("model"),
                "sources": sources,
                "lessons": list(st.get("lessons") or []),
                "calls": 0,
                "chars_read": 0,
                "seeded_from": os.path.abspath(src),
                "ts": time.time(),
            },
            fh,
            indent=1,
        )
    log(
        f"seeded from {src}: {len(sources)} agent list(s), {sum(len(v['lessons']) for v in sources.values())} lesson(s)"
    )
    return True


def write_lessons(
    paths: list[str],
    *,
    agent_dir: str,
    model: str,
    out_path: str | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    role: str | None = None,
    max_lessons: int | None = None,
    remerge: bool = False,
    seed_from: str | None = None,
    log=print,
) -> dict:
    """Read what is new in every run under `paths` (all of it on the first pass), fold it into
    the lessons kept from earlier passes, write lessons.md beside the runs and into `agent_dir`,
    where the brief reads it every cycle. lessons.json beside the .md keeps, per agent, the last
    cycle read and that agent's lessons, so the next pass costs only the new cycles."""
    if max_lessons is None:
        max_lessons = caps_for(agent_dir)[0]
    first = paths[0]
    base = out_path or os.path.join(
        first
        if not os.path.isfile(os.path.join(first, "trajectory.jsonl"))
        else os.path.dirname(os.path.normpath(first)),
        "lessons.md",
    )
    if seed_from:
        seed_state(base, seed_from, log=log)
    state = _load_state(_state_path(base))
    sources: dict = dict(state.get("sources") or {})
    if remerge:
        # The per-agent lists are kept; rank and fold them again without re-reading a byte.
        lists = {n: list(v.get("lessons") or []) for n, v in sources.items()}
        lessons, calls = merge_lessons(
            lists, model, max_lessons=max_lessons, previous=list(state.get("lessons") or []), log=log
        )
        _write_out(base, agent_dir, lessons, sources, model, calls, 0)
        log(
            f"re-merged {sum(len(v) for v in lists.values())} lesson(s) from {len(lists)} agent(s) into "
            f"{len(lessons)} ({calls} call) -> {base}" + (f" and {agent_dir}/lessons.md" if lessons else "")
        )
        return {"lessons": len(lessons), "workers": len(sources), "calls": calls, "chars_read": 0, "path": base}
    after = {n: int(v.get("cycles_read") or 0) for n, v in sources.items()}
    named = transcripts(paths, after)
    if not named:
        log("nothing new under " + ", ".join(paths) + (f" (state: {len(sources)} agent(s) read)" if sources else ""))
        return {
            "lessons": len(state.get("lessons") or []),
            "workers": len(sources),
            "calls": 0,
            "chars_read": 0,
            "path": base if sources else None,
        }
    if role is None:
        role = "searching agent" if any(n.startswith("attempt") for n, _, _ in named) else "worker"
    prior = {n: list(v.get("lessons") or []) for n, v in sources.items()}
    # Agents with nothing new keep their lists and still take part in the merge.
    merged_in = list(named) + [(n, "", after[n]) for n in prior if n not in {m[0] for m in named} and prior[n]]
    lessons, kept, calls = read_lessons(
        merged_in,
        model,
        role=role,
        chunk_chars=chunk_chars,
        prior_by_agent=prior,
        max_lessons=max_lessons,
        previous=list(state.get("lessons") or []),
        log=log,
    )
    for n, _text, last in named:
        sources[n] = {"cycles_read": last, "lessons": kept.get(n, [])}
    for n in kept:
        sources.setdefault(n, {})["lessons"] = kept[n]
    chars = sum(len(t) for _, t, _ in named)
    _write_out(base, agent_dir, lessons, sources, model, calls, chars)
    log(
        f"{len(lessons)} lesson(s) from {len(sources)} agent(s), {calls} model call(s), {chars:,} chars read -> {base}"
        + (f" and {agent_dir}/lessons.md" if lessons else " (none, agent copy untouched)")
    )
    return {"lessons": len(lessons), "workers": len(sources), "calls": calls, "chars_read": chars, "path": base}
