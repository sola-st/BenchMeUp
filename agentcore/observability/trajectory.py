"""TrajectoryWriter — event-driven run transcript (JSONL + per-cycle prompt dumps).

Subscribes to LoopAgent lifecycle events via EventEmitter.subscribe(); never
monkey-patches the loop. Writes:
  <out>/trajectory.jsonl       one line per event
  <out>/cycles/cycle_NNN.json  exact prompt messages sent each cycle
  <out>/result.json            final RunResult
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from agentcore.tools.parsing import clip

#: Longest string this writer will put in one field of one event. A LOG MUST BE BOUNDED:
#: the reducer bounds what the MODEL sees, and nothing bounded what went to DISK, so a check
#: that piped raw video to stdout (`ffmpeg ... -f rawvideo -`) would write a single
#: multi-gigabyte JSON record (JSON-escaping each non-UTF8 byte into six characters). 200k
#: chars is far above anything a model is ever shown, so this discards nothing anyone reads. BENCH_TRAJECTORY_MAX_CHARS overrides; 0 disables.
MAX_FIELD_CHARS = int(os.environ.get("BENCH_TRAJECTORY_MAX_CHARS") or 200_000)

#: Greppable, and it states the true length -- a truncated record must never read as a short
#: one. Anything counting or diffing trajectory text can find every place it was cut.
TRUNCATED = "BENCH_TRAJECTORY_TRUNCATED"


def _bounded(text: str) -> str:
    """One string, capped head+tail, announcing the cut and the original length."""
    if not MAX_FIELD_CHARS:
        return text
    return clip(text, MAX_FIELD_CHARS, label=f"{TRUNCATED}: ")


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump(mode="json"))
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, str):
        return _bounded(obj)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return _bounded(str(obj))


def _terminal_of(out_dir: str) -> list:
    """What this run's collector calls done, resolved from the run's own metadata.

    `run_meta.json` names the agent directory, whose `agent.yaml` declares `worklist.terminal`.
    Read here rather than plumbed through because the writer has the out_dir and nothing else,
    and a wrong answer is only ever a fallback to the conventional name.
    """
    try:
        with open(os.path.join(out_dir, "run_meta.json")) as fh:
            agent_dir = (json.load(fh) or {}).get("agent_dir") or ""
        import yaml

        with open(os.path.join(agent_dir, "agent.yaml")) as fh:
            wl = (yaml.safe_load(fh) or {}).get("worklist") or {}
        return [str(t) for t in (wl.get("terminal") or [])]
    except Exception:  # noqa: BLE001
        return []


def run_census(working_set, terminal=None) -> dict | None:
    """What the run produced, computed from the worklist rather than asked for.

    Returns None when there is no worklist: not every agent has one, and inventing
    `delivered: []` for a run that never had candidates is a claim the harness cannot support.

    WHY. An agent's final account of its own run can be factually wrong (a wrong candidate
    count, a wrong delivered project) while the truth sits in the same file. The delivered
    project, the candidate count and every candidate's state are in the worklist; nothing
    about them needs recalling from a long context at the end of a run. This does not read the account, does not
    refuse, and does not replace it -- the criteria ask for that account in as many words. It
    states the derivable half beside the claim so a reader sees both at once.
    """
    if not isinstance(working_set, dict):
        return None
    wl = working_set.get("worklist")
    if not isinstance(wl, dict):
        return None
    cands = wl.get("candidates")
    if not isinstance(cands, dict):
        return None
    # WHAT *THIS* TASK CALLS DONE. Testing `st == "DELIVERED"` would give `"delivered": []`
    # beside `"states": {"CONTRAST_VERIFIED": 3}` for a task whose terminal state has another
    # name, so the terminal states come from the spec.
    want = {str(t) for t in (terminal or [])} or {"DELIVERED"}
    states: dict = {}
    delivered = []
    for key, cand in cands.items():
        if not isinstance(cand, dict):
            continue
        st = str(cand.get("state") or "?")
        states[st] = states.get(st, 0) + 1
        if st in want:
            fields = cand.get("fields") or {}
            # The candidate's KEY is its identity in every task; a friendlier field name is
            # used when the record has one. some tasks' records carry `program_id`, which
            # is among the names below.
            name = next(
                (str(fields[k]) for k in ("project_id", "program_id", "project", "repo", "name") if fields.get(k)), None
            )
            delivered.append(name or str(key))
    return {"candidates": len(cands), "delivered": delivered, "states": states, "terminal": sorted(want)}


#: Every record's `kind`, from the event that produced it.
_KIND_OF_EVENT = {
    "on_message": "message",
    "on_refusal": "refusal",
    "on_closure": "closure",
    "on_decision": "decision",
    "on_prompt_sent": "prompt",
    "on_response": "response",
    "on_tool_result": "tool_result",
}
_DECISION_EVENTS = {
    "on_parse_path",
    "on_stage_transition",
    "on_output_reduced",
    "on_oracle_check",
    "on_budget_threshold",
    "on_model_retry",
    "on_stall",
    "on_slow_candidate",
    "on_intervention",
    "on_env_change",
}

#: How much of a blobbed text stays inline, so a record reads without opening the blob.
PREVIEW_CHARS = 200


def _run_and_worker(out_dir: str) -> tuple[str, str]:
    """`runs/<run>/worker_3` -> ("<run>", "worker_3"); a run with no worker
    directory is its own worker ("")."""
    out_dir = os.path.abspath(out_dir.rstrip("/"))
    leaf, parent = os.path.basename(out_dir), os.path.basename(os.path.dirname(out_dir))
    if leaf.startswith("worker_") or leaf.startswith("attempt-"):
        return parent, leaf
    return leaf, ""


def _git_commit() -> str:
    """The harness commit a run was produced by, or "" when git is not there to ask."""
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return ""


class TrajectoryWriter:
    """One append-only stream per worker, every record under one envelope.

    ENVELOPE: `id`, `run`, `worker`, `epoch`, `cycle`, `stage`, `candidate`,
    `kind`, `site`, `refs`, `author` on every record, so the five debugging levels are joins
    over one stream rather than reconstructions. `event` stays for the readers that key on
    it. Texts that can grow -- a message, a refusal -- go once to `blobs/<sha256>` and the
    record carries the hash and a preview.
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(os.path.join(out_dir, "cycles"), exist_ok=True)
        self._jsonl = open(os.path.join(out_dir, "trajectory.jsonl"), "a")
        self._cycle = 0
        self._seq = 0
        self._agent = None
        self._run, self._worker = _run_and_worker(out_dir)
        self._epoch = time.strftime("%Y%m%dT%H%M%S")
        self._pending_refs: list[str] = []  # refusal/decision ids since the last tool result

    def bind(self, agent) -> None:
        """Know the agent, so the envelope carries the stage and the current candidate."""
        self._agent = agent

    # ── envelope ──
    def _context(self) -> dict:
        a = self._agent
        stage = candidate = None
        if a is not None:
            try:
                stage = a.stages.name()
            except Exception:  # noqa: BLE001
                pass
            try:
                candidate = a._current_candidate_key() or None
            except Exception:  # noqa: BLE001
                pass
        return {"stage": stage, "candidate": candidate}

    def _log(self, event: str, **payload) -> str:
        self._seq += 1
        rid = f"{self._run}/{self._worker}/{self._cycle}/{self._seq}"
        kind = _KIND_OF_EVENT.get(event) or ("decision" if event in _DECISION_EVENTS else "lifecycle")
        context = self._context()
        # A record ABOUT a candidate (a refusal, a closure) names it; that name is the
        # envelope's candidate, over the one the agent happens to be serving.
        if payload.get("candidate"):
            context["candidate"] = str(payload.pop("candidate"))
        envelope = {
            "id": rid,
            "ts": time.time(),
            "run": self._run,
            "worker": self._worker,
            "epoch": self._epoch,
            "cycle": self._cycle,
            **context,
            "kind": kind,
            "site": payload.pop("site", None),
            "refs": payload.pop("refs", None) or [],
            "author": payload.pop("author", "harness"),
            "event": event,
        }
        # THE ENVELOPE WINS: `on_intervention` carries its own `kind` and `on_stall` its own
        # `stage`, which would overwrite the envelope's and hide the record from a query by
        # kind. A colliding payload key keeps its
        # value under `payload_<key>`; nothing is lost and nothing is misfiled.
        body = _jsonable(payload)
        for k in list(body):
            if k in envelope:
                body[f"payload_{k}"] = body.pop(k)
        rec = {**envelope, **body}
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()
        return rid

    def _blob(self, text: str) -> dict:
        """Store a text once by content and describe it: hash, size, preview."""
        import hashlib

        text = _bounded(str(text or ""))
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        path = os.path.join(self.out_dir, "blobs", digest)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8", errors="replace") as fh:
                fh.write(text)
            os.replace(tmp, path)
        return {"hash": digest, "bytes": len(text), "preview": text[:PREVIEW_CHARS]}

    # ── typed records ──
    def on_message(self, text="", **kw):
        self._log("on_message", **self._blob(text), **kw)

    def on_refusal(self, text="", **kw):
        rid = self._log("on_refusal", **self._blob(text), **kw)
        self._pending_refs.append(rid)

    def on_closure(self, **kw):
        self._log("on_closure", **kw)

    def on_decision(self, **kw):
        rid = self._log("on_decision", **kw)
        self._pending_refs.append(rid)

    # lifecycle handlers (names match EVENTS) ----------------------------------
    def on_run_start(self, **kw):
        # Provenance lives in the run dir, not the launcher's shell: which harness commit
        # and which resolved model produced this stream.
        meta = {}
        try:
            with open(os.path.join(self.out_dir, "run_meta.json")) as fh:
                meta = json.load(fh) or {}
        except Exception:  # noqa: BLE001
            pass
        self._log(
            "on_run_start", commit=_git_commit(), models=meta.get("models"), agent_dir=meta.get("agent_dir"), **kw
        )

    def on_cycle_start(self, **kw):
        self._cycle += 1
        self._log("on_cycle_start", **kw)

    def on_prompt_sent(self, messages=None, **kw):
        with open(os.path.join(self.out_dir, "cycles", f"cycle_{self._cycle:03d}.json"), "w") as f:
            json.dump(_jsonable(messages), f, indent=2)
        self._log("on_prompt_sent", n_messages=len(messages or []), **kw)

    def on_response(self, content="", **kw):
        self._log("on_response", content=content, author="model", **kw)

    def on_tool_result(self, name="", args=None, result=None, **kw):
        # The gate verdicts and decisions taken while this tool ran are what its result
        # carries to the model; the result references them.
        refs, self._pending_refs = self._pending_refs, []
        self._log("on_tool_result", name=name, args=args, result=result, refs=refs, author="tool", **kw)

    def on_cycle_end(self, **kw):
        self._log("on_cycle_end", **kw)

    # framework-decision events (the report's "why" layer) -----------------------
    def on_parse_path(self, **kw):
        self._log("on_parse_path", **kw)

    def on_stage_transition(self, **kw):
        self._log("on_stage_transition", **kw)

    def on_output_reduced(self, **kw):
        self._log("on_output_reduced", **kw)

    def on_oracle_check(self, **kw):
        self._log("on_oracle_check", **kw)

    def on_slow_candidate(self, **kw):
        # Emitted by loop._tick_worklist_cycle; recorded so the log shows the nudge the model
        # received.
        self._log("on_slow_candidate", **kw)

    def on_budget_threshold(self, **kw):
        self._log("on_budget_threshold", **kw)

    def on_model_retry(self, **kw):
        self._log("on_model_retry", **kw)

    def on_stall(self, **kw):
        self._log("on_stall", **kw)

    def on_intervention(self, **kw):
        self._log("on_intervention", **kw)

    def on_env_change(self, **kw):
        self._log("on_env_change", **kw)

    def on_finish(self, result=None, **kw):
        self._log("on_finish", result=result)
        doc = _jsonable(result)
        if isinstance(doc, dict):
            census = run_census(doc.get("working_set"), _terminal_of(self.out_dir))
            if census is not None:
                doc = dict(doc)
                doc["census"] = census
        with open(os.path.join(self.out_dir, "result.json"), "w") as f:
            json.dump(doc, f, indent=2)

    def on_run_end(self, **kw):
        self._log("on_run_end", **kw)
        try:
            self._jsonl.close()
        except Exception:
            pass
