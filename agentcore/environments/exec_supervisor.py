"""Supervised container exec: a watched command instead of a blind ceiling.

``exec_in_container`` enforces one wall-clock timeout and observes nothing in
between: a hung apt-get and a 2M-file input generator look identical until the
ceiling kills them both (and the killed command's processes would keep running
afterward if the timeout only abandoned them instead of stopping them). The supervisor watches the two signals that actually distinguish
work from a hang:

  output growth   -- the exec stream is consumed incrementally, and
  CPU-time growth -- summed jiffies over the processes the command spawned,
                     identified by diffing the container's process table against
                     a pre-launch baseline (so double-forked daemons that
                     reparent to PID 1 still count).

Only when BOTH are flat for ``stall_window`` seconds is the ``judge`` consulted
-- an injectable callable, non-blocking by construction on the stuck-advisor
contract: if it is absent, raises, or answers gibberish, the supervisor keeps
waiting, so it can only ever improve on the blind ceiling. A ``stuck`` verdict
kills the command's whole footprint (resolved to in-namespace pids, killed from
inside the container); ``normal`` re-arms the stall clock; ``needs_input`` is
recorded but not yet acted on -- the report data decides whether that arm is
worth building. The ceiling remains, as backstop, and now also kills what it
times out. After any completion the process table is diffed once more: survivors
are reported as stragglers, never killed -- agents start servers on purpose.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from typing import Any, Callable, Optional

DEFAULT_STALL_WINDOW = 600.0
DEFAULT_POLL_INTERVAL = 5.0
OUTPUT_TAIL_FOR_JUDGE = 4000

Judge = Callable[[dict], dict]


def unreachable_reason(exc: Exception) -> str:
    """The one useful sentence inside a docker-py exception, without the URL.

    docker-py spells an unreachable container as
    ``409 Client Error for http+docker://localhost/v1.55/containers/<64 hex>/exec:
    Conflict ("container <64 hex> is not running")`` -- 180 characters in which the
    only load-bearing words are the last five. Shown verbatim to a model, it would
    retry the same command against the same dead container.
    """
    text = str(exc)
    m = re.search(r'\("([^"]+)"\)\s*$', text)
    if m:
        text = m.group(1)
    elif ": " in text:
        text = text.rsplit(": ", 1)[1]
    # "container <64 hex> is not running" -> "the container is not running": scrubbing the
    # id alone left the model reading "container the container is not running".
    text = re.sub(r"\bcontainer\s+[0-9a-f]{12,}\b", "the container", text)
    text = re.sub(r"\b[0-9a-f]{12,}\b", "the container", text)
    return text.strip()[:200] or type(exc).__name__


# ── host-side process observation ────────────────────────────────────────────


def _top_rows(container) -> list[dict]:
    """The container's process table as [{pid, cmd}], host-namespace pids."""
    try:
        t = container.top()
    except Exception:
        return []
    titles = [str(x).upper() for x in (t.get("Titles") or [])]
    try:
        pid_i = titles.index("PID")
    except ValueError:
        return []
    cmd_i = titles.index("CMD") if "CMD" in titles else len(titles) - 1
    rows = []
    for row in t.get("Processes") or []:
        try:
            rows.append({"pid": int(row[pid_i]), "cmd": str(row[cmd_i])})
        except (ValueError, IndexError):
            continue
    return rows


def _jiffies(host_pid: int) -> Optional[int]:
    """utime+stime for a host pid; None once the process is gone."""
    try:
        with open(f"/proc/{host_pid}/stat", "rb") as f:
            data = f.read().decode("utf-8", errors="replace")
        # fields after the parenthesized comm (which may contain spaces)
        rest = data.rsplit(")", 1)[1].split()
        return int(rest[11]) + int(rest[12])  # utime, stime
    except Exception:
        return None


def _ns_pid(host_pid: int) -> Optional[int]:
    """The pid as the container sees it (last NSpid entry), for in-container kill."""
    try:
        with open(f"/proc/{host_pid}/status", "rb") as f:
            for line in f.read().decode("utf-8", errors="replace").splitlines():
                if line.startswith("NSpid:"):
                    parts = line.split()
                    return int(parts[-1])
    except Exception:
        pass
    return None


def _footprint(container, baseline_pids: set[int]) -> list[dict]:
    return [r for r in _top_rows(container) if r["pid"] not in baseline_pids]


def _kill_footprint(container, rows: list[dict]) -> list[int]:
    """SIGKILL the footprint from inside the container (execs run as the container
    user, typically root there, which the host-side python is not)."""
    ns = [(_ns_pid(r["pid"]), r["pid"]) for r in rows]
    targets = [str(n) for n, _ in ns if n and n != 1]
    killed = [h for n, h in ns if n and n != 1]
    if targets:
        try:
            container.exec_run(["/bin/sh", "-c", "kill -9 " + " ".join(targets)])
        except Exception:
            pass
    return killed


#: How much of a command's output the harness RETAINS. The stream is still consumed whole --
#: the byte count and a sha256 of every byte are computed as it arrives -- but only this much
#: is held in memory.
#:
#: WHY THIS EXISTS. Accumulating every chunk in a list and then joining and decoding it holds
#: the command's entire output twice in memory. A command writing gigabytes to stdout would
#: get the process OOM-killed mid-exec, and SIGKILL leaves no traceback and no record. The
#: supervisor already sums chunk lengths every poll for stall detection, so it knows exactly
#: how much it is holding and can bound it.
MAX_CAPTURED_BYTES = int(os.environ.get("AGENTCORE_MAX_CAPTURED_BYTES") or 256_000)


def _utf8_boundary(buf: bytes, i: int) -> int:
    """The largest index <= i at which `buf` can be cut without splitting a UTF-8 sequence.
    A window cut mid-character rendered two U+FFFD that the whole-buffer decode never did."""
    i = max(0, min(i, len(buf)))
    while i > 0 and i < len(buf) and (buf[i] & 0xC0) == 0x80:
        i -= 1
    return i


class _BoundedCapture:
    """Consume a byte stream whole; keep a bounded head+tail of it.

    Head 2/3, tail 1/3 -- the same cut `tools.parsing.clip` makes for text, for the same
    reason: a command's verdict is usually on its last line. What is dropped is announced
    with the TRUE total. The digest of the whole stream is kept on the object and in the
    supervisor's report for the record -- and deliberately NOT written into the output:
    every place the harness verifies an `output_sha256` re-runs the command through
    `sha256sum` itself (stdout only, in-container), so a number offered here would be one
    a model copies into a record field and a control then refuses. `limit <= 0` keeps
    everything, for the harness's own reads of files and reports.
    """

    def __init__(self, limit: int = MAX_CAPTURED_BYTES) -> None:
        self.limit = int(limit or 0)
        self.total = 0
        self._head = bytearray()
        self._tail = bytearray()
        self._sha = hashlib.sha256()
        self._head_cap = self.limit * 2 // 3
        self._tail_cap = self.limit - self._head_cap

    def add(self, blob: bytes) -> None:
        self.total += len(blob)
        self._sha.update(blob)
        if self.limit <= 0:
            self._head.extend(blob)
            return
        room = self._head_cap - len(self._head)
        if room > 0:
            self._head.extend(blob[:room])
            blob = blob[room:]
        if not blob:
            return
        cap = self._tail_cap
        if len(blob) >= cap:
            # The whole window is inside this one chunk: replace, do not append-then-trim.
            self._tail[:] = blob[-cap:]
            return
        self._tail.extend(blob)
        # Trim only once the tail has doubled: one memmove per window of input instead of
        # one per chunk, which for a multi-gigabyte stream was ~140k memmoves of ~85 kB.
        if len(self._tail) > 2 * cap:
            del self._tail[: len(self._tail) - cap]

    @property
    def digest(self) -> str:
        return self._sha.hexdigest()

    @property
    def kept(self) -> int:
        return len(self._head) + min(len(self._tail), self._tail_cap) if self.limit > 0 else len(self._head)

    def text(self) -> str:
        if self.limit <= 0 or self.total <= len(self._head) + len(self._tail):
            return bytes(self._head + self._tail).decode("utf-8", errors="replace")
        tail = self._tail[-self._tail_cap :] if len(self._tail) > self._tail_cap else self._tail
        head = self._head
        # Cut on character boundaries: the head keeps whole characters, the tail starts on one.
        hb = _utf8_boundary(bytes(head), len(head))
        j = 0
        while j < len(tail) and (tail[j] & 0xC0) == 0x80:
            j += 1
        head_s = bytes(head[:hb]).decode("utf-8", errors="replace")
        tail_s = bytes(tail[j:]).decode("utf-8", errors="replace")
        omitted = self.total - hb - (len(tail) - j)
        # One line, no hex, no words a contract's regex is likely to be looking for: the
        # note must not be able to satisfy `output matches ...` on the command's behalf.
        note = (
            f"\n[harness: {omitted} of {self.total} bytes of this command's output are "
            f"not shown; the harness keeps a bounded window. Digest fields are verified by "
            f"re-running your command, so printing large output gains nothing.]\n"
        )
        return head_s + note + tail_s


# ── the supervisor ───────────────────────────────────────────────────────────


def supervised_exec(
    container,
    cmd: str,
    *,
    timeout: float,
    workdir: Optional[str] = None,
    tty: bool = False,
    stall_window: float = DEFAULT_STALL_WINDOW,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    judge: Optional[Judge] = None,
    capture_limit: Optional[int] = None,
) -> tuple[int, str, dict]:
    """Run ``cmd`` in ``container`` under stall supervision.

    Returns ``(exit_code, output, report)``. ``report['interventions']`` records
    every judge consult and kill with timestamps; ``report['stragglers']`` lists
    processes the command left running. ``render_report`` turns it into the prose
    that belongs in the agent's tool result.
    """
    api = container.client.api
    report: dict[str, Any] = {"interventions": [], "stragglers": [], "stalled_total_s": 0.0}
    # `output` is filled at the end: bytes produced, bytes kept, sha256 of the whole stream.

    baseline_pids = {r["pid"] for r in _top_rows(container)}

    from agentcore.environments.docker_helpers import NO_TERMINAL_ENV

    create_kwargs: dict[str, Any] = {"stdout": True, "stderr": True, "tty": tty, "environment": list(NO_TERMINAL_ENV)}
    if workdir:
        create_kwargs["workdir"] = workdir
    try:
        exec_id = api.exec_create(container.id, ["/bin/sh", "-lc", cmd], **create_kwargs)["Id"]
    except Exception as e:  # noqa: BLE001 - a docker refusal is an observation, not a crash
        # THE SAME GUARD AS THE UNSUPERVISED PATH. The unsupervised exec
        # (docker_helpers.exec_in_container._run) wraps the whole exec and returns
        # "Error executing command: ..." for exactly this. Without an equivalent, a
        # paused/stopped/removed container would turn every exec into a raised APIError that
        # escapes DockerEnvironment.execute and reaches the model as a raw docker error. Same
        # condition, same recovery, so: same return shape.
        # THE SENTINEL THE VERDICT PATHS READ, not a shell exit code: the worklist's
        # re-derivation control and the variants gate classify an unreachable sandbox by
        # SANDBOX_UNREACHABLE / the HARNESS ERROR prefix and excuse it as the harness's
        # fault; 126 read as "the candidate's command could not run" and charged a strike.
        from agentcore.tools.containers import SANDBOX_UNREACHABLE, UNREACHABLE_PREFIX

        report["unreachable"] = unreachable_reason(e)
        return (
            SANDBOX_UNREACHABLE,
            (
                f"{UNREACHABLE_PREFIX}the container could not be reached to run this command "
                f"({report['unreachable']}). Nothing ran -- this is the sandbox being paused, "
                f"stopped or already removed, not your command failing. Recreate or restart it "
                f"and run the command again; retrying it as-is against this container will keep "
                f"failing."
            ),
            report,
        )

    capture = _BoundedCapture(MAX_CAPTURED_BYTES if capture_limit is None else capture_limit)
    done = threading.Event()
    stream_error: list[str] = []

    def _consume() -> None:
        try:
            for chunk in api.exec_start(exec_id, stream=True):
                if chunk:
                    capture.add(bytes(chunk))
        except Exception as e:
            stream_error.append(str(e))
        finally:
            done.set()

    t0 = time.monotonic()
    worker = threading.Thread(target=_consume, daemon=True)
    worker.start()

    def output_now() -> str:
        return capture.text()

    def finish_code() -> Optional[int]:
        # Up to five seconds: the stream can close a moment before the daemon records the
        # status. None when it never appears -- the caller says so instead of inventing one.
        for _ in range(50):
            try:
                code = api.exec_inspect(exec_id).get("ExitCode")
            except Exception:
                code = None
            if code is not None:
                return int(code)
            time.sleep(0.1)
        return None

    last_len = 0
    last_jiffies: Optional[int] = None
    last_progress = t0
    last_consult = 0.0
    killed_by_judge = False

    while not done.wait(timeout=poll_interval):
        now = time.monotonic()
        elapsed = now - t0

        rows = _footprint(container, baseline_pids)

        jsum_parts = [j for j in (_jiffies(r["pid"]) for r in rows) if j is not None]
        jsum = sum(jsum_parts) if jsum_parts else None

        out_len = capture.total
        progressed = (out_len != last_len) or (jsum != last_jiffies)
        last_len, last_jiffies = out_len, jsum
        if progressed:
            last_progress = now

        # Bounded grace. A progressing command is almost
        # always a WORKING command, so progress extends the ceiling -- but some commands progress
        # forever (a foreground server, tail -f, a loop with no exit), and a
        # synchronous exec that never returns wedges the worker for its whole
        # wallclock. So: a stalled command dies at the ceiling as always; a
        # progressing one earns up to 2x, then dies regardless, with the story
        # telling the agent to run servers/watchers detached.
        hard_cap = timeout * 2
        over_ceiling = elapsed >= timeout and not progressed
        over_cap = elapsed >= hard_cap
        if over_ceiling or over_cap:
            killed = _kill_footprint(container, rows)
            report["interventions"].append(
                {"kind": "ceiling_kill", "at_s": round(elapsed, 1), "pids": killed, "progressing": bool(progressed)}
            )
            done.wait(timeout=5)
            preview = cmd[:100] + "..." if len(cmd) > 100 else cmd
            out = output_now()
            note = (
                f"Command was still producing output/CPU but exceeded the "
                f"absolute cap ({int(hard_cap)}s = 2x the {int(timeout)}s "
                f"ceiling). If this is a server, watcher, or endless loop, run "
                f"it detached (nohup ... &) and poll it instead. Command"
                if over_cap and progressed
                else f"Command timed out after {int(timeout)} seconds"
            )
            return 124, (f"{note}: {preview}" + (f"\n--- partial output ---\n{out}" if out else "")), report
        if progressed:
            continue

        stalled_s = now - last_progress
        report["stalled_total_s"] = report.get("stalled_total_s", 0.0) + poll_interval

        if judge is not None and stalled_s >= stall_window and now - last_consult >= stall_window:
            last_consult = now
            out = output_now()
            info = {
                "cmd": cmd,
                "output_tail": out[-OUTPUT_TAIL_FOR_JUDGE:],
                "stalled_s": round(stalled_s, 1),
                "elapsed_s": round(elapsed, 1),
                # what the command is blocked IN is often the whole verdict: a
                # literal 'sleep 900' in the footprint is a self-inflicted finite
                # wait, an apt http method child is a network fetch, and so on
                "processes": [r["cmd"] for r in rows][:20],
            }
            try:
                verdict = judge(info) or {}
            except Exception as e:
                verdict = {"verdict": "normal", "reason": f"judge unavailable: {e}"}
            v = str(verdict.get("verdict", "normal")).lower()
            if v not in ("stuck", "normal", "needs_input"):
                v = "normal"
            report["interventions"].append(
                {
                    "kind": "judge",
                    "verdict": v,
                    "at_s": round(elapsed, 1),
                    "stalled_s": round(stalled_s, 1),
                    "reason": str(verdict.get("reason", ""))[:2000],
                }
            )
            if v == "stuck":
                # A kill destroys work, so a single verdict never triggers one: the
                # judge is re-consulted immediately and must say stuck TWICE. A single
                # verdict on a borderline scenario can flip; confirmation squares that away
                # while a wrong 'normal' merely waits one more window under the ceiling backstop.
                try:
                    confirm = judge(info) or {}
                except Exception as e:
                    confirm = {"verdict": "normal", "reason": f"judge unavailable: {e}"}
                cv = str(confirm.get("verdict", "normal")).lower()
                report["interventions"].append(
                    {
                        "kind": "judge",
                        "verdict": cv if cv in ("stuck", "normal", "needs_input") else "normal",
                        "at_s": round(time.monotonic() - t0, 1),
                        "stalled_s": round(stalled_s, 1),
                        "confirming": True,
                        "reason": str(confirm.get("reason", ""))[:2000],
                    }
                )
                if cv == "stuck":
                    killed = _kill_footprint(container, rows)
                    report["interventions"].append(
                        {"kind": "kill", "at_s": round(time.monotonic() - t0, 1), "pids": killed}
                    )
                    killed_by_judge = True
                    done.wait(timeout=5)
                    break
            # normal / needs_input / unconfirmed stuck: keep waiting; the stall
            # clock re-arms via last_consult

    # completion (natural, judged-stuck, or stream error)
    code = finish_code()
    report["output"] = {
        "bytes": capture.total,
        "kept": capture.kept,
        "sha256": capture.digest,
        "truncated": capture.total > capture.kept,
    }
    out = output_now()
    if stream_error and not out:
        return 1, f"Error executing command: {stream_error[0]}", report
    if code is None:
        # THE STATUS IS UNKNOWN, AND THE RECORD SAYS SO. The daemon never reported a status;
        # 137 (the number a SIGKILL leaves) is returned because every reader treats non-zero
        # as "did not succeed", and the story carries the truth, so a model does not read
        # "exit 137" as out-of-memory.
        report["exit_unknown"] = True
        report["interventions"].append(
            {"kind": "exit_unknown", "stream_error": stream_error[0] if stream_error else None}
        )
        code = 137

    for r in _footprint(container, baseline_pids):
        report["stragglers"].append({"pid": r["pid"], "cmd": r["cmd"]})

    if killed_by_judge:
        code = code if code not in (0, None) else 137
    if int(code) == 137 and not any(i.get("kind") in ("kill", "ceiling_kill") for i in report["interventions"]):
        # 137 IS SIGKILL AND NOTHING HERE SENT IT. A command carrying `pkill -f <pattern>`
        # (or killall) most likely killed the shell running it: `-f` matches whole command
        # lines, and the shell's own line contains the pattern (e.g. `pkill -9 -f
        # 'redis-server.*16383'` at the head of a command).
        if _SELF_KILL.search(cmd):
            report["interventions"].append({"kind": "self_kill_suspect", "match": _SELF_KILL.search(cmd).group(0)})
    return int(code), out, report


_SELF_KILL = re.compile(r"\b(pkill\s+(?:-\S+\s+)*-f\b|killall\b)")


# ── rendering for the agent's tool result ────────────────────────────────────


def render_report(report: dict) -> str:
    """The supervision story as prose the agent can learn from; empty when there is
    nothing to say, so the happy path stays noise-free."""
    lines: list[str] = []
    for i in report.get("interventions", []):
        kind = i.get("kind")
        if kind == "judge":
            lines.append(
                f"[supervisor] at {i.get('at_s')}s: no output and no CPU activity for "
                f"{i.get('stalled_s')}s; judged '{i.get('verdict')}'"
                + (f" ({i['reason']})" if i.get("reason") else "")
                + "."
            )
        elif kind == "kill":
            lines.append(
                f"[supervisor] command judged stuck and killed "
                f"(pids {i.get('pids')}). Partial output above is all there is."
            )
        elif kind == "ceiling_kill":
            lines.append(
                f"[supervisor] wall-clock ceiling reached at {i.get('at_s')}s; command killed (pids {i.get('pids')})."
            )
        elif kind == "self_kill_suspect":
            lines.append(
                f"[supervisor] exit 137 is SIGKILL, and the harness sent none -- the "
                f"sandbox was not out of memory as far as it knows. This command "
                f"contains `{i.get('match')}`: `pkill -f` matches whole command lines, "
                f"and the shell running this very command carries that pattern in its "
                f"own command line, so it killed itself before producing output. Match "
                f"the process name instead (`pkill -x <name>`), or pick a pattern the "
                f"command's own text does not match."
            )
        elif kind == "exit_unknown":
            why = f" ({i['stream_error']})" if i.get("stream_error") else ""
            lines.append(
                f"[supervisor] the command's exit status is UNKNOWN: docker's output "
                f"stream ended before the daemon reported one{why}. Nothing here "
                f"killed it, and it was not out of memory as far as the harness "
                f"knows -- it may still be running, or may have finished. Check its "
                f"effect (the files or packages it should have produced) before "
                f"treating it as a failure."
            )
    for s in report.get("stragglers", []):
        lines.append(
            f"[supervisor] the command finished but left a process running: "
            f"pid {s.get('pid')} ({s.get('cmd')}). If that is not a server you "
            f"meant to keep, clean it up."
        )
    return "\n".join(lines)


# ── production judge, on the stuck-advisor contract ──────────────────────────


def make_model_judge(model) -> Judge:
    """A judge backed by the run's model. Parse failure or transport failure yields
    'normal' upstream (supervised_exec already treats exceptions that way), so this
    can never be the reason work was lost."""

    def _judge(info: dict) -> dict:
        prompt = (
            "A command inside a sandboxed container has produced no output AND used "
            f"zero CPU for {info['stalled_s']}s (running {info['elapsed_s']}s total). "
            "Zero CPU for that whole span means the process is blocked in a wait, not "
            "computing -- a build, encode, or test run that was genuinely working "
            "would show CPU. Decide what it is blocked ON. Answer with FIRST LINE "
            "exactly one of: STUCK, NORMAL, NEEDS_INPUT. Then one sentence of reason. "
            "NEEDS_INPUT: the output tail shows a prompt waiting for interactive "
            "input. NORMAL: the wait is intended and will end on its own (an explicit "
            "sleep or wait in the command, a remote operation that plausibly still "
            "progresses server-side). STUCK: anything blocked with no credible "
            f"finisher -- locks, dead peers, prompts aside, work that should burn "
            f"CPU but is not.\n\nCOMMAND:\n{info['cmd'][:2000]}\n\n"
            "ITS PROCESSES RIGHT NOW (what it is actually blocked in -- an explicit "
            "sleep here is an intended, finite wait):\n"
            + ("\n".join(str(p)[:2000] for p in info.get("processes", [])) or "(none visible)")
            + f"\n\nOUTPUT TAIL:\n{info['output_tail'][-2000:] or '(no output at all)'}"
        )
        resp = model.query([{"role": "user", "content": prompt}])
        text = (getattr(resp, "content", "") or "").strip()
        first = text.splitlines()[0].strip() if text else ""
        # the model reliably STARTS with the token but often appends the
        # reason on the same line ("NEEDS_INPUT: the output shows..."); demanding a
        # bare token silently converted correct verdicts into the 'normal' fallback
        m = re.match(r"(STUCK|NORMAL|NEEDS_INPUT)\b", first.upper())
        verdict = {"STUCK": "stuck", "NORMAL": "normal", "NEEDS_INPUT": "needs_input"}.get(
            m.group(1) if m else "", "normal"
        )
        rest = first[m.end() :].lstrip(" :-") if m else ""
        reason = (rest or " ".join(text.splitlines()[1:2])).strip()
        return {"verdict": verdict, "reason": reason}

    return _judge
