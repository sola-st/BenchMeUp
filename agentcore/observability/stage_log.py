"""StageLog: the typed record stream for the pipeline stages that are not agent runs.

The formalizer, the designer and the verifier's repair loop make LLM decisions that would otherwise
leave only their output as a trace. This writes, beside a
stage's output, one append-only `<stage>.jsonl` under the same envelope as the trajectory
stream and stores texts once by hash under
`blobs/`. Three record kinds:

  llm_call   one model exchange: prompt and response as blobs, the model, the attempt, and
             whether the reply parsed
  decision   what the stage concluded from it: site, rule, output, consequence
  verdict    a gate's pass/refuse with its problems (preflight)

`call_llm_json` records every exchange into the CURRENT stage log when one is open, so the
stages themselves only open the log and write their decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from agentcore.observability.trajectory import PREVIEW_CHARS, _bounded, _git_commit

_CURRENT: list["StageLog"] = []


def current() -> "StageLog | None":
    """The stage log an LLM call should write to, or None outside any stage."""
    return _CURRENT[-1] if _CURRENT else None


class StageLog:
    def __init__(self, out_dir: str, stage: str, *, run: str = ""):
        self.out_dir = out_dir or "."
        self.stage = stage
        self.run = run or stage
        self._seq = 0
        self.calls = 0  # the `cycle` of the envelope: one per LLM call
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, f"{stage}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self._epoch = time.strftime("%Y%m%dT%H%M%S")
        self._log("run_start", kind="lifecycle", site=f"{stage}.start", commit=_git_commit())

    # ── context manager: while open, call_llm_json records into it ──
    def __enter__(self) -> "StageLog":
        _CURRENT.append(self)
        return self

    def __exit__(self, *exc) -> None:
        if _CURRENT and _CURRENT[-1] is self:
            _CURRENT.pop()
        self._log("run_end", kind="lifecycle", site=f"{self.stage}.end", error=repr(exc[1]) if exc and exc[1] else None)
        self.close()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    # ── records ──
    def _log(self, event: str, *, kind: str, site: str, refs=None, author: str = "harness", **payload: Any) -> str:
        self._seq += 1
        rid = f"{self.run}//{self.calls}/{self._seq}"
        rec = {
            "id": rid,
            "ts": time.time(),
            "run": self.run,
            "worker": "",
            "epoch": self._epoch,
            "cycle": self.calls,
            "stage": self.stage,
            "candidate": None,
            "kind": kind,
            "site": site,
            "refs": list(refs or []),
            "author": author,
            "event": event,
            **payload,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        return rid

    def blob(self, text: str) -> dict:
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

    def llm_call(
        self, site: str, *, model: str, prompt: str, response: str, attempt: int, parsed: bool, error: str | None = None
    ) -> str:
        self.calls += 1
        p, r = self.blob(prompt), self.blob(response)
        return self._log(
            "on_llm_call",
            kind="llm_call",
            site=site,
            model=model,
            attempt=attempt,
            parsed=parsed,
            error=error,
            prompt={**p},
            response={**r, "author": "model"},
        )

    def decision(self, site: str, *, rule: Any, output: Any, consequence: str, refs=None) -> str:
        return self._log(
            "on_decision", kind="decision", site=site, refs=refs, rule=rule, output=output, consequence=consequence
        )

    def verdict(self, site: str, *, ok: bool, problems: list, detail: Any = None) -> str:
        return self._log(
            "on_verdict", kind="verdict", site=site, ok=bool(ok), problems=[str(p) for p in problems], detail=detail
        )
