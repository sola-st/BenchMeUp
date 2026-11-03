"""The next epoch does not re-open what an earlier epoch already delivered or dropped.

`handover` takes an exclusion: identities read from a file, or collected from earlier epochs'
records. Delivered = a recipe directory with report.json; dropped = the harness's "X dropped:"
notice in the trajectory (the trajectory's own `candidate` field on a drop names the next one).
Matching is on the worker's normalized identity, so `.git`, case and a trailing slash do not let a
finished candidate back in.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKER = os.path.join(ROOT, "agents", "boca_worker")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _cand(repo, sha, program="prog"):
    return {"repo_url": repo, "commit_sha": sha, "program_name": program}


def _stock(tmp_path):
    rows = [
        _cand("https://github.com/x/alpha", "1" * 40),
        _cand("https://github.com/x/alpha", "2" * 40),
        _cand("https://github.com/x/beta", "3" * 40),
        _cand("https://github.com/x/gamma", "4" * 40),
        _cand("https://github.com/x/delta", "5" * 40),
    ]
    p = tmp_path / "candidates.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def test_excluded_identities_never_reach_a_worker(tmp_path):
    from benchcompiler import search

    stock = _stock(tmp_path)
    out = str(tmp_path / "fleet")
    # spelled differently from the stock on purpose: .git, upper case, trailing slash
    # the boca worker's identity has three fields: repository, commit, program
    exclude = [
        "https://github.com/X/alpha.git|" + "1" * 40 + "|prog",
        "https://github.com/x/beta/|" + "3" * 40 + "|prog",
    ]
    m = search.handover(stock, WORKER, out, workers=2, target=3, launch=False, exclude=exclude, log=print)
    print("[verbose]", {k: m[k] for k in ("candidates", "excluded")})
    assert m["excluded"] == 2 and m["candidates"] == 3
    handed = [
        c
        for i in range(m["workers"])
        for c in json.loads(open(os.path.join(out, f"worker_{i}.instances.jsonl")).readline())["candidates"]
    ]
    shas = {c["commit_sha"] for c in handed}
    assert "1" * 40 not in shas and "3" * 40 not in shas
    assert shas == {"2" * 40, "4" * 40, "5" * 40}


def test_finished_candidates_are_read_from_an_epochs_record(tmp_path):
    """delivered = recipes/<slug>/report.json + bench_last/instance.json; dropped = the harness
    notice 'X dropped: ...' in the trajectory (not the drop row's own candidate field)."""
    from benchcompiler import search

    epoch = tmp_path / "epoch_9"
    w = epoch / "worker_0"
    rec = w / "recipes" / "x-alpha-111" / "bench_last"
    rec.mkdir(parents=True)
    (rec.parent / "report.json").write_text("{}")
    (rec / "instance.json").write_text(
        json.dumps({"repo_url": "https://github.com/x/alpha", "commit_sha": "1" * 40, "program_name": "prog"})
    )
    rows = [
        # the drop row names the NEXT candidate; the harness notice names the dropped one
        {
            "cycle": 5,
            "event": "on_tool_result",
            "name": "drop_candidate",
            "candidate": "https://github.com/x/gamma|" + "4" * 40,
            "args": {"reason": "no fix"},
        },
        {
            "cycle": 5,
            "event": "on_message",
            "author": "harness",
            "site": "tool.result",
            "preview": "https://github.com/x/beta|" + "3" * 40 + "|prog dropped: no fix",
        },
        {"cycle": 6, "event": "on_cycle_start", "candidate": "https://github.com/x/gamma|" + "4" * 40},
    ]
    with open(w / "trajectory.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    fin = search.finished_candidates([str(epoch)], WORKER)
    print("[verbose]", fin)
    assert set(fin["delivered"]) == {"https://github.com/x/alpha|" + "1" * 40 + "|prog"}
    assert set(fin["dropped"]) == {"https://github.com/x/beta|" + "3" * 40 + "|prog"}
    assert not any("gamma" in d for d in fin["dropped"])  # still open, not dropped

    stock = _stock(tmp_path)
    m = search.handover(
        stock,
        WORKER,
        str(tmp_path / "fleet2"),
        workers=1,
        target=2,
        launch=False,
        exclude=fin["delivered"] + fin["dropped"],
        log=print,
    )
    assert m["excluded"] == 2 and m["candidates"] == 3
