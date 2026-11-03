"""The freeze takes the searching agent's final stock, not the union of every snapshot.

The live stock is the agent's last word; the verify-time copies of the stock are consulted only when
the live stock has nothing in it, so candidates the agent deleted do not come back.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchcompiler.search import _stock_files  # noqa: E402


def _write(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _rows(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            out += [json.loads(l) for l in fh if l.strip()]
    return out


def test_a_candidate_the_agent_removed_does_not_come_back(tmp_path):
    """The agent verified three, its check failed one, it removed that one and verified again."""
    run = str(tmp_path / "attempt_0")
    keep = [{"repo_url": "https://github.com/a/one"}, {"repo_url": "https://github.com/a/two"}]
    dropped = {"repo_url": "https://github.com/a/broken"}

    _write(os.path.join(run, "stock", "verify_001.candidates.jsonl"), keep + [dropped])
    _write(os.path.join(run, "stock", "verify_002.candidates.jsonl"), keep)
    _write(os.path.join(run, "stock", "candidates.jsonl"), keep)

    urls = [r["repo_url"] for r in _rows(_stock_files(run))]
    print("\n[verbose] freeze would read:", urls)
    assert dropped["repo_url"] not in urls, (
        "the freeze resurrected a candidate the agent deleted for failing its own check"
    )
    assert sorted(set(urls)) == sorted(r["repo_url"] for r in keep)


def test_a_wiped_stock_is_still_rescued_from_its_snapshots(tmp_path):
    """The live file was saved over with nothing; the newest verify copy stands in."""
    run = str(tmp_path / "attempt_0")
    found = [{"repo_url": "https://github.com/a/one"}, {"repo_url": "https://github.com/a/two"}]

    _write(os.path.join(run, "stock", "verify_001.candidates.jsonl"), found)
    _write(os.path.join(run, "stock", "candidates.jsonl"), [])

    urls = [r["repo_url"] for r in _rows(_stock_files(run))]
    print("[verbose] wiped stock recovered:", urls)
    assert sorted(urls) == sorted(r["repo_url"] for r in found), (
        "a wiped live stock must still be recovered from the snapshots kept at each verify"
    )


def test_the_newest_snapshot_wins_when_the_live_stock_is_gone(tmp_path):
    """Not the union of the snapshots either: the newest one is the agent's last word."""
    run = str(tmp_path / "attempt_0")
    _write(
        os.path.join(run, "stock", "verify_001.candidates.jsonl"),
        [{"repo_url": "https://github.com/a/one"}, {"repo_url": "https://github.com/a/gone"}],
    )
    _write(os.path.join(run, "stock", "verify_002.candidates.jsonl"), [{"repo_url": "https://github.com/a/one"}])

    urls = [r["repo_url"] for r in _rows(_stock_files(run))]
    print("[verbose] newest snapshot:", urls)
    assert "https://github.com/a/gone" not in urls
    assert urls == ["https://github.com/a/one"]


def test_a_plain_file_is_still_read_as_itself(tmp_path):
    p = str(tmp_path / "candidates.jsonl")
    _write(p, [{"repo_url": "https://github.com/a/one"}])
    assert _stock_files(p) == [p]
