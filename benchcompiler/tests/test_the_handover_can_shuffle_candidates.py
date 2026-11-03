"""With a shuffle, handover gives every candidate to exactly one worker, in a random order fixed by the
seed, instead of dealing by repository (which hands one worker long runs of the same repository).
"""

from benchcompiler.search import partition_stock

KEY = ["clone_url", "head_sha"]


def _stock():
    out = []
    for repo, n in (("jmeter", 29), ("camel", 15), ("pdfbox", 9), ("poi", 9), ("netty", 4)):
        out += [{"clone_url": f"https://github.com/apache/{repo}.git", "head_sha": f"{repo}{i:038d}"} for i in range(n)]
    out += [{"clone_url": f"https://github.com/o/r{i}.git", "head_sha": f"{i:040d}"} for i in range(30)]
    return out


def _longest_run(part):
    best = cur = 1
    for a, b in zip(part, part[1:]):
        cur = cur + 1 if a["clone_url"] == b["clone_url"] else 1
        best = max(best, cur)
    return best


def test_shuffled_each_candidate_goes_to_exactly_one_worker():
    entries = _stock()
    parts = partition_stock(entries, KEY, 3, shuffle_seed=7)
    ids = [e["head_sha"] for p in parts for e in p]
    print("sizes", [len(p) for p in parts], "longest same-repo run per worker", [_longest_run(p) for p in parts])
    assert sorted(ids) == sorted(e["head_sha"] for e in entries)  # every candidate, once
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    assert all(_longest_run(p) < 10 for p in parts)  # no 29-in-a-row queue
    assert parts == partition_stock(entries, KEY, 3, shuffle_seed=7)  # the seed fixes the order


def test_without_a_seed_the_deal_is_by_repository_as_before():
    parts = partition_stock(_stock(), KEY, 3)
    assert _longest_run(parts[0]) == 29
