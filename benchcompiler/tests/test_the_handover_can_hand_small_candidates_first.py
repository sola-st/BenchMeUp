"""With a size order, handover sorts candidates by their repository's size (smallest first; ties in a
seeded random order) and deals one per worker in turn, so every worker starts on the cheapest
builds.
"""

from benchcompiler.search import partition_stock, read_size_order

KEY = ["clone_url", "head_sha"]
SIZES = {"apache/camel": 174512, "apache/jmeter": 8929, "dropwizard/dropwizard": 3040, "apache/commons-email": 768}


def _stock():
    out = []
    for repo, n in (
        ("apache/camel", 6),
        ("apache/jmeter", 6),
        ("dropwizard/dropwizard", 3),
        ("apache/commons-email", 1),
        ("o/unknown", 2),
    ):
        out += [{"clone_url": f"https://github.com/{repo}.git", "head_sha": f"{repo[-3:]}{i:037d}"} for i in range(n)]
    return out


def _repo(e):
    return e["clone_url"].split("github.com/")[1].removesuffix(".git")


def test_every_worker_starts_on_the_smallest_repositories():
    entries = _stock()
    parts = partition_stock(entries, KEY, 3, order=SIZES, shuffle_seed=7)
    print([[_repo(e)[-8:] for e in p] for p in parts])
    assert sorted(e["head_sha"] for p in parts for e in p) == sorted(e["head_sha"] for e in entries)
    firsts = [_repo(p[0]) for p in parts]
    assert set(firsts) <= {"apache/commons-email", "dropwizard/dropwizard"}
    for p in parts:  # sizes never go down within a worker's queue
        sizes = [SIZES.get(_repo(e), float("inf")) for e in p]
        assert sizes == sorted(sizes)
    assert all(_repo(p[-1]) in ("apache/camel", "o/unknown") for p in parts)  # unknown size goes last


def test_a_candidate_entry_ranks_before_its_repository(tmp_path):
    # The rank is per candidate (repo@commit), so a rule-named commit of a big repository can come
    # before an unnamed one of a small repository.
    entries = _stock()
    camel0 = entries[0]
    order = dict(SIZES)
    order[f"apache/camel@{camel0['head_sha']}"] = 1.0
    parts = partition_stock(entries, KEY, 3, order=order, shuffle_seed=7)
    print([p[0]["head_sha"][:10] for p in parts])
    assert any(p[0] is camel0 for p in parts)
    assert sum(1 for p in parts for e in p) == len(entries)


def test_the_size_file_is_read(tmp_path):
    f = tmp_path / "sizes.csv"
    f.write_text("repo,java_source_kb,repo_size_kb\napache/camel,174512,1\nDropwizard/Dropwizard,3040,2\n")
    got = read_size_order(str(f))
    print(got)
    assert got == {"apache/camel": 174512.0, "dropwizard/dropwizard": 3040.0}
