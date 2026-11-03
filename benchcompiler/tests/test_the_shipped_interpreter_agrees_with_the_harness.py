"""The `bench` interpreter shipped in every packaged benchmark must reach the harness's verdicts.

A side counts as failed when any check fails (as in agentcore.recipes `_read_outcome`), and `submit`
builds the variant in a fresh directory, since a `git worktree add` checkout refuses an existing
directory. Docker is replaced by a fake that keeps a file system of variant directories and refuses
an existing directory exactly as `git worktree add` does.
"""

import importlib.machinery
import importlib.util
import json
import os
import types

from benchcompiler.artifact_recipes import INTERPRETER

IID = "inst-1"
VERBS = {
    "checkout": {"args": ["dir", "commit"]},
    "apply_patch": {"args": ["dir", "patch"]},
    "run_tests": {"args": ["dir"]},
    "check_all": {"args": ["dir"]},
}


def d4j_like(submit_placement="on_top"):
    """Two checks; buggy = checkout(buggy), fixed = checkout(fixed); the answer is a patch."""
    return {
        "materialize": {
            "buggy": [{"verb": "checkout", "args": {"commit": "{buggy}"}}],
            "fixed": [{"verb": "checkout", "args": {"commit": "{fixed}"}}],
        },
        "check": [
            {"verb": "run_tests", "reader": "exit-code", "select": None, "compare": "exact"},
            {"verb": "check_all", "reader": "exit-code", "select": None, "compare": "exact"},
        ],
        "oracle": {"kind": "fail-to-pass", "fails_on": "buggy", "passes_on": "fixed"},
        "submit": {
            "verb": "apply_patch",
            "arg": "patch",
            "variant": "buggy",
            "placement": submit_placement,
            "reference_answers": ["/bench/fix.patch", "/bench/empty.patch"],
        },
    }


def swt_like():
    """One check; prefix = checkout(base) + apply_patch(test patch); the answer REPLACES it."""
    return {
        "materialize": {
            "prefix": [
                {"verb": "checkout", "args": {"commit": "{base}"}},
                {"verb": "apply_patch", "args": {"patch": "/bench/test_patch.patch"}},
            ],
            "postfix": [{"verb": "checkout", "args": {"commit": "{merge}"}}],
        },
        "check": [{"verb": "run_tests", "reader": "exit-code", "select": None, "compare": "exact"}],
        "oracle": {"kind": "fail-to-pass", "fails_on": "prefix", "passes_on": "postfix"},
        "submit": {
            "verb": "apply_patch",
            "arg": "patch",
            "variant": "prefix",
            "placement": "replace",
            "reference_answers": ["/bench/test_patch.patch", "/bench/empty.patch"],
        },
    }


class FakeContainer:
    """Variant directories as a dict: dir -> {"commit": ..., "patches": [...]}.

    run_tests passes when `passes(state)` says so; check_all always passes (a guard
    that holds on both sides, as on the d4j instances)."""

    def __init__(self, passes):
        self.dirs, self.passes, self.calls = {}, passes, []

    def __call__(self, i, argv, cwd="/work", timeout=3600):
        self.calls.append(argv)
        r = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[:2] == ["mkdir", "-p"]:
            return r
        verb = os.path.basename(argv[1])[:-3]
        args = argv[2:]
        if verb == "checkout":
            if args[0] in self.dirs:
                r.returncode, r.stderr = 128, "fatal: '%s' already exists" % args[0]
            else:
                self.dirs[args[0]] = {"commit": args[1], "patches": []}
        elif verb == "apply_patch":
            self.dirs[args[0]]["patches"].append(args[1])
        elif verb == "run_tests":
            r.returncode = 0 if self.passes(self.dirs.get(args[0], {})) else 1
        elif verb == "check_all":
            r.returncode = 0
        return r


def load_bench(tmp_path, iface, fields, fake):
    root = tmp_path / "bench_dir"
    (root / "instances" / IID).mkdir(parents=True)
    (root / "contract.json").write_text(json.dumps({"interface": iface, "contract": {"verbs": VERBS}}))
    (root / "manifest.json").write_text(json.dumps({"instances": [{"id": IID}]}))
    (root / "instances" / IID / "instance.json").write_text(json.dumps(fields))
    (root / "bench").write_text(INTERPRETER)
    loader = importlib.machinery.SourceFileLoader("shipped_bench_%d" % id(fake), str(root / "bench"))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(mod)
    mod.dexec = fake
    return mod


def d4j_passes(state):
    return state.get("commit") == "f1" or "/bench/fix.patch" in state.get("patches", [])


def test_a_regression_guard_passing_on_the_buggy_side_does_not_undo_the_verdict(tmp_path, capsys):
    fake = FakeContainer(d4j_passes)
    bench = load_bench(tmp_path, d4j_like(), {"buggy": "b1", "fixed": "f1"}, fake)
    rc = bench.judge(IID)
    out = capsys.readouterr().out
    print(out)
    assert rc == 0, "the triggering tests fail on buggy and pass on fixed: the instance holds"
    assert "VERDICT: holds" in out


def test_judge_still_refuses_when_no_check_fails_on_the_failing_side(tmp_path, capsys):
    fake = FakeContainer(lambda s: True)  # the triggering tests pass everywhere
    bench = load_bench(tmp_path, d4j_like(), {"buggy": "b1", "fixed": "f1"}, fake)
    rc = bench.judge(IID)
    out = capsys.readouterr().out
    print(out)
    assert rc == 1 and "does NOT hold" in out


def test_submit_after_judge_builds_in_a_fresh_directory(tmp_path, capsys):
    fake = FakeContainer(d4j_passes)
    bench = load_bench(tmp_path, d4j_like(), {"buggy": "b1", "fixed": "f1"}, fake)
    bench.judge(IID)
    gold = bench.submit(IID, "/bench/fix.patch")
    wrong = bench.submit(IID, "/bench/empty.patch")
    again = bench.judge(IID)
    out = capsys.readouterr().out
    print(out)
    print("directories built:", sorted(fake.dirs))
    assert gold == 0, "the developer's fix must be accepted after a judge"
    assert wrong == 1, "an empty patch must be rejected"
    assert again == 0, "judging twice must work too"
    assert "already exists" not in out


def test_a_replacing_submission_after_judge_is_built_not_refused(tmp_path, capsys):
    fake = FakeContainer(lambda s: s.get("commit") == "m1" or "/bench/test_patch.patch" not in s.get("patches", []))
    bench = load_bench(tmp_path, swt_like(), {"base": "b1", "merge": "m1"}, fake)
    assert bench.judge(IID) == 0
    rc = bench.submit(IID, "/bench/empty.patch")
    out = capsys.readouterr().out
    print(out)
    print("directories built:", sorted(fake.dirs))
    assert "did not build" not in out, "the submission must be built, not refused by a stale directory"
    assert rc == 0
