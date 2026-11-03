"""The prompt that derives a family's INSTANCE CONTRACT from its NL spec (recipes plan, phase 6).

The contract is the config file the replay engine (agentcore.recipes) reads: which scripts an
instance ships, how each is called, and the numbers from the spec. The builder's per-candidate
stages are derived from it mechanically (environment, variants, observable, solver/measure),
so this is the one place a model designs anything on the construction side, and what it
designs is small and typed: script names, their argument shapes, and one line each on what
the script must do.

`build_contract_prompt(raw_spec)` renders the prompt; `parse_contract_answer(text)` reads the
model's JSON and validates it with the same checker the engine uses, so a wrong answer is
refused with the field named. `draw_contract(raw_spec, model)` does both with retries, feeding
the checker's refusal back.
"""

from __future__ import annotations

import json
import os
from typing import Any

from benchcompiler.llm import call_llm_json, strip_code_fence

HERE = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD = os.path.abspath(os.path.join(HERE, ".."))
if SCAFFOLD not in os.sys.path:
    os.sys.path.insert(0, SCAFFOLD)

SCHEMA = """
{
  "verbs": {                       // the scripts every instance ships, <name>.sh; 2 to 6 of them
    "<name>": {
      "args": ["dir", "..."],      // positional arguments, in order. `dir` and `input` are RESERVED:
                                   // the harness binds `dir` to a variant's directory and `input`
                                   // to a file the inputs verb wrote. Any other name is bound from a
                                   // variant's setup step or from the solver's probe value.
      "does": "one sentence: what the script must do and what it must print on stdout"
    }
  },
  "variants": {                    // the sides the oracle compares; names from the spec's artifact_variants
    "<name>": {"setup": [{"verb": "<name>", "args": {"<arg>": "<value>"}}]}
  },
  "observable": {                // ONE check, or a LIST of checks [{...}, {...}]: as many as the family needs
    "verb": "<name>",              // the verb whose STDOUT is the observable
    "compare": "exact" | "numeric:<tol>",
    "reader": null | "junit-xml" | "exit-code",  // junit-xml when stdout is a framework's report (with `select`); exit-code when the exit status is the verdict
    "select": null | "{<instance.json field naming the test>}"
  },                               // with several checks the harness runs all of them, in order, and reads them together: pass when every check passes
  "oracle": {"kind": "reference-equivalence"}                     // observable equal across variants
         | {"kind": "fail-to-pass", "fails_on": "<variant>", "passes_on": "<variant>"},
  "negative_control": null | {"verb": "<name>", "variant": "<name>", "expect": "observable_changes"},
  "solver_input": null | {"verb": "<name>", "arg": "<arg>", "variant": "<name>",
                          "probe_values": ["<v1>", "<v2>"]},
  "measure": null | {"inputs": "<verb>", "repetitions": N, "floor_ms": N, "ceiling_ms": N,
                     "min_ratio": R, "ratio_of": ["<slow variant>", "<fast variant>"]},
  "predicates": [{"run": "<shell command over {instance.json fields}>", "require": "exit zero"}],
  "instance_fields": ["repo_url", "commit_hash", "..."],          // what instance.json must carry
  "numbers_quoted": {"floor_ms": "<the spec sentence the number came from>", "...": "..."},
  "stages": [                      // the ORDER the builder agent writes the files in, 2 to 6 stages
    {"name": "<short action phrase>", // e.g. "install the JDK and clone"; the first stage always writes Dockerfile, setup.sh, instance.json
     "files": ["Dockerfile", "setup.sh", "instance.json"],
     "goal": "two or three sentences, in THIS benchmark's terms: what the builder must get right here, and the traps",
     "done_when": "what the checker will confirm before the builder may move on"},
    {"name": "...", "files": ["<verb>.sh", "..."], "goal": "...", "done_when": "..."}
  ]
}
"""

RULES = """
RULES
1. A script's stdout IS the evidence. The observable verb prints the program's result and
   nothing else: no timings, no logs, no progress. The harness times the process and compares
   its stdout itself. A verb that must print a framework's report prints the raw report.
2. One script, one job. Building is not running; generating inputs is not running. A verb
   that both builds and runs makes the build's output part of the observable (unstable).
3. `dir` is the first argument of every verb that touches a built variant. Variant directories
   have random names; a script never assumes where it is.
4. The solver's input reaches exactly ONE verb, as an argument, and the two probe values must
   produce a visibly different result (different binaries, or a different observable).
5. Every number in `measure` and every threshold is copied from the spec text, and
   `numbers_quoted` cites the sentence. Never invent a threshold; if the spec has none, the
   block is null.
6. Every verb named anywhere (variants, observable, negative_control, solver_input, measure)
   is declared in `verbs`, and every declared verb is named somewhere. Reserved names `dir`
   and `input` never appear in a setup step's args.
7. The negative control corrupts what the observable SEES (the built program's computation,
   or the fix), not merely the presence of a file.
8. Prefer few verbs with obvious names (build, run, checkout, compile, run_tests, apply,
   gen_inputs, break). A consumer of the benchmark will call them by hand.
9. `predicates` are cheap shell checks over instance.json fields that prove identity (the
   commit exists in /work/repo; a source file the identity names exists). They run in the
   fresh clone right after setup.sh and BEFORE any verb, so they can only test what the
   clone holds at the commit: a field that names something a verb produces (the executable
   the build makes, a generated file) is proven by the stage that runs that verb, never by
   a predicate. A file predicate is written only over a field that IS a path in the
   repository at the commit; a field that names something else (a program, a test, a rule)
   yields no file predicate, and turning it into a path by guessing an extension
   (`{program_name}.c`) is not a predicate but a guess. A predicate never tests a /bench file
   (a patch a stage writes is proven by that stage, not by identity). Write git's peel syntax
   as ^{{commit}}.
10. Answer with ONE JSON object and nothing else.
11. The solver's input is what a SOLVER SUBMITS to be judged (a flag string, a patch, a test
    file, a rewritten comment) -- never an instance field like a commit hash. When the spec
    declares a solver input, the contract MUST carry a `solver_input` block with a verb that
    applies it, and probe values that are two real, different submissions the harness can
    apply (the gold answer and an empty or wrong one, given as paths under /bench or
    literal values). If the submission is applied by a verb already in a variant's setup,
    name that verb and its argument. When the submission is itself a file of program input
    (a test input, a seed, a trace), the argument is still NOT the reserved name `input`:
    call it `submission` and let the verb copy it where the program reads it.
12. instance.json carries identity and locations only. It must NOT contain the answer a
    solver is judged against in plain text (a gold comment, a gold patch body): keep such
    answers out of instance.json, as files a variant's setup writes, and say so in `does`.
13. Choose the oracle kind by what the check does. Two variants whose observable must be
    IDENTICAL (two builds of one program): `reference-equivalence`, reader null. A check
    that must FAIL on one variant and PASS on the other (a test, an analyzer, a comparison
    script): `fail-to-pass`, and the reader is `exit-code` when the verb is a check script
    that exits 0/nonzero, or `junit-xml` with `select` when it prints a JUnit report. Never
    declare a reader for a report the verb does not print.
14. Stages are the builder's work order, and they are YOURS to shape for this benchmark: as
    many as the work has natural steps (two to six), no more. Name each stage by what the
    builder does, as a short action phrase ("install the JDK and clone", "check out both
    commits and build", "write the test runner and the break", "apply a patch and rerun"),
    never by an abstract label. Every instance file appears in exactly one stage; the first
    stage sets up the environment (Dockerfile, setup.sh, instance.json); the scripts that
    build the versions come before the script that produces the result (so a failing build
    is found before anything runs); the break script goes with the result script; the
    solver's script last. A stage's `goal` is written for the builder agent, in this
    benchmark's own vocabulary: what a good script looks like here, what the checker will
    run, and the mistakes that cost checks. A stage that closes no check is refused.
15. A variant is a side the HARNESS builds by itself, before any solver exists: its setup
    arguments are literal values, instance.json fields in braces, or files under /bench. A
    placeholder such as <solver_input> in a setup step is refused. The solver's submission
    enters only through `solver_input`, applied on top of the named variant.
16. Every verb has a `does` sentence. A verb named by no block is refused; an argument the
    harness cannot bind (see the reserved names) is refused; `measure` is null unless the
    spec states runtime thresholds.
"""

EXAMPLE = """
WORKED EXAMPLE (a compiler-flags family: two builds of one program must agree and separate):
{
  "verbs": {
    "build":      {"args": ["dir", "flags"], "does": "compile the program into <dir> with CFLAGS=<flags>; exit 0 when it built (its output is not read)"},
    "run":        {"args": ["dir", "input"], "does": "run <dir>'s program on <input>; stdout is exactly the program's output"},
    "break":      {"args": ["dir"],          "does": "corrupt <dir>'s built program's computation so run's output changes"},
    "gen_inputs": {"args": ["dir"],          "does": "write several deterministic inputs of rising cost into <dir>"}
  },
  "variants": {"default": {"setup": [{"verb": "build", "args": {"flags": "-O0"}}]},
               "optimized": {"setup": [{"verb": "build", "args": {"flags": "-O3"}}]}},
  "observable": {"verb": "run", "compare": "exact", "reader": null, "select": null},
  "oracle": {"kind": "reference-equivalence"},
  "negative_control": {"verb": "break", "variant": "default", "expect": "observable_changes"},
  "solver_input": {"verb": "build", "arg": "flags", "variant": "default", "probe_values": ["-O0", "-O3"]},
  "measure": {"inputs": "gen_inputs", "repetitions": 5, "floor_ms": 1000, "ceiling_ms": 10000,
              "min_ratio": 1.05, "ratio_of": ["default", "optimized"]},
  "predicates": [{"run": "git -C /work/repo rev-parse --verify {commit_hash}^{{commit}}", "require": "exit zero"},
                 {"run": "test -f /work/repo/{main_source_file}", "require": "exit zero"}],
  "instance_fields": ["repo_url", "commit_hash", "main_source_file"],
  "numbers_quoted": {"floor_ms": "must bring the default build's median runtime to at least 1 second",
                     "ceiling_ms": "and to at most 10 seconds",
                     "min_ratio": "at least 1.05 times the -O3 build's median",
                     "repetitions": "median over at least 5 runs"},
  "stages": [
    {"name": "install the compiler and clone the program", "files": ["Dockerfile", "setup.sh", "instance.json"],
     "goal": "Install the compiler and every library the project's build needs in the Dockerfile; clone the repository at the exact commit in setup.sh (a depth-1 fetch of that commit is enough). A package that exists only in your scratch container is missing in the fresh container the scripts run in.",
     "done_when": "the image builds, setup.sh exits 0 and the identity predicates hold in the fresh clone"},
    {"name": "build the program with the given flags", "files": ["build.sh"],
     "goal": "build.sh must honour the flags it is given: it is run once with -O0 and once with -O3 into two directories, and later with a solver's flags. A Makefile that hardcodes -O2 must be overridden on the make command line.",
     "done_when": "both variants compile in the fresh container"},
    {"name": "run it on generated inputs and break it on purpose", "files": ["run.sh", "gen_inputs.sh", "break.sh"],
     "goal": "Pick the command where this program does its own computation on data it reads (compress, hash, parse, transform), not startup, not I/O, not a version banner. run.sh prints only the program's output; gen_inputs.sh writes a few deterministic inputs of rising size, small for now; break.sh corrupts the computation so run.sh's output changes.",
     "done_when": "the output is identical across runs, differs between inputs, matches between -O0 and -O3, and changes after break.sh"},
    {"name": "grow the workload into the band", "files": [],
     "goal": "Grow the largest input until the -O0 build takes 1 to 10 seconds on it and runs at least 1.05x slower than -O3. The report shows the runtime of every input; if the curve is flat the program is not doing the work and you need a different command, not a bigger file.",
     "done_when": "the costliest input is in the band and the ratio clears 1.05x"}
  ]
}

SECOND WORKED EXAMPLE (a bug-reproduction family: a test must fail before the fix and pass after it;
the solver submits a test). Note where the gold test is applied and how the submission replaces it:
{
  "verbs": {
    "checkout":    {"args": ["dir", "commit"],  "does": "clone the repository into <dir> at <commit>; exit 0"},
    "apply_patch": {"args": ["dir", "patch"],   "does": "apply the unified diff <patch> (a path under /bench) to <dir> with git apply; exit 0"},
    "install":     {"args": ["dir"],            "does": "install <dir>'s project and its test dependencies so its test suite runs; exit 0"},
    "run_tests":   {"args": ["dir"],            "does": "run the test named in /bench/instance.json in <dir> and print the framework's JUnit XML report on stdout, unmodified; exit 0 whether the test passes or fails"},
    "revert_fix":  {"args": ["dir"],            "does": "apply the reverse of /bench/fix.patch to <dir> so the fixed code is gone; exit 0"}
  },
  "variants": {
    "before": {"setup": [{"verb": "checkout", "args": {"commit": "{base_commit}"}},
                         {"verb": "apply_patch", "args": {"patch": "/bench/gold_test.patch"}},
                         {"verb": "install", "args": {}}]},
    "after":  {"setup": [{"verb": "checkout", "args": {"commit": "{fix_commit}"}},
                         {"verb": "install", "args": {}}]}   // the fix commit already carries the test: applying the patch again would fail
  },
  "observable": {"verb": "run_tests", "compare": "exact", "reader": "junit-xml", "select": "{test_name}"},
  "oracle": {"kind": "fail-to-pass", "fails_on": "before", "passes_on": "after"},
  "negative_control": {"verb": "revert_fix", "variant": "after", "expect": "observable_changes"},
  "solver_input": {"verb": "apply_patch", "arg": "patch", "variant": "before",
                   "probe_values": ["/bench/gold_test.patch", "/bench/empty.patch"]},
  "measure": null,
  "predicates": [{"run": "git -C /work/repo rev-parse --verify {base_commit}^{{commit}}", "require": "exit zero"},
                 {"run": "git -C /work/repo rev-parse --verify {fix_commit}^{{commit}}", "require": "exit zero"}],
  "instance_fields": ["repo_url", "base_commit", "fix_commit", "test_name"],
  "numbers_quoted": {},
  "stages": ["...as above: environment first; checkout, apply_patch and install before run_tests; revert_fix with run_tests..."]
}
The gold test lives in a file a setup step applies to BOTH sides, so the harness itself sees it
fail before and pass after; the solver's test replaces it through solver_input on the side named
there. The observable takes `dir` only and reads the test's name from instance.json; `select`
names that field in braces.
"""


def build_contract_prompt(
    raw: dict, feedback: str = "", previous: Any = None, identity_fields: list[str] | None = None
) -> str:
    raw_text = "\n".join(f"{k}: {v}" for k, v in raw.items() if k != "data" or True)
    identity = ""
    if identity_fields:
        # The candidate's identity is decided by the spec's search step, not here: the
        # worker registers candidates by these fields, so instance.json must carry them
        # under these exact names and the predicates must read them.
        identity = (
            "\n\nIDENTITY FIELDS, FIXED: the search step identifies a candidate by "
            + ", ".join(identity_fields)
            + ". `instance_fields` must include exactly these "
            "names (add others if the family needs them), and `predicates` read them."
            "\n\nTHOSE FIELDS ARE ALL THE BUILDER IS GIVEN. Every other name you put in "
            "`instance_fields` is one the builder must ESTABLISH from the candidate before "
            "it can write instance.json: by walking the repository's history for the "
            "commit the definition describes, by running a tool over the source to find "
            "the location, by reading the build's own report for the test names. That is "
            "the largest part of the work and it comes first. So when `instance_fields` "
            "is longer than the identity above, the FIRST stage's `goal` must say which "
            "of those fields the builder establishes and how it looks for them in this "
            "benchmark's terms, and its `done_when` must include that they were found and "
            "written into instance.json. Never write a stage that reads as though a "
            "commit, a file, a line or a test name the search did not hand over is "
            "already in hand."
        )
    retry = ""
    if feedback:
        # The whole previous answer is shown: if it is cut short the model cannot repair it and
        # rewrites from scratch each round. llm.PREVIOUS_ATTEMPT_CHARS admits a whole config.
        from benchcompiler.llm import PREVIOUS_ATTEMPT_CHARS

        prev = json.dumps(previous, indent=1) if previous is not None else "(unparseable)"
        if len(prev) > PREVIOUS_ATTEMPT_CHARS:
            prev = prev[:PREVIOUS_ATTEMPT_CHARS] + "\n... (truncated)"
        retry = (
            f"\n\nYOUR PREVIOUS ANSWER WAS REFUSED BY THE CHECKER:\n{feedback}\n"
            f"Previous answer:\n{prev}\n"
            f"REPAIR that answer: change only what the refusal names, keep everything else "
            f"exactly as it was, and answer again with the whole object."
        )
    return f"""THE BIG PICTURE

We are building a system that turns a written description of a software-engineering benchmark
into the benchmark itself, automatically. A researcher writes one page in plain English: what
the task is (for example "fix this bug so the failing test passes", or "find compiler flags
that make this program faster"), what a valid instance looks like, where instances may be
mined from, and how a solver's answer is judged. From that page the system must produce a set
of real instances that anyone can download, run and use to evaluate a tool.

HOW THE SYSTEM WORKS, IN FOUR STEPS

1. This step (yours). Read the researcher's page and decide the SHAPE that every instance of
   this benchmark will have: which small shell scripts an instance ships, what arguments each
   script takes, which two (or more) versions of the code are compared, how the verdict is
   read, what a solver submits, and which numeric thresholds apply. You write this once per
   benchmark, as one JSON object. Everything downstream is driven by it.

2. Builder agents. For each candidate project found in the wild, an LLM agent writes the
   actual scripts you declared (a Dockerfile, a setup script, and your verbs) so that the
   instance rebuilds itself from nothing. The agent works inside a scratch container and gets
   an automatic check report after each attempt.

3. The checker. A harness takes the agent's directory, builds a FRESH container from its
   Dockerfile, runs setup, runs the scripts exactly as you declared, and derives every fact
   itself: exit codes, outputs, wall-clock times. Nothing the agent typed is trusted.
   From those runs it decides a fixed list of checks: does it build, is the output stable,
   does it depend on the input, do the versions relate as the benchmark requires, does the
   deliberate break change the verdict, does the solver's submission reach the result, and,
   for timing benchmarks, is the workload in the required band and does it separate.

4. The pool. Instances that pass every check are shipped as directories of scripts with a
   small command-line tool whose commands ARE those scripts. A user runs them by hand; a
   solver is evaluated by submitting its answer through the one script you name for that.

WHAT YOU ARE DECIDING, CONCRETELY

An instance is a directory:
  Dockerfile      installs the toolchain and dependencies (no source, no build)
  setup.sh        clones the repository at the exact commit into /work/repo
  instance.json   identity fields (repository, commits, file paths, test names)
  <verb>.sh ...   the scripts YOU declare below, one per action, run from /work

Your job is to name those verbs and their arguments, say in one sentence what each must do,
and fill in the other blocks so the checker knows how to use them. Keep it small and obvious:
a user of the finished benchmark will type these script names by hand.

THE BENCHMARK THIS IS ABOUT

=== THE SPEC (the researcher's own words; the only source of truth) ===
{raw_text}

=== OUTPUT SCHEMA ===
{SCHEMA}
{RULES}
{EXAMPLE}{identity}{retry}
"""


def parse_contract_answer(text: str, identity_fields: list[str] | None = None) -> tuple[dict | None, str]:
    """(contract, '') when the answer validates; (raw-or-None, why) when it does not."""
    from agentcore import recipes

    try:
        raw = json.loads(strip_code_fence(text))
    except Exception as e:  # noqa: BLE001
        return None, f"not a JSON object: {e}"
    if not isinstance(raw, dict):
        return None, "the answer is not a JSON object"
    # The checker reads only the engine's blocks; the prompt's extra fields ride beside them.
    import copy

    core = {
        k: copy.deepcopy(raw.get(k))
        for k in (
            "verbs",
            "variants",
            "observable",
            "oracle",
            "negative_control",
            "solver_input",
            "measure",
            "predicates",
        )
    }
    from benchcompiler.stage_prompt import _asks_to_record, harness_claims

    # Wording worth a look is noted, never refused.
    notes: list[str] = []
    for name, v in (core.get("verbs") or {}).items():
        if isinstance(v, dict):
            does = str(v.get("does") or "").strip()
            if not does:
                return raw, f"verb {name!r} has no `does` sentence; say what the script must do and print"
            if _asks_to_record(does):
                notes.append(f"verb {name!r}: `does` asks the builder to record something ({does!r})")
            if harness_claims(does):
                notes.append(f"verb {name!r}: `does` describes what the harness does ({does!r})")
            v.pop("does", None)
    try:
        contract = recipes.validate_contract(core)
    except recipes.ContractError as e:
        return raw, str(e)
    contract["_does"] = {
        k: (raw["verbs"][k].get("does") if isinstance(raw["verbs"].get(k), dict) else "") for k in contract["verbs"]
    }
    try:
        contract["_stages"] = recipes.validate_stage_plan(contract, raw.get("stages"))
    except recipes.ContractError as e:
        return raw, str(e)
    for st in contract["_stages"]:
        for field in ("goal", "done_when"):
            for hit in harness_claims(str(st.get(field) or "")):
                notes.append(
                    f"stage {st['name']!r}: its `{field}` describes what the harness or the system does ({hit!r})"
                )
    contract["_draw_notes"] = notes
    contract["_instance_fields"] = list(raw.get("instance_fields") or [])
    missing = [f for f in (identity_fields or []) if f not in contract["_instance_fields"]]
    if missing:
        # The identity is the search step's, fixed before generation; a config that names the
        # same thing differently (main_source_file for source_file_path) cannot register what
        # the search found. Refused here so the retry corrects it, not after the draw.
        return raw, (
            f"instance_fields must include the identity fields {missing} under exactly these "
            f"names (the search step identifies a candidate by {list(identity_fields)}); "
            f"rename your field and use the same name in `predicates`"
        )
    contract["_numbers_quoted"] = dict(raw.get("numbers_quoted") or {})
    return contract, ""


def draw_contract(
    raw: dict, model: str, rounds: int = 6, identity_fields: list[str] | None = None, audit: bool = True
) -> tuple[dict | None, list[str]]:
    """Draw the contract from the spec, feeding the checker's refusal back up to `rounds`.

    Two checkers, in order: the schema checker (shape, verbs, fields) and, once the shape
    holds, the audit (benchcompiler.contract_audit): every spec sentence about what is built,
    applied, compared or corrupted must be realized by a config block. A config can pass the
    first and fail the second, e.g. by never applying the test the spec applies to both
    snapshots."""
    from benchcompiler.contract_audit import audit_contract, audit_table

    feedback, previous, log = "", None, []
    for i in range(rounds):
        result = call_llm_json(
            model,
            build_contract_prompt(raw, feedback, previous, identity_fields),
            temperature=0,
            site="contract",
            max_tokens=32000,
        )
        text = result if isinstance(result, str) else json.dumps(result)
        contract, why = parse_contract_answer(text, identity_fields)
        for n in ((contract or {}).get("_draw_notes") or []) if isinstance(contract, dict) else []:
            log.append(f"round {i + 1}: noted, not refused -- {n}")
        if contract is not None and not why and audit:
            ok, why, report = audit_contract(raw, contract, model)
            shape = {
                k: contract.get(k) for k in ("variants", "observable", "oracle", "negative_control", "solver_input")
            }
            log.append(f"round {i + 1}: audited config\n" + json.dumps(shape, indent=1))
            log.append(f"round {i + 1}: audit\n" + audit_table(report))
            if not ok:
                why = "the schema holds, but the config does not realize the spec. " + why
        if contract is not None and not why:
            log.append(f"round {i + 1}: accepted")
            return contract, log
        log.append(f"round {i + 1}: refused -- {why}")
        # The whole history of refusals is fed back, not only the last: otherwise rounds can
        # alternate between two mistakes, each round fixing one and reintroducing the other.
        earlier = [ln.split(" -- ", 1)[1] for ln in log[:-1] if ": refused -- " in ln]
        feedback = why + (
            "\n\nEARLIER ROUNDS WERE REFUSED FOR THESE, WHICH MUST STAY FIXED:\n- "
            + "\n- ".join(e[:600] for e in earlier)
            if earlier
            else ""
        )
        previous = contract
    return None, log
