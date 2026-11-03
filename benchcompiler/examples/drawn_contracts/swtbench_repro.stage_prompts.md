

################################################################################
# STAGE 1
################################################################################

STAGE 1 of 3: environment

Where you are
=============
You are building an instance of a benchmark that turns a real GitHub issue and its pre-fix repository snapshot into a reproducible test. This is the first stage of three. Nothing exists yet. You will write a Dockerfile, setup.sh, instance.json, test_patch.patch, and fix_patch.patch. The Dockerfile must install Python and pytest. setup.sh must shallow-clone the repository at the base_commit from instance.json into /work/repo. The patch files must be plain text.

Goal of this stage
==================
In this stage you must create the minimal environment that the benchmark's correctness check relies on: a container with Python and pytest, a shallow clone of the repository at the exact pre-fix commit, and the two patch files (test_patch.patch and fix_patch.patch) as plain text. This matters because the benchmark's measurement depends on the test patch failing on the pre-fix snapshot and the fix patch passing on the post-fix snapshot. If the base_commit is wrong or the patches are missing or malformed, the entire measurement is invalid.

What to write into /bench
=========================
  Dockerfile     FROM a base image; install the toolchain and every dependency the build
                 and the checks need. No clone, no build, no source in the image.
  setup.sh       runs once inside that container from /bench, with the network: clone the
                 repository at the exact commit into /work/repo and leave /work ready. No build.
  instance.json  a JSON object with base_commit, test_patch_file, merge_commit, test_name, fix_patch_file, repo_url, solver_test_path.
  test_patch.patch
  fix_patch.patch

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - container builds, setup runs [replays]
  - commit and files exist [identity]
The identity commands, run with your instance.json values:
  git -C /work/repo rev-parse --verify {base_commit}^{{commit}}   -> require: exit zero
  git -C /work/repo rev-parse --verify {merge_commit}^{{commit}}   -> require: exit zero
  test -f /work/{test_patch_file}   -> require: exit zero
  test -f /work/{fix_patch_file}   -> require: exit zero
  test -f /bench/gold_test.py   -> require: exit zero
  test -f /bench/dummy_test.py   -> require: exit zero
  python -m py_compile /bench/gold_test.py   -> require: exit zero
  python -m py_compile /bench/dummy_test.py   -> require: exit zero

Traps
=====
  - Using a full clone instead of a shallow clone (depth 1) of the base_commit; this wastes time and disk space and may fail if the repository is large. Use `git clone --depth 1 --branch <commit>` or `git fetch --depth 1`.
  - Writing instance.json with missing or incorrectly named fields (e.g., 'base_commit' instead of 'base_commit', or omitting 'repo', 'patch', 'test_patch', 'problem_statement', 'FAIL_TO_PASS'). The instance.json must contain exactly those fields.
  - Forgetting to install pytest in the Dockerfile; the test execution will later fail because pytest is required to run the test.
  - Placing the clone in a directory other than /work/repo; setup.sh must clone into that exact path because the subsequent stages rely on that path.
  - Writing the patch files with binary or non-UTF-8 content; they must be plain text so that git apply can use them later.

Done when
=========
bench_check reports every check above as ok. Then call bench_deliver(sandbox=<name>): the stage closes and you get the next stage's instruction. Do not write the next stage's files before that.

Example of a passing report for this stage
==========================================
   1 container builds, setup runs [replays]             ok
   2 commit and files exist [identity]                  ok


################################################################################
# STAGE 2
################################################################################

STAGE 2 of 3: setup_scripts

Where you are
=============
You are building an instance of this benchmark as a self-contained directory of scripts that, when run in a fresh container, reconstructs a bug-reproducing test from a real GitHub issue. Earlier stages established that the container builds and the setup script runs successfully, leaving the repository at the pre-fix commit with all dependencies installed and the test suite runnable. The commit and files exist in the expected locations. Nothing else has been created yet.

Goal of this stage
==================
In this stage, you must write checkout.sh and apply_patch.sh to robustly handle git operations without printing to stdout, and create gold_test.py (the exact test that reproduces the bug, extracted from the test_patch) and dummy_test.py (a trivial passing test). These scripts and files are essential because the harness will later use them to set up both the prefix and postfix variants, and the gold test is the oracle that the solver's output must match.

What to write into /bench
=========================
  checkout.sh <dir> <commit>
      Check out the given commit in the repository at <dir>/repo using git checkout. Prints nothing on stdout.
  apply_patch.sh <dir> <patch_file>
      Apply the patch file at <patch_file> to the repository at <dir>/repo using git apply. Prints nothing on stdout.
  gold_test.py
  dummy_test.py
The harness materializes each version into its own randomly named directory <dir>, in this order:
  prefix: checkout.sh <dir> {base_commit}; then apply_patch.sh <dir> {test_patch_file}
  postfix: checkout.sh <dir> {merge_commit}

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - both versions build [variants_build]

Traps
=====
  - Do not let checkout.sh or apply_patch.sh print anything to stdout; the harness expects silent operation, so redirect any git output to stderr or /dev/null.
  - Do not hardcode the repository directory path; the scripts receive the directory as the first argument, so use that parameter instead of a fixed path like /repo.
  - Do not assume the repository is already at the target commit; checkout.sh must handle the case where the desired commit is already checked out without erroring.
  - Do not write gold_test.py by copying the entire test_patch verbatim; extract only the test function or class that reproduces the bug, and ensure it is a valid, runnable Python file.
  - Do not forget that dummy_test.py must be a trivial test that always passes (e.g., def test_dummy(): assert True) so the harness can verify the test runner works.
  - Do not include any import or dependency that is not already installed in the container; the gold test must run in the pre-fix environment without additional packages.

Done when
=========
bench_check reports every check above as ok. Then call bench_deliver(sandbox=<name>): the stage closes and you get the next stage's instruction. Do not write the next stage's files before that.

Example of a passing report for this stage
==========================================
   3 both versions build [variants_build]               ok


################################################################################
# STAGE 3
################################################################################

STAGE 3 of 3: test_scripts

Where you are
=============
You are building an instance of a benchmark that turns a real GitHub issue and repository snapshot into a reproducible bug test. Earlier stages have set up the container, installed the project, and ensured both the pre-fix (prefix) and post-fix (postfix) snapshots build and run their test suites. The repository is checked out at the base commit, the fix patch is applied, and the test patch (the PR's own test) is in place. Now you need to write the scripts that the harness will use to evaluate a solver's submission: run_test.sh, break.sh, and apply_test.sh.

Goal of this stage
==================
In this stage, you must write three scripts that enable the harness to measure whether a submitted test reproduces the bug. run_test.sh runs pytest with --junitxml=- on the test path specified in instance.json, outputting JUnit XML to stdout. break.sh reverts the fix patch using git apply -R, so the bug is present. apply_test.sh copies a submission file to the test path. These scripts are essential for the oracle: the harness runs them on prefix (expects test failure) and postfix (expects test pass), and also checks that breaking the fix causes the test to fail.

What to write into /bench
=========================
  run_test.sh <dir>
      Run the test at the path specified in instance.json's solver_test_path (relative to <dir>/repo) using pytest with --junitxml=- to output JUnit XML to stdout. The script exits with the test runner's exit code.
  break.sh <dir>
      Revert the fix patch on the repository at <dir>/repo by applying the reverse of the fix patch file (from instance.json's fix_patch_file) using git apply -R. Prints nothing on stdout.
  apply_test.sh <dir> <submission>
      Copy the file at <submission> to <dir>/repo/<solver_test_path> (from instance.json), overwriting any existing file. Prints nothing on stdout.
run_test.sh prints the test framework's own JUnit XML report on stdout, unmodified; the harness reads test {test_name} from it.
break.sh <dir> corrupts the postfix version's computation so that the result changes; a break the result cannot reveal is refused.

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - same output every run [stable]
  - output depends on the input [input_sensitive]
  - fails on prefix, passes on postfix [oracle]
  - the break is detected [can_fail]
  - a submission changes the result [solver_input]
The check must FAIL on prefix and PASS on postfix.

Traps
=====
  - Using a hardcoded test path instead of reading solver_test_path from instance.json; always parse the JSON with jq or a shell variable.
  - Forgetting to handle the directory argument: run_test.sh <dir> must operate on <dir>/repo; change directory or prefix paths accordingly.
  - In break.sh, applying the wrong patch (e.g., the test patch) instead of the fix_patch_file; use git apply -R on the correct file from instance.json.
  - In apply_test.sh, printing any output to stdout (e.g., cp -v) will corrupt the harness's measurement; use cp with no flags or redirect stderr.
  - Not ensuring the target directory for the test file exists before copying; use mkdir -p if needed, but suppress any output.

Done when
=========
bench_check reports every check above as ok. Then call bench_deliver(sandbox=<name>): the candidate is delivered.

Example of a passing report for this stage
==========================================
   4 same output every run [stable]                     ok
   5 output depends on the input [input_sensitive]      ok
   6 fails on prefix, passes on postfix [oracle]        ok
   7 the break is detected [can_fail]                   ok
   8 a submission changes the result [solver_input]     ok
