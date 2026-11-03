

################################################################################
# STAGE 1
################################################################################

STAGE 1 of 3: environment

Where you are
=============
You are building an instance of a Defects4J-style bug benchmark from a real Java project. This is the first stage of three; nothing exists yet. You will write a Dockerfile, a setup.sh script, and an instance.json file. The Dockerfile must install Java JDK 11 or 17, the project's build tool (Maven, Gradle, or Ant), and git. The setup.sh must clone the repository at the buggy commit with depth 1 into /work/repo, fetch the fix commit, and create two patch files: /bench/test_patch.diff (only test changes from the fix commit) and /bench/gold_patch.diff (the full fix diff). The harness will build the image and run setup.sh; a missing tool or failed clone causes failure.

Goal of this stage
==================
Your goal is to set up a reproducible environment that captures the buggy state and the fix's test changes. The test_patch.diff is essential because the harness will apply it to the buggy checkout to make the triggering test fail (the bug is observable). The gold_patch.diff is the full fix used later for validation. Without correct patches, the oracle cannot distinguish buggy from fixed behavior. This stage ensures the project builds and the required revisions are accessible, forming the foundation for all subsequent checks.

What to write into /bench
=========================
  Dockerfile     FROM a base image; install the toolchain and every dependency the build
                 and the checks need. No clone, no build, no source in the image.
  setup.sh       runs once inside that container from /bench, with the network: clone the
                 repository at the exact commit into /work/repo and leave /work ready. No build.
  instance.json  a JSON object with buggy_commit, fixed_commit, triggering_tests, build_file, repo_url, project_id.

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - container builds, setup runs [replays]
  - commit and files exist [identity]
The identity commands, run with your instance.json values:
  git -C /work/repo rev-parse --verify {buggy_commit}^{{commit}}   -> require: exit zero
  git -C /work/repo rev-parse --verify {fixed_commit}^{{commit}}   -> require: exit zero
  test -f /work/repo/{build_file}   -> require: exit zero

Traps
=====
  - Cloning with full history instead of depth 1: use `git clone --depth 1 --branch <buggy_commit>` to avoid huge downloads and ensure the buggy revision is the only commit in the clone.
  - Forgetting to fetch the fix commit after cloning: you must fetch the fix commit (e.g., `git fetch origin <fixed_commit_hash>`) to create diffs; the clone only has the buggy commit.
  - Creating test_patch.diff that includes non-test changes: use `git diff <buggy_commit> <fixed_commit> -- <test_files>` to isolate only the test modifications.
  - Assuming the project uses Maven when it might use Gradle or Ant: check the repository for build files (pom.xml, build.gradle, build.xml) and install the correct tool in the Dockerfile.
  - Hardcoding Java version 11 when the project requires 17: install both or detect from the project's configuration; the benchmark allows either 11 or 17.
  - Writing setup.sh that fails silently: ensure every command checks for errors (e.g., `set -e`) so the harness detects failures immediately.

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

STAGE 2 of 3: checkout

Where you are
=============
You are building an instance of this benchmark: a real Java bug-fix commit pair, packaged as scripts that rebuild from nothing. Earlier stages have built the container and run setup; they established that the commit and its files exist. Nothing else is established. Now you must write checkout.sh, which will be run twice: once for the buggy revision (with the fix's test patch applied) and once for the fixed revision (without the patch). The script must check out the given revision into a target directory, apply the test patch if provided, and build the project using its build system (Maven, Gradle, or Ant).

Goal of this stage
==================
Your goal is to produce a checkout.sh that, for each variant, checks out the correct revision, applies the test patch (which adds or modifies the triggering test from the fix commit) to the buggy variant, and builds the project successfully. This is critical because the benchmark requires that the buggy revision has the failing test present and builds, and the fixed revision builds without regression. A failed build or missing test file means the buggy variant does not have the failing test, or the fixed variant does not build, which violates the benchmark's correctness criteria.

What to write into /bench
=========================
  checkout.sh <dir> <revision> <patch_file>
      Check out the given revision into <dir>/repo, apply the patch from <patch_file> if provided (the patch should add/modify test files from the fix commit), and build the project using its build system (Maven/Gradle/Ant); print nothing on stdout.
The harness materializes each version into its own randomly named directory <dir>, in this order:
  buggy: checkout.sh <dir> {buggy_commit} /bench/test_patch.diff
  fixed: checkout.sh <dir> {fixed_commit}

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - both versions build [variants_build]

Traps
=====
  - Applying the patch to the wrong directory: the patch file is relative to the repository root; apply it inside <dir>/repo, not outside or to a submodule.
  - Assuming a single build system: the project may use Maven, Gradle, or Ant; your script must detect and invoke the correct build command (e.g., mvn compile, gradle build, ant) without hardcoding.
  - Printing output to stdout: checkout.sh must print nothing on stdout; any build output or error messages must go to stderr or be suppressed.
  - Forgetting to handle an empty patch file: if no patch is provided (for the fixed variant), the script must skip the patch step entirely, not fail.
  - Not verifying that the test file exists after patching: if the patch fails silently, the build may succeed but the test will be missing, which means the buggy variant will not have the failing test present, invalidating the instance.
  - Using a hardcoded branch or tag name: the revision is given as a commit hash; checkout must use that exact hash, not a branch name that may move.

Done when
=========
bench_check reports every check above as ok. Then call bench_deliver(sandbox=<name>): the stage closes and you get the next stage's instruction. Do not write the next stage's files before that.

Example of a passing report for this stage
==========================================
   3 both versions build [variants_build]               ok


################################################################################
# STAGE 3
################################################################################

STAGE 3 of 3: testing

Where you are
=============
You are building a benchmark instance from a real Java project with a bug-fix commit. Earlier stages established that the container builds, the setup script runs, both the buggy and fixed revisions exist as checked-out repositories, and both revisions build successfully. The project is a real, public Java project with a genuine test suite, and the fix commit adds or modifies a test that fails on the parent commit and passes on the fix commit. You now have the two revision directories ready for testing.

Goal of this stage
==================
In this stage, you must write three scripts: run_tests.sh, apply_patch.sh, and break.sh. run_tests.sh must run the full test suite and output raw JUnit XML to stdout. apply_patch.sh must apply a given patch file to the buggy source tree using git apply, then rebuild the project. break.sh must revert the fix commit's changes so the triggering test fails again. These scripts are essential because the harness uses them to verify that the bug is observable, the fix works, and the fix is minimal.

What to write into /bench
=========================
  run_tests.sh <dir>
      Run the project's full test suite using the build system and print the raw JUnit XML report to stdout.
  apply_patch.sh <dir> <patch_file>
      Apply the patch from <patch_file> to the source tree in <dir>/repo using 'git apply' and rebuild the project; print nothing on stdout.
  break.sh <dir>
      Revert the fix commit's changes in the source tree (by reverting the patch or deleting the transplanted test file) so that the triggering test fails; print nothing on stdout.
run_tests.sh prints the test framework's own JUnit XML report on stdout, unmodified; the harness reads test {triggering_tests} from it.
break.sh <dir> corrupts the fixed version's computation so that the result changes; a break the result cannot reveal is refused.

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - same output every run [stable]
  - output depends on the input [input_sensitive]
  - fails on buggy, passes on fixed [oracle]
  - the break is detected [can_fail]
  - a submission changes the result [solver_input]
The check must FAIL on buggy and PASS on fixed.

Traps
=====
  - Do not filter or summarize the test output; run_tests.sh must print the raw JUnit XML report to stdout exactly as the build system produces it, because the harness parses that XML to count failures.
  - Do not hardcode the triggering test name or the patch file path; apply_patch.sh must accept <patch_file> as an argument and apply it to the repo subdirectory, not to the current working directory.
  - Do not use mvn test or ant test alone if the project has a custom test suite configuration; run_tests.sh must invoke the exact command that runs the full test suite and produces JUnit XML output, typically mvn surefire:test or ant junitreport.
  - Do not forget to rebuild after applying the patch in apply_patch.sh; the patch changes source files, so you must run the build command again before the tests can reflect the patch.
  - Do not delete the entire test file in break.sh; instead revert only the fix commit's changes (e.g., git revert or git checkout of the parent for the changed files) so that the triggering test still exists but fails.
  - Do not assume the build system is Maven; check the project's build files and use the correct build command (mvn, ant, or gradle) for both building and running tests.

Done when
=========
bench_check reports every check above as ok. Then call bench_deliver(sandbox=<name>): the candidate is delivered.

Example of a passing report for this stage
==========================================
   4 same output every run [stable]                     ok
   5 output depends on the input [input_sensitive]      ok
   6 fails on buggy, passes on fixed [oracle]           ok
   7 the break is detected [can_fail]                   ok
   8 a submission changes the result [solver_input]     ok
