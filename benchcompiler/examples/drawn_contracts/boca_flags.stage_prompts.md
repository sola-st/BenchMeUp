

################################################################################
# STAGE 1
################################################################################

STAGE 1 of 4: environment

Where you are
=============
You are building an instance of a compiler-autotuning benchmark as a set of scripts that rebuild everything from nothing. This is the first of four stages; nothing exists yet. The instance will consist of a C program at a specific commit, reference inputs, and reference compilations. The harness will later build a Docker image from your Dockerfile, run setup.sh to clone the repository and install dependencies, then execute later stages to compile, generate inputs, and measure runtimes. Your work now establishes the reproducible environment that every subsequent stage depends on.

Goal of this stage
==================
In this stage you must install the compiler (gcc or clang) and all libraries required to build the chosen C program, and clone the repository at the exact commit specified in instance.json. The setup.sh script must perform the clone with a depth-1 fetch of that commit only. The main source file must exist after setup. This matters because the benchmark's oracle compares outputs and measures runtimes across compilations; a missing library or wrong commit would break every later check and make the instance unusable.

What to write into /bench
=========================
  Dockerfile     FROM a base image; install the toolchain and every dependency the build
                 and the checks need. No clone, no build, no source in the image.
  setup.sh       runs once inside that container from /bench, with the network: clone the
                 repository at the exact commit into /work/repo and leave /work ready. No build.
  instance.json  a JSON object with commit_hash, main_source_file, repo_url.

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - container builds, setup runs [replays]
  - commit and files exist [identity]
The identity commands, run with your instance.json values:
  git -C /work/repo rev-parse --verify {commit_hash}^{{commit}}   -> require: exit zero
  test -f /work/repo/{main_source_file}   -> require: exit zero

Traps
=====
  - Installing the compiler via a package manager command that requires interactive confirmation or a non‑default repository – use non‑interactive flags (e.g., DEBIAN_FRONTEND=noninteractive) and pin a specific version to avoid environment drift.
  - Cloning the full repository history instead of a depth‑1 fetch of the target commit – this wastes time and disk; use `git clone --depth 1 --branch <commit>` or fetch the commit directly with `git fetch origin <commit> && git checkout <commit>`.
  - Forgetting to install build‑system tools (make, cmake, autoconf) or language‑specific libraries that the project’s build file expects – read the project’s build instructions and install every dependency listed, not just the compiler.
  - Assuming the main source file path from the proposal is correct without verifying it exists after the clone – the commit may have moved or the path may be relative; check with `test -f` and fail the script if missing.
  - Using a compiler that ignores exported CFLAGS (e.g., a makefile that hardcodes -O2) – ensure the build recipe honours CFLAGS by testing with a trivial override; if it does not, the project is inadmissible and you must reject it now.

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

STAGE 2 of 4: build

Where you are
=============
You are building an instance of a compiler-autotuning benchmark: a C program that, when compiled with different optimization flags, produces identical output but different runtimes. Earlier stages cloned the repository at the specified commit and verified that the source files exist. The setup script has run, ensuring the build environment is ready. Now you must write build.sh, which will be called by the harness to compile the program with any given flag string into a specified output directory. This script is the only way the harness produces binaries for measurement.

Goal of this stage
==================
Write build.sh that accepts two positional arguments: an output directory and a flag string (e.g., '-O0' or '-O3'). It must compile the program into that directory using the given flags, honouring the CFLAGS environment variable or overriding any hardcoded optimization flags in the build system. This is critical because the harness will use build.sh to build the default (-O0) and optimized (-O3) reference compilations, and later the solver's candidate configurations. A correct build.sh ensures that the only difference between binaries is the optimization flags, enabling fair runtime comparison.

What to write into /bench
=========================
  build.sh <dir> <flags>
      Compile the program into <dir> with CFLAGS=<flags>; print nothing that matters.
The harness materializes each version into its own randomly named directory <dir>, in this order:
  default: build.sh <dir> -O0
  optimized: build.sh <dir> -O3

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - both versions build [variants_build]

Traps
=====
  - Assuming the project's Makefile respects CFLAGS when it hardcodes -O2 or -O3; instead, override by exporting CFLAGS before make or by passing CFLAGS='$flags' on the make command line.
  - Ignoring the output directory argument and building in the current directory; build.sh must place the compiled binary (and any intermediate files) inside the given directory, not pollute the source tree.
  - Using a fixed flag string like '-O2' instead of the second argument; the harness passes different flags each time, so the script must use the exact flag string provided.
  - Forgetting to clean previous builds before recompiling; if the output directory already contains a binary, make may skip recompilation, leading to stale binaries with wrong flags.
  - Handling only Makefile-based builds but the project uses CMake or configure; build.sh must detect and invoke the correct build system (e.g., cmake -DCMAKE_C_FLAGS='$flags' or ./configure CFLAGS='$flags').
  - Building with additional flags beyond the given ones (e.g., adding -march=native); the flag string must be used exactly as provided, with no extra flags that could affect runtime or output.

Done when
=========
bench_check reports every check above as ok. Then call bench_deliver(sandbox=<name>): the stage closes and you get the next stage's instruction. Do not write the next stage's files before that.

Example of a passing report for this stage
==========================================
   3 both versions build [variants_build]               ok


################################################################################
# STAGE 3
################################################################################

STAGE 3 of 4: observable

Where you are
=============
You are building an instance of this compiler-autotuning benchmark as a set of scripts that rebuild everything from nothing. Earlier stages established that the container builds, the setup script runs, the commit and source files exist, and both the default (-O0) and optimized (-O3) versions build successfully. No scripts beyond build.sh and setup.sh exist yet. Now you must write the scripts that make the program's computation observable: run.sh, gen_inputs.sh, and break.sh.

Goal of this stage
==================
In this stage you must write run.sh so that it prints only the program's output (no extra text), gen_inputs.sh so that it generates several deterministic inputs of rising cost that exercise the program's own computation, and break.sh so that it corrupts the built binary or a constant to change the output. This matters because the benchmark's oracle compares output digests, and its measurement requires a runtime band where the default build is at least 1.05× slower than -O3 on the costliest input.

What to write into /bench
=========================
  run.sh <dir> <input>
      Run <dir>'s program on <input>; stdout is exactly the program's output.
  gen_inputs.sh <dir>
      Write several deterministic inputs of rising cost into <dir>; each input file records the default build's median runtime on it.
  break.sh <dir>
      Corrupt <dir>'s built program's computation so that run's output changes (e.g., replace binary with a no-op or corrupt a constant).
run.sh prints the program's result and nothing else: no timings, no logs, no progress lines (send those to stderr).
gen_inputs.sh <dir> writes SMALL inputs for now (a few sizes; the largest can take under a second); growing them is the last stage.
break.sh <dir> corrupts the default version's computation so that the result changes; a break the result cannot reveal is refused.

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - inputs are generated [inputs]
  - same output every run [stable]
  - output depends on the input [input_sensitive]
  - the versions give the same output [oracle]
  - the break is detected [can_fail]
The versions must agree: default and optimized give the same output on every input.

Traps
=====
  - Writing run.sh that prints a version banner, timing, or any extra text will cause output digest mismatches; print only what the program writes to stdout.
  - Generating inputs with random data without a fixed seed makes output non-reproducible across runs; use a deterministic generator or fixed content.
  - Making break.sh corrupt a file the program does not read (e.g., a comment in source) will not change the output; corrupt the binary itself or a constant the output depends on.
  - Assuming the binary is at a fixed path like ./a.out instead of using the directory argument <dir> will break when the harness runs different versions; always invoke the binary from <dir>.
  - Generating inputs that do not scale the program's own domain (e.g., just appending zeros) may not increase runtime meaningfully; scale parameters like corpus size, wordlist length, or image dimensions.
  - For programs with floating-point output, ensure the output is deterministic across runs (same rounding, no time-dependent values) so that the numeric tolerance comparison works.

Done when
=========
bench_check reports every check above as ok. Then call bench_deliver(sandbox=<name>): the stage closes and you get the next stage's instruction. Do not write the next stage's files before that.

Example of a passing report for this stage
==========================================
   4 inputs are generated [inputs]                      ok
   5 same output every run [stable]                     ok
   6 output depends on the input [input_sensitive]      ok
   7 the versions give the same output [oracle]         ok
   8 the break is detected [can_fail]                   ok


################################################################################
# STAGE 4
################################################################################

STAGE 4 of 4: workload

Where you are
=============
You are building an instance of a compiler autotuning benchmark: a C program at a specific commit, with scripts that rebuild from nothing. Earlier stages have established that the container builds, setup runs, the commit and files exist, both default (-O0) and optimized (-O3) builds compile and produce identical output on the small reference input, inputs are generated by gen_inputs.sh, output is deterministic and depends on input, and the break detection works. Now you must ensure the largest generated input makes the default build run between 1 and 10 seconds and at least 1.05x slower than the optimized build.

Goal of this stage
==================
In this stage, you must adjust the input generation script (gen_inputs.sh) to produce several reference inputs of increasing cost, culminating in one that pushes the default build's median runtime into the 1–10 second band and shows at least 5% slowdown relative to -O3. This ensures the workload is dominated by the program's own computation, making compiler flag differences observable. If the program cannot be pushed into the band, you must leave gen_inputs.sh as is (the harness will detect the exception from the measured runtimes).

What to write into /bench
=========================
  (no new file; this stage edits what earlier stages wrote)
The harness materializes each version into its own randomly named directory <dir>, in this order:
  default: build.sh <dir> -O0
  optimized: build.sh <dir> -O3
The solver submits <flags> to build.sh on the default version. The harness applies '-O0' and '-O3' and requires a different result; if a probe value names a file under /bench, write it.
Grow gen_inputs.sh's largest input until default's median over 5 runs is between 1000 and 10000 ms, and default / optimized is at least 1.05x.

How it is checked
=================
bench_check(sandbox=<name>) copies /bench out of your sandbox, builds a FRESH container from
your Dockerfile, runs setup.sh, and runs your scripts there. Nothing in your sandbox is used.
This stage's checks, in order (the report stops at the first failure):
  - a submission changes the result [solver_input]
  - runtime inside the band [in_band]
  - default is slower than optimized by the margin [separates]
The report shows the runtime of every input; a flat curve means the program is not doing the
work, and you need a different command, not a bigger file.

Traps
=====
  - Do not assume byte size predicts runtime; a small config file can be the longest-running input if it triggers heavy computation, so measure runtime directly rather than scaling by file size.
  - Do not stop after one input type; if the first scaling attempt yields a flat runtime curve (e.g., I/O-bound), try a different kind of workload (more computation per byte, less I/O) before concluding the program is flag-insensitive.
  - Do not generate an input that takes the default build beyond 10 seconds; the harness will re-run it many times, and an overly long input wastes compute and may become I/O-bound, hiding the flag effect.
  - Do not forget that the program must be deterministic on the generated inputs; if your input introduces randomness (e.g., unseeded random), fix the seed in gen_inputs.sh or run.sh to ensure reproducible output digests.
  - Do not modify run.sh or build.sh; only gen_inputs.sh is the script that controls input generation. The harness measures runtimes by running run.sh with the generated inputs, so ensure gen_inputs.sh produces inputs that run.sh can consume.

Done when
=========
bench_check reports every check above as ok. Then call bench_deliver(sandbox=<name>): the candidate is delivered.

Example of a passing report for this stage
==========================================
   9 a submission changes the result [solver_input]     ok
  10 runtime inside the band [in_band]                  ok
  11 default is slower than optimized by the margin [separates]ok
