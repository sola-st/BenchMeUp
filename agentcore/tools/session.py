"""Run the CONSUMER's own commands before the instance is delivered.

`consumer_session` is the command sequence the packaged artifact offers, and the same
sequence the merge executes. Construction verifies the agent's own scripts; this gate verifies
the verbs a consumer would run against the fields the agent recorded. A typical failure is
`exec <id> -- git -C {clone_path} apply {fix_patch}` exiting 128, or a command-not-found (127):
a field that worked in the builder's sandbox and fails when replayed.

What this gate runs, and what it deliberately does not:

  * RUNS the steps that template recorded fields into a shell command (`exec <id> -- ...`),
    because that is where a recorded value meets the shell for the first time.
  * RUNS `rebuild-<variant>` / `check-<variant>` too. The variants gate already rebuilds and
    checks each variant in ISOLATION, but only the SEQUENCE asserts what nothing else
    does: check fails, the reference fix is applied, the SAME check now passes. That last
    step is the whole product. The rehearsal cannot borrow the variants gate's
    `_verify_<name>` copy either -- the break probe runs there and leaves it broken -- so
    the rehearsal builds `_use_<name>`, the same name the packaged artifact builds, and
    the session's own `dir` expectations render against the tree they name.
  * SKIPS consumer built-ins the artifact CLI implements itself (`info`, `start`, `stop`,
    `measure`): they exist only in the packaged CLI, and inventing an equivalent here would
    be a second implementation of the thing the merge already checks.

The materialisation rule matches the merge's (benchcompiler/artifact.py): a value that
spans lines becomes a FILE inside the sandbox and the command sees its path. A patch
templated into argv is not a command, and that single defect cost six instances.
"""

from __future__ import annotations

import re

from agentcore.tools.variants import declared_variants, informative_tail, render_cmd

_PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Verbs the packaged CLI implements itself, or that another gate already executes.
_CLI_BUILTINS = ("info", "start", "stop", "measure", "verbs", "list")

_EXPECT_SHAPES = (
    "the string last_exit_zero / last_exit_nonzero, "
    "{results: <kind>, outcome: passed|failed[, dir: ...]}, or "
    "{observable: exit_code|<BENCH_OBS name>, "
    "equals|at_least|at_most|present: ...}"
)


def _results_problem(
    i: int, command: str, expect: dict, use: str, sandbox: str, run_in, code: int, inst: dict | None = None
) -> str:
    """Judge a {results: ...} claim by dumping this step's report. "" or a problem."""
    from agentcore import reports

    kind = str(expect.get("results") or "").strip().lower()
    if kind not in reports.supported_kinds():
        return (
            f"step {i} `{command}` expects results {kind!r}, which this harness has no "
            f"reader for (it supports {reports.supported_kinds()})"
        )
    patterns = reports.discovery_patterns(inst, kind)
    dump_cmd = reports.dump_command(kind, patterns)
    if not dump_cmd:
        return (
            f"HARNESS/SPEC FAULT: step {i} `{command}` expects results {kind!r} but "
            f"this instance declares no `report_discovery` patterns for it, so the "
            f"harness does not know where that ecosystem writes its reports. Declare "
            f"them in the spec; an undeclared convention is a missing rule, not a "
            f"permissive one."
        )
    _, dump = run_in(sandbox, f"cd {use} && {dump_cmd}", 120)
    # THE SEAM: the verdict rule lives in one place (`agentcore.verdict`) and is asked here,
    # not re-implemented, so an errored run that happens to satisfy the floor cannot read as a pass.
    #
    # The session states its expectation as `outcome: failed|passed`, which is the contract's
    # `expect_failures` in different words -- translated here, decided there.
    from agentcore.verdict import verdict_of

    want_failed = str(expect.get("outcome") or "").strip().lower() == "failed"
    ok, why = verdict_of({"results": kind, "expect_failures": 1 if want_failed else 0}, int(code), dump or "")
    if ok:
        return ""
    tail = (
        " This step is what says the bug is present in the tree a solver is handed."
        if want_failed
        else " This step is the claim the whole instance sells -- that the reference fix "
        "makes the triggering test pass."
    )
    return f"step {i} `{command}` under {use}: {why}.{tail}"


def _observable_problem(i: int, command: str, expect: dict, code: int, out: str) -> str:
    """Judge an {observable: ...} claim against this step's own run. "" or a problem.

    `exit_code` is the step's exit status; any other name must appear as a
    `BENCH_OBS <name>=<value>` line in the step's output -- the same convention the
    packaged CLI's measurement path prints, so a claim that holds here holds there.
    """
    name = str(expect.get("observable") or "").strip()
    if name == "exit_code":
        seen = [str(int(code))]
    else:
        seen = []
        for ln in str(out or "").splitlines():
            ln = ln.strip()
            if ln.startswith("BENCH_OBS "):
                k, _, v = ln[len("BENCH_OBS ") :].partition("=")
                if k.strip() == name:
                    seen.append(v.strip())
    if "present" in expect:
        want = bool(expect["present"])
        if bool(seen) == want:
            return ""
        return (
            f"step {i} `{command}`: {name} was "
            f"{'not emitted' if want else 'emitted'} (claim: present: "
            f"{expect['present']})"
        )
    if not seen:
        return (
            f"step {i} `{command}` emitted no BENCH_OBS {name}=... line, so its {name} "
            f"claim has no evidence to judge -- exit {code} says nothing about it"
        )
    last = seen[-1]
    if "equals" in expect:
        want = str(expect["equals"]).strip()
        if str(last).strip() == want:
            return ""
        return f"step {i} `{command}`: {name}={last!r}, but the session claims equals {want!r}"
    bounds = [(k, expect[k]) for k in ("at_least", "at_most") if k in expect]
    if bounds:
        try:
            got = float(last)
        except (TypeError, ValueError):
            return f"step {i} `{command}`: {name}={last!r} is not a number"
        for key, raw in bounds:
            try:
                want = float(raw)
            except (TypeError, ValueError):
                return f"step {i} `{command}`: {key} {raw!r} is not a number"
            if (got < want) if key == "at_least" else (got > want):
                return f"step {i} `{command}`: {name}={got:g} is outside the claim ({key} {want:g})"
        return ""
    return f"step {i} `{command}`: the {name} claim states no test (give it equals, at_least/at_most, or present)"


def _dict_expect_problem(
    i: int, command: str, expect: dict, use: str, sandbox: str, run_in, code: int, out: str, inst: dict | None = None
) -> str:
    """ONE dispatch for every dict expectation a session step may declare.

    Exec steps and check steps share it, so a dict expectation is never silently left
    unjudged (a vacuous gate); check steps alone would understand only
    `results:`, refusing the observable vocabulary with an unactionable "no reader". A
    shape neither judge understands FAILS the step naming the judged shapes: a claim
    nothing judges must never pass silently.
    """
    if "results" in expect:
        return _results_problem(i, command, expect, use, sandbox, run_in, code, inst)
    if "observable" in expect:
        return _observable_problem(i, command, expect, code, out)
    return (
        f"step {i} `{command}` declares an expectation this harness does not judge: "
        f"{expect!r}. Judged shapes: {_EXPECT_SHAPES}. A claim nothing judges would "
        f"pass silently, so it fails here instead."
    )


def _variant_step(i: int, command: str, verb: str, step: dict, fields: dict, inst: dict, sandbox: str, run_in) -> str:
    """Run one `rebuild-<variant>` / `check-<variant>` session step. Returns "" or a problem.

    Deliberately built the same way the packaged CLI builds it, because a rehearsal that
    runs something else certifies something else: the rebuild is the shipped
    `setup_<name>.sh` into `{variants_root}/_use_<name>`, and the check is the shipped
    `check_<name>.sh` when present, else the contract's `check_cmd` -- the agent owns the
    HOW, the spec owns the EXPECTATION.
    """
    kind_of, name = verb.split("-", 1)
    root = str(fields.get("variants_root") or "").strip()
    if not root:
        return (
            f"step {i} `{command}` names a variant, but this candidate has not recorded "
            f"variants_root, so there is no directory to run it in"
        )
    v = next((x for x in declared_variants(inst) if str(x.get("name")) == name), None)
    if v is None:
        return (
            f"step {i} `{command}` names variant {name!r}, which this task's contract "
            f"does not declare (declared: "
            f"{[str(x.get('name')) for x in declared_variants(inst)] or 'none'})"
        )
    use = f"{root}/_use_{name}"
    timeout = int(v.get("timeout") or 900)

    if kind_of == "rebuild":
        code, out = run_in(sandbox, f"rm -rf {use} && mkdir -p {use} && bash {root}/setup_{name}.sh {use}", timeout)
        if code != 0:
            return (
                f"step {i} of the CONSUMER session failed: `{command}` -> exit {code}. "
                f"{informative_tail(out)}. This is the first thing a user of this "
                f"benchmark does, and the packaging gate does it too."
            )
        # The setup ran into an EMPTY directory this step just made, so "it produced
        # nothing" is decidable without knowing anything about the task.
        code, _ = run_in(sandbox, f'test -n "$(ls -A {use} 2>/dev/null)"', 60)
        if code != 0:
            return f"step {i} `{command}` exited 0 but left {use} empty, so the tree a solver is handed does not exist"
        return ""

    # THE MARKER GOES DOWN BEFORE THE CHECK RUNS -- report discovery reads only files newer
    # than it, so a report left by an EARLIER step of this same session cannot witness this
    # one. Without it the second check-buggy reads the first one's failures and the session
    # can never show a fix working.
    run_in(sandbox, "touch /tmp/.bench_check_started", 30)
    code, out = run_in(sandbox, f"test -f {root}/check_{name}.sh && echo HAVE_SCRIPT", 60)
    if "HAVE_SCRIPT" in str(out or ""):
        code, out = run_in(sandbox, f"cd {use} && bash {root}/check_{name}.sh {use}", timeout)
    else:
        cmd, err = render_cmd(str(v.get("check_cmd") or "true"), fields)
        if cmd is None:
            return f"step {i} `{command}`: {err}"
        code, out = run_in(sandbox, f"cd {use} && {cmd}", timeout)

    expect = step.get("expect")
    if not isinstance(expect, dict):
        wants_zero = str(expect or "last_exit_zero") == "last_exit_zero"
        if wants_zero and code != 0:
            return f"step {i} of the CONSUMER session failed: `{command}` -> exit {code}. {informative_tail(out)}"
        if not wants_zero and code == 0:
            return f"step {i} `{command}` was expected to fail and exited 0"
        return ""

    return _dict_expect_problem(i, command, expect, use, sandbox, run_in, code, out, inst)


def consumer_session_runs(fields: dict, ctx, run_in) -> tuple[bool, str]:
    """Execute the declared consumer session's field-templating steps in this candidate's
    own sandbox. Returns (ok, detail) like every other gate here.

    `run_in(sandbox, cmd, timeout) -> (code, output)` is supplied by the collector's
    validators module, so execution goes through the same plumbing as every other check.
    """
    inst = (getattr(ctx, "instance", {}) or {}) if ctx is not None else {}
    session = [s for s in (inst.get("consumer_session") or []) if isinstance(s, dict)]
    if not session:
        return True, "no consumer session is declared, so there is nothing to rehearse"

    sandbox = str(fields.get("sandbox") or "").strip()
    ran, problems = [], []
    for i, step in enumerate(session, start=1):
        command = str(step.get("command") or "").strip()
        if not command:
            continue
        verb = command.split()[0]
        if verb in _CLI_BUILTINS:
            continue
        if verb.startswith(("rebuild-", "check-")):
            problem = _variant_step(i, command, verb, step, fields, inst, sandbox, run_in)
            ran.append(command)
            if problem:
                problems.append(problem)
            continue
        if verb == "exec":
            _, _, shell = command.partition("--")
            shell = shell.strip()
        else:
            shell = command
        if not shell:
            continue
        # THE SEAM: how a recorded field becomes a command is one rule (`render_command`).
        # A field the session names and the candidate never recorded is still named here --
        # that refusal is this gate's whole point -- but the rule that finds it is the one rule.
        from agentcore.render import render_command
        from agentcore.tools.variants import _field_writer

        with_ids = dict(fields)
        with_ids.setdefault("id", str(inst.get("instance_id") or "this-instance"))
        with_ids.setdefault("instance_id", with_ids["id"])
        cmd, err = render_command(shell, with_ids, _field_writer(sandbox, run_in))
        if cmd is None:
            problems.append(
                f"step {i} `{command}` {err}. A consumer runs this command verbatim, so "
                f"every field it templates must be recorded on the instance -- fix that "
                f"(or the step) before delivering."
            )
            continue
        expect = step.get("expect")
        if isinstance(expect, dict) and "results" in expect:
            # Only reports THIS step writes may witness its claim -- same marker rule as
            # the check steps above.
            run_in(sandbox, "touch /tmp/.bench_check_started", 30)
        code, out = run_in(sandbox, cmd, 900)
        ran.append(command)
        if isinstance(expect, dict):
            # The dict vocabulary is judged on exec steps too, so a failing step cannot
            # be reported as "ran clean".
            use_dir = "."
            raw_dir = str(expect.get("dir") or "").strip()
            if raw_dir:
                rendered_dir, dir_err = render_command(raw_dir, with_ids, _field_writer(sandbox, run_in))
                if rendered_dir is None:
                    problems.append(f"step {i} `{command}`: its expectation's dir {raw_dir!r} {dir_err}")
                    continue
                use_dir = rendered_dir
            problem = _dict_expect_problem(i, command, expect, use_dir, sandbox, run_in, code, out, inst)
            if problem:
                problems.append(problem)
            continue
        wants_zero = str(expect or "last_exit_zero") == "last_exit_zero"
        if not wants_zero and code == 0:
            # The string form's failing half was also unchecked on exec steps.
            problems.append(f"step {i} `{command}` was expected to fail and exited 0")
            continue
        if wants_zero and code != 0:
            # SAY WHICH FIELD BECAME A FILE. A rendered command can show
            # `-Dtest=/tmp/bench_fields/triggering_test` -- a path the
            # agent never wrote and has never seen. `render_command` materialises any reachable
            # field whose value spans lines, because "a value that spans lines can never be a
            # shell argument", and that is right for `git apply {fix_patch_path}` and fatal for
            # `-Dtest={triggering_test}`. Without this note the agent has a path with no
            # provenance; with it, the fix (record ONE test id, or a one-line list) is obvious.
            # The rule itself is `agentcore/render.py`, a single-home seam, and is untouched.
            from agentcore.render import referenced

            reach, _missing = referenced(shell, with_ids)
            spanned = [n for n in reach if isinstance(with_ids.get(n), str) and "\n" in str(with_ids[n]).strip()]
            note = ""
            if spanned:
                note = (
                    f" NOTE: {', '.join(sorted(spanned))} span(s) lines, so the harness "
                    f"wrote {'them' if len(spanned) > 1 else 'it'} into the sandbox as a "
                    f"file and passed the PATH -- a multi-line value cannot be a shell "
                    f"argument. If this step needs the VALUE, record a single-line one."
                )
            problems.append(
                f"step {i} of the CONSUMER session failed: `{command}` -> exit {code}. "
                f"What ran: `{cmd}`. What it printed: {informative_tail(out)}.{note} This is "
                f"the sequence a user of the finished benchmark runs, and the packaging gate "
                f"runs it too -- an instance whose session does not complete cannot ship, "
                f"so fix the recorded field or the step now, in this sandbox."
            )

    if problems:
        head = (
            f"{len(problems)} problem(s) rehearsing the consumer session -- these are "
            f"the commands a user of this benchmark will run:"
        )
        return False, head + " " + " ".join(f"[{n}] {p}" for n, p in enumerate(problems, start=1))
    return True, (
        f"the consumer session ran clean in this candidate's sandbox: {len(ran)} step(s) -- {', '.join(ran)}"
        if ran
        else "the consumer session declares no step this gate can run here (its steps "
        "are all CLI built-ins the packaged artifact implements itself)"
    )
