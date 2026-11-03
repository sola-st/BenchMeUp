"""Delivered means MATERIALIZED: every declared variant rebuilt from its own script, checked.

Watching the fail-to-pass transition happen once, live, in whatever container state the
agent's particular sequence of actions produced, freezes that end state. Re-demonstrating the
instance later -- checkout the buggy tree, overlay the fix's tests, rebuild, watch the test
fail -- would be a NEW procedure nobody had ever run.

So the deliverable contract is materialization. The task's spec declares its ARTIFACT VARIANTS -- for a
repair benchmark: the fixed tree (triggering test passes), the buggy tree with the fix's tests
overlaid (triggering test fails), the pristine buggy tree (builds) -- and a candidate reaches
the terminal state only after the agent has materialized each variant as a folder PLUS a setup
script, and this validator has re-run each script into a FRESH directory and executed the
variant's check there. The scripts ship with the instance; the packager stops reverse-
engineering procedures because the procedures are part of the product.

Contract shape, in the derived instance (instances.jsonl -> ctx.instance):

    "artifact_variants": [
      {"name": "buggy",
       "description": "parent tree with the fix's test files overlaid",
       "check_cmd": "mvn -q -Dsurefire.failIfNoSpecifiedTests=true -Dtest={triggering_test} test",
       "expect": "fail",                    # "pass" | "fail" (exit code of check_cmd)
       "timeout": 900},
      ...
    ]

`check_cmd` templates over the candidate's own recorded fields ({triggering_test} etc.).
Conventions the agent is instructed to follow, and this validator enforces:

    {variants_root}/setup_<name>.sh   takes ONE argument: the target directory; sets the
                                      variant up there from scratch (clone/checkout/overlay)
    {variants_root}/<name>/           the agent's own materialized copy (its working proof)

The re-run happens in {variants_root}/_verify_<name>, so the agent's copy is never touched and
a script that secretly depends on the agent's copy fails here instead of at a consumer's desk.
"""

from __future__ import annotations

import json
import re

from agentcore.tools import parsing as _parsing

_SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./-]{1,200}$")
# THE rule for what a variant may be called, exported because two copies disagreed. This one
# allowed underscores, capitals and 41 characters; benchcompiler's allowed none of those and 31,
# so `buggy_pristine` was materialised, re-run from its setup script and checked here -- the full
# construction gate -- and then filtered out of the artifact's plan by a list comprehension, with
# nothing counting the difference. The permissive rule wins because instances have already been
# proven under it; narrowing would retroactively invalidate work that was correctly done.
VARIANT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,40}$")
_NAME = VARIANT_NAME  # the original spelling, kept so this module reads unchanged
_PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")


# Noise footer suppression. Maven-specific and knowingly so: it only ever SUPPRESSES lines,
# so an ecosystem it does not know about loses nothing.
_BOILERPLATE = (
    "read the following articles",
    "cwiki.apache.org",
    "[Help",
    "re-run Maven",
    "-X switch",
    "full stack trace",
)

# Failure-signal vocabulary. The primary set is the shared, ecosystem-blind one in
# tools/parsing (_FAILURE_PATTERNS: make, npm, python tracebacks, compiler diagnostics,
# exit statuses, segfaults) -- this engine is task-blind and may not know only Java. The
# literals below stay as a SUPPLEMENT for the summary lines that carry no failure word of
# their own ("Tests run: 4, Failures: 0", "BUILD FAILURE") and for the bare "FAIL" that
# `go test` prints, none of which the shared patterns match.
_SIGNAL = ("ERROR", "FAIL", "error:", "Tests run", "AssertionError", "BUILD", "Exception", "assert")

# Cheap pre-gate for the regex arm. EVERY _FAILURE_PATTERNS regex requires one of these
# substrings to be present (case-folded), so a line matching none of them cannot match any
# pattern and the regex loop can be skipped: on a real Maven log nearly every line carries no
# failure vocabulary at all. Keep in step with _FAILURE_PATTERNS: a pattern whose vocabulary is
# not represented here becomes unreachable, which is what
# tests/test_the_failure_hint_pregate_covers_every_shared_pattern.py pins.
_PATTERN_HINTS = (
    "error",
    "fail",
    "exception",
    "traceback",
    "fatal",
    "fault",
    "assert",
    "denied",
    "not found",
    "no such file",
    "modulenotfound",
    "importerror",
    "exit",
    "make",
    "npm err",
    "bug",
    "[-w",
    "[core.",
    "warning:",
)


def _has_failure_signal(ln: str) -> bool:
    """Whether one line carries failure signal, by the shared vocabulary or the local one."""
    if any(w in ln for w in _SIGNAL):
        return True
    low = ln.lower()
    if not any(h in low for h in _PATTERN_HINTS):
        return False
    return any(p.search(ln) for p in _parsing._FAILURE_PATTERNS)


def noop_sweep_cmd(probe_find: str) -> str:
    """The command that replaces every found executable with /bin/true, one cp per file.

    NOT `-exec cp /bin/true {} +`: `+` hands cp ALL matches in one invocation, and with
    more than one file cp requires the last argument to be a directory -- it errors and
    replaces nothing, silently. That shape refused every multi-binary candidate (freerdp,
    git, qemu, coturn, curl) as 'a check that cannot fail' while the
    binaries sat unbroken. The per-file loop replaces each match whatever
    their number or their names.
    """
    return probe_find + " -exec sh -c 'for f; do cp /bin/true \"$f\"; done' _ {} +"


def executables_find(root_dir: str) -> str:
    """The ONE definition of 'a built executable' under a tree.

    The no-op sweep and the solver-channel probe must agree about what they are breaking
    and hashing, so both build their find from here. Script suffixes are spared: an
    interpreter running ELF bytes fails for a reason that says nothing about the check,
    and a hash over copied scripts measures the copy, not the build.
    """
    return f"find {root_dir} -type f -executable -not -name '*.sh' -not -name '*.py' -not -name '*.pl'"


def probe_listing_cmd(find_cmd: str, limit: int = 20) -> str:
    """The find whose output names the executables the no-op probe's refusal will quote.

    Timestamped and sorted newest-first -- see ordered_probe_targets for why. `-printf` is
    GNU find; where it is missing the whole pipeline fails and the caller falls back to the
    plain find, because an empty listing skips the probe entirely.
    """
    return f"{find_cmd} -printf '%T@ %p\\n' | sort -rn | head -{int(limit)}"


def ordered_probe_targets(listing) -> list[str]:
    """Paths from a probe listing, newest build output first.

    The refusal quotes the first five of these, and a CMake candidate could spend
    all five on compiler-detection leftovers (`CMakeFiles/3.22.1/CompilerIdC/a.out` and
    friends) while its own program sat sixth. A configure step writes its probe binaries
    before anything is compiled and the program links last, so recency separates what THIS
    build produced from what the build system produced while deciding how to build -- true
    of cmake, autotools and a hand-written Makefile alike, and it puts no build system's name
    in the engine.

    Lines without a leading timestamp are kept in the order they arrived, so a sandbox whose
    find lacks `-printf` still gets a working probe instead of a silently skipped one.
    """
    timed: list[tuple[float, int, str]] = []
    plain: list[str] = []
    for i, raw in enumerate(str(listing or "").splitlines()):
        line = raw.strip()
        if not line:
            continue
        head, sep, rest = line.partition(" ")
        if sep and rest.strip():
            try:
                timed.append((float(head), i, rest.strip()))
                continue
            except ValueError:
                pass
        plain.append(line)
    # `i` breaks ties so the order is total and the same listing always reads the same way.
    return [p for _, _, p in sorted(timed, key=lambda t: (-t[0], t[1]))] + plain


def informative_tail(out: str, limit: int = 300) -> str:
    """The CAUSE lines of a failed command, never its boilerplate footer.

    Observed live: the last-200-bytes tail of failed Maven output is, by construction, the
    help footer ("read the following articles... MojoExecutionException") -- the one part
    that says nothing. The model, refused with that, deleted its own work and rebuilt blind.
    Select lines that carry failure signal, drop known boilerplate, keep the most recent few.
    """
    # strip_ansi (not a colour-only sub): cursor and erase sequences from npm/gradle/docker
    # progress output are exactly the noise this function exists to keep out of the message.
    text = _parsing.strip_ansi(str(out or ""))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Only the last five matches are ever shown, so scan BACKWARDS and stop at five instead
    # of testing all 11k lines of a build log. Same five lines, O(1) in the log's length.
    picked = []
    for ln in reversed(lines):
        if _has_failure_signal(ln) and not any(b in ln for b in _BOILERPLATE):
            picked.append(ln)
            if len(picked) == 5:
                break
    picked.reverse()
    if not picked:
        picked = lines[-3:]
    joined = " | ".join(picked)
    if len(lines) <= len(picked) and len(joined) <= limit:
        return joined
    # SAY THAT THIS IS AN EXCERPT. The caller's sentence reads "What the check
    # printed:", which claims a completeness a five-line filtered selection does not have --
    # and an agent that concludes "the build printed nothing about X" from it is reasoning
    # from a view it was never told was pruned.
    return (
        f"{joined[:limit]} [excerpt: {len(picked)} of {len(lines)} lines, chosen for "
        f"failure signal; the full output is longer]"
    )


def variant_run_cmds_problem(raw, want: list[str]) -> str:
    """Empty string if this value is a shape the packaged benchmark can look a variant up
    in; otherwise the refusal saying why.

    ONE HOME. This rule decides whether a delivered instance can measure anything
    at all. It must be askable about a value a model just produced, with no sandbox anywhere,
    and a scenario that re-implements the rule measures its own copy. So the gate and the
    scenario both call this.
    """
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001
            # A PYTHON DICT LITERAL IS A MAPPING TOO. A recorded value like
            # `{'default': 'sh check_default.sh .', ...}` is not malformed -- it is one mapping with
            # one meaning, which `json.loads` would refuse because it will not take single quotes. Resolve
            # the notation, do not refuse it: same class as the git object-peel and the
            # JSON-string digest map.
            #
            # literal_eval, NEVER eval: `raw` is a model's free text arriving unchecked, and
            # literal_eval evaluates literals only -- a call expression raises instead of
            # running. Anything it cannot read still falls through to the refusal below,
            # which prints the exact shape to write.
            try:
                import ast

                parsed = ast.literal_eval(raw)
            except Exception:  # noqa: BLE001 - genuinely not a mapping in any spelling
                parsed = None
    missing = [n for n in want if not isinstance(parsed, dict) or not parsed.get(n)]
    if not missing:
        # AND THEY MUST AGREE ABOUT THE INPUT. A record such as
        # `{"default": ".../test.elf", "optimized": ".../test.elf {input_path}"}`
        # satisfies every rule above and is still unusable: this benchmark's task is
        # "make it faster on the SAME input with the same output", so a pair where one command
        # takes the reference input and the other does not is not a noisy comparison, it is two
        # different questions. Agreement, not presence -- variants that legitimately take no
        # input all lack the placeholder and still agree.
        with_input = [n for n in want if "{input_path}" in str(parsed.get(n))]
        if with_input and len(with_input) != len(want):
            without = [n for n in want if n not in with_input]
            return (
                f"variant_run_cmds must give every variant the SAME input: "
                f"{with_input} take {{input_path}} and {without} do not. This instance is "
                f"measured by running each variant on one reference input and comparing "
                f"runtime and output, so a command that ignores the input is answering a "
                f"different question -- its runtime and its digest are not comparable with "
                f"the others'. Put {{input_path}} in {without} too, or take it out of "
                f"{with_input} if these variants genuinely read no input."
            )
        return ""
    return (
        f"variant_run_cmds must be a mapping from VARIANT NAME to the command that "
        f"runs that variant -- the packaged benchmark looks up {want} in it to "
        f"measure what a solver submitted, and this instance "
        + (
            "recorded nothing at all"
            if raw in (None, "")
            else f"recorded a mapping without {missing}"
            if isinstance(parsed, dict)
            else "recorded a string that is not JSON"
            if parsed is None
            else f"recorded a {type(parsed).__name__} instead"
        )
        + f", so {missing} cannot be found. Record it as "
        + '{"'
        + '": "<command>", "'.join(want)
        + '": "<command>"}'
        + " (JSON), with each command runnable in this instance's own sandbox."
    )


def declared_variants(instance: dict) -> list[dict]:
    out = []
    for v in (instance or {}).get("artifact_variants") or []:
        if isinstance(v, dict) and _NAME.match(str(v.get("name") or "")):
            out.append(v)
    return out


#: The recorded fields that stand in for the measurement input, most specific first. The
#: design's `measure:` block declares `{input_path}` as a placeholder the MACHINERY fills
#: (designer.MACHINERY_FILLED); the record spells the input it measured on under one of
#: these. Packaging and construction resolve them the same way, so one gate never demands
#: `{input_path}` while another refuses it.
MACHINERY_INPUT_FIELDS = ("costliest_input_path", "largest_input_path", "initial_input_path")


def with_machinery_input(fields: dict) -> dict:
    """`fields` plus `input_path`, bound to the recorded input the record names, when the
    record does not carry `input_path` itself. THE one resolution, for the construction-time
    re-derivation and for packaging alike: what the harness will substitute at measurement
    time is what it re-derives against now."""
    if str(fields.get("input_path") or "").strip():
        return fields
    for alt in MACHINERY_INPUT_FIELDS:
        if str(fields.get(alt) or "").strip():
            return {**fields, "input_path": str(fields[alt])}
    return fields


def render_cmd(template: str, fields: dict, sandbox: str = "", run_in=None):
    """(command, error) for a template from the CONTRACT (a check_cmd, a step).

    Shares one rendering rule with `render_break`, so a check_cmd naming a
    multi-line field gets a file rather than a diff across its argv.

    `sandbox`/`run_in` are optional so the many callers that template only single-line
    fields need not thread them; a multi-line value without them is refused, never sprayed.
    """
    from agentcore.render import render_command

    writer = _field_writer(sandbox, run_in) if run_in is not None else None
    return render_command(template, fields, writer)


def _field_writer(sandbox: str, run_in):
    """The injection point: how THIS package puts a file inside a container."""

    def write(name: str, content: str) -> str:
        return write_field_file(name, content, sandbox, run_in)

    return write


def render_break(field: str, fields: dict, sandbox: str, run_in, passes: int = 5):
    """(command, error) for a recorded break, rendered the way the ARTIFACT renders it.

    The break is a recorded FIELD holding a template, so this is `render_command` over
    `{field}` -- which is literally what the packaged negative control's break step is
    (`["{corruption_cmd}"]`). Same string, same rules, both layers.
    """
    if not field:
        return None, "no break field named"
    if not str(fields.get(field) or "").strip():
        return None, (
            f"the declared break names {field!r}, which this candidate has not recorded, so nothing was broken here"
        )
    from agentcore.render import render_command

    cmd, err = render_command("{" + field + "}", fields, _field_writer(sandbox, run_in))
    if cmd is None:
        return None, f"the declared break ({field}) could not be built: {err}"
    return cmd, ""


FIELD_DIR = "/tmp/bench_fields"


def write_field_file(name: str, value: str, sandbox: str, run_in, directory: str = FIELD_DIR) -> str:
    """Put a recorded value into the sandbox as a file and return its path.

    THE ONE WRITER for this runtime. Pasting the value into a heredoc is unsafe: the values
    are patches, arbitrary text whose own lines can equal the delimiter, ending the document
    early and handing the rest to the shell. The content travels base64-encoded so it is
    data and never syntax, and the write is CHECKED: `git apply <empty file>` exits 0, so an
    unverified write turns a step that did nothing into a step that passed.

    Raises RuntimeError if the file could not be written; callers turn that into a refusal.
    """
    import base64

    body = value if value.endswith("\n") else value + "\n"
    blob = base64.b64encode(body.encode("utf-8", "replace")).decode("ascii")
    path = f"{directory}/{name}"
    code, why = run_in(sandbox, f"mkdir -p {directory} && printf %s {blob} | base64 -d > {path} && test -s {path}", 120)
    if code != 0:
        raise RuntimeError(
            f"could not write the recorded field {name!r} into the sandbox as a file "
            f"(exit {code}: {str(why)[:2000]}). The step that templates it was not "
            f"attempted -- a command pointing at an empty file is not a rehearsal."
        )
    return path


def _tree_signature(sandbox: str, workdir: str, run_in) -> str | None:
    """What the checked tree looks like right now, or None when it cannot be observed.

    Mirror of benchcompiler/artifact.py's `_tree_signature` -- the two packages do not
    import each other. None (not a git checkout, or git absent) means the caller falls back
    to the break's exit code.
    """
    if not workdir:
        return None
    code, out = run_in(
        sandbox,
        f"cd {workdir} 2>/dev/null && "
        f"git rev-parse --is-inside-work-tree >/dev/null 2>&1 && "
        f"{{ echo __BENCH_GIT_TREE__; git status --porcelain; }} "
        f"|| echo __BENCH_NO_GIT_TREE__",
        120,
    )
    text = str(out or "")
    # The observation announces itself; silence means we could not look, and the caller
    # falls back to the break's exit code rather than reading "unchanged" into it.
    if code != 0 or "__BENCH_GIT_TREE__" not in text:
        return None
    return text


def _declared_break_probe(
    inst_doc,
    fields,
    sandbox,
    fresh,
    name,
    check_script,
    probe_out,
    contract_cmd,
    timeout,
    run_in,
    kind=None,
    want_failures=0,
    run_report=False,
):
    """(broke, why): run the break the SPEC declares, and see whether the verdict flips.

    `negative_control: {break_cmd: <field name>}` names the field holding the command --
    task content, declared once, executed by the harness. Returns:
      True   the verdict flipped: this check can fail
      False  it did not: the check, or the command recorded in that field, proves nothing
      None   nothing to run (no declaration, or the field was never recorded), which must
             stay VISIBLE rather than read as proven
    """
    nc = (inst_doc or {}).get("negative_control") or {}
    field = str(nc.get("break_cmd") or "").strip()
    if not field:
        return None, "no break is declared for this benchmark, so nothing was broken here"
    # THE SPEC NAMES THE VARIANT THE BREAK IS FOR. For example a spec may declare
    # `{break_cmd: corruption_cmd, check: fixed, describes: "revert the fix ... so the
    # triggering test must stop passing"}`. Reverting the fix is meaningful on the FIXED
    # variant and a no-op on the buggy one, which is already buggy -- so probing `buggy` with
    # it left the tree byte-identical every time and reported UNPROBED for a variant the spec
    # never nominated. Worse under `run_report=True`, where the same call returns False and
    # lands in `problems` as "this check cannot fail", refusing a sound instance for a break
    # that was not written for it. Absent `check`, nothing is nominated and every variant is
    # probed as before.
    want = str(nc.get("check") or "").strip()
    if want and name and want != name:
        return None, (
            f"{NOT_NOMINATED} the declared break is for the {want!r} variant "
            f"(negative_control.check), not for {name!r}, so nothing was broken here"
        )
    command, why = render_break(field, fields, sandbox, run_in)
    if command is None:
        return None, why
    before_tree = _tree_signature(sandbox, fresh, run_in)
    code, out = run_in(sandbox, f"cd {fresh} && {command}", timeout)
    after_tree = _tree_signature(sandbox, fresh, run_in)
    # DID THE ARTIFACT CHANGE, not what did the command exit (mirroring
    # benchcompiler/artifact.py). The lifecycle asks an agent to show a corruption AND the
    # check that demonstrates it, so a break is routinely recorded as
    # `sed -i ... && mvn test ...` -- whose exit code is the TEST's verdict. Reading that as
    # the break's success refuses precisely the instances whose break worked. A break that
    # leaves the checked tree byte-identical still did not apply, whatever it exited.
    if before_tree is not None and after_tree is not None:
        if after_tree == before_tree:
            return None, (
                f"the declared break ({field}) left {fresh} byte-identical, so "
                f"nothing about the check was broken (it exited {code}). A break "
                f"that edits some other directory cannot demonstrate anything "
                f"about the tree the check reads: {informative_tail(out)}"
            )
    elif code != 0:
        return None, (
            f"the declared break ({field}) did not apply here (exit {code}) and "
            f"{fresh} is not a git tree, so its exit code is the only evidence "
            f"available: {informative_tail(out)}"
        )
    if run_report:
        from agentcore import reports

        # The report discovery reads files NEWER than this marker, and it was dropped before
        # the FIRST check. Without re-dropping it, a runner that writes to new filenames
        # (Gradle, pytest) hands back the pre-break AND post-break reports merged, so the
        # pre-break failures are still in the count and the probe refuses a sound check.
        run_in(sandbox, "touch /tmp/.bench_check_started", 30)
        bcode, _bout = (
            run_in(sandbox, f"cd {fresh} && bash {check_script} {fresh}", timeout)
            if "CHECK_SCRIPT_PRESENT" in str(probe_out or "")
            else run_in(sandbox, f"cd {fresh} && {contract_cmd}", timeout)
        )
        dump_cmd = reports.dump_command(kind, reports.discovery_patterns(inst_doc, kind))
        _, dump = run_in(sandbox, f"cd {fresh} && {dump_cmd}", 120)
        after = reports.parse_report(kind, dump or "")
        if after is None or after.tests == 0:
            return True, (
                f"after the declared break ({field}) the run produced no usable "
                f"verdict at all, so the check does react to it"
            )
        # THE SAME QUESTION IN BOTH DIRECTIONS: does the verdict this variant ASSERTS stop
        # holding? One break_cmd cannot carry both polarities (a bug-demonstrating variant's
        # failing test starting to PASS), so the question is asked of the asserted verdict.
        #
        # Asked through the seam: "the assertion stopped holding" is exactly
        # "the contract is no longer satisfied", so it is that one rule negated rather than
        # a second floor comparison living beside it.
        from agentcore.verdict import verdict_of

        still_holds, why_after = verdict_of({"results": kind, "expect_failures": want_failures}, bcode, dump or "")
        return (not still_holds), (
            f"the declared break ({field}) was applied and the report became {after.describe()} -- {why_after}"
        )
    # The exit-code-judged probe: ONE run is the verdict, and it runs only on this
    # path so the report path does not pay for the identical check twice.
    if "CHECK_SCRIPT_PRESENT" in str(probe_out or ""):
        bcode, _bout = run_in(sandbox, f"cd {fresh} && bash {check_script} {fresh}", timeout)
    else:
        bcode, _bout = run_in(sandbox, f"cd {fresh} && {contract_cmd}", timeout)
    return bcode != 0, (f"the declared break ({field}) was applied and the check exited {bcode}")


# A CHECK THAT PASSED WITHOUT BEING SHOWN ABLE TO FAIL. Emitted into the gate's SUCCESS
# message, deliberately: some task families legitimately cannot be probed here, so this is not a
# refusal. `worklist_tools.advance_note` imports this to decide whether an advance is worth
# annotating, which is why it is a named constant rather than a literal in two files.
UNPROBED = "UNPROBED"

# A variant the spec did not nominate for its break is not "unprobed" in the sense that word
# carries -- nothing went wrong and there is nothing for the agent to fix. It must not be
# labelled UNPROBED, because `worklist_tools.advance_note` keys on that word to warn about a
# check that proved less than it looks, and a warning that fires on every delivered instance is
# one nobody reads on the instance where it is real.
NOT_NOMINATED = "OUT OF SCOPE:"


def _solver_channel_probe(inst_doc, root, name, script, timeout, sandbox, run_in):
    """(verdict, why): does the DECLARED solver parameter actually reach the binary?

    The spec declares the knob -- `solver_input: {kind: parameter, name: <p>,
    probe_values: [...]}` -- and packaging hands the solver's value to every rebuild as
    the environment variable BENCH_<P> (benchcompiler/artifact_plan.py exports it onto
    each rebuild verb). A setup script that hardcodes its own value builds the same
    program for every submission, which makes the variation control unfalsifiable -- and
    would otherwise be found only at packaging, after the sandbox is gone.

    So: rebuild the SAME recorded script into two fresh scratch dirs, once per probe
    value with BENCH_<P> set, and hash the built executables. Returns one of:
      (None, "")           nothing declared (no parameter kind, or fewer than two
                           probe values) -- the probe never ran, no builds spent
      ("live", why)        the two rebuilds differ: the channel carries
      ("dead", refusal)    byte-identical across values AND reproduced by a SAME-value
                           rebuild (deterministic, so the sameness is the script's
                           doing) -- refuse, naming the env var and the fix shape
      ("unprobed", why)    the comparison cannot mean anything here (a probe rebuild
                           failed, nothing hashable, or the build is nondeterministic:
                           a SAME-value rebuild differs too) -- the harness's news,
                           never the candidate's fault

    The SAME-value rebuild runs only when "dead" is on the table, to keep cost down:
    builds are expensive, and an honest channel is proven by the first two.
    """
    si = (inst_doc or {}).get("solver_input") or {}
    if not isinstance(si, dict) or str(si.get("kind") or "").strip() != "parameter":
        return None, ""
    pname = str(si.get("name") or "").strip()
    values = [str(x) for x in (si.get("probe_values") or []) if str(x).strip()]
    if not pname or len(values) < 2:
        return None, ""
    import shlex

    env = f"BENCH_{pname.upper()}"

    def rebuild(tag, value):
        """(hashes, why-not): run the recorded script with the declared env set into a
        fresh scratch dir, then hash what it built (paths relative, so two dirs
        compare). None hashes mean the comparison cannot happen, and why."""
        d = f"{root}/_channel_{name}_{tag}"
        code, out = run_in(
            sandbox, f"rm -rf {d} && mkdir -p {d} && {env}={shlex.quote(value)} bash {script} {d}", timeout
        )
        if code != 0:
            return None, (
                f"rebuilding {name!r} with {env}={value} exited {code} "
                f"({informative_tail(out)}) though the same script had just "
                f"rebuilt cleanly without it, so nothing comparable was built"
            )
        hcode, hashes = run_in(sandbox, f"cd {d} && {executables_find('.')} -exec sha256sum {{}} + | sort", 300)
        text = str(hashes or "").strip()
        if hcode != 0 or not text:
            return None, (
                f"no built executable could be hashed under the {env}={value} "
                f"rebuild of {name!r} (exit {hcode}), so there is nothing to "
                f"compare"
            )
        return text, ""

    first, second = values[0], values[1]
    ha, why = rebuild("a", first)
    if ha is None:
        return "unprobed", why
    hb, why = rebuild("b", second)
    if hb is None:
        return "unprobed", why
    if ha != hb:
        return "live", (f"{env}={first} and {env}={second} rebuilds of {name!r} produced different executables")
    # Identical across values would refuse -- SELF-CHECK first: only a build that
    # reproduces itself under the SAME value can make byte-identity mean anything.
    h2, why = rebuild("a2", first)
    if h2 is None:
        return "unprobed", why
    if h2 != ha:
        return "unprobed", (
            f"{env}={first} and {env}={second} rebuilds of {name!r} were byte-identical, "
            f"but rebuilding twice with the SAME {env}={first} ALSO produced different "
            f"executables -- the build is nondeterministic, so byte-identity across "
            f"values cannot show whether the channel is dead. This says nothing against "
            f"the candidate; the variation control at packaging remains the gate that "
            f"must prove it"
        )
    n = len(ha.splitlines())
    return "dead", (
        f"variant {name!r}: the declared solver input ({pname}) never reaches the "
        f"binary. I re-ran `bash {script} <fresh-dir>` with {env}={first}, again with "
        f"{env}={second}, and all {n} built executable(s) came out byte-identical "
        f"across the two values (a third rebuild with {env}={first} reproduced them "
        f"exactly, so the build is deterministic -- the sameness is the script's doing, "
        f"not noise). The packaged benchmark hands the solver's {pname} to every "
        f"rebuild as the environment variable {env}; a setup script that hardcodes its "
        f"own value builds the same program for every submission, so the variation "
        f"control cannot distinguish anything. Make every variant's setup script "
        f'consume it -- e.g. compile with {pname.upper()}="${{{env}:-<your current '
        f'default>}}" or pass "${env}" into the build the equivalent way -- then '
        f"re-run this state."
    )


#: The record fields this gate reads. Declared so the worklist treats a re-record of any of
#: them as a re-run of the gate (Worklist._validator_reads): `variant_run_cmds` is required
#: by a later state, and without the declaration a corrected value replayed the refusal.
VARIANTS_GATE_READS = (
    "sandbox",
    "variants_root",
    "variant_run_cmds",
    "costliest_input_path",
    "largest_input_path",
    "initial_input_path",
)


def _measurement_runs(raw, variants: list, root: str, sandbox: str, fields: dict, run_in, timeout: int) -> list[str]:
    """Run each variant's measurement command ONCE, inside its materialized copy, on the
    recorded input, and refuse a command that does not exit 0.

    A recorded command can have the right SHAPE and still be wrong: `./jq {input_path}` with no
    filter times a usage error, not the measured workload. A field the consumer executes is accepted by executing it. When no
    input is recorded yet (the gate also runs at states before inputs exist) there is
    nothing to bind, and the shape check stands alone until there is.
    """
    try:
        mapping = raw if isinstance(raw, dict) else json.loads(str(raw))
    except (TypeError, ValueError):
        return []
    if not isinstance(mapping, dict):
        return []
    bound = with_machinery_input(fields)
    input_path = str(bound.get("input_path") or "").strip()
    if not input_path:
        return []
    out: list[str] = []
    for v in variants:
        name = str(v.get("name"))
        cmd = str(mapping.get(name) or "").strip()
        if not cmd:
            continue
        rendered = cmd.replace("{input_path}", input_path)
        mdir = f"{root}/{name}"
        code, text = run_in(
            sandbox,
            f"cd {mdir} && s=$(date +%s%N); ( {rendered} ) >/dev/null 2>&1; rc=$?; "
            f"e=$(date +%s%N); echo MEASURE_EXIT=$rc MEASURE_MS=$(( (e - s) / 1000000 ))",
            timeout,
        )
        m = re.search(r"MEASURE_EXIT=(\d+) MEASURE_MS=(\d+)", str(text or ""))
        if not m:
            continue  # the harness's own plumbing failed to report; not the candidate's
        rc, ms = int(m.group(1)), int(m.group(2))
        if rc != 0:
            out.append(
                f"variant {name!r}: its measurement command (variant_run_cmds[{name!r}] = "
                f"{cmd!r}) exited {rc} after {ms} ms when run inside {mdir} on the recorded "
                f"input {input_path}. The packaged benchmark TIMES exactly this command in "
                f"exactly this directory; a command that fails there measures nothing. Run it "
                f"yourself in {mdir}, fix it, and re-record variant_run_cmds."
            )
    return out


def variants_are_materialized(fields: dict, ctx, run_in) -> tuple[bool, str]:
    """The generic gate: each declared variant's script rebuilds it fresh, and the check holds.

    `run_in(sandbox, cmd, timeout) -> (code, output)` is supplied by the collector's
    validators module so execution goes through the same sandbox plumbing as every other
    check. Admits immediately when the instance declares no variants -- tasks opt in through
    their spec, and a task without a variant contract keeps its old delivery semantics.
    """
    inst_doc = (getattr(ctx, "instance", {}) or {}) if ctx is not None else {}
    variants = declared_variants(inst_doc)
    if not variants:
        return True, "no artifact variants declared"

    root = str(fields.get("variants_root") or "").strip()
    if not _SAFE_PATH.match(root):
        return False, (
            "record variants_root: the absolute directory inside this candidate's "
            "sandbox holding setup_<variant>.sh scripts and one folder per variant "
            "(e.g. /workspace/variants)"
        )
    sandbox = str(fields.get("sandbox") or "").strip()
    if not sandbox:
        # Probing with an empty container name produces refusals about scripts the agent
        # cannot fix; the missing thing is the FIELD, so name it.
        return False, (
            "this candidate records no `sandbox`, so there is no container to "
            "verify the variants in. Record the sandbox name with "
            "worklist_record before this state can run."
        )
    from agentcore.tools.containers import SANDBOX_UNREACHABLE, UNREACHABLE_PREFIX

    proven = []
    # ONE refusal, every problem. Returning at the first defect would make the
    # agent learn its requirements across several round trips, and each round trip re-runs
    # the setup script into a fresh directory -- a full from-scratch build per lesson.
    # So accumulate, then report.
    problems: list[str] = []
    # One record per variant of what was actually seen, so a future rule can be replayed against
    # this run instead of argued about. The exit codes were known here and discarded when this
    # returned (ok, detail); the harness moves them into the evidence block and pops the key.
    observations = fields.setdefault("_observations", [])
    observations.clear()
    # ONE variant carries the solver-channel probe: the one the negative
    # control nominates (`negative_control.check`), falling back to the first -- once per
    # gate pass, not per variant, because each probe is two (sometimes three) full builds.
    _vnames = [str(v["name"]) for v in variants]
    _nc_check = str((inst_doc.get("negative_control") or {}).get("check") or "").strip()
    channel_target = _nc_check if _nc_check in _vnames else _vnames[0]
    for v in variants:
        name = str(v["name"])
        script = f"{root}/setup_{name}.sh"
        fresh = f"{root}/_verify_{name}"
        timeout = int(v.get("timeout") or 900)
        want_fail = str(v.get("expect") or "pass").lower() == "fail"

        # Each step below abandons THIS variant on failure and moves to the next, so one
        # refusal carries every variant's problem instead of the first one found.
        code, out = run_in(sandbox, f"test -f {script} && test -d {root}/{name} && echo BOTH_PRESENT", 60)
        if int(code) == SANDBOX_UNREACHABLE or UNREACHABLE_PREFIX in str(out or ""):
            # A dead daemon or an unrevivable sandbox is the harness's fault: the
            # probe's silence must not fall into the missing-scripts branch below, telling
            # the agent to write scripts nothing can run and charging a strike per attempt.
            # The marker is what _harness_fault looks for.
            problems.append(
                f"variant {name!r}: {UNREACHABLE_PREFIX}the sandbox could not be reached "
                f"to check its layout ({informative_tail(out) or f'exit {code}'}). "
                f"Nothing was executed, so this says nothing about your scripts. If this "
                f"repeats while the sandbox should be alive, book it with "
                f"worklist_drop(blocked=true) naming the sandbox as the fault."
            )
            continue
        if "BOTH_PRESENT" not in (out or ""):
            problems.append(
                f"variant {name!r}: {script} and {root}/{name}/ must both exist. Write "
                f"the setup script (it takes the target directory as its only argument), "
                f"run it to materialize {root}/{name}, and keep both -- they ship with "
                f"the instance."
            )
            continue

        # The re-run: the script must reproduce the variant somewhere it has never been.
        code, out = run_in(sandbox, f"rm -rf {fresh} && mkdir -p {fresh} && bash {script} {fresh}", timeout)
        if code != 0:
            problems.append(
                f"variant {name!r}: setup_{name}.sh failed when re-run into a fresh "
                f"directory (exit {code}: {informative_tail(out)}). The script must "
                f"reproduce the variant from scratch, not depend on state it left behind "
                f"the first time."
            )
            continue

        # THE DECLARED SOLVER INPUT MUST REACH THE BINARY. A setup script that hardcodes
        # compiler flags makes the injected BENCH_<NAME> never change what was compiled,
        # which would fail the packaging variation control late. The spec already
        # declares the knob (`solver_input: {kind: parameter, probe_values: [...]}`) and
        # this gate has just proven the script rebuilds from scratch, so the channel is
        # probed HERE, in the candidate's sandbox, where the agent can still fix the
        # script. A probe that could not run above (script missing, rebuild failed)
        # never reaches this line -- those verdicts are the gate's own, not duplicated.
        if name == channel_target:
            chan, chan_why = _solver_channel_probe(inst_doc, root, name, script, timeout, sandbox, run_in)
            if chan is not None:
                observations.append({"variant": name, "solver_channel_probe": chan, "detail": chan_why})
                if chan == "dead":
                    problems.append(chan_why)
                    continue
                if chan == "unprobed":
                    proven.append(f"solver-input channel {UNPROBED} -- {chan_why}")
                else:
                    proven.append(f"solver-input channel live ({chan_why})")

        # SELF-CONTAINED, OR IT DOES NOT SURVIVE BEING AN IMAGE. A local
        # `git clone` HARDLINKS the object packs, which works while everything is in one
        # writable layer and fails the moment the sandbox is committed -- a hardlink cannot
        # cross into a lower image layer ("fatal: hardlink different from source"). Committing
        # an image per delivery to rehearse in would be costly; this checks the same property
        # with one `find`.
        # The probe ANNOUNCES ITSELF: a runner that returns nothing must read as "could not look", never as
        # "looked and found sharing" -- and never as "looked and found none" either.
        _c, shared = run_in(sandbox, f"echo __BENCH_LINKS__; find {fresh} -type f -links +1 2>/dev/null | head -5", 120)
        text = str(shared or "")
        examples = (
            [x.strip() for x in text.split("__BENCH_LINKS__", 1)[1].splitlines() if x.strip()][:3]
            if "__BENCH_LINKS__" in text
            else []
        )
        if examples:
            problems.append(
                f"variant {name!r}: setup_{name}.sh produced a copy that SHARES FILES with "
                f"the workspace instead of copying them ({', '.join(examples)}). That works "
                f"here and breaks the moment this sandbox becomes an image -- a hardlink "
                f"cannot cross into a lower layer, and the consumer's `rebuild-{name}` dies "
                f"with 'fatal: hardlink different from source'. The usual cause is a local "
                f"`git clone`, which hardlinks object packs: use `git clone --no-hardlinks`, "
                f"or copy the tree (`cp -a`, not `cp -al`)."
            )
            continue

        # The agent's own check script wins over the contract's command. For example
        # the contract may say bare `mvn`, which resolves to the system Maven while the
        # project demands a newer one -- the agent can fix ITS scripts but could
        # do nothing about the command the spec hardcodes. The spec owns the EXPECTATION
        # (this check must pass / must fail); the agent owns the HOW.
        check_script = f"{root}/check_{name}.sh"
        _, probe_out = run_in(sandbox, f"test -f {check_script} && echo CHECK_SCRIPT_PRESENT", 60)
        # THE MARKER GOES DOWN BEFORE THE CHECK RUNS. Report discovery reads only files newer
        # than it -- so that a report left behind by an earlier run cannot witness this one --
        # and touching it AFTER the check would make every report older than the marker. The find
        # would then match nothing, always: "its check ran (exit 0) but produced no junit-xml
        # report", whatever the check did.
        run_in(sandbox, "touch /tmp/.bench_check_started", 30)
        cmd = None  # the contract's fallback command, when the agent shipped no script
        if "CHECK_SCRIPT_PRESENT" in str(probe_out or ""):
            code, out = run_in(sandbox, f"cd {fresh} && bash {check_script} {fresh}", timeout)
        else:
            cmd, err = render_cmd(str(v.get("check_cmd") or "true"), fields)
            if cmd is None:
                problems.append(f"variant {name!r}: {err}")
                continue
            code, out = run_in(sandbox, f"cd {fresh} && {cmd}", timeout)

        # RESULT-FILE VERDICTS. When the contract declares a report kind, the report the
        # tool RECORDED is the verdict -- never the exit code of a script whose branch can
        # swallow an environment error into the expected answer.
        kind = str(v.get("results") or "").strip().lower()
        if kind and kind != "none":
            from agentcore import reports

            # FIX THE RUN, NEVER THE REPORT: a model could answer an
            # INCONCLUSIVE refusal by appending a sed to its check script that rewrites
            # <error> into <failure> in its own TEST-*.xml. The report IS this variant's
            # verdict channel, so what runs is scanned for report-editing before any report
            # is believed: the agent's script when one exists, else the contract's command.
            if cmd is None:
                _, checked_text = run_in(sandbox, f"cat {check_script}", 60)
            else:
                checked_text = cmd
            patterns = reports.discovery_patterns(inst_doc, kind)
            tamper = reports.report_tampering(checked_text, patterns)
            if tamper:
                problems.append(
                    f"variant {name!r}: its check {tamper}. Fix the RUN so the tests "
                    f"genuinely pass or fail -- never the report. Remove the "
                    f"report-editing command from "
                    f"{f'check_{name}.sh' if cmd is None else 'the check command'} and "
                    f"re-run this state."
                )
                continue
            if kind not in reports.supported_kinds():
                problems.append(
                    f"variant {name!r} declares results: {kind!r}, which this harness has "
                    f"no reader for (it supports {reports.supported_kinds()}). Declare a "
                    f"supported kind, or results: none with the reason."
                )
                continue
            dump_cmd = reports.dump_command(kind, patterns)
            if not dump_cmd:
                problems.append(
                    f"HARNESS/SPEC FAULT: variant {name!r} declares results: {kind!r} but "
                    f"this instance declares no `report_discovery` patterns for it, so the "
                    f"harness does not know where that ecosystem writes its reports. "
                    f"Declare them in the spec; an undeclared convention is a missing "
                    f"rule, not a permissive one."
                )
                continue
            # (the marker was dropped BEFORE the check ran, above)
            # THE DUMP'S OWN EXIT CODE IS NOT NOISE. A dump that could not run -- dead
            # sandbox, a `find` that errored -- produces empty text, which must not be reported
            # as "your check produced no report": a harness fault charged to the candidate.
            dump_code, dump = run_in(sandbox, f"cd {fresh} && {dump_cmd}", 120)
            if dump_code not in (0, 1) and not str(dump or "").strip():
                problems.append(
                    f"HARNESS ERROR: variant {name!r}: the command that collects test "
                    f"reports could not run in this sandbox (exit {dump_code}). Nothing "
                    f"was read, so this says nothing about your check. If this repeats "
                    f"while the sandbox is alive, book it with worklist_drop(blocked=true) "
                    f"naming the dump as the fault -- do not change evidence that was fine."
                )
                continue
            results = reports.parse_report(kind, dump or "")
            observations.append(
                {
                    "variant": name,
                    "expect": "fail" if want_fail else "pass",
                    "exit": int(code),
                    "test_results": (results.__dict__ if results else None),
                }
            )
            if results is None and str(dump or "").strip():
                # FOUND BUT UNPARSEABLE IS NOT "NO REPORT": report files existed, the dump
                # collected them, the parser returned None -- and a "produced no report" message
                # would send the agent chasing a report location that was never the problem.
                problems.append(
                    f"variant {name!r}: report files WERE found under {fresh} but their "
                    f"content did not parse as {kind} (first 200 chars: "
                    f"{str(dump).strip()[:2000]!r}). If your runner emits a nonstandard "
                    f"format, emit standard junit XML (a <testsuite> root with <testcase> "
                    f"children) into the reports directory instead."
                )
                continue
            if results is None or results.tests == 0:
                # SHOW WHAT THE CHECK SAID. Without it this refusal cannot be acted on: "no
                # report" reads the same whether the build failed, the tests never ran, or the
                # reports landed outside the directory the harness searched -- three different
                # fixes.
                # AND STATE THE RULE THAT EXCLUDED: report files may exist but be older than
                # the marker, and nothing would say so. List what exists-but-did-not-qualify,
                # with mtimes.
                _, stale = run_in(sandbox, f"cd {fresh} && {reports.compose_stale_command(patterns)}", 60)
                stale_note = ""
                if str(stale or "").strip():
                    stale_note = (
                        f" Report files that DO exist under {fresh} but did not qualify "
                        f"(mtime shown): {str(stale).strip()[:500]}. Only files NEWER "
                        f"than /tmp/.bench_check_started -- touched by the harness just "
                        f"before your check ran -- can witness this run: if these are "
                        f"yours, your runner reused old report files or the tests ran "
                        f"before this state's re-run; make the check re-run them."
                    )
                problems.append(
                    f"variant {name!r}: its check ran (exit {code}) but produced no {kind} "
                    f"report under {fresh}, so nothing was recorded that could decide "
                    f"anything -- whatever the exit code says. What the check printed: "
                    f"{informative_tail(out)} -- if the build failed, fix the build; if the "
                    f"tests never ran, fix the selector; if the runner wrote its report "
                    f"somewhere else, write it under the directory the check is given "
                    f"($1), because that is where the harness looks. THE DISCOVERY RULE "
                    f"(the framework's, not this task's): report files are collected by "
                    f"the standard report-name patterns under {fresh}, and only files "
                    f"NEWER than the marker count.{stale_note}"
                )
                continue
            if results.errors > 0:
                problems.append(
                    f"variant {name!r}: INCONCLUSIVE -- {results.errors} case(s) ended in "
                    f"an error rather than a verdict: "
                    f"{', '.join(results.error_names[:5])}. An error is not a failing "
                    f"case; fix the environment so the check genuinely runs. Do NOT edit "
                    f"the report to convert errors into failures -- the check is scanned "
                    f"for report-editing commands and a check that forges its own evidence "
                    f"is refused outright."
                )
                continue
            # THE REPORT'S POLARITY IS `expect_failures`, NOT `expect`. They are different
            # questions and a spec states both: `expect: pass|fail` is what the check SCRIPT's
            # exit code must be, `expect_failures: <int>` is how many failing tests its report
            # must show. defects4j's buggy variant declares BOTH -- expect: pass (the script
            # exits 0 when it observes the bug) and expect_failures: 1 (the report shows the
            # triggering test failing) -- and judging the report by `expect` refused every
            # correct instance with "expected a clean pass but the report shows failures".
            # THE SEAM. The merge's negative control, the packaging claim and the certification
            # sweep judge the same floor comparison. One rule, asked here, and `benchcompiler/tests/test_the_verdict_rule_is_one_rule.py` holds
            # this package's copy equal to the packager's.
            from agentcore.verdict import verdict_of, want_failures_of

            want_failures = want_failures_of(v)
            satisfied, why = verdict_of(v, int(code), dump or "")
            if not satisfied:
                problems.append(f"variant {name!r}: {why}")
                continue
            # A REPORT-JUDGED VARIANT MUST ALSO BE SHOWN ABLE TO FAIL. The
            # probe below this branch replaces built EXECUTABLES, which is meaningless for a
            # Maven or Gradle project, so this branch has its own probe. What must break in
            # those tasks is the FIX, which is task content, so the spec declares it:
            # negative_control.break_cmd names the field holding the command. Running it here
            # also catches the corruption_cmd collision at construction: a command that
            # perturbs an INPUT (what the lifecycle teaches) does not flip a test verdict.
            broke, why = _declared_break_probe(
                inst_doc,
                fields,
                sandbox,
                fresh,
                name,
                check_script,
                probe_out,
                cmd,
                timeout,
                run_in,
                kind,
                want_failures,
                run_report=True,
            )
            observations[-1]["break_probe"] = why
            if broke is False:
                problems.append(
                    f"variant {name!r}: after the declared break ({why}) its report still "
                    f"reads the same verdict, so this check cannot fail -- and a check that "
                    f"cannot fail is `return True`. Either the command recorded there does "
                    f"not break what the test observes (a probe that perturbs an INPUT is "
                    f"not a break of the ARTIFACT), or the check is not reading the run it "
                    f"claims to."
                )
                continue
            # UNPROBED SAYS WHY. The probe returns its reason, so a candidate whose break
            # does not run -- exit 2, sed can't read the path -- is told why rather than only
            # the word. It still does not refuse (some task families legitimately
            # cannot be probed here); it states what it already knows.
            if broke:
                label = results.describe()
            elif str(why).startswith(NOT_NOMINATED):
                label = f"{results.describe()}, {why}"
            else:
                label = f"{results.describe()}, {UNPROBED} -- {why}"
            proven.append(f"{name}({label})")
            continue

        observations.append({"variant": name, "expect": "fail" if want_fail else "pass", "exit": int(code)})
        # AN INFRA EXIT IS NOT A VERDICT. 124/125 is the bound killing the
        # command, 126/127 the shell failing to run it at all, and the sandbox marker means
        # the exec never reached the container. None of them observed the variant -- and
        # for an `expect: fail` variant they would read as "fails as required", so an
        # instance could deliver on an oracle that was never executed. The report-judged
        # branch above already refuses its equivalent (errored cases); same rule here.
        infra = ""
        if int(code) in (124, 125):
            infra = f"the {timeout}s bound killed it (exit {code})"
        elif int(code) in (126, 127):
            infra = f"the shell could not run it at all (exit {code}: command not found or not executable)"
        elif int(code) == SANDBOX_UNREACHABLE:
            infra = f"the sandbox was unreachable (exit {code})"
        if infra:
            problems.append(
                f"variant {name!r}: its check produced no verdict -- {infra}, so nothing "
                f"was observed in either direction. What it printed: "
                f"{informative_tail(out)}. Fix the command or toolchain (or re-create the "
                f"sandbox) and re-run this state; an unexecuted check cannot demonstrate "
                f"anything, least of all the failure this variant exists to show."
            )
            continue
        if want_fail and code == 0:
            problems.append(
                f"variant {name!r}: its check was expected to FAIL (that is what this "
                f"variant demonstrates) but exited 0. The usual cause is that the check "
                f"ran zero cases because this variant is missing the artefact the check "
                f"needs -- whatever this benchmark's own instruction says must be placed "
                f"here. A check that observes nothing cannot demonstrate anything."
            )
            continue
        if not want_fail and code != 0:
            problems.append(
                f"variant {name!r}: its check was expected to pass but exited {code}. "
                f"What failed: {informative_tail(out)}. If the failure is the CHECK's "
                f"toolchain rather than the variant, write {root}/check_{name}.sh (target "
                f"dir as $1) invoking the toolchain that works -- it takes precedence over "
                f"the contract's default command."
            )
            continue
        # A CHECK THAT PASSES IS NOT YET AN ORACLE: a check that measures, emits BENCH_OBS
        # lines and exits 0 would also pass with `cp /bin/true <binary>`, unless it is
        # required to be able to FAIL. This is the harness's own break, needing no declared
        # field (recorded break fields cannot be trusted): replace
        # every executable the setup just built with a no-op and run the same check again;
        # it must now refuse. The fresh copy is scratch, so nothing is restored. Script
        # suffixes are spared: an interpreter running ELF bytes fails for a reason that
        # says nothing about whether the check compares anything.
        if not want_fail:
            probe_find = executables_find(fresh)
            _, listing = run_in(sandbox, probe_listing_cmd(probe_find), 60)
            targets = ordered_probe_targets(listing)
            if not targets:
                # `-printf` is GNU find. Where it is absent the timestamped pipeline emits
                # nothing, and an empty target list does not fail the probe -- it SKIPS it,
                # which is the worst outcome available: the gate that catches a check unable
                # to fail would quietly stop running. Fall back to the plain listing.
                _, listing = run_in(sandbox, f"{probe_find} | head -20", 60)
                targets = ordered_probe_targets(listing)
            if targets:
                run_in(sandbox, noop_sweep_cmd(probe_find), 120)
                # DID THE SABOTAGE LAND? A sweep of `-exec cp /bin/true {} +` with more than
                # one match hands cp several targets and no directory, errors out, and replaces
                # NOTHING -- the check then honestly passes its unbroken binaries and the gate
                # would refuse the candidate for the harness's own miss, which would silently
                # shape the population toward single-binary programs. A saboteur that failed to sabotage is a HARNESS fault and
                # must say so; it may never count against the candidate.
                import shlex as _shlex

                vcode, _ = run_in(sandbox, f"cmp -s /bin/true {_shlex.quote(targets[0])}", 60)
                if vcode != 0:
                    problems.append(
                        f"variant {name!r}: HARNESS ERROR -- the no-op sweep did not "
                        f"actually replace {targets[0]} (cmp exit {vcode}), so nothing "
                        f"was broken and this probe proves nothing about the check "
                        f"either way. This is the harness's fault, not the "
                        f"candidate's; do not book the candidate on this refusal."
                    )
                    continue
                if "CHECK_SCRIPT_PRESENT" in str(probe_out or ""):
                    bcode, bout = run_in(sandbox, f"cd {fresh} && bash {check_script} {fresh}", timeout)
                else:
                    bcode, bout = run_in(sandbox, f"cd {fresh} && {cmd}", timeout)
                observations[-1]["break_probe_exit"] = int(bcode)
                if bcode == 0:
                    # THE REFUSAL QUOTES WHAT IT DID AND WHAT IT SAW. A refusal that asserts
                    # "every built executable" and names none cannot be acted on, and an agent
                    # could conclude the harness contradicts its tests (see
                    # tests/test_the_noop_probe_refusal_carries_its_evidence.py). The
                    # hypothesis it cannot otherwise test is real and common: a check script that
                    # rebuilds from source inside itself overwrites the no-op and legitimately
                    # passes. Naming the paths lets them see that; comparing the output before
                    # and after tells them which of the two fixes applies.
                    # The listing is `head -20`, so 20 entries means "at least 20".
                    n = f"{len(targets)}+" if len(targets) >= 20 else str(len(targets))
                    shown = ", ".join(targets[:5])
                    if len(targets) > 5:
                        shown += f", +{len(targets) - 5} more"
                    same = "byte-identical to" if str(bout or "") == str(out or "") else "different from"
                    ran = (
                        check_script
                        if "CHECK_SCRIPT_PRESENT" in str(probe_out or "")
                        else f"the contract's `{cmd}` inside {fresh} (no {check_script} exists to take precedence)"
                    )
                    problems.append(
                        f"variant {name!r}: I replaced {n} built executable(s) under {fresh} "
                        f"with /bin/true ({shown}) and re-ran {ran}: it STILL exited 0 -- a check "
                        f"that cannot fail is `return True`: it certifies a solver that did "
                        f"nothing. What the check printed with those binaries broken, "
                        f"{same} what it printed before: {informative_tail(bout)}. Which fix "
                        f"you need depends on that output: if check_{name}.sh REBUILT those "
                        f"paths from source it never ran a no-op at all -- make it run what "
                        f"setup_{name}.sh left in $1 instead of building its own; if it ran "
                        f"them and passed anyway, make it COMPARE what the artifact produced "
                        f"against the recorded reference (e.g. the stored output digest, per "
                        f"input) and exit nonzero on mismatch. Emitting runtime and digest "
                        f"observables is measurement, not a verdict."
                    )
                    continue
            else:
                # UNPROBED IS NOT PROVEN, AND MUST NOT READ AS PROVEN: the pass string must
                # differ from a probed variant's so a reader can tell them apart.
                #
                # It does NOT refuse: this probe replaces compiled executables, and an
                # interpreted project (e.g. Python repos) legitimately has none.
                # Refusing here would make a whole task family unbuildable to close a hole
                # that the merge's declared negative control -- which breaks the artifact
                # in the task's own terms, e.g. reverting the fix -- already covers.
                # So the verdict carries the word: the label says UNPROBED, the evidence
                # records it, and enforcement stays where a task-appropriate break exists.
                observations[-1]["break_probe"] = (
                    "no compiled executable under the fresh copy: this variant's check was "
                    "NOT shown able to fail here; the packaging negative control is the "
                    "gate that must prove it"
                )
                proven.append(f"{name}(passes, {UNPROBED}: no binary to break)")
                continue
        proven.append(f"{name}({'fails as required' if want_fail else 'passes'})")

    # THE MEASUREMENT COMMAND IS READ THE WAY THE CONSUMER WILL READ IT. The consumer
    # needs `variant_run_cmds` as a mapping keyed by variant name; a plain string, a JSON
    # list or a mapping keyed by input filename would each be unusable, and nothing else
    # reads the field before packaging, when the agent can no longer fix it.
    raw = fields.get("variant_run_cmds")
    # Absence is the same defect as a wrong shape, and the cheaper mistake for an agent to
    # make: the packaged benchmark still cannot measure anything.
    if variants and ((getattr(ctx, "instance", {}) or {}).get("measure")):
        problem = variant_run_cmds_problem(raw, [str(v["name"]) for v in variants])
        if problem:
            problems.append(problem)
        else:
            problems.extend(_measurement_runs(raw, variants, root, sandbox, fields, run_in, timeout))

    if problems:
        # THE CHECK'S PATH IS PART OF THE LAYOUT. Naming only the setup, materialized and
        # re-run paths, with a bare check name, lets an agent write its check where nothing
        # runs it (e.g. {root}/default/check.sh) while {root}/check_default.sh is what is
        # executed. Name what runs, and where.
        layout = "; ".join(
            f"{str(v['name'])}: setup {root}/setup_{v['name']}.sh, materialized "
            f"{root}/{v['name']}/, re-run into {root}/_verify_{v['name']}, check "
            f"{root}/check_{v['name']}.sh if present (else the contract's "
            f"`{str(v.get('check_cmd') or 'true')}` inside the re-run directory)"
            for v in variants
        )
        head = (
            f"{len(problems)} problem(s) with the declared variants -- fix ALL of "
            f"them before re-recording. THE LAYOUT I CHECK: {layout}. "
        )
        return False, head + " ".join(f"[{i + 1}] {p}" for i, p in enumerate(problems))
    return True, f"all variants rebuilt fresh from their scripts and checked: {', '.join(proven)}"
