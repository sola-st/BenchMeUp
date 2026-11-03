"""Data shapes for BenchSpec's front-end: the raw (researcher-written) spec,
the canonical spec it formalizes into, and the typed diagnostics returned on
rejection. See spec/benchspec-v0.md for the design this implements.

Requires: pip install pyyaml
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing dependency: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REQUIRED_FIELDS = ["task", "instance", "data", "correctness", "interface", "scale"]
OPTIONAL_FIELDS = ["criteria", "quality", "examples", "references", "constraints", "access"]

DIAGNOSTIC_CODES = {
    "MISSING_INFO",  # a required field is absent or empty
    "AMBIGUOUS_TERM",  # a vague term with no safe default was used
    "NON_OPERATIONAL_CORRECTNESS",  # correctness is circular / not a decision procedure
    "UNPRICEABLE_CORRECTNESS",  # correctness needs human judgment but no budget was given
    "INVALID_DELEGATION",  # DELEGATE was used on a field that can't be delegated wholesale
    # A spec that never says what a source has to be. Distinct from AMBIGUOUS_TERM, which
    # fires on a vague WORD -- silence is not a word, so a spec such as "Real, public C/C++
    # projects with a Git history" would otherwise pass.
    "MISSING_ADMISSIBILITY",
    # A ratio clause over a field no executed step derives: if the fields are agent-typed the
    # ratio is arithmetic over invented numbers. A compared number must come from an execution
    # the harness ran (a corroboration run step's derive), never from the model's keyboard.
    "UNEXECUTED_COMPARISON",
    # A declared gate the task-blind engine cannot read: an entry-contract clause
    # outside the grammar, a corroboration step of no known kind, a sampling block
    # missing its required keys. Distinct from MISSING_INFO -- the block is there,
    # but enforcing it as written is impossible, and an unreadable rule enforced as
    # a silent pass is how a declared control stops existing.
    "UNREADABLE_CONTRACT",
    "UNMEASURABLE_ADMISSIBILITY",  # criteria were stated but no program could check them
    # A lead-testable viability criterion the spec states (a runtime floor, a
    # configuration-separation ratio) with no cross-field ratio clause projecting
    # it into sampling -- the model juggles one projection out when adding the
    # other; this diagnostic holds the conjunction.
    "MISSING_VIABILITY_PROJECTION",
    # The same gap at the OTHER end of the pipeline. `admissibility` governs what may go in;
    # these govern whether what comes out can be verified. A spec that declares its variant
    # contract gets an interface rendered mechanically; otherwise an LLM must invent one,
    # which the certifier may not be able to drive.
    "MISSING_VARIANTS",  # the spec never says how a finished instance is checked
    "UNCHECKABLE_VARIANTS",  # variants were declared but not in runnable terms
    # A check that navigates away from the directory the harness rebuilt observes
    # something other than the rebuild -- the one thing the verification exists to test.
    "CHECK_LEAVES_VARIANT",
    # The spec declares an oracle pattern the registry does not name, so nothing can
    # decide whether a variant contract applies to it.
    "UNCLASSIFIED_ORACLE",
    # Every variant must say what evidence its check is judged on; an exit code is not
    # a verdict.
    "UNDECLARED_CHECK_EVIDENCE",
    # criteria that merely restate the source filters leave the benchmark's defining
    # property unenforced, because criteria are what lifecycle states must enforce.
    "CRITERIA_ARE_SOURCE_FILTERS",
    # Nothing says how a solver hands the benchmark its answer, so nothing can check that the
    # shipped artifact accepts one (e.g. a task "find a flag configuration" needs a verb that
    # accepts a flag string).
    "NO_SOLVER_INPUT",
    # Nothing says how to break an instance on purpose, so nothing can prove a shipped check
    # is able to fail (e.g. `cp /bin/true <binary>` must not pass the default check).
    "NO_NEGATIVE_CONTROL",
}

# Stage 2 (planner) diagnostic codes -- distinct namespace from stage 1's,
# since a plan can fail for reasons a canonical spec alone can't reveal.
# ── The declared-gate grammar, MIRRORED from the task-blind engine ────────────
#
# The engine that executes these declarations is agentcore.agent.field_contract;
# the two packages do not import each other, so the forms are restated here and a
# test (test_formalizer_emits_the_declared_gates.py) holds this mirror equal to
# the engine -- the same held-equal pattern as report discovery. The formalizer's
# prompt is BUILT from these lists and check_canonical validates against them, so
# the rule the model reads, the rule the validator applies and the rule the
# engine executes are one list.

#: `require` forms an entry-contract clause may use, exactly as the engine reads
#: them. Beyond these: "matches <regex>", "is a command invoking {field}", and the
#: admissibility five-form grammar (>= N / within N / is true / one of [...]).
ENTRY_REQUIRE_FORMS = (
    "is non-empty",
    "is a clone url",
    "is a full http url",
    "is a git sha",
    "is an atomic name",
    "is a qualified name",
    "is a relative path to one file",
    "is a plain absolute path",
    "is a plain container name",
    "is a positive integer",
    "is a number",
    "is affirmed",
    "is a hex digest",
    "is a hex digest of non-empty output",
    "is at least the spec-stated minimum",
    "is not a pull request url",
    "resolves to a tracker item that is not a pull request",
    "is distinct across leads",
    # parametric: <N> and the peer field name are the spec's to choose
    "is at least 10 times field scale_probe_small_ms",
)

#: Step kinds a corroboration contract (or a collector state contract) may declare;
#: one of these keys must be present in every step object.
CORROBORATION_STEP_KINDS = (
    "field",
    "run",
    "tracker",
    "marker_search",
    "resolve_shas",
    "report",
    "diff_check",
    "shared",
    "digests",
    "stated_reference",
    "measurements",
    "comparisons",
    "variant_cmds",
    "exclusions_on",
    # THE HARNESS'S CLOCK: {"timed": "{cmd_field}", "cwd": "{path_field}",
    # "repetitions": N, "derive": <field>, "timeout": <s>} runs a recorded command N times,
    # the wall-clock taken inside the sandbox, and derives the integer median in ms. The
    # only step kind whose derived number is the harness's rather than the program's or the
    # agent's -- and so the only one a ratio clause may compare (see the check below).
    "timed",
)

#: `require` forms an executed `run` step may use (prefix-matched where the form
#: embeds a field or a list).
RUN_REQUIRE_FORMS = (
    "exit zero",
    "exit zero and some output line",
    "output is a git sha",
)
#: (`_run_require_readable` asks `agentcore.agent.field_contract`, not a local list; the FORMS
#: tuple above stays only because the formalizer PROMPT teaches those spellings.)

#: The keys a canonical `sampling` block must declare (the sampler generator
#: refuses without them; refusing HERE puts the message in the formalize loop,
#: where a retry round can fix it).
SAMPLING_REQUIRED_KEYS = (
    "pool_requires",
    "spread_by",
    "pool_exclusion_fields",
    "lead_requires",
    "lead_key",
    "normalize",
    "caps",
    "search_hints",
)


def _entry_require_readable(require: str) -> bool:
    """Would the task-blind engine read this require string? Asked of the ENGINE
    itself, never a mirror: an unreadable rule comes back with the grammar's own
    "cannot read" refusal, while a readable rule either passes or refuses on the
    probe value's content. `benchcompiler` depends on `agentcore` by design (the
    product layer on the runtime, same as the admissibility branch above).
    ENTRY_REQUIRE_FORMS above is prompt TEXT built from the same grammar, held
    equal to the engine by test_formalizer_emits_the_declared_gates."""
    from agentcore.agent.field_contract import evaluate_clause

    r = " ".join(str(require or "").strip().split())
    if not r:
        return False
    _, why = evaluate_clause({"field": "probe_field", "require": r}, {"probe_field": "probe-value"})
    return "cannot read" not in str(why)


def _run_require_readable(require: str) -> bool:
    """Asked of the ENGINE, never of a mirror -- the same rule the entry-clause check above
    follows. A local list of forms would drift from the executor: `_run_step` reads
    `output <value-form>` by handing the output line to the field grammar, and a stale list
    would refuse contracts the engine can run."""
    from agentcore.agent.field_contract import run_require_readable

    return run_require_readable(require)


def _viability_projection_diagnostics(get) -> list:
    """A sampler that screens on runtime AT ALL must screen on every pair the criteria state.

    The model juggles: asked for two probes it may emit one, dropping the other (a
    runtime-growth clause without a separation pair, or the reverse). Prompt wording cannot hold a
    conjunction; a diagnostic the retry loop feeds back can. Task-blind: both
    triggers read the spec's OWN criteria text, and both demands are grammar
    shapes, not field names.

    The demand is not made on the criteria alone: otherwise every spec whose criteria mention
    a runtime floor and a speedup would oblige a sampler to clone, build twice, generate inputs
    and time them before a lead exists. WHERE viability is proven -- at mining, or at construction -- is the
    researcher's decision, and the harness must not make it: the criteria describe the
    finished INSTANCE, which the constructor builds. So the demand is conditional on the
    canonical having opted into runtime screening at all."""
    import re as _re

    out: list = []
    sampling = get("sampling", None) or {}
    if not isinstance(sampling, dict) or not sampling:
        return out
    crit_text = " ".join(str(c) for c in (get("criteria", None) or [])).lower()
    clauses = " ".join(
        str(cl.get("require") or "") for cl in (get("entry_contract", None) or []) if isinstance(cl, dict)
    )
    ratio_clauses = clauses.count("times field")

    states_floor = bool(_re.search(r"at least \d+(\.\d+)?\s*(second|ms|millisecond)", crit_text))
    states_compare = bool(_re.search(r"(faster than|speedup|separat\w+ by|ratio|time to beat)", crit_text))
    wanted = int(states_floor) + int(states_compare)
    # ZERO is a legitimate division of labour: the sampler screens on what a repository
    # says about itself and the constructor proves viability by building and measuring.
    # ONE is not: the half that was kept is arbitrary, and that is the juggling this
    # diagnostic exists to stop.
    if wanted and 0 < ratio_clauses < wanted:
        missing = []
        if states_floor:
            missing.append("a runtime-growth pair (small vs scaled input, 'is at least N times field <small>')")
        if states_compare:
            missing.append(
                "a configuration-separation pair (baseline vs optimized "
                "on the same input, 'is at least N times field "
                "<optimized>')"
            )
        out.append(
            Diagnostic(
                code="MISSING_VIABILITY_PROJECTION",
                field="sampling",
                detail=(
                    f"the criteria state {wanted} lead-testable viability "
                    f"criterion(s) but the entry contract carries only "
                    f"{ratio_clauses} cross-field ratio clause(s). Required, EACH "
                    f"with its evidence fields in lead_requires: " + "; ".join(missing)
                ),
                fix_hint=(
                    "project BOTH: keep every probe pair already present and ADD "
                    "the missing one -- emitting one at the expense of the other "
                    "re-raises this same diagnostic."
                ),
            )
        )
    return out


def _contract_diagnostics(get) -> list:
    """Diagnostics for the declared-gate blocks: validate-if-present, plus the couplings the downstream
    fail-closed gates would otherwise report only at scaffold or mining time --
    a sampling block needs the contracts, the entry contract may not demand a
    field the lead never collects, and a declared report kind needs its
    discovery patterns."""
    out: list = []

    entry = get("entry_contract", None) or []
    for i, cl in enumerate(entry):
        if not isinstance(cl, dict) or not str(cl.get("field") or "").strip():
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="entry_contract",
                    detail=f"clause #{i + 1} is not an object naming its `field`.",
                    fix_hint='each clause is {"field": ..., "require": ...} with optional '
                    "`teach`, `optional`, `resolve`.",
                )
            )
            continue
        # A spec-stated bound must say where the spec states it. The clause
        # reads its number back from the spec's text through `spec.pattern`; without one
        # the engine finds no bound and passes ("the spec states no bound, so none is
        # enforced"): a floor projected as this clause with no `spec` block is readable and
        # accepted, yet every lead would clear it.
        _req_norm = " ".join(str(cl.get("require") or "").split()).lower()
        if _req_norm == "is at least the spec-stated minimum":
            _spec = cl.get("spec") if isinstance(cl.get("spec"), dict) else {}
            _pat = str(_spec.get("pattern") or "")
            _groups = 0
            if _pat:
                try:
                    import re as _re_pat

                    _groups = _re_pat.compile(_pat).groups
                except _re_pat.error:
                    _groups = 0
            if _groups < 1:
                out.append(
                    Diagnostic(
                        code="UNREADABLE_CONTRACT",
                        field="entry_contract",
                        detail=(
                            f"clause for {cl.get('field')!r} requires 'is at least the "
                            f"spec-stated minimum' but declares no `spec.pattern` that "
                            f"captures the number from the spec's text"
                            f"{' (the pattern has no capture group)' if _pat else ''}: "
                            f"the engine would find no bound and enforce nothing."
                        ),
                        fix_hint=(
                            "add `spec: {pattern: <regex with ONE capture group over the "
                            "spec's own sentence, e.g. 'at least (\\\\d+) second'>, "
                            "default: <number>, scale: <multiplier from the sentence's "
                            "unit to the field's, e.g. 1000 for seconds -> ms>}` to the "
                            "clause, so the bound lives in the spec's text and nowhere "
                            "else."
                        ),
                    )
                )
                continue
            # THE PATTERN MUST READ, FROM THIS CANONICAL'S OWN TEXT, THE NUMBER THE DEFAULT
            # MEANS. Draw 6 declared pattern "at least (\d+) second" with default 1000 on a
            # millisecond field: the pattern reads 1, the default means 1000, the runtime
            # enforces 1 ms. Read the bound the way the engine will (pattern x scale over
            # the criteria/task text) and hold it to the default -- no unit is known here,
            # and none needs to be: two declared sources of one number must agree.
            _text = " ".join(str(x) for x in (get("criteria", None) or [])) + " " + str(get("task", "") or "")
            _m = None
            try:
                _m = _re_pat.search(_pat, _text, _re_pat.I)
            except _re_pat.error:
                _m = None
            _read = None
            if _m:
                try:
                    _read = float(_m.group(1)) * float(_spec.get("scale") or 1)
                except (TypeError, ValueError):
                    _read = None
            _default = _spec.get("default")
            try:
                _default = float(_default) if _default not in (None, "") else None
            except (TypeError, ValueError):
                _default = None
            if _read is None:
                out.append(
                    Diagnostic(
                        code="UNREADABLE_CONTRACT",
                        field="entry_contract",
                        detail=(
                            f"clause for {cl.get('field')!r} declares spec.pattern "
                            f"{_pat[:60]!r}, which captures no number from this spec's own "
                            f"criteria/task text: the 'spec-stated minimum' would never be "
                            f"read and only the default would ever run."
                        ),
                        fix_hint=(
                            "make the pattern match the sentence in the criteria that "
                            "states the bound, capturing its number, and declare "
                            "`scale` if the sentence's unit differs from the field's."
                        ),
                    )
                )
                continue
            if _default is not None and _default > 0 and _read > 0 and max(_read / _default, _default / _read) >= 10:
                out.append(
                    Diagnostic(
                        code="UNREADABLE_CONTRACT",
                        field="entry_contract",
                        detail=(
                            f"clause for {cl.get('field')!r} ('is at least the spec-stated "
                            f"minimum'): spec.pattern reads "
                            f"{_read:g} from the spec's text while spec.default says "
                            f"{_default:g}. Two declared sources of one bound disagree by "
                            f"{max(_read / _default, _default / _read):g}x -- the sentence "
                            f"states the bound in a different unit than the field records "
                            f"(seconds vs ms, say), and the runtime would enforce "
                            f"{_read:g}."
                        ),
                        fix_hint=(
                            "declare `scale` on the clause's `spec` (the multiplier from "
                            "the sentence's unit to the field's unit, e.g. 1000 for "
                            "seconds -> ms) so pattern x scale equals what the default "
                            "means, or fix the default."
                        ),
                    )
                )
                continue
        if not _entry_require_readable(cl.get("require")):
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="entry_contract",
                    detail=(
                        f"clause for {cl.get('field')!r} states require="
                        f"{str(cl.get('require'))[:60]!r}, which the task-blind engine "
                        f"cannot read -- an unreadable rule refuses every lead at "
                        f"runtime."
                    ),
                    fix_hint=(
                        "use one of the exact forms (e.g. 'is a clone url', "
                        "'is a git sha', 'is non-empty'), 'matches <regex>', "
                        "'is a command invoking {field}', or the admissibility "
                        "five-form grammar."
                    ),
                )
            )

    corro = get("corroboration_contract", None) or []
    for i, st in enumerate(corro):
        if not isinstance(st, dict):
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="corroboration_contract",
                    detail=f"step #{i + 1} is not an object.",
                    fix_hint="each step is an object declaring exactly one step kind.",
                )
            )
            continue
        kinds = [k for k in CORROBORATION_STEP_KINDS if k in st]
        if len(kinds) != 1:
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="corroboration_contract",
                    detail=(
                        f"step #{i + 1} declares "
                        f"{kinds or 'no known step kind'}; exactly one of "
                        f"{list(CORROBORATION_STEP_KINDS)} must be present."
                    ),
                    fix_hint="one step, one kind.",
                )
            )
            continue
        if "run" in st:
            import re as _re

            cmd = str(st.get("run") or "")
            # THE EXECUTION MODEL IS A FACT ABOUT THE ENGINE: a run step executes
            # in the harness's own working directory, never inside any repository.
            # A git command with no recorded-clone anchor hits "not a git
            # repository" on EVERY candidate, so all leads would be blocked.
            if (
                _re.search(r"(^|[;&|]\s*|\s)git\s", cmd)
                and not _re.search(r"git\s+-C\s+\{[a-z_][a-z0-9_]*\}", cmd)
                and not _re.search(r"(^|[;&|]\s*)cd\s+\{[a-z_][a-z0-9_]*\}", cmd)
            ):
                out.append(
                    Diagnostic(
                        code="UNREADABLE_CONTRACT",
                        field="corroboration_contract",
                        detail=(
                            f"run step #{i + 1} invokes git with no repository anchor "
                            f"({cmd[:70]!r}). Run steps execute in the harness's own "
                            f"working directory, never inside a clone, so this fails "
                            f"with 'not a git repository' on every candidate."
                        ),
                        fix_hint=(
                            "anchor every repository command to a field the miner "
                            'records: a `{"field": "clone_path", "require": '
                            '"is non-empty"}` clause first, then '
                            "`git -C {clone_path} ...` in each run step."
                        ),
                    )
                )
        if "run" in st:
            # Shell quoting is the engine's, never the template's. The engine
            # shell-quotes every substituted value, so a {placeholder} inside the template's
            # own quotes is nested quoting that cannot come out right, and a {placeholder}
            # standing as a whole command runs the recorded string as ONE word. A
            # recorded command runs through `sh -c {field}`.
            import re as _re_q

            cmd_q = _re_q.sub(r"\^\{(commit|tree|tag|blob|object)\}", "", str(st.get("run") or ""))
            quoted_ph = []
            for m in _re_q.finditer(r"'[^']*'|\"[^\"]*\"", cmd_q):
                quoted_ph.extend(_re_q.findall(r"\{([a-z][a-z0-9_]*)\}", m.group(0)))
            if quoted_ph:
                _head = " ".join(cmd_q.split())[:70]
                out.append(
                    Diagnostic(
                        code="UNREADABLE_CONTRACT",
                        field="corroboration_contract",
                        detail=(
                            f"run step #{i + 1} (`{_head}`) puts {{{quoted_ph[0]}}} inside "
                            f"quotes. The engine shell-quotes every substituted value itself, "
                            f"so a placeholder inside the template's own quotes is nested "
                            f"quoting: a recorded value containing a quote breaks the command "
                            f"and it prints nothing."
                        ),
                        fix_hint=(
                            "never quote a placeholder; write `cd {clone_path} && sh -c "
                            "{cmd_field}` -- the engine quotes each value, and `sh -c` hands "
                            "a recorded command to a shell whole."
                        ),
                    )
                )
            else:
                bare = _re_q.findall(r"(?:^|&&|\|\||;|\|)\s*\{([a-z][a-z0-9_]*)\}\s*(?:$|&&|\|\||;|\||>)", cmd_q)
                if bare:
                    _head = " ".join(cmd_q.split())[:70]
                    out.append(
                        Diagnostic(
                            code="UNREADABLE_CONTRACT",
                            field="corroboration_contract",
                            detail=(
                                f"run step #{i + 1} (`{_head}`) runs {{{bare[0]}}} as a whole "
                                f"command. The engine substitutes it as ONE shell-quoted word, "
                                f"so a recorded command line ('time (./prog < in)') is looked "
                                f"up as a program of that name and is not found."
                            ),
                            fix_hint=(
                                f"run a recorded command through a shell: `sh -c {{{bare[0]}}}` "
                                f"(e.g. `cd {{clone_path}} && sh -c {{{bare[0]}}}`), and never "
                                f"quote a placeholder."
                            ),
                        )
                    )
        if "timed" in st:
            # the harness's clock: it must say what it derives, and its command must be a
            # placeholder handed whole (the wrapper quotes it), never inline shell text
            if not str(st.get("derive") or "").strip():
                out.append(
                    Diagnostic(
                        code="UNREADABLE_CONTRACT",
                        field="corroboration_contract",
                        detail=(
                            f"timed step #{i + 1} derives nothing: a `timed` step exists "
                            f"to produce a harness-measured number."
                        ),
                        fix_hint="add `derive: <field>` naming the median-in-ms field it fills.",
                    )
                )
            import re as _re_t

            _t = str(st.get("timed") or "").strip()
            if not _re_t.fullmatch(r"\{[a-z][a-z0-9_]*\}", _t):
                out.append(
                    Diagnostic(
                        code="UNREADABLE_CONTRACT",
                        field="corroboration_contract",
                        detail=(
                            f"timed step #{i + 1} times {_t[:60]!r}; `timed` names ONE "
                            f"recorded command field, as `{{field}}`, which the harness "
                            f"hands to a shell whole and times around."
                        ),
                        fix_hint=(
                            '`{"timed": "{run_scaled_cmd}", "cwd": "{clone_path}", '
                            '"repetitions": 3, "derive": "probe_baseline_ms"}`'
                        ),
                    )
                )
        if "run" in st and not _run_require_readable(st.get("require")):
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="corroboration_contract",
                    detail=(
                        f"run step #{i + 1} states require="
                        f"{str(st.get('require'))[:60]!r}, which the engine cannot "
                        f"read."
                    ),
                    fix_hint=(
                        "one of: 'exit zero', 'exit zero and some output line', "
                        "'output is a git sha', 'output lines are exactly "
                        "[{field}]', 'some output line ends with one of [...]', "
                        "'output matches <regex>', 'last output line is the commit "
                        "{field}'."
                    ),
                )
            )

    # every placeholder a run template names must be a field something provides:
    # a lead field, a corroboration field clause, or an earlier step's derive/save.
    # (git peel suffixes ^{commit} etc. are notation the engine resolves, not
    # placeholders.) r79s2 blocked all 73 of its leads on `{commit}` -- an
    # unprovided placeholder refuses every candidate at runtime, so it refuses
    # HERE, inside the formalize retry loop, instead.
    import re as _re2

    sampling_peek = get("sampling", None) or {}
    provided = (
        {str(f) for f in (sampling_peek.get("lead_requires") or [])} if isinstance(sampling_peek, dict) else set()
    )
    for _pos, st in enumerate(corro, start=1):
        if not isinstance(st, dict):
            continue
        if "field" in st and "run" not in st:
            provided.add(str(st.get("field")))
        for k in ("derive", "save"):
            if st.get(k):
                provided.add(str(st[k]))
        if isinstance(st.get("marker_search"), dict) and st["marker_search"].get("derive"):
            provided.add(str(st["marker_search"]["derive"]))
        if isinstance(st.get("policy"), dict) and st["policy"].get("kind_into"):
            provided.add(str(st["policy"]["kind_into"]))
        if "run" in st or "timed" in st:
            _src = str(st["run"]) if "run" in st else str(st.get("timed") or "") + " " + str(st.get("cwd") or "")
            tmpl = _re2.sub(r"\^\{(commit|tree|tag|blob|object)\}", "", _src)
            unknown = sorted({m for m in _re2.findall(r"\{([a-z][a-z0-9_]*)\}", tmpl)} - provided)
            if unknown:
                # Named, not counted: each offending step gets its own diagnostic naming all of
                # its unknown placeholders, so the model can tell several mistakes from one and
                # repair each step in one round.
                _head = " ".join(_src.split())[:60]
                _names = ", ".join("{" + u + "}" for u in unknown)
                out.append(
                    Diagnostic(
                        code="UNREADABLE_CONTRACT",
                        field="corroboration_contract",
                        detail=(
                            f"run step {_pos} (`{_head}`) names {_names}, which no "
                            f"lead field, field clause or earlier derive provides -- "
                            f"an unprovided placeholder refuses every candidate at "
                            f"runtime."
                        ),
                        fix_hint=(
                            "every {placeholder} must be a sampling.lead_requires "
                            "field, a corroboration field clause, or a preceding "
                            "step's derive/save. Git peel suffixes like ^{commit} "
                            "are fine -- the engine reads them as notation."
                        ),
                    )
                )

    # A COMPARED NUMBER IS AN EXECUTED NUMBER. A ratio clause over agent-typed values is
    # arithmetic over invented numbers, and a model-written canonical can contain such clauses
    # with nothing deriving the compared fields. So the rule lives here, task-blind: every field a
    # cross-field ratio clause names must be provided by a run step's derive/save, i.e. by
    # an execution the harness performs. A field clause is a typed value and does not count.
    # "Derived by an executed step" is not enough: a recorded command that prints "the median",
    # or the first output line of a build-plus-run step, is still the AGENT's number. A `timed`
    # step's number is the harness's clock; only that may be compared.
    executed = set()
    for st in corro:
        if isinstance(st, dict) and "timed" in st:
            for k in ("derive", "save"):
                if st.get(k):
                    executed.add(str(st[k]))
    compared: list[str] = []
    for cl in get("entry_contract", None) or []:
        if not isinstance(cl, dict):
            continue
        req = str(cl.get("require") or "")
        others = _re2.findall(r"\btimes\s+field\s+([a-z][a-z0-9_]*)", req)
        if not others:
            continue
        for name in [str(cl.get("field") or "").strip(), *others]:
            if name and name not in compared:
                compared.append(name)
    typed = [f for f in compared if f not in executed]
    if typed:
        out.append(
            Diagnostic(
                code="UNEXECUTED_COMPARISON",
                field="corroboration_contract",
                detail=(
                    f"the entry contract compares {', '.join(typed)} by ratio, but no "
                    f"`timed` corroboration step derives "
                    f"{'them' if len(typed) > 1 else 'it'}: a compared number must be the "
                    f"HARNESS's clock. A number the miner types, or one its own recorded "
                    f"command prints, or the first output line of a run step, is the "
                    f"miner's number."
                ),
                fix_hint=(
                    "project each compared field as a `timed` step: "
                    '{"timed": "{run_scaled_cmd}", "cwd": "{clone_path}", '
                    '"repetitions": 3, "derive": "probe_baseline_ms", "timeout": '
                    "1800} -- the harness runs the recorded command that many times in the "
                    "miner's own clone, takes the wall-clock itself and derives the median "
                    "in ms. Build first with a run step that exports the configuration: "
                    '"cd {clone_path} && env CFLAGS=-O0 sh -c {build_cmd}" (then the -O3 '
                    "build before the optimized timing). The recipe and workload commands "
                    "are lead fields the miner records; the numbers never are."
                ),
            )
        )
    sampling = get("sampling", None) or {}
    if sampling and not isinstance(sampling, dict):
        out.append(
            Diagnostic(
                code="UNREADABLE_CONTRACT",
                field="sampling",
                detail="`sampling` is not an object.",
                fix_hint="an object with the sampler's declarations.",
            )
        )
        sampling = {}
    if sampling:
        missing = [k for k in SAMPLING_REQUIRED_KEYS if not sampling.get(k)]
        if missing:
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="sampling",
                    detail=f"the sampling block is missing {missing}.",
                    fix_hint=(
                        "declare them; the sampler generator refuses without them "
                        "and nothing invents a value for a spec."
                    ),
                )
            )
        if not entry or not corro:
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="sampling",
                    detail=(
                        "a sampling block is declared but "
                        + ("entry_contract" if not entry else "corroboration_contract")
                        + " is empty -- a sampler without gates cannot exist."
                    ),
                    fix_hint="emit entry_contract and corroboration_contract alongside sampling.",
                )
            )
        # normalize must cover exactly the lead_key fields: a key field with no
        # declared normal form is an identity decision nobody made, and a
        # normalize entry naming a non-key field is a declaration nothing reads
        # (both are what the lifecycle audit refuses downstream -- saying it HERE
        # puts the fix in the formalize retry loop).
        key_fields = [str(f) for f in (sampling.get("lead_key") or [])]
        norm = sampling.get("normalize") if isinstance(sampling.get("normalize"), dict) else {}
        unnormed = [f for f in key_fields if f not in norm]
        orphans = sorted(set(norm) - set(key_fields))
        if key_fields and (unnormed or orphans):
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="sampling",
                    detail=(
                        "sampling.normalize must cover exactly the lead_key fields: "
                        + (f"key field(s) {unnormed} declare no normal form; " if unnormed else "")
                        + (f"normalize names non-key field(s) {orphans}, which nothing reads" if orphans else "")
                    ).rstrip("; "),
                    fix_hint=(
                        "give every lead_key field an entry in normalize -- 'none' "
                        "when the generic treatment is right, 'git_url' for clone "
                        "URLs, 'full_sha' for a commit (the full 40-hex hash, so an "
                        "abbreviation is not a second candidate) -- and remove entries "
                        "for fields not in lead_key."
                    ),
                )
            )
        lead = [str(f) for f in (sampling.get("lead_requires") or [])]
        uncovered = sorted(
            {str(cl.get("field")) for cl in entry if isinstance(cl, dict) and cl.get("field")} - set(lead)
        )
        if lead and uncovered:
            out.append(
                Diagnostic(
                    code="UNREADABLE_CONTRACT",
                    field="sampling",
                    detail=(
                        f"entry_contract names field(s) {uncovered} that "
                        f"sampling.lead_requires does not list -- the gate would demand "
                        f"what the state never collects."
                    ),
                    fix_hint="add them to sampling.lead_requires or drop the clauses.",
                )
            )

    # a declared report kind needs its discovery patterns (the same rule
    # `bench instantiate` enforces; stating it here puts it in the retry loop)
    disc = get("report_discovery", None) or {}
    kinds_used = set()
    for v in get("artifact_variants", None) or []:
        k = str((v or {}).get("results") or "").strip().lower() if isinstance(v, dict) else ""
        if k and k != "none":
            kinds_used.add(k)
    undeclared = sorted(k for k in kinds_used if not (disc.get(k) if isinstance(disc, dict) else None))
    if undeclared:
        out.append(
            Diagnostic(
                code="UNREADABLE_CONTRACT",
                field="report_discovery",
                detail=(
                    f"the contracts declare report kind(s) {undeclared} but no "
                    f"report_discovery patterns for them -- the harness knows no "
                    f"ecosystem's report paths of its own."
                ),
                fix_hint=(
                    "declare, per kind, the find patterns the ecosystem's runner "
                    "writes reports under (e.g. junit-xml: "
                    "['*surefire-reports/TEST-*.xml', 'TEST-*.xml'])."
                ),
            )
        )
    return out


PLAN_DIAGNOSTIC_CODES = {
    "BUDGET_EXCEEDED",  # estimated cost for some resource exceeds the declared cap
    "MISSING_ACCESS",  # a needed platform/LLM credential has no `access` entry at all
    "ACCESS_BUDGET_EXCEEDED",  # a platform's own rate/cost budget (from `access`) would be exceeded
}


def load_raw_spec(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a YAML mapping (got {type(data).__name__})")
    return data


@dataclass
class Diagnostic:
    code: str
    field: str
    detail: str
    fix_hint: str

    def __str__(self) -> str:
        return f"{self.code}({self.field}): {self.detail}\n    fix: {self.fix_hint}"


@dataclass
class Assumption:
    # Defaulted so a reply SHORT of a key is recorded partially rather than raising out of
    # `formalize`. `Assumption(**a)` on a real model reply raised TypeError and the caller got
    # a traceback instead of a diagnostic. An empty `resolved_to` is then refused by
    # `check_canonical` as the silent default it is -- a typed refusal, which is the contract.
    field: str = ""
    resolved_to: str = ""
    reason: str = ""


@dataclass
class Delegation:
    # Defaulted for the same reason as Assumption above.
    field: str = ""
    term: str = ""
    resolved_to: str = ""
    rationale: str = ""


@dataclass
class CanonicalSpec:
    """The structured form a raw spec formalizes into. Field shape follows
    the split/array/enum refinement noted while building the paper-mining
    extraction schema — a canonical spec is meant to be consumed
    programmatically by later pipeline stages, not read as prose."""

    spec_id: str
    task: str
    instance_input: str
    instance_output: str
    data_source: str
    candidate_definition: str
    oracle_strategy: str  # one of registry.ORACLE_PATTERNS keys
    oracle_detail: str
    interface: str
    scale_instance_count: str
    scale_compute_budget: str
    scale_human_budget: str
    criteria: list[str] = field(default_factory=list)
    quality_gates: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    delegations: list[Delegation] = field(default_factory=list)
    # The two declarations the front end refuses a spec for omitting. Carried VERBATIM from
    # the raw spec, never from the model's paraphrase: they are structured blocks the harness
    # validates (check_admissibility / check_verifiability), and re-wording them in an LLM
    # call would reintroduce the drift the formalizer exists to prevent. Empty means "the raw
    # spec declared none" -- to_dict omits them then, because the variants gate admits when
    # none are declared and a spurious empty contract would promise a check nobody wrote.
    admissibility: list = field(default_factory=list)
    # A benchmark that mines nothing says so, and the sentence saying it travels SEPARATELY.
    # It does not ride in `admissibility` itself, where `instantiate` reads truthiness to decide
    # whether to wire the gate: the gate would be wired onto a string and `adm.evaluate` would
    # raise `AttributeError` on every candidate.
    admissibility_exemption: str = ""
    artifact_variants: list = field(default_factory=list)
    # HOW A SOLVER SUBMITS, and HOW AN INSTANCE IS BROKEN ON PURPOSE. Both carried verbatim
    # like the two above, for the same reason -- the harness validates them structurally
    # (check_canonical: NO_SOLVER_INPUT / NO_NEGATIVE_CONTROL) and executes them at packaging.
    # Without them an artifact may have no verb that accepts the thing its task is about, and
    # every check in it may survive `cp /bin/true <binary>`.
    solver_input: dict = field(default_factory=dict)
    negative_control: dict = field(default_factory=dict)
    parameters: list = field(default_factory=list)
    # The declared gates and the sampler's per-task facts:
    # the task-blind engine executes these and the sampler generator is derived
    # from them, so a canonical without them cannot drive the pipeline. The
    # formalizer WRITES them from the NL spec (there is no structured raw form to
    # carry verbatim -- when a raw spec does ship them structured, they are
    # carried like admissibility clauses) and check_canonical holds every clause
    # to the engine's own grammar via the mirror above.
    entry_contract: list = field(default_factory=list)
    exclusions_on: list = field(default_factory=list)
    corroboration_contract: list = field(default_factory=list)
    corroboration_accept: str = ""
    sampling: dict = field(default_factory=dict)
    report_discovery: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "spec_id": self.spec_id,
            "task": self.task,
            "instance_input": self.instance_input,
            "instance_output": self.instance_output,
            "data_source": self.data_source,
            "candidate_definition": self.candidate_definition,
            "oracle_strategy": self.oracle_strategy,
            "oracle_detail": self.oracle_detail,
            "interface": self.interface,
            "scale": {
                "instance_count": self.scale_instance_count,
                "compute_budget": self.scale_compute_budget,
                "human_budget": self.scale_human_budget,
            },
            "criteria": self.criteria,
            "quality_gates": self.quality_gates,
            "examples": self.examples,
            "constraints": self.constraints,
            "assumptions": [a.__dict__ for a in self.assumptions],
            "delegations": [d.__dict__ for d in self.delegations],
        }
        if self.admissibility:
            d["admissibility"] = self.admissibility
        if self.admissibility_exemption:
            d["admissibility_exemption"] = self.admissibility_exemption
        if self.artifact_variants:
            d["artifact_variants"] = self.artifact_variants
        if self.solver_input:
            d["solver_input"] = self.solver_input
        if self.negative_control:
            d["negative_control"] = self.negative_control
        if self.parameters:
            d["parameters"] = self.parameters
        if self.entry_contract:
            d["entry_contract"] = self.entry_contract
        if self.exclusions_on:
            d["exclusions_on"] = self.exclusions_on
        if self.corroboration_contract:
            d["corroboration_contract"] = self.corroboration_contract
        if self.corroboration_accept:
            d["corroboration_accept"] = self.corroboration_accept
        if self.sampling:
            d["sampling"] = self.sampling
        if self.report_discovery:
            d["report_discovery"] = self.report_discovery
        return d


@dataclass
class StageEstimate:
    stage: str  # e.g. "2-harvest", matching spec/pipeline-v0.md's numbering
    description: str
    instances_in: str  # rough count/description of what enters this stage
    instances_out: str  # rough count/description of what survives it
    compute_estimate: str  # e.g. "5 CPU-hours"
    llm_cost_estimate: str  # e.g. "$2.50"
    wallclock_estimate: str  # e.g. "30 minutes"
    platform_calls: dict = field(default_factory=dict)  # platform -> estimated call count (str)


@dataclass
class Plan:
    spec_id: str
    stages: list[StageEstimate] = field(default_factory=list)
    total_compute: str = ""
    total_llm_cost: str = ""
    total_wallclock: str = ""
    total_human_hours: str = ""
    access_usage: list[dict] = field(
        default_factory=list
    )  # [{platform, estimated_calls, declared_budget, within_budget}]

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "stages": [s.__dict__ for s in self.stages],
            "totals": {
                "compute": self.total_compute,
                "llm_cost": self.total_llm_cost,
                "wallclock": self.total_wallclock,
                "human_hours": self.total_human_hours,
            },
            "access_usage": self.access_usage,
        }


@dataclass
class PlanResult:
    outcome: str  # "GO" | "NO_GO"
    plan: Plan | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def go(self) -> bool:
        return self.outcome == "GO"


@dataclass
class FormalizeResult:
    outcome: str  # "ACCEPT" | "REJECT" | "ACCEPT_SUBSTITUTED"
    canonical_spec: CanonicalSpec | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    # populated only for ACCEPT_SUBSTITUTED: full auditability of what was
    # tried first and why the substitution happened, so this is never silent
    original_diagnostics: list[Diagnostic] = field(default_factory=list)
    substitution: dict | None = None  # {oracle_strategy, mechanism, human_budget_implication, tradeoff_vs_stated}

    @property
    def accepted(self) -> bool:
        return self.outcome in ("ACCEPT", "ACCEPT_SUBSTITUTED")


_ADMISSIBILITY_HINT = (
    "Add an `admissibility:` list to the spec -- one entry per property a source must have, "
    "each with three keys:\n"
    "    - name:    the value to record (a plain field name)\n"
    "    - observe: HOW to obtain it, concretely enough that two people fetch the same number\n"
    "    - require: the threshold, as `>= 100`, `within 12 months`, `is false` or "
    "`one of [a, b]`\n"
    "Which properties matter is yours to decide and differs by task. State at least one."
)


_VARIANTS_HINT = (
    "Declare `artifact_variants:` -- one entry per state of the instance the benchmark needs to "
    "distinguish. Each needs `name` (what the artifact's verbs are built from), `check_cmd` "
    "(the command that decides it, with {placeholders} for fields each instance records) and "
    "`expect: pass|fail`. At least one variant must be expected to pass, or a working baseline "
    "cannot be demonstrated."
)


# A `cd` anywhere in a check command. The gate runs `cd {fresh} && {check_cmd}` so the check
# starts in a directory the setup script has just rebuilt and the agent has never touched;
# navigating out of it means observing something else. Placeholders count -- boca's
# `cd {source_path}/..` escaped via a field recorded as an absolute path, so matching only a
# literal `/` would have missed the case this rule exists for.
_CD_IN_CHECK = re.compile(r"(^|[;&|]|\bthen\b|\bdo\b)\s*cd\s", re.I)


def _check_stays_put(variants) -> list[str]:
    """Which declared checks navigate away from the variant they are given.

    Not "a check may not invoke a build tool", which was the first version of this rule and
    would have rejected `mvn ... test` -- the only contract in the project that certifies. The
    rule is about the working directory, because that is what actually differs: of the seven
    live check commands, the five that work contain no `cd` and the two that are broken both
    begin with one.
    """
    out = []
    for v in variants:
        if not isinstance(v, dict):
            continue
        cmd = str(v.get("check_cmd") or "")
        if _CD_IN_CHECK.search(cmd):
            out.append(f"variant {str(v.get('name') or '?')!r}: check_cmd changes directory ({cmd[:60]!r})")
    return out


def check_verifiability(raw: dict) -> list[Diagnostic]:
    """Does this spec say how a finished instance is CHECKED, in terms a program can run?

    The sibling of `check_admissibility`, and the same lesson at the other end. That one exists
    because silence about sources is not a vague word, so nothing caught a spec that simply
    never said what a source must be. This exists because silence about VERIFICATION is not a
    vague word either.

    A spec that declares this gets an interface rendered mechanically from the declaration and
    every instance re-executed against it. A spec that declares nothing sends an LLM to invent
    an interface, which may be one the certifier cannot drive, or none at all when instances do
    not share a shape. So the lever is declaration, not model capability: where the contract exists no model is needed, and where it is absent no model
    can supply it.

    Nothing here names a task, a toolchain or an ecosystem. What the variants ARE, and what
    decides them, is the spec's business and differs per benchmark.
    """
    # NOT required here. Whether a variant contract applies depends on the oracle pattern, and
    # `oracle_strategy` is assigned by the formalizer -- it does not exist on a raw spec. Demanding
    # it at this stage forced every benchmark into one verification style, including the eleven
    # shipped examples and the `llm-judge` / `human-annotation` / `external-authority` patterns
    # the registry itself advertises. `check_canonical` requires it of the patterns it fits.
    #
    # What IS judged here needs no strategy: if a contract is declared at all, it must be
    # runnable. Well-formedness is unconditional; the requirement is not.
    variants = raw.get("artifact_variants")
    if not variants or not isinstance(variants, list):
        return []

    problems = []
    for i, v in enumerate(variants):
        where = f"artifact_variants[{i}]"
        if not isinstance(v, dict):
            problems.append(f"{where} is not a mapping")
            continue
        if not str(v.get("name") or "").strip():
            problems.append(f"{where} has no `name`, so no verb can be built from it")
        if not str(v.get("check_cmd") or "").strip():
            problems.append(f"{where} declares no `check_cmd`, so nothing decides it")
        expect = str(v.get("expect") or "").strip().lower()
        if expect not in ("pass", "fail"):
            # Polarity has to be structural. It was once recovered by searching the English
            # description for "expected to fail", so a variant whose wording differed certified
            # with the polarity inverted -- and an instance whose check was SUPPOSED to fail was
            # marked broken for failing.
            problems.append(f"{where} declares expect={v.get('expect')!r}; it must be exactly `pass` or `fail`")
    if isinstance(variants, list) and variants and not problems:
        if not any(str((v or {}).get("expect", "")).strip().lower() == "pass" for v in variants if isinstance(v, dict)):
            problems.append(
                "no variant is expected to pass, so the artifact cannot "
                "demonstrate a working baseline -- a wholly broken instance and a "
                "correct one would be indistinguishable"
            )
    if problems:
        return [
            Diagnostic(
                code="UNCHECKABLE_VARIANTS",
                field="artifact_variants",
                detail="; ".join(problems),
                fix_hint=_VARIANTS_HINT,
            )
        ]

    wandered = _check_stays_put(variants)
    if wandered:
        return [
            Diagnostic(
                code="CHECK_LEAVES_VARIANT",
                field="artifact_variants",
                detail="; ".join(wandered),
                fix_hint=(
                    "The harness rebuilds each variant into a fresh directory and runs the check "
                    "THERE, so the check must observe that directory. Move anything that builds, "
                    "fetches or arranges state into setup_<name>.sh -- which takes the target "
                    "directory as its only argument -- and leave the check to observe the result. "
                    "If a check needs a subdirectory, pass it as an argument (most tools take one, "
                    "e.g. a `-C`/`-f`/path option) rather than changing directory. Separating the "
                    "two is also what makes a failure attributable: a broken setup and a genuine "
                    "negative result exit the same way, and only the stage tells them apart."
                ),
            )
        ]
    return []


#: Nouns that name the SOURCE. A criterion whose SUBJECT is one of these is about the
#: repository; the same words appearing later in a sentence are not ("build with the project's
#: configured build tool", "run the project's test harness" are both genuine criteria).
#: Position is the whole rule -- measured over the 61 criteria shipped across eleven
#: collectors, a prefix match on admissibility clause NAMES finds 0 of the 3 real ones.
_SOURCE_SUBJECT = re.compile(
    r"^\s*(the\s+|a\s+|an\s+|each\s+|every\s+|all\s+)?"
    r"(sources?|projects?|repositor(?:y|ies)|repos?|codebases?)\b",
    re.I,
)
#: A criterion that defers to admissibility is definitionally a source filter, and enforces
#: nothing a state could prove.
_DEFERS_TO_ADMISSIBILITY = re.compile(r"admissibilit", re.I)


def _source_filter_reason(text: str, names: set) -> str:
    """Why this criterion is a source filter, or "" if it is not."""
    low = str(text).strip()
    if _DEFERS_TO_ADMISSIBILITY.search(low):
        return "defers to the admissibility block instead of stating a guarantee"
    if _SOURCE_SUBJECT.match(low):
        return "is a sentence about the source rather than about one instance"
    head = low.lower().split(":", 1)[0].strip()
    if names and (head in names or any(low.lower().startswith(n) for n in names)):
        return "restates the admissibility clause it is named after"
    return ""


#: The prose fields a later stage reads and cannot do its job without, and what each is FOR --
#: the message has to say why an empty one matters, not merely that it is empty.
_LOAD_BEARING_FIELDS = {
    "task": "what the evaluated system must do",
    "candidate_definition": "the single event or artifact that counts as ONE instance, which "
    "is what the sampler mines for",
    "instance_input": "what the evaluated system receives",
    "instance_output": "what it must produce",
    "data_source": "where raw instances come from",
    "oracle_detail": "precisely how correctness is decided",
}


def check_canonical(canonical) -> list[Diagnostic]:
    """Checks that need the FORMALIZER's answers, not the author's raw prose.

    `mechanical_precheck` runs before an oracle strategy has been assigned, so it cannot tell a
    fail-to-pass benchmark from an llm-judge one and must treat them alike. This runs after, and
    asks each spec only for what its own declared pattern can supply.

    Accepts a CanonicalSpec or a plain dict, so a caller holding either can use it.
    """
    from benchcompiler.registry import ORACLE_PATTERNS, VARIANT_ORACLE_PATTERNS

    get = canonical.get if isinstance(canonical, dict) else lambda k, d=None: getattr(canonical, k, d)
    strategy = str(get("oracle_strategy", "") or "").strip().lower()
    variants = get("artifact_variants", None)

    out: list[Diagnostic] = []
    # DOES THE SPEC SAY ANYTHING AT ALL. Checked before everything else, because every rule
    # below reasons about content and none of them notice its absence: a reply of
    # `{"outcome":"ACCEPT","canonical_spec":{}}` would otherwise come back a clean ACCEPT with a real
    # spec_id and empty everything, and `derive_collector_task` would render it into a
    # collector whose source_requirements is the boilerplate addendum and nothing else.
    #
    # The reason this needs a gate rather than trust: a degraded or truncated reply does not
    # LOOK like a failure. It produces a plausible canonical spec, and every stage downstream
    # treats it as authoritative -- the sampler is told to mine "", the designer to design for
    # "". `candidate_definition` is the sharpest case, since it is the subject of the prompt's
    # own check #1 and nothing verified it.
    for name, what in _LOAD_BEARING_FIELDS.items():
        if not str(get(name, "") or "").strip():
            out.append(
                Diagnostic(
                    code="MISSING_INFO",
                    field=name,
                    detail=(
                        f"the formalized spec has no {name}: {what}. A canonical spec missing "
                        f"it is not a spec -- every later stage reads this field and has "
                        f"nothing to fall back on."
                    ),
                    fix_hint=(
                        f"This is the formalizer's own output, not the researcher's input. "
                        f"An empty {name} means the model's reply was degraded or truncated; "
                        f"re-run. If the raw spec genuinely does not support answering it, "
                        f"the answer is a REJECT naming the missing input, not an empty "
                        f"field."
                    ),
                )
            )

    out.extend(_contract_diagnostics(get))
    out.extend(_viability_projection_diagnostics(get))

    # A DELEGATION IS "A BINDING, RECORDED CHOICE, NOT A SILENT DEFAULT" -- the prompt's own
    # words. It was recorded and never checked: `check_canonical` did not look at `delegations`
    # at all, so a delegation naming `correctness` came back a clean ACCEPT despite check 5
    # forbidding exactly that ("the system cannot invent what the benchmark is FOR, or its core
    # correctness mechanism"). The vocabulary is read from the registry rather than copied, so
    # the rule and the registry cannot drift apart.
    #
    # NOT CLOSED HERE: a delegation still binds nothing downstream -- no construction state can
    # fail because a delegated resolution was ignored. That needs it carried into the collector
    # and gated on, which is `instantiate.py`.
    from benchcompiler.registry import DELEGABLE_WHOLE_FIELDS

    for d in get("delegations", None) or []:
        rec = d if isinstance(d, dict) else getattr(d, "__dict__", {})
        fname = str(rec.get("field") or "").strip()
        # A sub-item ("criteria.dedup") is delegable wherever its parent list is; the prompt
        # offers that shape explicitly, so only WHOLE-field delegations are judged here.
        whole = fname and "." not in fname
        if whole and fname not in DELEGABLE_WHOLE_FIELDS:
            out.append(
                Diagnostic(
                    code="INVALID_DELEGATION",
                    field="delegations",
                    detail=(
                        f"the spec delegates {fname!r}, which cannot be delegated as a whole "
                        f"field. DELEGATE is valid on "
                        f"{', '.join(sorted(DELEGABLE_WHOLE_FIELDS))}, and on sub-items within "
                        f"criteria/quality. What the benchmark is FOR and how correctness is "
                        f"decided cannot be invented for the researcher."
                    ),
                    fix_hint=(
                        f"State {fname!r} in the raw spec. If it genuinely is not yet "
                        f"decided, that is a spec that is not ready to compile, not a "
                        f"delegation."
                    ),
                )
            )
        for key in ("resolved_to", "rationale"):
            if fname and not str(rec.get(key) or "").strip():
                out.append(
                    Diagnostic(
                        code="INVALID_DELEGATION",
                        field="delegations",
                        detail=(
                            f"the delegation of {fname!r} records no {key}. A delegation is a "
                            f"binding choice ON RECORD; one with no {key} is the silent "
                            f"default it exists to replace, and a reviewer cannot tell them "
                            f"apart."
                        ),
                        fix_hint=f"Record what {fname!r} was resolved to, and why.",
                    )
                )

    # Well-formedness first, and unconditionally: a declared contract must be runnable whatever
    # the pattern, so a judge-oracle spec that also ships variants is still held to them.
    if variants:
        out += [d for d in check_verifiability({"artifact_variants": variants})]
        # Every variant states what its check's EVIDENCE is. Variants pass VERBATIM from here
        # to the collector, and when no stage asks what a check proves, every layer degrades to
        # exit codes (e.g. counting 'command not found' as the bug being observable). Refused
        # at the EARLIEST gate, with the fix named.
        for v in variants:
            vname = str((v.get("name") if isinstance(v, dict) else "") or "?")
            kind = v.get("results") if isinstance(v, dict) else None
            problem = ""
            if kind is None:
                problem = (
                    "declares no `results`: what evidence is this variant's check "
                    "judged on? An exit code is not a verdict -- a script's "
                    "else-branch can turn any error into the expected outcome."
                )
            elif str(kind).lower() == "junit-xml":
                if "expect_failures" not in v:
                    problem = (
                        "results: junit-xml without expect_failures -- state how "
                        "many failing tests this variant must show (bug-showing: "
                        ">=1, fixed-style: 0); polarity is never defaulted."
                    )
            elif str(kind).lower() == "none":
                if not str(v.get("results_none_reason") or "").strip():
                    problem = (
                        "results: none without results_none_reason -- write down "
                        "why no test framework runs and what the real evidence "
                        "is (e.g. digest comparison of program output)."
                    )
            else:
                problem = f"results: {kind!r} is not a kind the harness implements (junit-xml, or none with a reason)."
            if problem:
                out.append(
                    Diagnostic(
                        code="UNDECLARED_CHECK_EVIDENCE",
                        field=f"artifact_variants[{vname}]",
                        detail=f"variant {vname!r} {problem}",
                        fix_hint=(
                            "Add `results: junit-xml` plus `expect_failures: <int>` when "
                            "the check runs a test framework (the harness judges on the "
                            "runner's own report files -- Surefire/Gradle write them by "
                            "default, pytest via --junitxml), or `results: none` with "
                            "`results_none_reason` naming the actual evidence."
                        ),
                    )
                )

    # WHAT A SOLVER SUBMITS, AND HOW A CHECK CAN FAIL. Both are things every benchmark has;
    # without them an artifact can pass every other gate and still be unusable:
    #
    #   solver_input     -- a task such as "find a flag configuration that beats -O3" needs a
    #                       verb that accepts a flag string; otherwise the artifact cannot be
    #                       used for its own task, and no other gate notices because they are
    #                       all about verdicts.
    #   negative_control -- `cp /bin/true <binary>` must not pass the default check. The
    #                       break is task content (what "break it" means here), so it is
    #                       declared and the harness only executes it and demands the verdict
    #                       flips.
    #
    # Declared on the canonical spec, carried into the artifact contract by
    # artifact_plan.plan_from_contract, and executed at packaging.
    si = get("solver_input", None)
    if not isinstance(si, dict) or not str(si.get("describes") or "").strip():
        out.append(
            Diagnostic(
                code="NO_SOLVER_INPUT",
                field="solver_input",
                detail=(
                    "nothing states how a solver hands this benchmark its answer, so nothing "
                    "can check that the shipped artifact accepts one. A benchmark whose task "
                    "cannot be attempted through its own interface is not a benchmark."
                ),
                fix_hint=(
                    "Declare solver_input: {kind: parameter|workdir|file, "
                    "name/path: ..., describes: '<what a solver submits>'}. `parameter` "
                    "means a value the consumer passes (a flag string, a config); "
                    "`workdir`/`file` mean the solver edits the artifact itself (a patch, "
                    "an edited source tree)."
                ),
            )
        )
    elif str(si.get("kind") or "").strip().lower() not in ("parameter", "workdir", "file"):
        out.append(
            Diagnostic(
                code="NO_SOLVER_INPUT",
                field="solver_input",
                detail=f"solver_input kind {si.get('kind')!r} is not one of parameter, workdir, file",
                fix_hint="Pick the channel a solver actually uses to submit its answer.",
            )
        )
    elif str(si.get("kind")).lower() == "parameter":
        params = {str(p.get("name")) for p in (get("parameters", None) or []) if isinstance(p, dict)}
        if str(si.get("name") or "") not in params:
            out.append(
                Diagnostic(
                    code="NO_SOLVER_INPUT",
                    field="solver_input",
                    detail=(
                        f"solver_input names parameter {si.get('name')!r}, which is not in "
                        f"`parameters` ({sorted(params)}) -- so the artifact's interface will "
                        f"not offer it and a solver cannot submit anything."
                    ),
                    fix_hint="Declare it in `parameters:` too, with a kind and a default.",
                )
            )
        probes = [str(x) for x in (si.get("probe_values") or [])]
        if len(probes) != 2 or probes[0] == probes[1]:
            out.append(
                Diagnostic(
                    code="NO_SOLVER_INPUT",
                    field="solver_input",
                    detail=(
                        "solver_input declares no two `probe_values`, so nothing can check "
                        "that submitting different answers produces different outcomes. A verb "
                        "that accepts a value and ignores it passes every structural check "
                        "there is -- nine of sixteen shipped instances were prebuilt binaries "
                        "whose rebuild did nothing."
                    ),
                    fix_hint=(
                        "Add probe_values: [<one answer>, <a materially different answer>] "
                        "-- e.g. ['-O0', '-O3']. The harness submits each and requires the "
                        "declared observable to differ."
                    ),
                )
            )

    if variants:
        nc = get("negative_control", None)
        # The same cross-check `solver_input.name` gets against `parameters`, on the sibling
        # field. A `check` naming a variant that does not exist (e.g. `fixed` against variants
        # called `prefix` and `postfix`) is turned by `artifact_plan.plan_from_contract` into
        # verbs no instance ships, and `_check_negative_control` would refuse it only at
        # PACKAGING, many stages downstream of the typo. `check`
        # stays OPTIONAL (the packager falls back to the first variant); what is refused is a
        # name that is wrong.
        if isinstance(nc, dict) and str(nc.get("check") or "").strip():
            declared = [str((v.get("name") if isinstance(v, dict) else "") or "") for v in variants]
            declared = [n for n in declared if n]
            named = str(nc.get("check")).strip()
            if declared and named not in declared:
                out.append(
                    Diagnostic(
                        code="NO_NEGATIVE_CONTROL",
                        field="negative_control",
                        detail=(
                            f"negative_control.check names variant {named!r}, which is not one "
                            f"of the declared variants ({declared}). The packager builds the "
                            f"rebuild/check verb pair from this name, so no verb exists to run "
                            f"and the control is refused at packaging -- after construction "
                            f"has been paid for."
                        ),
                        fix_hint=(
                            f"Set `check:` to the variant whose verdict must flip when the "
                            f"instance is broken -- one of {declared}."
                        ),
                    )
                )
        if not isinstance(nc, dict) or not str(nc.get("break_cmd") or "").strip():
            out.append(
                Diagnostic(
                    code="NO_NEGATIVE_CONTROL",
                    field="negative_control",
                    detail=(
                        "nothing states how to BREAK an instance on purpose, so nothing can "
                        "prove a shipped check is able to fail. A check that survives a "
                        "deliberate break certifies a solver that did nothing."
                    ),
                    fix_hint=(
                        "Declare negative_control: {break_cmd: <field name holding the "
                        "command that breaks this instance>, restore_cmd: <optional field>, "
                        "check: <variant name whose verdict must flip>, describes: '...'}. "
                        "The FIELD is recorded per instance by the builder -- what breaks "
                        "one program is not what breaks another -- and the harness only "
                        "runs it and requires the check to fail afterwards."
                    ),
                )
            )

    # CRITERIA vs ADMISSIBILITY. A formalization can emit criteria -- language, stars,
    # is_fork, commit_count, ... -- that are all SOURCE filters, and drop "the commit's own
    # test fails on its parent and passes on the commit itself". criteria drive criteria_coverage: each one must be enforced
    # by some lifecycle state, so a criteria list made only of repo filters means the
    # designed lifecycle is never required to prove what the benchmark is ABOUT. This refuses
    # the outcome whatever the wording.
    # THE FIELD THE HARNESS WILL EXECUTE. Whatever produced it -- carried verbatim from a raw
    # clause list, or resolved by the model from prose -- what leaves here is what `instantiate`
    # wires `validators:candidate_is_admissible` onto, and what `adm.evaluate` is called with
    # per candidate. So it is held to the same parser the gate uses, at the last point a
    # diagnostic is still cheaper than a run. An exemption stated on the record is the one way
    # to carry none.
    admis_raw = get("admissibility", None)
    exemption = str(get("admissibility_exemption", "") or "").strip()
    if not isinstance(admis_raw, list):
        out.append(
            Diagnostic(
                code="MISSING_ADMISSIBILITY",
                field="admissibility",
                detail=(
                    f"admissibility left the formalizer as {type(admis_raw).__name__}, not a "
                    f"list of clauses. The gate calls `evaluate` on each entry, so anything "
                    f"else raises inside a validator -- which is charged to the harness, not "
                    f"the candidate, and refuses every candidate forever."
                ),
                fix_hint=_ADMISSIBILITY_HINT,
            )
        )
    elif not admis_raw and not exemption:
        out.append(
            Diagnostic(
                code="MISSING_ADMISSIBILITY",
                field="admissibility",
                detail=(
                    "the formalized spec states no measurable floor for what counts as an "
                    "admissible source, so the sampler will decide for itself what is worth "
                    "mining. If the raw spec described its sources in prose, the model was "
                    "asked to resolve that into clauses and returned none."
                ),
                fix_hint=_ADMISSIBILITY_HINT,
            )
        )
    else:
        from agentcore.agent import admissibility as adm

        for i, clause in enumerate(admis_raw):
            problems = adm.clause_problems(clause) + observe_problems(clause)
            if problems:
                named = (clause.get("name") if isinstance(clause, dict) else None) or f"#{i + 1}"
                out.append(
                    Diagnostic(
                        code="UNMEASURABLE_ADMISSIBILITY",
                        field="admissibility",
                        detail=f"clause {named}: " + "; ".join(problems),
                        fix_hint=_ADMISSIBILITY_HINT,
                    )
                )

    admis = get("admissibility", None) or []
    crit = [str(c) for c in (get("criteria", None) or [])]
    if admis and crit:
        names = {str(a.get("name") or "").strip().lower() for a in admis if isinstance(a, dict)}
        names = {n for n in names if n}

        for c in crit:
            why = _source_filter_reason(c, names)
            if not why:
                continue
            out.append(
                Diagnostic(
                    code="CRITERIA_ARE_SOURCE_FILTERS",
                    field="criteria",
                    detail=(
                        f"criterion {c.strip()[:120]!r} {why}. It states a property of the "
                        f"SOURCE, not of an INSTANCE. Every criterion must be enforced by "
                        f"some construction state, so this one spends a state re-proving "
                        f"something the admissibility gate already settled for free, before "
                        f"the clone was paid for -- while whatever the benchmark is actually "
                        f"ABOUT goes unenforced."
                    ),
                    fix_hint=(
                        "Keep source filters in `admissibility` and put the per-instance "
                        "guarantees in `criteria`, one per property the oracle decides "
                        "(e.g. 'the added test fails at the parent commit and passes at "
                        "the fix commit, confirmed by an actual test run')."
                    ),
                )
            )

    if strategy in VARIANT_ORACLE_PATTERNS and not variants:
        out.append(
            Diagnostic(
                code="MISSING_VARIANTS",
                field="artifact_variants",
                detail=(
                    f"oracle_strategy is {strategy!r}, which establishes ground truth by "
                    f"contrasting two states of the artifact, but the spec names no variants -- "
                    f"so nothing downstream can re-execute the contrast, and the artifact's "
                    f"interface would have to be guessed. A guess cannot be certified."
                ),
                fix_hint=_VARIANTS_HINT,
            )
        )
    elif strategy and strategy not in ORACLE_PATTERNS:
        out.append(
            Diagnostic(
                code="UNCLASSIFIED_ORACLE",
                field="oracle_strategy",
                detail=(
                    f"oracle_strategy {strategy!r} is not one the registry names, so whether a "
                    f"variant contract applies to it cannot be decided."
                ),
                fix_hint=(
                    "Use one of registry.ORACLE_PATTERNS, or state the decision procedure "
                    "precisely enough that it can be classified."
                ),
            )
        )
    elif strategy == "other":
        out.append(
            Diagnostic(
                code="UNCLASSIFIED_ORACLE",
                field="oracle_strategy",
                detail=(
                    "oracle_strategy is 'other', the registry's deliberate escape hatch, so "
                    "whether this benchmark's instances can be re-executed is undecided. "
                    "Exempting it silently would let anyone skip the contract by declining to "
                    "classify; requiring one would close an escape hatch that exists on purpose."
                ),
                fix_hint=(
                    "Either name a pattern from registry.ORACLE_PATTERNS, or declare "
                    "artifact_variants explicitly so the artifact is certifiable anyway."
                ),
            )
        )
    return out


# An author who mines nothing writes it down. Recognised several ways because they will spell it
# several ways, and refusing a correct exemption over punctuation is the over-refusal this
# codebase has paid for repeatedly.
_EXEMPTION = re.compile(r"^\s*(not[\s_-]*applicable|n/?a|none)\b[\s:,.;-]*", re.I)
# Enough words after the marker to be a reason rather than a shrug. The point of the exemption
# is a decision on record; a bare marker records that someone wanted to skip the gate, which a
# reviewer cannot tell apart from not applying.
_MIN_EXEMPTION_REASON = 12


def _declared_inapplicable(value) -> tuple[bool, str]:
    """(is an exemption, the reason). Prose that merely discusses sources is NOT an exemption.

    Modelled on the worklist's `self_graded: true`, described in its own schema as "an explicit,
    greppable admission ... so the gap is a decision on record rather than an omission nobody
    sees". `check_admissibility` exists because silence is not a vague word and nothing caught a
    spec that never mentioned sources; but not every benchmark HAS sources -- cruxeval's data
    field says "Not mined from real repositories -- fully synthetic generation" -- and requiring
    it to invent a floor is asking for a number about nothing.

    Keying this on `oracle_strategy` the way verifiability now is would not work: `injection`
    builds instances by perturbing known-good artifacts and still needs its sources bounded, so
    the oracle pattern does not say whether a population is mined.
    """
    if not isinstance(value, str):
        return False, ""
    m = _EXEMPTION.match(value)
    if not m:
        return False, ""
    return True, value[m.end() :].strip()


#: A choice of procedure inside one `observe`. Word-bounded so `origin`, `for`, `word` and a
#: URL query containing the letters do not match; the comma form is the one that shipped.
_OBSERVE_DISJUNCTION = re.compile(r"(?:,\s*)?\bor\b", re.I)

#: An observe that tells the reader to go and form an impression. The formalizer prompt names
#: exactly this shape as its worked example -- "'look at the project' is NOT a procedure --
#: reject it with UNMEASURABLE_ADMISSIBILITY" -- and `clause_problems`' blocklist holds
#: "look at it" and "inspect it", so the prompt's own example passes on the extra word.
#: Anchored at the start, so a real procedure that happens to contain the verb ("GET .../
#: contents/ -> the root listing; look at the first name") is untouched, and checked against
#: every observe shipped in this repo: none begins with one of these.
_OBSERVE_JUDGMENT = re.compile(
    r"^\s*(look|inspect|check|examine|review|see|read|judge|assess|evaluate|consider)\b", re.I
)


def observe_problems(clause) -> list[str]:
    """What the formalizer demands of an `observe` beyond what the harness's parser can.

    `clause_problems` (agentcore) checks `observe` for emptiness, a blocklist of judgment
    phrases, and a 12-character floor -- length, essentially, while `require` is machine-checked
    by `parse()`. The prompt's actual standard is stricter and unenforced: "concretely enough
    that two people fetch the same number".

    ONE defect in the shipped corpus meets that standard's plain reading and nothing caught it.
    Over 169 clause instances and 11 distinct (name, observe) pairs, exactly one offers a
    choice:

        commit_count: "git rev-list --count HEAD in a shallow clone, or the contributors API"

    Two procedures, two different numbers, no record of which was used -- and one branch
    (a shallow clone returns 1) refuses every source against a `>= 1000` threshold. The boca
    family states the same field as "in a full clone".

    DELIBERATELY NOT a shape rule (`GET <url> -> <field>` | a shell command). Measured against
    the same corpus, a shape rule refuses `build_file`, whose observe is a followable
    natural-language instruction, across 13 shipped clauses -- and catches nothing the
    disjunction rule does not. This lives here rather than in `clause_problems` because it is a
    rule about what a SPEC may declare, not about what the runtime gate can evaluate: tightening
    the runtime parser would refuse clauses already shipped inside live collectors.
    """
    if not isinstance(clause, dict):
        return []
    observe = " ".join(str(clause.get("observe") or "").split())
    if not observe:
        return []
    out = []
    if _OBSERVE_DISJUNCTION.search(observe):
        out.append(
            f"`observe` {observe!r} offers two ways to obtain one value. Two procedures return "
            f"two different numbers and nothing records which was used, so the threshold means "
            f"something different depending on who ran it. Name the one procedure that decides "
            f"this field."
        )
    if _OBSERVE_JUDGMENT.match(observe):
        out.append(
            f"`observe` {observe!r} asks the reader to go and form an impression, which is what "
            f"a clause exists to replace. Say where the value comes from: an API "
            f"call and the field it returns, or a command and what it prints."
        )
    return out


def admissibility_shape(value) -> str:
    """What kind of admissibility declaration this is: clauses | exempt | prose | absent.

    The four shapes are treated differently on purpose, and the difference is WHO can resolve
    them:

      clauses  a structured block the harness executes. Carried VERBATIM -- an LLM paraphrase
               of a predicate is the drift the formalizer exists to prevent -- so it must be
               valid as written, and is checked before any model call.
      exempt   a benchmark that mines nothing, saying so on the record (`_declared_inapplicable`).
               Never carried into `admissibility`: `instantiate` wires the admissibility gate
               whenever that field is truthy, and `adm.evaluate` raised AttributeError on the
               string, which routes to `harness_origin` and charges no strike -- so every
               candidate was refused forever by a harness fault.
      prose    the researcher described their sources in a sentence. This is exactly what the
               formalizer's term table is FOR ("popular" -> stars >= 1000, recorded as an
               assumption), so it is a question for the model, not a rejection. Refusing it
               here would make that whole paragraph of the prompt unreachable.
      absent   silence. Still refused mechanically: nothing can resolve silence without
               inventing a floor, and an invented floor admits arbitrary sources.
    """
    if isinstance(value, list):
        return "clauses" if value else "absent"
    if isinstance(value, str) and value.strip():
        return "exempt" if _declared_inapplicable(value)[0] else "prose"
    return "absent"


def check_admissibility(raw: dict) -> list[Diagnostic]:
    """Does this spec say what a SOURCE has to be, in terms a program can check?

    The formalizer's ambiguity lint fires on vague WORDS -- it resolves "popular" through
    TERM_DEFAULTS and refuses to guess at "large project". Silence is not a word, so a spec
    that simply never mentions scale has nothing to catch ("Real, public C/C++ projects with
    a Git history" passes the lint). Without this check the lint would punish the researcher
    who reached for a fuzzy adjective and wave through the one who said nothing at all.

    What is required here is only that criteria EXIST and are measurable. Which dimensions,
    and what the numbers are, is the spec's business and differs per task -- so nothing below
    names a dimension. `measurable` is decided by the same parser the harness enforces with, so
    a spec cannot pass review carrying a predicate that would refuse every source at runtime.
    """
    from agentcore.agent import admissibility as adm

    clauses = raw.get("admissibility")

    exempt, reason = _declared_inapplicable(clauses)
    if exempt:
        if len(reason) < _MIN_EXEMPTION_REASON:
            return [
                Diagnostic(
                    code="MISSING_ADMISSIBILITY",
                    field="admissibility",
                    detail=(
                        "admissibility is marked not-applicable with no reason. The exemption "
                        "exists so a benchmark that mines nothing can say so on the record; a "
                        "bare marker records only that someone wanted to skip the gate, which "
                        "a reviewer cannot tell apart from it genuinely not applying."
                    ),
                    fix_hint=(
                        "Say why there are no sources to bound -- e.g. 'not applicable: "
                        "instances are generated synthetically', or 'not applicable: "
                        "instances come from a fixed published corpus'. If sources ARE "
                        "mined, state the clauses instead."
                    ),
                )
            ]
        return []

    if not clauses or not isinstance(clauses, list):
        return [
            Diagnostic(
                code="MISSING_ADMISSIBILITY",
                field="admissibility",
                detail=(
                    "The spec states no measurable floor for what counts as an admissible "
                    "source, so the sampler will decide for itself what is worth mining."
                ),
                fix_hint=_ADMISSIBILITY_HINT,
            )
        ]

    out = []
    for i, clause in enumerate(clauses):
        problems = adm.clause_problems(clause) + observe_problems(clause)
        if problems:
            named = (clause.get("name") if isinstance(clause, dict) else None) or f"#{i + 1}"
            out.append(
                Diagnostic(
                    code="UNMEASURABLE_ADMISSIBILITY",
                    field="admissibility",
                    detail=f"clause {named}: " + "; ".join(problems),
                    fix_hint=_ADMISSIBILITY_HINT,
                )
            )
    return out


def mechanical_precheck(raw: dict) -> list[Diagnostic]:
    """Cheapest possible check, no LLM call: are the required fields even
    present and non-empty? Catches the most trivial incomplete submissions
    for free before spending an LLM call on semantic judgment."""
    diagnostics = []
    for f in REQUIRED_FIELDS:
        value = raw.get(f)
        if value is None or (isinstance(value, str) and not value.strip()):
            diagnostics.append(
                Diagnostic(
                    code="MISSING_INFO",
                    field=f,
                    detail=f"Field '{f}' is required and was not provided.",
                    fix_hint=f"Add a `{f}:` field to the spec. See spec/benchspec-v0.md for what it must state.",
                )
            )
    # PROSE IS HELD BACK FOR THE MODEL (see `admissibility_shape`). Everything else -- silence,
    # a bare exemption marker, an unmeasurable clause -- is decided here, for free, because no
    # model call can resolve any of it. What comes back from the model is held to this same
    # function by `check_canonical`, so a resolution cannot smuggle in a clause the gate could
    # not evaluate.
    if admissibility_shape(raw.get("admissibility")) != "prose":
        diagnostics += check_admissibility(raw)
    diagnostics += check_verifiability(raw)
    return diagnostics


def _head_chars() -> int:
    """The real cap, read from where it lives, so this message cannot drift from it."""
    from benchcompiler.observables import HEAD_CHARS

    return HEAD_CHARS


#: Fields the harness fills by TRUNCATING (observables.HEAD_CHARS). They describe a stream;
#: they cannot identify it.
EXCERPT_SOURCES = ("stdout_head", "stderr_head", "output_head")
#: Observable kinds that DECIDE rather than describe.
VERDICT_KINDS = ("boolean", "bool")


def excerpt_sourced_verdicts(contract: dict) -> list:
    """Observables that decide something while reading a truncated excerpt.

    Example: `correct` (kind: boolean) captured `from:
    stdout_head`, so two program outputs agreeing for 200 characters and diverging after
    would both be correct -- in a benchmark whose entire oracle is output equality. The digest
    of the same stream (`stdout_sha256` / `output_sha256`) is in the same vocabulary.

    An excerpt captured under an honest name and a describing kind (`output_excerpt`,
    kind: text) is untouched: the problem was never the excerpt, it was the verdict.
    """
    kinds = {
        str(o.get("name")): str(o.get("kind") or "").strip().lower()
        for o in (contract.get("observables") or [])
        if isinstance(o, dict)
    }
    out = []
    for cap in (contract.get("measure") or {}).get("capture") or []:
        if not isinstance(cap, dict):
            continue
        src, name = str(cap.get("from") or ""), str(cap.get("name") or "")
        if src in EXCERPT_SOURCES and kinds.get(name) in VERDICT_KINDS:
            digest = src.replace("_head", "_sha256")
            out.append(
                f"observable {name!r} decides ({kinds.get(name)}) but is captured from "
                f"{src!r}, which is only the first {_head_chars()} characters of that stream "
                f"-- two different outputs agreeing that far would both pass. Capture it "
                f"from {digest!r} instead."
            )
    return out
