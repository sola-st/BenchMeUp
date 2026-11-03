#!/usr/bin/env python3
"""
formalizer.py — BenchSpec stage 1: formalize-or-reject.

Takes a researcher-written spec (YAML, see spec/benchspec-v0.md) and either
produces a CANONICAL SPEC (structured, defaults/delegations recorded) or
REJECTS with typed diagnostics naming exactly what's missing or ambiguous.
No human-in-the-loop dialogue: this is compile-or-reject, like a compiler.
Not implemented here: planning/pricing (stage 2) or the build pipeline
(spec/pipeline-v0.md) — this module only covers stage 1.

Requirements:
    pip install litellm pyyaml

Usage:
    export OPENAI_API_KEY=...        # or whatever litellm needs for --model
    python -m benchcompiler.formalizer specs/<task>/spec.yaml
    python -m benchcompiler.formalizer specs/<task>/spec.yaml --model <litellm model>
"""

import argparse
import dataclasses
import hashlib
import json
import os
import sys

from benchcompiler.llm import call_llm_json, retry_block
from benchcompiler.registry import (
    DEFAULT_QUALITY_GATES,
    DELEGABLE_WHOLE_FIELDS,
    ORACLE_PATTERNS,
    TERM_DEFAULTS,
)
from benchcompiler.rules_audit import judge_rules
from benchcompiler.spec import (
    DIAGNOSTIC_CODES,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    Assumption,
    CanonicalSpec,
    Delegation,
    Diagnostic,
    FormalizeResult,
    admissibility_shape,
    check_canonical,
    load_raw_spec,
    mechanical_precheck,
)


def _spec_id(raw: dict) -> str:
    blob = json.dumps(raw, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


FORMALIZE_SCHEMA_DESC = """{
  "outcome": "ACCEPT" | "REJECT",

  // present only if outcome == "REJECT":
  "diagnostics": [
    {"code": "MISSING_INFO" | "AMBIGUOUS_TERM" | "NON_OPERATIONAL_CORRECTNESS" | "UNPRICEABLE_CORRECTNESS" | "INVALID_DELEGATION",
     "field": "<which raw spec field>",
     "detail": "<what exactly is wrong>",
     "fix_hint": "<what the researcher should add/change to fix it>"}
  ],

  // present only if outcome == "ACCEPT":
  "canonical_spec": {
    "task": "<cleaned-up 1-3 sentence restatement>",
    "instance_input": "<what the evaluated system receives>",
    "instance_output": "<what it must produce>",
    "data_source": "<where raw instances come from>",
    "candidate_definition": "<the single event/artifact that counts as one instance>",
    "oracle_strategy": "<one of the named patterns below, or 'other'>",
    "oracle_detail": "<precisely how correctness is decided>",
    "interface": "<how a consumer uses the benchmark>",
    "scale": {"instance_count": "...", "compute_budget": "...", "human_budget": "..."},
    "criteria": ["<what EVERY shipped instance guarantees>", "..."],
    "quality_gates": ["<validation step>", "..."],
    "examples": [{"valid": true, "description": "..."}],
    "constraints": ["..."],
    "assumptions": [{"field": "...", "resolved_to": "...", "reason": "..."}],
    "delegations": [{"field": "...", "term": "...", "resolved_to": "...", "rationale": "..."}],

    // THE DECLARED GATES AND THE SAMPLER'S FACTS -- the pipeline executes these
    // (see the grammar and rules after the schema). Emit them whenever the task
    // mines instances from real public sources, which is every spec whose `data`
    // describes repositories or datasets to search:
    "entry_contract": [
      {"field": "<lead field>", "require": "<one grammar form>", "teach": "<optional guidance shown on refusal>"}
    ],
    "exclusions_on": ["<lead fields matched against the spec's exclusion roster>"],
    "corroboration_contract": [
      {"field": "<field>", "require": "is non-empty", "teach": "..."},
      {"run": "<command template over {fields}>", "require": "<run form>", "derive": "<optional field to fill>"},
      {"tracker": "<url field>", "policy": {"jira_types": ["..."], "label_words": ["..."], "text_pattern": "<regex>", "kind_into": "<field>"}},
      {"marker_search": {"clone": "<field>", "path": "<field>", "system": "<field>", "derive": "<field>", "families": {"<family>": ["<marker file>", "..."]}}}
    ],
    "corroboration_accept": "<one sentence over {fields} shown when every step holds>",
    "sampling": {
      "pool_requires": ["<fields every source-pool entry records>"],
      "spread_by": "<pool field whose distinct values measure spread>",
      "pool_exclusion_fields": ["<pool fields matched against the roster>"],
      "lead_requires": ["<every field a registered lead must carry -- superset of entry_contract fields>"],
      "lead_key": ["<the fields that IDENTIFY a lead, matching the constructor's identity>"],
      "normalize": {"<key field>": "none | git_url | full_sha"},
      "caps": [{"dimension": "<name>", "keys": ["<lead fields>"], "limit": 1}],
      "found_enforces": ["<one line per property the entry gate enforces>"],
      "delivered_enforces": ["<one line per property corroboration enforces>"],
      "search_hints": "<where this task's spec says to look: APIs, indexes, what to verify>",
      "strategies": [{"kind": "<name>", "strategy": "<one mining approach>"}],
      "mining_quality_gates": ["<what every delivered lead guarantees>"]
    },
    "report_discovery": {"<report kind>": ["<find pattern the ecosystem's runner writes reports under>"]}
  }
}

CRITERIA vs ADMISSIBILITY -- different lists, never duplicates of each other:
  "admissibility" (carried verbatim from the raw spec) filters SOURCES: what a
      repository or dataset must BE to be mined at all -- language, activity,
      popularity, not-a-fork.
  "criteria" states what EVERY SHIPPED INSTANCE guarantees: the properties a consumer
      relies on, which the construction pipeline must PROVE for each instance. Each
      one becomes an obligation some construction state has to enforce, so a property
      missing here is a property nothing will ever check.
      Repair benchmark: "the fix commit's own added/modified test fails at the parent
      commit and passes at the fix commit, confirmed by an actual test run"; "both
      revisions build with the project's own build tool".
      Optimization benchmark: "the optimized build reproduces the default build's
      output byte-for-byte on every recorded input".
      Crash-reproduction benchmark: "the proof-of-concept triggers the reported
      sanitizer class on the vulnerable revision and not on the fixed one".
      Restating "language must be Java" or "stars >= 100" as a criterion is a SOURCE
      filter in the wrong list and is refused: those belong to admissibility.
"""


#: The output slot for a resolved admissibility block, added to the schema ONLY when the raw
#: spec wrote its sources as prose. When the author already wrote clauses they are carried
#: verbatim and the model's copy would be discarded, so offering the slot then would invite a
#: paraphrase that silently loses -- which is the drift `admissibility` is carried verbatim to
#: prevent. `admissibility_shape` is the one place that decides which case this is.
#:
#: Numeric thresholds come back well-formed (`>= 200`, `>= 1000`) but non-numeric ones tend
#: to be unparseable (`is C or C++`, `== 'Python'`, `true`, `is a redistributable license`).
#: The grammar is stated in check 3, so it is restated HERE, beside the field being written,
#: with a worked example of each failing shape.
ADMISSIBILITY_SLOT = """
    "admissibility": [
      {"name": "<plain field name>", "observe": "<how to fetch it>", "require": "<predicate>"}
    ],
"""

#: The `require` grammar, verbatim beside the slot. Kept next to it deliberately.
_REQUIRE_GRAMMAR = """
  `require` IS A GRAMMAR, NOT A SENTENCE. Exactly one of these five forms, and nothing else
  parses -- a clause the parser cannot read refuses every source at runtime:
      >= N   <= N   > N   < N   == N        a number
      within N days|weeks|months|years      a recency
      is true   |   is false                a boolean
      one of [a, b, c]                      a membership, lowercase, comma-separated
  The mistakes made on this exact field before, each with what to write instead:
      "is C or C++"                  ->  "one of [c, c++]"
      "== 'Python'" / "Python"       ->  "one of [python]"
      "true"                         ->  "is true"
      "is a redistributable license" ->  "one of [mit, apache-2.0]" IF the spec names them.
                                         If it does not, which licenses count is the
                                         researcher's call and not yours: leave the clause
                                         out and record the omission in `assumptions`. Do
                                         not choose a set, and do not reject the whole spec
                                         over it -- see the note below.
      "is true (not in the list)"    ->  invert the field instead: name it
                                         `is_excluded_project` and require "is false"
  If a property cannot be written in one of the five forms, it is not an admissibility clause.
  Say so in `assumptions` and leave it out rather than writing prose into `require`.

  THIS RULE IS ABOUT `require` AND NOTHING ELSE. It does not widen the ambiguity lint in check
  4, and a term you cannot fit into this grammar is not thereby an AMBIGUOUS_TERM. Check 4
  fires on a term whose readings would MINE DIFFERENT INSTANCES -- "popular", "large project",
  "recent" -- not on ordinary prose that a competent reader would act on identically. "not a
  merge commit", "C/C++ only" and "touches only C/C++ files" are precise enough; demanding that
  a spec define them refuses every spec ever written, which is worse than the silence it
  replaced.

  `observe` NAMES ONE PROCEDURE: an API call and the field it returns, or a command and what
  it prints. Not two joined by "or" -- two procedures return two different numbers and nothing
  records which was used -- and not an instruction to go and look. The mistakes made on this
  exact field before, each with what to write instead:
      "Check project primary language via GitHub API or repository contents: exists a
       `pytest.ini`, `setup.cfg` with pytest, or similar indicator"
                                     ->  "GET /repos/{org}/{project} -> language"
      "Check the latest commit date: git log -1 --format=%cd (or GitHub API 'pushed_at')"
                                     ->  "GET /repos/{org}/{project} -> pushed_at"
      "Check if the repository name appears in the known list of excluded projects"
                                     ->  NOT AN ADMISSIBILITY CLAUSE AT ALL. A roster of named
                                         projects a benchmark refuses is read straight out of
                                         this spec's prose by the harness. Leave it in
                                         `constraints` where you found it and write no clause.
  Every one of those began with "Check". If the sentence starts with check, look, inspect,
  examine, review or see, it is not a procedure yet.
"""

_CONTRACTS_TASK_TEMPLATE = """
THE DECLARED GATES ARE EXECUTED, NOT READ. A task-blind engine runs
`entry_contract` on every lead the miner registers and `corroboration_contract`
against a real clone before a lead is DELIVERED; the sampler agent itself is
GENERATED from `sampling`. Nothing task-specific lives in code, so anything you
leave out of these blocks does not exist, and anything outside the grammar
refuses every lead at runtime.

`entry_contract` -- one clause per lead field that has a checkable shape.
`require` is a grammar, not a sentence. The exact forms:
{entry_forms}
plus: "matches <regex>", "is a command invoking {{<field>}}" (the named field's
value must sit in the command's invocation position; version/help banners are
refused), and the admissibility five-form grammar (>= N / within N months /
is true / one of [a, b]). A `teach` sentence per clause is shown to the miner on
refusal -- write the guidance a refused miner needs, e.g. what a correct value
looks like and where it is observed. Use "is distinct across leads" for any
field whose value is an observation about ONE lead (a command, an excerpt): the
same value on many leads is a template, not an observation.

`corroboration_contract` -- the steps the harness runs itself, in order, inside
the clone the miner records; the first failing step refuses. Each step declares
exactly one kind: a field clause; {{"run": <command template over {{fields}}>,
"require": <form>, "save"/"derive": <name>}}; {{"tracker": <url field>,
"policy": ...}} to fetch a tracker item live and judge it by declared vocabulary
(prefer this over a run step whenever the task involves issues or pull
requests); {{"marker_search": ...}} to find a declared marker file family along
a path; {{"timed": "{{cmd_field}}", "cwd": "{{path_field}}", "repetitions": 3,
"derive": "<field>", "timeout": <seconds>}} to have the HARNESS run a recorded
command that many times, take the wall-clock itself and derive the median in
milliseconds. Derived fields (a resolved sha, a tracker's kind, a found marker, a
timing) are filled by the harness -- never ask the miner to type them. EVERY
FIELD A RATIO CLAUSE COMPARES ("is at least N times field <other>") MUST BE
DERIVED BY A `timed` STEP: a number the miner's own command prints is the
miner's number. A build that the timing depends on is a run step that exports
the configuration and hands the recorded recipe to a shell whole, e.g.
      {{"run": "cd {{clone_path}} && env CFLAGS=-O0 sh -c {{build_cmd}}",
       "require": "exit zero", "timeout": 1800}}
and never puts a placeholder inside quotes or runs one as a whole command.

RUN STEPS EXECUTE IN THE HARNESS'S OWN WORKING DIRECTORY, never inside any
repository. Every command that operates on a repository must anchor to a place
the MINER recorded: first a field clause such as {{"field": "clone_path",
"require": "is non-empty", "teach": "record where you cloned this repository"}},
then `git -C {{clone_path}} ...` in each git step. The mistake made on this
exact block before, with what to write instead:
      {{"run": "git fetch --depth=2 origin {{merge_commit}}"}}
          ->  {{"run": "git -C {{clone_path}} rev-parse --verify {{merge_commit}}^{{commit}}",
               "require": "output is a git sha"}}
(a naked git command failed with 'not a git repository' on every candidate of a
live run; the fetch is also unnecessary -- the miner already cloned).

A run step's `require` IS A GRAMMAR TOO. These seven forms and nothing else --
an entry-clause form inside a run step does not parse:
      "exit zero"
      "exit zero and some output line"
      "output is a git sha"
      "output lines are exactly [{{field}}]"
      "some output line ends with one of [.c, .h]"
      "output matches <regex>"
      "last output line is the commit {{field}}"
The mistake made on this exact field before, with what to write instead:
      "output is a positive integer"  ->  "output matches ^[0-9]+$"

`sampling` -- every per-task fact the generated sampler needs: what a source-pool
entry records (pool_requires), how spread is measured (spread_by),
pool_exclusion_fields (which POOL entry fields -- typically its name and URL
fields -- get matched against the spec's exclusion roster), the lead's
identity (lead_key -- it must match how the CONSTRUCTOR identifies instances),
every field a lead carries (lead_requires -- a superset of the entry_contract's
fields; the gate may not demand what the state never collects), diversity caps
with their limits, per-state enforces lines, search_hints (where the spec says
to look, concretely: which APIs, which indexes, what to verify per source),
strategies (2-5 distinct mining approaches), and mining_quality_gates.
ALL of {sampling_required} are REQUIRED in a sampling block -- one missing key
refuses the whole spec, and nothing downstream invents a value.

`report_discovery` -- whenever any contract or variant declares a report kind
(e.g. results: junit-xml), declare where that ecosystem's runner writes report
files, as find patterns (e.g. "*surefire-reports/TEST-*.xml", "TEST-*.xml").
The harness composes its discovery command from this and knows no paths of its
own.

A LEAD IS THE MINER'S UNIT, NOT THE FINISHED INSTANCE. lead_requires and the
entry_contract describe what the MINER can observe cheaply and register: the
source's identity plus the concrete POINTER to one candidate exactly as
candidate_definition states it, plus small facts a listing or API call PRINTS.
The test: could `git log`, an API response or a directory listing have printed
this value? If producing the value needs extracting a diff, reading file
contents, or RUNNING anything, it belongs to construction and is never a lead
field -- a miner asked for it will fabricate it. The mistake made on this exact
block before, with what to write instead: for a merged-pull-request candidate,
    lead fields like "patch", "test_patch", "FAIL_TO_PASS", "problem_statement"
        ->  the POINTER instead: the merge commit sha, the base sha, the pull
            request or closed-issue URL, the org and repo -- the construction
            stage extracts diffs and runs tests later.

HOW MUCH THE FINDING STAGE PROVES IS THE SPEC'S DECISION, AND YOU READ IT OFF
THE SPEC. Some specs want a cheap finder: it screens each source on what the
source says about ITSELF -- metadata, the presence of documented build
instructions, the project's own description of the work it does -- and every
build, input and measurement belongs to the construction stage. Other specs
want the finder to prove a per-candidate property first, so construction never
pays a clone and a build for a candidate that cannot satisfy it. Both are
legitimate; the spec says which, in its own words about who does what, and you
project THAT. Never add screening the spec does not ask for, and never drop
screening it does.

  * WHEN THE SPEC SAYS THE FINDER DOES NOT BUILD OR MEASURE: the lead carries
    NO timings and no build of its own. Project the screening the spec DOES
    describe, and make it evidence rather than opinion: the fields naming
    where each documented fact was found, and corroboration steps that verify
    the recorded evidence exists where the lead says it does. A prose field
    gated only on non-emptiness is not evidence: one round shipped 30 of 30
    leads carrying the same boilerplate sentence, and the instances built from
    them cost 2.3x the cycles per delivery. So a recorded QUOTE gets a step
    that finds it in the file the lead names, not just a step that checks the
    file exists:
      {{"field": "doc_file", "require": "is a relative path to one file"}},
      {{"field": "doc_quote", "require": "is non-empty"}},
      {{"run": "grep -Fq -- {{doc_quote}} {{clone_path}}/{{doc_file}}",
       "require": "exit zero",
       "teach": "copy the line VERBATIM from that file; a paraphrase is not there"}}
    -- one such step per recorded quote, and the file-existence step becomes
    redundant once the grep runs.

  * WHEN THE SPEC ASKS THE FINDER TO PROVE A RUNTIME PROPERTY: project it as
    EXECUTED evidence -- the recipes the lead records, run steps that build
    each configuration under its exported flags, `timed` steps that derive
    each compared number, and entry clauses over those derived fields. A
    number the miner types, or one its own command prints, is not evidence
    (in practice a miner's own timings are often placeholders). Whatever you project, project it WHOLE: a floor
    without the comparison, or a comparison without the floor, screens on the
    arbitrary half. Two timed steps compared to each other must not time the
    same recorded command unless a build step between them changes what it
    measures -- otherwise both sides measure the same thing and the clause can
    only fail (81 candidates, 1 lead, 66 refusals of one such clause).

Either way `search_hints` steers the miner toward the program classes the
spec's own criteria describe, and away from the ones it rules out. THE SCOPE
OF ALL OF THIS: it projects what the SPEC STATES, and only for specs that mine
candidates from sources. A spec that generates its instances synthetically, or
states no per-candidate screening, gets NO invented floor, probe, or
admissibility demand -- writing one in would be authoring task content the
spec never stated.

WHERE AN EXCLUSION ROSTER LIVES: the named projects a benchmark refuses stay
VERBATIM in `constraints` (the harness reads the roster out of the spec's own
prose), and `exclusions_on` names which lead fields get matched against it.
Never restate the roster in `criteria` -- a criterion is a per-instance
guarantee some construction state must prove, and "not on the list" was already
settled at the gate, for free.

Emit all of these for a spec that mines real sources. Numbers you cannot ground
in the spec (cap limits, thresholds) follow the same rule as everywhere else:
resolve through the term table where one applies and record the resolution in
`assumptions`; never invent silently.
"""


def _contracts_task() -> str:
    from benchcompiler.spec import ENTRY_REQUIRE_FORMS, RUN_REQUIRE_FORMS, SAMPLING_REQUIRED_KEYS

    return _CONTRACTS_TASK_TEMPLATE.format(
        entry_forms="\n".join(f'      "{f}"' for f in ENTRY_REQUIRE_FORMS),
        run_forms=", ".join(f'"{f}"' for f in RUN_REQUIRE_FORMS),
        sampling_required=", ".join(SAMPLING_REQUIRED_KEYS),
    )


_ADMISSIBILITY_TASK = (
    """
ADMISSIBILITY IS PROSE IN THIS SPEC, SO RESOLVING IT IS YOUR JOB HERE.
The `admissibility` field above is a sentence, not clauses. Turn it into the clause list
described in check 3 -- one clause per property the sentence actually claims, each with
`name`/`observe`/`require` -- and emit it as `admissibility` in canonical_spec. Resolve every
fuzzy word through the term table (check 5) and record each resolution as an assumption naming
the word and the number you chose. Do not add a dimension the sentence does not claim, and do
not drop one it does. If the sentence claims nothing checkable about sources at all, REJECT
with MISSING_ADMISSIBILITY rather than inventing a floor.
"""
    + _REQUIRE_GRAMMAR
)


def build_prompt(raw: dict, feedback: str = "", previous=None) -> str:
    raw_text = "\n".join(f"{k}: {v}" for k, v in raw.items())
    term_defaults_text = "\n".join(f'  - "{k}": {v}' for k, v in TERM_DEFAULTS.items())
    oracle_patterns_text = "\n".join(f"  - {k}: {v}" for k, v in ORACLE_PATTERNS.items())
    default_gates_text = "\n".join(f"  - {g}" for g in DEFAULT_QUALITY_GATES)
    delegable_text = ", ".join(sorted(DELEGABLE_WHOLE_FIELDS))
    schema_desc = FORMALIZE_SCHEMA_DESC
    admissibility_task = ""
    if admissibility_shape(raw.get("admissibility")) == "prose":
        schema_desc = schema_desc.replace(
            '    "assumptions":', ADMISSIBILITY_SLOT.strip("\n") + '\n    "assumptions":', 1
        )
        admissibility_task = _ADMISSIBILITY_TASK
    contracts_task = _contracts_task()
    retry = retry_block(feedback, previous)

    return f"""You are the formalization front-end of BenchMeUp. A researcher
submitted the spec below. Formalize it into a canonical spec, or REJECT it
with precise diagnostics. This is compile-or-reject: no dialogue, no
guessing past what's stated or safely defaultable.

=== RAW SPEC ===
{raw_text}
=== END RAW SPEC ===

Required fields: {", ".join(REQUIRED_FIELDS)} (all present per a mechanical
pre-check that already ran; your job is SEMANTIC sufficiency, not presence).
Optional fields: {", ".join(OPTIONAL_FIELDS)}.

CHECKS TO PERFORM, IN ORDER:

1. `data` must state a specific unit of instancehood — the single
   event/artifact that counts as ONE instance (e.g. "a merged PR that
   closes an issue and modifies tests", not just "GitHub data"). If this is
   missing or too vague to mine from, reject with MISSING_INFO(data, ...).

2. `correctness` must be OPERATIONAL: an executable check, a named external
   authority (a real process — a judge platform, a CVE registry, a
   maintainer's PR-merge decision), or explicit human-judgment criteria
   PLUS a validation plan. Circular statements ("should be correct",
   "matches the expected behavior" without saying how) get
   NON_OPERATIONAL_CORRECTNESS. If it names human judgment but no human
   budget is stated anywhere (including `scale`), that's
   UNPRICEABLE_CORRECTNESS — human judgment is fine, but it must be priced.
   Map the resolved mechanism onto ONE of these named oracle patterns
   (or "other" if genuinely novel, with a precise oracle_detail regardless):
{oracle_patterns_text}

3. `admissibility` must state, in checkable terms, what makes a SOURCE worth
   sampling from -- one clause per property, each with:
     - name:    the value to record (a plain field name)
     - observe: HOW to obtain it, concretely enough that two people fetch the
                same number ("GET /repos/{{org}}/{{project}} -> stargazers_count",
                "git rev-list --count HEAD"). "look at the project" is NOT a
                procedure -- reject it with UNMEASURABLE_ADMISSIBILITY.
     - require: `>= N`, `<= N`, `within N months`, `is true/false`,
                `one of [a, b]`. Anything else is not measurable.
   Which properties matter is the RESEARCHER's call and differs by task (stars
   and commits for a repository, entrants for a contest round, downloads for a
   package index) -- do not impose a dimension they did not ask for. If the spec
   is silent, reject with MISSING_ADMISSIBILITY rather than inventing a floor:
   a benchmark whose sources nobody bounded gets whatever the sampler happens
   to find, which has meant student homework and tutorial repositories.
   If the researcher stated a fuzzy word ("popular", "active"), resolve it
   through the term table below INTO a clause and record the assumption.

4. Ambiguity linting — scan every field for vague, unquantified terms.
   Known terms and their resolutions (apply if the term appears, record as
   an assumption; several are explicitly NOT safely defaultable, meaning
   you must reject with AMBIGUOUS_TERM instead of guessing):
{term_defaults_text}
   For any OTHER vague/unquantified term not in this list, use judgment:
   if a reasonable, narrow default exists, apply it and record it as an
   assumption; if genuinely ambiguous (multiple materially different
   readings), reject with AMBIGUOUS_TERM and list the candidate readings.

5. DELEGATE handling — a field or sub-item may say DELEGATE instead of a
   value. DELEGATE is valid on: {delegable_text}, on `scale`'s instance
   count specifically, and on sub-items within `criteria`/`quality`. It is
   INVALID as a whole-field answer for task/instance/data/correctness
   themselves (the system cannot invent what the benchmark is FOR, or its
   core correctness mechanism) — reject those with INVALID_DELEGATION. For
   every valid DELEGATE, resolve it yourself and record it as a delegation
   (field, term, resolved_to, rationale) — this is a binding, recorded
   choice, not a silent default.

6. THE THREE DECLARATIONS YOU DO NOT WRITE, AND MUST STILL CHECK FOR. These are
   carried into the canonical spec VERBATIM from the raw spec above -- a
   paraphrase of `cp /bin/true {{binary}}` is not a command, so you are not
   asked to produce them. But a spec that omits one is REJECTED after you
   answer, by a check you cannot see, and the researcher then gets a bare
   code instead of your reading of their spec. So check them yourself and
   reject with these codes, naming exactly what is absent:
     solver_input       how a solver hands this benchmark its answer.
                        Absent -> NO_SOLVER_INPUT. A benchmark whose task
                        cannot be attempted through its own interface is not
                        a benchmark; one shipped pool's task was "find a flag
                        configuration" and its artifact had no verb that
                        accepts a flag string.
     negative_control   how an instance is broken on purpose, so a check can
                        be shown able to FAIL. Required whenever
                        artifact_variants are declared. Absent ->
                        NO_NEGATIVE_CONTROL. `cp /bin/true <binary>` passed
                        every check in a shipped pool of sixteen.
     artifact_variants  the two or more states of the instance the oracle
                        contrasts. Required when your oracle_strategy is one
                        that establishes truth by contrast (fail-to-pass-tests,
                        execution-replay, injection, reference-equivalence) --
                        which is YOUR classification, made in check 2, so only
                        you can tell whether this spec needs them. Absent ->
                        MISSING_VARIANTS.

   WORKED, from a real spec that declares none of the three. Its `correctness`
   says "apply the candidate bug-injection patch, run the repo's test suite,
   keep the candidate only if at least one test flips fail->pass"; its
   `instance` says "expected output: a patch (.diff)". Nowhere in it are the
   words solver_input, negative_control or artifact_variants. The three
   diagnostics that spec must draw, and their exact shape:

     {{"code": "NO_SOLVER_INPUT", "field": "solver_input",
      "detail": "the task expects a patch (.diff) and nothing states how a
                 solver submits one, so nothing can check the artifact accepts
                 it",
      "fix_hint": "declare solver_input: {{kind: workdir, path: '{{clone_path}}',
                   describes: 'the solver edits the checked-out tree'}}"}}

     {{"code": "MISSING_VARIANTS", "field": "artifact_variants",
      "detail": "oracle_strategy is fail-to-pass-tests, which decides by
                 CONTRASTING the broken and fixed states, and the spec names
                 neither",
      "fix_hint": "declare artifact_variants: one entry per state, each with
                   name, check_cmd, expect, results and expect_failures"}}

     {{"code": "NO_NEGATIVE_CONTROL", "field": "negative_control",
      "detail": "nothing states how to break an instance on purpose, so no
                 shipped check can be shown able to FAIL",
      "fix_hint": "declare negative_control: {{break_cmd: <field holding the
                   command>, check: <variant whose verdict must flip>}}"}}

   REJECT with all three at once -- not one, and not silently accepting because
   the rest of the spec reads well. A spec can be excellent prose and still be
   missing every block that makes it executable, which is the common case.

7. If `quality` was thin or omitted, ADD these defaults to quality_gates
   (in addition to anything the researcher stated) and record as an
   assumption:
{default_gates_text}

8. If ALL checks pass (possibly with assumptions/delegations recorded),
   output ACCEPT with the full canonical_spec. If ANY check fails, output
   REJECT with ALL diagnostics found (not just the first one) — a
   researcher fixing a rejected spec should see everything wrong at once.

{admissibility_task}
{contracts_task}
{retry}
Respond with ONLY a JSON object matching this shape (no other text):
{schema_desc}"""


def _carried_admissibility(raw: dict, canonical_spec: dict) -> list:
    """The clause list that leaves stage 1, by shape of what the author wrote."""
    shape = admissibility_shape(raw.get("admissibility"))
    if shape == "clauses":
        return raw["admissibility"]
    if shape == "prose":
        resolved = (canonical_spec or {}).get("admissibility")
        return resolved if isinstance(resolved, list) else []
    return []


def _admissibility_exemption(raw: dict) -> str:
    value = raw.get("admissibility")
    return value.strip() if admissibility_shape(value) == "exempt" else ""


#: Diagnostics the MODEL authored, and can therefore fix by being told. Everything else is
#: about a block carried VERBATIM from the raw spec, or about the researcher's own prose -- the
#: model cannot change one of those by trying again, so retrying is money spent confirming a
#: foregone conclusion, and compile-or-reject stays the stance.
#:
#: `UNMEASURABLE_ADMISSIBILITY` is unambiguous here even though a researcher can also write
#: clauses: a researcher-written clause list is validated by `mechanical_precheck` BEFORE any
#: call, so one reaching `check_canonical` was resolved from prose by the model.
MODEL_AUTHORED_CODES = frozenset(
    {
        "MISSING_VIABILITY_PROJECTION",
        "UNMEASURABLE_ADMISSIBILITY",
        "CRITERIA_ARE_SOURCE_FILTERS",
        "UNCLASSIFIED_ORACLE",
        # The declared-gate diagnostics are the model's own writing by definition -- an
        # unreadable clause, an incomplete sampling block, a normalize map misaligned with the
        # lead key. Listing them here lets the retry loop engage instead of a final reject.
        "UNREADABLE_CONTRACT",
        # A ratio clause over fields no run step derives: the corroboration contract is the
        # model's projection of the spec, so the fix is the model's.
        "UNEXECUTED_COMPARISON",
    }
)

#: Rounds the formalizer gets to fix ITS OWN answer. The `require` grammar is taught beside
#: the output slot, yet a re-draw can still find a fresh way out of it -- a name with spaces,
#: a comparison and a recency conjoined, a paragraph. That tail is not enumerable by worked
#: examples; it is recoverable in a few rounds, each with one named diagnosis. A round costs
#: only on failure paths, and a retry happens ONLY for a diagnosis the model itself can fix.
FORMALIZE_ROUNDS = 6


def _records(entries, cls):
    """Build `cls` from model-emitted dicts without splatting them into the constructor.

    `Assumption(**a)` raised `TypeError: unexpected keyword argument 'term'` on a real reply --
    `term` belongs to `Delegation`, the two blocks sit adjacent in the schema and share two key
    names, and the confusion is an easy one to make. That exception came OUT of `formalize`: the
    caller got a traceback instead of a diagnostic, `--lock` wrote nothing, and the retry loop
    never ran because nothing returned. A reply that is well-formed JSON with one key too many
    is the likeliest way for a reply to be slightly wrong, and it took the whole stage down.

    Unknown keys are dropped, missing ones default, and a non-object entry is skipped: the
    stage's contract is compile-or-reject, and a traceback is neither.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    out = []
    for e in entries or []:
        if isinstance(e, dict):
            out.append(cls(**{k: str(v) for k, v in e.items() if k in known}))
    return out


def _resolved_criteria(raw: dict, cs: dict) -> tuple[list, list]:
    """(criteria, extra_assumptions): criteria the model emitted, minus the ones
    that RESTATE a source filter whose content verifiably lives where it belongs.

    The harness resolves notation, it does not refuse it (the standing rule; this
    was its seventh instance): a criterion like "the project is not in the
    exclusion list (...)" has one unambiguous resolution -- drop it, because the
    roster already sits verbatim in `constraints` and the exclusion gate reads it
    from there -- and the model re-emitted it through two rounds of feedback
    saying exactly that. Only the VERIFIABLY duplicated case is resolved (the
    roster's names appear in constraints/admissibility); a source filter whose
    content lives nowhere else still refuses via check_canonical, since dropping
    it would lose the researcher's meaning. Every drop is recorded as an
    assumption, so nothing leaves the spec silently."""
    from benchcompiler.spec import _source_filter_reason

    crit = [str(c) for c in (cs.get("criteria") or [])]
    admis = cs.get("admissibility") or raw.get("admissibility") or []
    names = {str(a.get("name") or "").strip().lower() for a in admis if isinstance(a, dict)}
    names = {n for n in names if n}
    elsewhere = " ".join(str(x) for x in (raw.get("constraints") or [])).lower()
    elsewhere += " " + " ".join(str(x) for x in (cs.get("constraints") or [])).lower()

    kept, dropped = [], []
    for c in crit:
        why = _source_filter_reason(c, names)
        if not why:
            kept.append(c)
            continue
        words = [w.strip(" ,.()").lower() for w in c.split() if len(w.strip(" ,.()")) > 3]
        duplicated = why == "restates the admissibility clause it is named after" or (
            words and sum(w in elsewhere for w in words) >= max(2, len(words) // 3)
        )
        if duplicated:
            dropped.append((c, why))
        else:
            kept.append(c)  # not resolvable -- check_canonical will say so
    extra = [
        Assumption(
            field="criteria",
            resolved_to=f"dropped: {c[:120]}",
            reason=(
                f"this criterion {why}; its content already lives in "
                f"the source-filter blocks the gates read, so keeping "
                f"it would spend a construction state re-proving a "
                f"settled fact"
            ),
        )
        for c, why in dropped
    ]
    return kept, extra


def _raw_criteria_items(raw: dict) -> list[str]:
    """The researcher's criteria as they wrote them, one per bullet: a YAML list as is, a
    prose block split on its `- ` bullets (a bullet's continuation lines folded into it).
    Empty when the raw spec writes no criteria block or no bullets -- then the model's
    list stands, since there is nothing to carry."""
    crit = (raw or {}).get("criteria")
    if isinstance(crit, list):
        return [" ".join(str(c).split()) for c in crit if str(c).strip()]
    if not isinstance(crit, str):
        return []
    items: list[str] = []
    for line in crit.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
        elif stripped and items:
            items[-1] = items[-1] + " " + stripped
    return [" ".join(i.split()) for i in items if i.strip()]


def build_canonical(raw: dict, cs: dict) -> CanonicalSpec:
    """The CanonicalSpec a model reply becomes. ONE home.

    The criteria are the researcher's words. A model that rewrites the raw `criteria` block
    into its own shorter list loses ordering and precision ("working setup FIRST", "coefficient
    of variation" -> "variance reported"). The model projects gates FROM the criteria; it does
    not re-word them. When the raw spec writes them as bullets they are carried verbatim, one
    per bullet, the same way the structured blocks below are; the source-filter resolution
    still applies to them.

    Extracted from `formalize` so a scenario harness cannot grade a DIFFERENT dict
    from the one production builds. It did: the harness hand-rolled nine keys and
    dropped `candidate_definition`, `instance_input`, `instance_output`,
    `data_source` and `oracle_detail`, so the emptiness check added the same day
    fired on fields the HARNESS had lost and reported them as the model's omissions.
    """
    scale = cs.get("scale", {}) or {}
    verbatim = _raw_criteria_items(raw)
    cs_crit = {**cs, "criteria": verbatim} if verbatim else cs
    return CanonicalSpec(
        spec_id=_spec_id(raw),
        task=cs.get("task", ""),
        instance_input=cs.get("instance_input", ""),
        instance_output=cs.get("instance_output", ""),
        data_source=cs.get("data_source", ""),
        candidate_definition=cs.get("candidate_definition", ""),
        oracle_strategy=cs.get("oracle_strategy", "other"),
        oracle_detail=cs.get("oracle_detail", ""),
        interface=cs.get("interface", ""),
        scale_instance_count=scale.get("instance_count", ""),
        scale_compute_budget=scale.get("compute_budget", ""),
        scale_human_budget=scale.get("human_budget", "0"),
        criteria=_resolved_criteria(raw, cs_crit)[0],
        quality_gates=cs.get("quality_gates", []) or [],
        examples=cs.get("examples", []) or [],
        constraints=cs.get("constraints", []) or [],
        assumptions=(_records(cs.get("assumptions"), Assumption) + _resolved_criteria(raw, cs_crit)[1]),
        delegations=_records(cs.get("delegations"), Delegation),
        # Verbatim from the RAW spec, not from the model's canonical_spec: these are
        # structured blocks the harness validates, and the model paraphrasing them on the
        # way through is exactly the drift this module exists to prevent.
        # ...with ONE exception, and it is not a loosening. `admissibility` is carried
        # verbatim only when the author wrote CLAUSES -- a structured block an LLM must not
        # re-word. Prose is not a structured block: it is the fuzzy sentence the term table
        # exists to resolve, and refusing it before the call made that whole mechanism
        # unreachable, so the model's resolution is taken and then held to the same parser
        # by `check_canonical`. An exemption travels in its own field, because
        # `instantiate` wires the admissibility gate on this one's truthiness and
        # `adm.evaluate` cannot be called on a sentence.
        admissibility=_carried_admissibility(raw, cs),
        admissibility_exemption=_admissibility_exemption(raw),
        artifact_variants=raw.get("artifact_variants") or [],
        # Same rule, same reason: how a solver submits its answer and how an
        # instance is broken on purpose are structured blocks the harness executes at
        # packaging. A paraphrase of "cp /bin/true {binary_path}" is not a command.
        solver_input=raw.get("solver_input") or {},
        negative_control=raw.get("negative_control") or {},
        parameters=raw.get("parameters") or [],
        # The declared gates and the sampler's facts: carried verbatim when the
        # raw spec ships them structured; otherwise the model WROTE them from the
        # prose (that is the task the prompt sets), and check_canonical holds
        # every clause to the engine's own grammar.
        entry_contract=raw.get("entry_contract") or cs.get("entry_contract") or [],
        exclusions_on=raw.get("exclusions_on") or cs.get("exclusions_on") or [],
        corroboration_contract=(raw.get("corroboration_contract") or cs.get("corroboration_contract") or []),
        corroboration_accept=str(raw.get("corroboration_accept") or cs.get("corroboration_accept") or ""),
        sampling=raw.get("sampling") or cs.get("sampling") or {},
        report_discovery=(raw.get("report_discovery") or cs.get("report_discovery") or {}),
    )


def formalize(
    raw: dict, model: str, rounds: int = FORMALIZE_ROUNDS, *, judge: bool = True, judge_model: str | None = None
) -> FormalizeResult:
    """Specification -> search rules (the canonical spec), or a rejection with diagnostics.

    Rules that pass the mechanical checks are reviewed by a second model call against the
    specification (`judge`, by default with the same model as the generator). A refusal goes
    back to the generating model as feedback; when the rounds run out, the diagnostics go to
    the researcher instead of an unreviewed rule set."""
    pre_diagnostics = mechanical_precheck(raw)
    if pre_diagnostics:
        return FormalizeResult(outcome="REJECT", diagnostics=pre_diagnostics)
    feedback, previous, post = "", None, []
    for attempt in range(rounds):
        outcome = _formalize_once(raw, model, feedback, previous)
        if isinstance(outcome, FormalizeResult):
            if judge and outcome.outcome == "ACCEPT":
                rules = outcome.canonical_spec.to_dict()
                accepted, review = judge_rules(raw, rules, judge_model or model)
                _decide(
                    "the review accepted the rules" if accepted else "the review refused the rules",
                    "ACCEPT" if accepted else "RETRY",
                )
                if not accepted:
                    post = [Diagnostic(code="AMBIGUOUS_TERM", field="(search rules)", detail=review, fix_hint="")]
                    feedback = "\n".join(f"  - {d.detail}" for d in post)
                    previous = {"outcome": "ACCEPT", "canonical_spec": rules}
                    continue
            return outcome
        result, post = outcome  # a retryable rejection, with the reply
        feedback = "\n".join(f"  - {d}" for d in post)
        if str(result.get("outcome") or "") == "REJECT":
            # The model refused rather than fixed. Naming whose blocks these are is the
            # whole content of the round: without it the model has already shown it will
            # hand the same list back. The list is DERIVED -- the diagnosed fields that the
            # raw spec does not contain -- never restated by hand, which could become false
            # the moment a raw spec shipped one of them.
            yours = _blocks_the_model_writes(post, raw)
            if yours:
                feedback = (
                    f"  - You answered REJECT, but every diagnostic below is about a block "
                    f"YOU write, not one the researcher wrote: {', '.join(yours)} appear "
                    f"nowhere in the raw spec. A spec cannot be rejected for fields it does "
                    f"not contain. Emit ACCEPT with those blocks written correctly, or "
                    f"reject only for something the raw spec itself says.\n" + feedback
                )
        # A reject carries no canonical_spec; keep the last attempt that had one on screen,
        # or the retry shows the model a refusal instead of the answer it must repair.
        if result.get("canonical_spec"):
            previous = result
    return FormalizeResult(
        outcome="REJECT", diagnostics=[*post, _unconverged(rounds, _blocks_the_model_writes(post, raw))]
    )


def resolve_notation(raw: dict, cs: dict) -> tuple[dict, list[str]]:
    """Resolve, on record, what the model wrote in a shape the harness can read past.

    Two defects are not a verdict on the spec: a lead_key field with no entry in sampling.normalize (the generic
    treatment -- case, trailing slash, .git -- is what `none` means, so that IS the answer),
    and a delegation of a whole field the raw spec states (`scale` was stated; the model
    delegated it anyway). Refusing either is the harness refusing notation it can resolve
    (user rule); resolving it puts the choice in the canonical's assumptions, where a reader
    finds it. Returns the amended canonical_spec dict and one note per resolution."""
    notes: list[str] = []
    assumptions = cs.get("assumptions") if isinstance(cs.get("assumptions"), list) else []
    sampling = cs.get("sampling") if isinstance(cs.get("sampling"), dict) else None
    if sampling:
        key = [str(f) for f in (sampling.get("lead_key") or [])]
        norm = dict(sampling.get("normalize") or {}) if isinstance(sampling.get("normalize"), dict) else {}
        missing = [f for f in key if f not in norm]
        orphans = sorted(set(norm) - set(key))
        if key and (missing or orphans):
            for f in missing:
                norm[f] = "none"
            for f in orphans:
                norm.pop(f, None)
            sampling["normalize"] = norm
            note = (
                f"sampling.normalize: {', '.join(missing) or 'nothing'} given the normal form "
                f"'none' (the generic treatment: case, a trailing slash and .git do not change "
                f"the identity)" + (f"; dropped entries for non-key fields {orphans}" if orphans else "")
            )
            assumptions.append(
                {
                    "field": "sampling.normalize",
                    "resolved_to": ", ".join(f"{f}: none" for f in missing) or "orphans dropped",
                    "reason": note,
                }
            )
            notes.append(note)
    # Only the delegations the checker would refuse (a whole field outside the delegable
    # set) are looked at, and only when the raw spec states that field rather than marking
    # it DELEGATE: then the delegation is the model's slip, not the researcher's choice.
    from benchcompiler.registry import DELEGABLE_WHOLE_FIELDS

    delegations = cs.get("delegations") if isinstance(cs.get("delegations"), list) else []
    kept, stated = [], []
    for d in delegations:
        rec = d if isinstance(d, dict) else {}
        fname = str(rec.get("field") or "").strip()
        whole = bool(fname) and "." not in fname and fname not in DELEGABLE_WHOLE_FIELDS
        value = (raw or {}).get(fname) if whole else None
        if whole and value not in (None, "", [], {}) and str(value).strip() and "DELEGATE" not in str(value):
            stated.append(fname)
            continue
        kept.append(d)
    if stated:
        cs["delegations"] = kept
        note = (
            f"delegations of {', '.join(stated)} dropped: the raw spec states {', '.join(stated)} "
            f"and a stated field is not delegated"
        )
        assumptions.append({"field": "delegations", "resolved_to": "dropped: " + ", ".join(stated), "reason": note})
        notes.append(note)
    if notes:
        cs["assumptions"] = assumptions
    return cs, notes


def _raw_ships(raw: dict, field) -> bool:
    """Does the raw spec SHIP this block structured? A structured block wins in
    build_canonical, so the model cannot repair it and a retry is money spent confirming
    that. Prose (admissibility written as a sentence) is the model's to resolve -- that is
    the whole reason UNMEASURABLE_ADMISSIBILITY is a model-authored code."""
    top = str(field or "").split(".")[0].split("(")[0].strip()
    return bool(top) and isinstance((raw or {}).get(top), (list, dict))


#: The blocks build_canonical carries VERBATIM from the raw spec when it ships them. A model
#: diagnostic on one of these decides nothing: the model cannot change the block and the
#: harness validates it itself. `scale`, `data`, `correctness` are the model's to judge.
CARRIED_VERBATIM = (
    "artifact_variants",
    "solver_input",
    "negative_control",
    "parameters",
    "entry_contract",
    "exclusions_on",
    "corroboration_contract",
    "corroboration_accept",
    "sampling",
    "report_discovery",
)


def _carried_verbatim(raw: dict, field) -> bool:
    top = str(field or "").split(".")[0].split("(")[0].strip()
    return top in CARRIED_VERBATIM and _raw_ships(raw, top)


def _harness_flags_on_shipped_blocks(raw: dict) -> set[str]:
    """The shipped blocks the harness's own checks refuse, read the way check_canonical
    will read them after an ACCEPT: the canonical built from the raw spec alone. A field in
    this set is the researcher's problem whatever the model says; one absent from it is
    clean by the only judge that counts."""
    try:
        canonical = build_canonical(raw, {})
        flagged = {
            str(getattr(d, "field", "") or "").split(".")[0].split("(")[0].strip() for d in check_canonical(canonical)
        }
    except Exception:  # noqa: BLE001 - an unbuildable raw spec flags nothing here; the
        return set()  # precheck before the call is where that is refused
    return {f for f in flagged if f in CARRIED_VERBATIM}


def _blocks_the_model_writes(diags, raw: dict) -> list[str]:
    """The diagnosed top-level fields the raw spec does NOT contain -- the blocks only the
    model's answer has. Empty when every diagnosed field is the researcher's."""
    fields = {str(getattr(d, "field", "") or "").split(".")[0].split("(")[0].strip() for d in diags}
    fields.discard("")
    return sorted(f for f in fields if not _raw_ships(raw, f))


def _unconverged(rounds: int, yours: list[str]) -> Diagnostic:
    """The last word when the model never wrote a canonical that passes its own checks.

    Without it the researcher is handed diagnostics about blocks their spec does not contain
    and reads them as a verdict on their spec. `(system)` is the discriminator -- the same
    field the call-failed diagnostic uses -- and the blocks are named from the diagnostics,
    not restated.
    """
    named = f" ({', '.join(yours)})" if yours else ""
    return Diagnostic(
        code="MISSING_INFO",
        field="(system)",
        detail=(
            f"The formalizer could not produce a canonical spec that passes its own "
            f"checks in {rounds} rounds. The diagnostics above name blocks the "
            f"formalizer writes{named}, not ones you wrote."
        ),
        fix_hint=(
            "Re-run, or run with a stronger --model. Change your spec only if a "
            "diagnostic names a field your spec actually contains."
        ),
    )


def _decide(rule: str, output: str, codes: list | None = None) -> None:
    """The round's conclusion, into the open stage log (a no-op outside `bench formalize`)."""
    from agentcore.observability import stage_log

    log = stage_log.current()
    if log is None:
        return
    log.decision(
        "formalizer.verdict",
        rule={"why": rule, "codes": list(codes or [])},
        output=output,
        consequence="retry" if output == "RETRY" else "final",
    )


def _formalize_once(raw: dict, model: str, feedback: str, previous):
    """One cycle. Returns a FormalizeResult when the answer is final, or (reply, diagnostics)
    when every blocker is something the model itself wrote and another round could fix it."""
    # The canonical spec is a long structured answer; at the provider's default budget the
    # reply can come back empty. Same budget as the other generation sites.
    result = call_llm_json(model, build_prompt(raw, feedback, previous), site="formalizer.formalize", max_tokens=32000)
    if result is None:
        _decide("call failed", "REJECT")
        return FormalizeResult(
            outcome="REJECT",
            diagnostics=[
                Diagnostic(
                    code="MISSING_INFO",
                    field="(system)",
                    detail="Formalization call failed or returned invalid JSON.",
                    fix_hint="Retry; if this persists, check your --model / API key setup.",
                ),
            ],
        )

    outcome = result.get("outcome")
    if outcome == "REJECT":
        diags = []
        for d in result.get("diagnostics", []):
            code = d.get("code")
            if code not in DIAGNOSTIC_CODES:
                code = "AMBIGUOUS_TERM"
            diags.append(
                Diagnostic(
                    code=code,
                    field=d.get("field", "(unspecified)"),
                    detail=d.get("detail", ""),
                    fix_hint=d.get("fix_hint", ""),
                )
            )
        if not diags:
            diags = [
                Diagnostic(
                    code="AMBIGUOUS_TERM",
                    field="(unspecified)",
                    detail="Model rejected the spec but returned no diagnostics.",
                    fix_hint="Re-run; this is a formalizer-side issue, not necessarily your spec.",
                )
            ]
        # A REJECT NAMING ONLY THE MODEL'S OWN BLOCKS IS NOT A VERDICT ON THE SPEC.
        # Fed its own `check_canonical` diagnostics as retry feedback, the model can
        # answer REJECT and restate them as faults of the researcher's spec:
        # "add to entry_contract a clause relating probe_baseline_ms to probe_optimized_ms",
        # addressed to a spec with no entry_contract, no sampling block and no normalize map.
        # Those blocks exist only in the model's own answer, so the spec cannot be refused
        # for them, and restating the feedback is the cheapest exit from a retry round.
        # `MODEL_AUTHORED_CODES` already draws this line for the checker's diagnostics;
        # this is the same line on the model's.
        if all(d.code in MODEL_AUTHORED_CODES for d in diags) and not any(_raw_ships(raw, d.field) for d in diags):
            _decide("the model rejected the spec for blocks it writes itself", "RETRY", [d.code for d in diags])
            return result, diags
        # A DIAGNOSTIC ON A BLOCK THE RAW SPEC SHIPS STRUCTURED IS MOOT. A REJECT can name
        # declarations the raw spec ships as structured blocks -- which build_canonical
        # carries VERBATIM and the harness's own checks validate. The prompt asks the model to
        # flag those blocks when ABSENT; it judged their content ("`corruption_cmd` is a
        # placeholder, not a command" -- it is the field the builder records the command in).
        # The model cannot change a shipped block and the harness does not take its word on
        # one, so the diagnostic decides nothing: dropped on record. Naming only shipped
        # blocks buys a round that says so; naming the researcher's prose as well is final
        # on the prose alone, so the researcher reads a verdict about their spec and nothing
        # the harness already settled.
        # ...but only where the HARNESS, reading the same shipped block, finds nothing wrong.
        # A shipped block the harness's own checks refuse (an unreadable clause in a raw
        # corroboration_contract) is the researcher's to fix and the model was right; a
        # shipped block the harness reads clean is not the model's to reject. And only the
        # blocks build_canonical carries verbatim count as shipped: `scale` is a dict in the
        # raw spec and still the model's to judge.
        harness_flags = _harness_flags_on_shipped_blocks(raw)
        moot = [
            d
            for d in diags
            if _carried_verbatim(raw, d.field) and str(d.field).split(".")[0].split("(")[0].strip() not in harness_flags
        ]
        kept = [d for d in diags if d not in moot]
        if moot and not kept:
            _decide(
                "the model rejected only blocks the raw spec ships structured, which the "
                "harness carries verbatim and validates itself",
                "RETRY",
                [d.code for d in moot],
            )
            shipped = sorted({str(d.field).split(".")[0] for d in moot})
            return result, [
                Diagnostic(
                    code=d.code,
                    field=d.field,
                    detail=(
                        f"{d.field} is SHIPPED by the raw spec as a structured block and is "
                        f"carried into the canonical verbatim; the harness validates it, not "
                        f"you. You were asked to flag it only if ABSENT, and it is present "
                        f"({', '.join(shipped)}). Do not judge its content; emit ACCEPT."
                    ),
                    fix_hint="",
                )
                for d in moot
            ]
        if moot:
            _decide(
                "the model rejected the spec; its diagnostics on shipped blocks are moot and dropped",
                "REJECT",
                [d.code for d in kept],
            )
            return FormalizeResult(outcome="REJECT", diagnostics=kept)
        _decide("the model rejected the spec", "REJECT", [d.code for d in diags])
        return FormalizeResult(outcome="REJECT", diagnostics=diags)

    if outcome == "ACCEPT":
        cs = result.get("canonical_spec", {})
        cs, resolved = resolve_notation(raw, cs)
        for note in resolved:
            _decide("notation resolved on record: " + note, "RESOLVE")
        canonical = build_canonical(raw, cs)
        # Checks that need the formalizer's OWN answers rather than the author's raw prose.
        # `mechanical_precheck` above runs before an oracle strategy exists, so it cannot tell a
        # fail-to-pass benchmark from an llm-judge one -- and demanding a variant contract of
        # both is what rejected all eleven shipped examples and forbade four patterns the
        # registry advertises. Here the strategy is known, so each spec is asked only for what
        # its own oracle can supply.
        post = check_canonical(canonical)
        if post:
            # A code in MODEL_AUTHORED_CODES says the block is usually the model's; a raw
            # spec that SHIPS that block structured (build_canonical lets it win) makes it the
            # researcher's, and retrying cannot change it.
            if all(d.code in MODEL_AUTHORED_CODES for d in post) and not any(_raw_ships(raw, d.field) for d in post):
                _decide("the model accepted; every blocker is model-authored", "RETRY", [d.code for d in post])
                return result, post
            _decide("the model accepted; a blocker is the author's", "REJECT", [d.code for d in post])
            return FormalizeResult(outcome="REJECT", diagnostics=post)
        _decide("the model accepted and the canonical checks pass", "ACCEPT")
        return FormalizeResult(outcome="ACCEPT", canonical_spec=canonical)

    # malformed outcome field
    return FormalizeResult(
        outcome="REJECT",
        diagnostics=[
            Diagnostic(
                code="MISSING_INFO",
                field="(system)",
                detail=f"Formalizer returned an unrecognized outcome: {outcome!r}",
                fix_hint="Retry.",
            ),
        ],
    )


PROPOSE_SCHEMA_DESC = """{
  "proposals": [
    {
      "oracle_strategy": "<one of the named patterns, or 'other'>",
      "mechanism": "<a concrete, operational decision procedure -- specific enough that AMBIGUOUS_TERM/NON_OPERATIONAL_CORRECTNESS checks would pass on it as written>",
      "human_budget_implication": "<0h, or a real estimate if this strategy still needs human judgment>",
      "tradeoff_vs_stated": "<1-2 sentences: what this strategy measures differently/worse/better than what the raw spec currently states, if anything is currently stated>"
    }
  ]
}"""


def propose_oracle_strategies(model: str, raw: dict) -> dict | None:
    """Given a raw spec (regardless of whether it would formalize cleanly),
    propose 1-3 concrete alternative oracle strategies for `correctness` --
    not just critique the stated one. Each proposal must be operational
    enough to survive the same sufficiency checks as formalize() applies."""
    raw_text = "\n".join(f"{k}: {v}" for k, v in raw.items())
    oracle_patterns_text = "\n".join(f"  - {k}: {v}" for k, v in ORACLE_PATTERNS.items())
    prompt = f"""You are the oracle-strategy advisor for BenchMeUp. Given the
spec below, propose 1-3 CONCRETE alternative ways to establish correctness
for this task -- including automated proxies the researcher may not have
considered, not only refinements of what they already wrote. Each proposal
must be operational: precise enough that it would pass an operational-
correctness check on its own (state exact procedures, thresholds, sample
sizes -- not vague gestures at a strategy name).

=== RAW SPEC ===
{raw_text}
=== END RAW SPEC ===

Named oracle patterns to choose from (or propose "other" with an equally
precise mechanism):
{oracle_patterns_text}

Prioritize proposing at least one strategy with human_budget=0h if the task
plausibly admits an automated or semi-automated proxy (e.g. execution-based
differential testing, injection, external-authority), even if the raw spec
currently names a human-judgment oracle. If the task genuinely requires
human judgment with no credible automated proxy, say so as one proposal and
explain why automation is not viable for this specific task.

Respond with ONLY a JSON object matching this shape (no other text):
{PROPOSE_SCHEMA_DESC}"""
    return call_llm_json(model, prompt, site="formalizer.propose")


CORRECTNESS_CODES = {"UNPRICEABLE_CORRECTNESS", "NON_OPERATIONAL_CORRECTNESS"}


def formalize_with_fallback(
    raw: dict, model: str, *, judge: bool = True, judge_model: str | None = None
) -> FormalizeResult:
    """The general-case pipeline. Try to formalize as-is; if it rejects
    SOLELY because the stated correctness mechanism is unpriced or
    non-operational (not because of other structural gaps), automatically
    try the lowest-human-budget proposed alternative before giving up.

    This exists because "the paper's stated oracle doesn't meet the bar" is
    not a rare edge case -- across this project's own literature study, a
    real fraction of published papers have exactly this shape (see
    changeguard.yaml). Requiring a human to notice, run --propose, and
    hand-author a new spec every single time doesn't scale to processing
    many papers. This makes that loop automatic -- but the substitution is
    never silent: FormalizeResult.original_diagnostics and .substitution
    record exactly what was tried first and what replaced it, so nothing
    is hidden the way silent DELEGATE-on-correctness would be (which the
    design forbids -- see registry.DELEGABLE_WHOLE_FIELDS).
    """
    first = formalize(raw, model, judge=judge, judge_model=judge_model)
    if first.accepted:
        return first

    # CODE AND FIELD ARE READ TOGETHER, PER DIAGNOSTIC. Computing them separately made the gate
    # field-blind: a NON_OPERATIONAL_CORRECTNESS raised against `field: "data"` satisfied
    # neither exclusion -- its code is a correctness code so it did not count as "other fields
    # broken", and its presence flipped `correctness_is_blocker` true -- so the pipeline
    # substituted an ORACLE to fix a problem with the DATA field and reported success.
    def _is_a_correctness_blocker(d) -> bool:
        return d.code in CORRECTNESS_CODES and d.field in ("correctness", "", "(unspecified)")

    correctness_is_blocker = any(_is_a_correctness_blocker(d) for d in first.diagnostics)
    other_fields_broken = any(not _is_a_correctness_blocker(d) for d in first.diagnostics)
    # Only attempt substitution when correctness is the actual blocker. If
    # task/data/instance/interface are ALSO broken, fix those first --
    # substituting an oracle won't help and would mask the real problem.
    if not correctness_is_blocker or other_fields_broken:
        return first

    proposals = propose_oracle_strategies(model, raw)
    if not proposals or not proposals.get("proposals"):
        return first  # no viable alternative found; surface the original rejection

    def _budget_rank(p):
        b = (p.get("human_budget_implication") or "").lower()
        return 0 if b.startswith("0") or "0h" in b else 1

    ranked = sorted(proposals["proposals"], key=_budget_rank)
    for candidate in ranked:
        substituted_raw = dict(raw)
        substituted_raw["correctness"] = f"{candidate.get('oracle_strategy')}: {candidate.get('mechanism')}"
        second = formalize(substituted_raw, model, judge=judge, judge_model=judge_model)
        if second.accepted:
            # THE ID NAMES WHAT THE RESEARCHER SUBMITTED. `formalize` hashed whatever raw dict
            # it was handed, and on this pass that is the SUBSTITUTED one -- while `--lock`
            # records the sha256 of the original file, so the lockfile's two halves described
            # different documents on every substituted accept and the drift guard compared
            # them. What was substituted is not lost: it is recorded in full below.
            second.canonical_spec.spec_id = _spec_id(raw)
            return FormalizeResult(
                outcome="ACCEPT_SUBSTITUTED",
                canonical_spec=second.canonical_spec,
                original_diagnostics=first.diagnostics,
                substitution=candidate,
            )
    # None of the proposed alternatives resolved cleanly either -- surface
    # the original rejection rather than a partially-fixed one.
    return first


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec_path", help="path to a BenchSpec YAML file")
    ap.add_argument("--model", required=True, help="litellm model string")
    ap.add_argument("--json", action="store_true", help="print the raw result as JSON instead of a formatted report")
    ap.add_argument(
        "--lock",
        metavar="PATH",
        help="on ACCEPT, also write a lockfile: the canonical spec plus a sha256 "
        "of the raw file it was formalized from. The drift-guard test "
        "compares the lockfile against both the raw spec (freshness) and the "
        "live collector spec (agreement), so CI never needs an LLM call. "
        "Refresh it with this flag whenever the raw spec changes.",
    )
    ap.add_argument(
        "--propose",
        action="store_true",
        help="also propose 1-3 alternative oracle strategies for `correctness`, "
        "including automated proxies -- runs regardless of accept/reject",
    )
    ap.add_argument(
        "--auto-substitute",
        action="store_true",
        help="the general-case pipeline: if rejected solely on an unpriced/non-"
        "operational correctness mechanism, automatically retry with the "
        "lowest-human-budget proposed alternative. Substitution is always "
        "reported, never silent -- see ACCEPT_SUBSTITUTED output.",
    )
    args = ap.parse_args()

    raw = load_raw_spec(args.spec_path)

    if args.propose:
        proposals = propose_oracle_strategies(args.model, raw)
        print("=== PROPOSED ORACLE STRATEGIES ===\n")
        if proposals is None:
            print("  (proposal call failed)")
        else:
            for i, p in enumerate(proposals.get("proposals", []), 1):
                print(f"  [{i}] {p.get('oracle_strategy')} (human_budget: {p.get('human_budget_implication')})")
                print(f"      mechanism: {p.get('mechanism')}")
                print(f"      tradeoff: {p.get('tradeoff_vs_stated')}\n")
        print()

    from agentcore.observability.stage_log import StageLog

    with StageLog(os.path.dirname(os.path.abspath(args.lock or args.spec_path)), "formalizer"):
        result = formalize_with_fallback(raw, args.model) if args.auto_substitute else formalize(raw, args.model)

    if args.lock:
        if result.accepted:
            with open(args.spec_path, "rb") as fh:
                raw_sha = hashlib.sha256(fh.read()).hexdigest()
            lock = {
                "raw_spec_path": args.spec_path,
                "raw_sha256": raw_sha,
                "model": args.model,
                "outcome": result.outcome,
                "canonical_spec": result.canonical_spec.to_dict(),
            }
            with open(args.lock, "w") as fh:
                json.dump(lock, fh, indent=2)
                fh.write("\n")
            print(f"lockfile written: {args.lock}", file=sys.stderr)
        else:
            # A rejected spec must not silently leave a stale lockfile behind that still
            # matches an OLDER accepted version of the raw file by content.
            print(f"REJECT -- lockfile NOT written ({args.lock})", file=sys.stderr)

    if args.json:
        payload = {"outcome": result.outcome}
        if result.accepted:
            payload["canonical_spec"] = result.canonical_spec.to_dict()
        if result.outcome == "ACCEPT_SUBSTITUTED":
            payload["original_diagnostics"] = [d.__dict__ for d in result.original_diagnostics]
            payload["substitution"] = result.substitution
        if not result.accepted:
            payload["diagnostics"] = [d.__dict__ for d in result.diagnostics]
        print(json.dumps(payload, indent=2))
        sys.exit(0 if result.accepted else 1)

    if result.outcome == "ACCEPT_SUBSTITUTED":
        print(f"ACCEPT (SUBSTITUTED) — canonical spec {result.canonical_spec.spec_id}\n")
        print(
            f"  The spec as submitted was REJECTED on {len(result.original_diagnostics)} issue(s), "
            f"all tied to `correctness`:"
        )
        for d in result.original_diagnostics:
            print(f"    {d.code}: {d.detail}")
        print(
            f"\n  Substituted oracle_strategy: {result.substitution.get('oracle_strategy')} "
            f"(human_budget: {result.substitution.get('human_budget_implication')})"
        )
        print(f"  Mechanism used: {result.substitution.get('mechanism')}")
        print(f"  Tradeoff vs. what was originally stated: {result.substitution.get('tradeoff_vs_stated')}\n")
        print(json.dumps(result.canonical_spec.to_dict(), indent=2))
    elif result.accepted:
        print(f"ACCEPT — canonical spec {result.canonical_spec.spec_id}\n")
        print(json.dumps(result.canonical_spec.to_dict(), indent=2))
    else:
        print(f"REJECT — {len(result.diagnostics)} issue(s):\n")
        for d in result.diagnostics:
            print(f"  {d}\n")

    sys.exit(0 if result.accepted else 1)


if __name__ == "__main__":
    main()
