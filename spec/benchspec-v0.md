# BenchSpec v0 — input format for BenchCompiler

## Design goals

1. **Low effort**: a usable spec fits in 10–20 lines; no DSL to learn. Field
   values are natural language, lists, or small schemas — the agent
   interprets them.
2. **General**: any benchmark class expressible — the format constrains
   *what must be stated*, never *what kinds of benchmarks exist*.
3. **Sufficient**: the required core covers exactly the information the
   agent cannot invent on the researcher's behalf (grounded in the 22-entry
   study: every cataloged construction is reconstructible from these
   fields).
4. **Trust hooks built in**: seed examples double as acceptance tests for
   the compiled pipeline; every default the agent fills in must be reported
   as an explicit assumption.

## Format

A YAML (or equivalently markdown-sectioned) document. **6 required fields,
6 optional.** (`scale` — including the target instance count — is required:
it is part of the user's requirements, not a tuning knob; every answer may
be `DELEGATE`.)

### Guided intake (default authoring mode, v0.3)

Researchers do not have to author this document freehand. The default mode
is a **question-led intake**: the system asks a fixed sequence of targeted
questions, one per required field plus conditional follow-ups, and the
answers fill the spec. Authoring the document directly remains available
for power users — the wizard and the document are two views of the same
schema, and rejection diagnostics always point back to the corresponding
question.

Fixed core sequence (wording frozen for variance control):
1. What should the evaluated system do? (one to three sentences)
2. What does one instance consist of — what does the system receive, and
   what must it produce?
3. Where do instances come from, and what event or artifact counts as one
   candidate? (e.g. "bug-fixing commits in popular Python repos")
4. How is a submitted output judged correct? (an executable check, a named
   authority, or stated judgment criteria — "it should just be correct" is
   not an answer, and the wizard says so upfront with examples)
5. How many instances do you need (count or range), and what are your
   compute and human-effort budgets?
6. How will consumers interact with the benchmark? (CLI verbs, files,
   harness style — "like Defects4J/SWE-bench" is a valid answer)

Conditional follow-ups are triggered by answers, not curiosity: an
execution-based answer to Q4 triggers "which test framework / how many
reruns?"; a heuristic answer triggers "state the judgment criteria; what
sample will validate the judgment?"; a source without a stock connector
triggers access questions. Every question displays: why it is asked, two
example answers, and what the system will do if the answer is `DELEGATE`.

This mode reduces description variance at the origin (same questions, same
order, same wording), guarantees the required information is asked for
rather than hoped for, and lowers effort — answering six questions is
easier than authoring a document against a schema.

### Required

```yaml
task: |
  1–3 sentences: what the system under evaluation must do.

instance: |
  What one benchmark instance consists of:
  - input: what the evaluated system receives
  - expected output: what it must produce
  Prose or schema, whichever is natural.

data: |
  Where instances come from, and what event/artifact constitutes one
  candidate instance. ("Merged GitHub PRs that close an issue and modify
  tests, in popular Python repos" / "wrong-then-accepted submission pairs
  from Codeforces" / "inject faults into reference implementations of X")

correctness: |
  How to decide that a system's output on an instance is right.
  Must be OPERATIONAL: an executable check ("tests that failed before the
  patch pass after, nothing else breaks"), a named external authority
  ("the platform judge's verdict" / "maintainer accepts the PR"), or —
  if it genuinely requires human judgment — say so and give the judgment
  criteria (this becomes a priced manual step, not silent magic).

interface: |
  How a consumer uses the benchmark: invocation style (CLI verbs, files,
  API), what a submission looks like, what the evaluator returns.
  ("defects4j-style: checkout / compile / test" / "SWE-bench-style docker
  harness returning resolved-or-not + log")
```

### Also required: the blocks the front end refuses a spec for omitting

Everything above is prose the formalizer reads. These are STRUCTURED blocks it does not
paraphrase -- they are carried into the canonical spec verbatim and executed by the harness at
packaging, so a re-wording would change what runs. Omitting one is a rejection with the code
named beside it.

```yaml
admissibility:       # REQUIRED. MISSING_ADMISSIBILITY / UNMEASURABLE_ADMISSIBILITY
  # What a SOURCE must BE to be worth mining at all. One clause per property:
  - name: stars                                    # the value to record, a plain field name
    observe: "GET /repos/{org}/{project} -> stargazers_count"   # ONE procedure, never two
    require: ">= 100"                              # the predicate, in the grammar below
  # `require` is a grammar, not a sentence. Exactly one of:
  #     >= N   <= N   > N   < N   == N          a number
  #     within N days|weeks|months|years        a recency
  #     is true | is false                      a boolean
  #     one of [a, b, c]                        a membership, lowercase
  # `observe` names ONE procedure -- an API call and the field it returns, or a command and
  # what it prints. Not two joined by "or" (two procedures return two different numbers and
  # nothing records which was used), and not "check the project", which is not a procedure.
  #
  # Which dimensions matter is YOURS to choose and differs by task -- stars and commits for a
  # repository, entrants for a contest round, downloads for a package index. Nothing here
  # imposes one. But silence is refused: a benchmark whose sources nobody bounded gets
  # whatever the sampler happens to find, which has meant student homework.
  #
  # You may write this as PROSE instead ("popular, actively maintained Java projects") and the
  # formalizer will resolve it into clauses, recording each resolution as an assumption.
  #
  # If your benchmark mines nothing, say so ON THE RECORD and say why -- a bare marker is
  # still refused, because a reviewer cannot tell it from an omission:
  #     admissibility: "not applicable: instances are generated synthetically from
  #                     standard-library function names; no sources are mined"

artifact_variants:   # REQUIRED when your oracle establishes truth by CONTRAST
  # (fail-to-pass-tests, execution-replay, injection, reference-equivalence).
  # Not required for llm-judge, human-annotation, external-authority or rerun-labeling.
  # Absent when required: MISSING_VARIANTS.
  - name: buggy                      # the artifact's verbs are built from this: rebuild-buggy,
                                     # check-buggy
    description: "the project at the fix commit's parent, with the fix commit's own test applied"
    check_cmd: "sh check_buggy.sh ."   # {placeholders} may name fields each instance records
    expect: pass                       # pass|fail -- at least one variant must be expected to
                                       # pass, or no working baseline can be demonstrated
    results: junit-xml                 # REQUIRED. What the check is judged ON.
    expect_failures: 1                 # REQUIRED with junit-xml: how many tests must fail.
                                       # Polarity is never defaulted.
  # `results: none` is legal and needs `results_none_reason` naming the real evidence
  # (e.g. "digest comparison of program output"). Omitting `results` is
  # UNDECLARED_CHECK_EVIDENCE: an exit code is not a verdict, because a script's else-branch
  # can turn any error into the expected outcome.

solver_input:        # REQUIRED, always. NO_SOLVER_INPUT.
  # How a solver hands this benchmark its answer. A benchmark whose task cannot be attempted
  # through its own interface is not a benchmark (for a flag-selection task the interface
  # needs a verb that accepts a flag string).
  kind: parameter                  # parameter | workdir | file
  name: cflags                     # for `parameter`: must also appear in `parameters` below
  describes: "the flag configuration the solver proposes"
  probe_values: ["-O0", "-O3"]     # for `parameter`: two MATERIALLY DIFFERENT answers. The
                                   # harness submits each and requires the observable to
                                   # differ -- a verb that accepts a value and ignores it
                                   # passes every structural check there is.
  # `workdir`/`file` mean the solver edits the artifact itself (a patch, an edited tree) and
  # take `path:` instead of `name:`/`probe_values:`.

parameters:          # REQUIRED when solver_input.kind is `parameter`. NO_SOLVER_INPUT.
  - name: cflags
    kind: string
    default: "-O0"

negative_control:    # REQUIRED whenever artifact_variants are declared. NO_NEGATIVE_CONTROL.
  # How to break an instance ON PURPOSE, so a shipped check can be shown able to FAIL. A check
  # that survives a deliberate break certifies a solver that did nothing, for example one that
  # replaced the binary with `/bin/true`.
  break_cmd: corruption_cmd        # the NAME OF A FIELD each instance records, not a command:
                                   # what breaks one program is not what breaks another
  restore_cmd: rebuild_cmd         # optional
  check: buggy                     # which variant's verdict must flip. Must name a variant
                                   # declared above (a name that matches none of them is only
                                   # caught at packaging).
  describes: "revert the fix in the working tree"
```

### Also required: scale

```yaml
scale: |             # REQUIRED (v0.3) — part of the user's requirements
  - target instance count or range ("~300", "500-1000", or DELEGATE);
    compute budget; human-hours budget (0 by default); deadline
```

### Optional (defaults applied and reported if omitted)

```yaml
criteria: |          # what EVERY SHIPPED INSTANCE guarantees -- NOT source filters
  - the properties a consumer relies on in every instance, which construction
    must PROVE per instance: "the fix commit's own added/modified test fails at
    the parent commit and passes at the fix commit, confirmed by an actual test
    run"; "both revisions build with the project's own build tool"
  # Each one becomes an obligation some construction state has to enforce, so a
  # property missing here is a property nothing will ever check.
  #
  # A SOURCE filter ("language must be Java", "stars >= 100") does NOT belong here
  # -- it belongs in `admissibility`, and a criteria list made only of source
  # filters is refused with CRITERIA_ARE_SOURCE_FILTERS. The distinction is what a
  # property is ABOUT: `admissibility` is about the repository, `criteria` is about
  # the instance.
  # default: agent proposes per-instance guarantees typical for the oracle and lists them

quality: |           # validation demands on the benchmark itself
  - e.g. "instances must be reproducible across 3 clean reruns",
    "include negative controls", "labels must survive corruption checks",
    "dedup near-duplicates"
  # default: reproducibility check + corruption sensitivity + dedup

examples: |          # seed instances — cheap for the researcher, high value
  - a few known-valid instances (and ideally known-INVALID ones).
  # used two ways: few-shot guidance for mining, AND acceptance tests the
  # compiled pipeline must classify correctly before the build is trusted

references: |        # anchors to existing work
  - "like Defects4J but for X", links to papers/benchmarks whose
    construction should be imitated or improved upon

constraints: |       # policy
  - licensing, languages, privacy, contamination cutoff dates,
    "no proprietary sources", "must run offline", ...

access: |            # credentials + budgets for every platform construction touches
  - platform: github, key_env: GITHUB_TOKEN, budget: 5000 req/hour (standard authenticated limit)
  - platform: openai, key_env: OPENAI_API_KEY, budget: $50 total (mining/filtering/oracle calls)
  - platform: codeforces, key_env: none (public API), budget: 1 req/2s (documented courtesy limit)
  # every data source AND every LLM provider the construction pipeline will call must be
  # listed here -- construction itself burns LLM budget (mining assistance, judging,
  # the agent's own reasoning), not just the human/compute budget in `scale`. Never put
  # the raw secret in the spec -- only the env var name it will be read from.
```

## Agent contract (what the format buys the researcher)

On receiving a spec, BenchCompiler must:

1. **Formalize or reject (no human in the loop)**: the first stage
   transforms the spec into a formal canonical spec. If it cannot — missing
   information, ambiguity it can't resolve from the spec itself, an
   unpriceable correctness condition — it REJECTS with structured
   diagnostics (which field, which phrase, what's missing, what would fix
   it), like a compiler error. The researcher edits and resubmits; the loop
   is edit–recompile, not a dialogue.
2. **Plan + price, enforce budgets**: from the canonical spec, produce the
   construction plan and its estimated cost (compute, API calls,
   human-hours if correctness requires judgment). Budgets declared in
   `scale` are enforced automatically: a plan exceeding them is rejected
   with the estimate and the overshoot, not silently truncated.
3. **Fill gaps loudly**: every omitted optional field gets a default, and
   every default appears in the build report as an explicit assumption.
4. **Prove the build**: run the compiled pipeline's checks against
   `examples` (if given) and the generic quality gates; ship the benchmark
   with a per-instance provenance/audit record and the pipeline itself (so
   the benchmark can be refreshed/regenerated).
5. **Fail informatively**: if the spec can't be compiled (unreachable data,
   unpriceable correctness, gates keep failing), say exactly why and what
   would fix it.

## Two filled examples

### Minimal — differential-testing benchmark

```yaml
task: Generate a test input that makes two versions of a program behave differently.
instance: |
  input: two Python programs P (buggy) and Q (fixed) + the problem statement
  expected output: a test input on which P and Q produce different outputs
data: |
  Codeforces submissions (via Code4Bench): per author+problem, the last
  wrong-answer submission paired with the first accepted one.
correctness: |
  Execute both programs on the submitted input: accept iff outputs differ.
interface: |
  CLI: `bench fetch <id>`, `bench eval <id> <input-file>` → accept/reject + both outputs.
scale: |
  - DELEGATE instance count; compute ≤ 20 CPU-hours; human budget 0
admissibility: |
  not applicable: the instances come from a fixed published corpus (Code4Bench's
  snapshot of Codeforces submissions), not from a population sampled per run.
solver_input:
  kind: file
  path: "{input_path}"
  describes: "the test input a solver submits, on which P and Q must differ"
```

A short spec: prose plus the blocks the front end requires. `admissibility` is an exemption here because
nothing is mined -- stated on the record, with the reason, which is the only way to skip that
gate. `solver_input` is not optional for anyone: it is what makes the artifact usable for its
own task.

No `artifact_variants` and so no `negative_control`: this oracle is `differential-execution`,
which decides by running two programs the instance already ships rather than by contrasting two
states of one artifact. Everything else defaults -- the agent proposes gates (determinism
reruns, corruption check) and scale, and reports them for approval.

### Fuller (~1 page) — issue-resolution benchmark, Python slice

```yaml
task: |
  Given a GitHub issue and the repository at the pre-fix commit, produce a
  patch that resolves the issue.
instance: |
  input: repo snapshot + issue text
  expected output: unified diff
data: |
  Merged PRs that close an issue and modify test files, from popular PyPI
  packages (top ~200 by downloads).
correctness: |
  Split the developer PR into code patch and test patch. A submission is
  correct iff the tests that fail at the pre-fix commit pass after applying
  the submitted patch, and previously-passing tests still pass. Verified by
  execution in a per-instance container.
interface: |
  SWE-bench-compatible docker harness: `run --instance <id> --prediction
  <patch>` → resolved | unresolved + execution log.
admissibility:
  - name: language
    observe: "GET /repos/{org}/{project} -> language"
    require: "one of [python]"
  - name: stars
    observe: "GET /repos/{org}/{project} -> stargazers_count"
    require: ">= 1000"
  - name: last_commit
    observe: "GET /repos/{org}/{project} -> pushed_at (YYYY-MM-DD)"
    require: "within 12 months"
  - name: is_fork
    observe: "GET /repos/{org}/{project} -> fork"
    require: "is false"
artifact_variants:
  - name: prefix
    description: "the repo at the pre-fix commit with the PR's own test applied; the check runs
      that test and passes only if it FAILS here -- the issue is reproducible"
    check_cmd: "sh check_prefix.sh ."
    expect: pass
    results: junit-xml
    expect_failures: 1
  - name: postfix
    description: "the repo at the PR's merge commit; the check runs the same test and passes
      only if it PASSES here"
    check_cmd: "sh check_postfix.sh ."
    expect: pass
    results: junit-xml
    expect_failures: 0
solver_input:
  kind: workdir
  path: "{clone_path}"
  describes: "the solver edits the checked-out pre-fix tree; the check re-runs the PR's test"
negative_control:
  break_cmd: corruption_cmd
  check: postfix
  describes: "revert the fix in the working tree so the PR's test must stop passing"
criteria: |
  - the PR's own added/modified test fails at the pre-fix commit and passes at the
    merge commit, confirmed by an actual test run
  - both snapshots install and run the project's test suite
  - the gold patch touches only Python (setup/config files allowed)
  # note: "python", "stars >= 1000" and "not a fork" are SOURCE properties and live
  # in `admissibility` above, not here -- see the note on `criteria` in Optional
quality: |
  - environment must build deterministically; fail→pass set non-empty
  - 3× rerun stability; corruption check (a reversed gold patch must be rejected)
scale: |
  - 100–300 instances; compute ≤ 200 CPU-hours; human budget 0
examples: |
  - known-valid: <link to 2 PRs that would qualify>
  - known-invalid: <a PR that closes an issue without tests — must be filtered out>
constraints: |
  - permissive licenses only; contamination cutoff 2025-01-01
```

## Variance control (v0.1 addition)

Problem: free-form values mean the same intended benchmark, described
differently (by different people, or the same person twice), may compile to
different outcomes. Description-induced variance must be engineered down and
measured.

### 1. Two layers: flexible source → canonical form (confirmed)

Like a compiler front-end: the researcher writes the flexible spec above;
stage 1 **transforms it into a formal canonical spec, or rejects it** — no
human-in-the-loop confirmation. Rejection carries structured diagnostics:
which field, which phrase is ambiguous or missing, and what information
would make it compile (compiler-error semantics; the researcher edits the
spec and resubmits).

- Phrasing variance collapses at normalization: "popular Python projects",
  "well-known PyPI packages", "widely-used Python repos" all normalize to
  e.g. `population: pypi-packages, rank: top-200-by-downloads` — and if the
  normalizer can't resolve a phrase confidently from the spec (plus its
  documented defaults registry), it must reject with a diagnostic, never
  guess.
- The canonical spec is the ground truth artifact: it is diffable, so when
  two specs produce different benchmarks, the difference is visible at the
  canonical layer, not buried in the agent's behavior.
- Build determinism: same canonical spec (+ same data snapshot date) →
  same benchmark. All sampling seeded; all agent choices recorded in the
  canonical spec, not made ad hoc downstream.

### 2. Anchor semantics with canonical patterns + examples

- For the highest-variance fields (`correctness`, `data`'s unit of
  instancehood), the normalizer maps descriptions onto **named,
  parameterized patterns** drawn from the study's catalog (fail-to-pass
  tests(k), N-rerun labeling(n, envs), authority-verdict(source),
  injection(mutators, realism-audit), human-judgment(criteria, budget)) —
  with a free-form escape hatch for genuinely new patterns, which then get
  added to the pattern registry once validated.
- `examples` (seed known-valid/known-invalid instances) are the strongest
  disambiguators: concrete instances pin down meaning that words leave
  open, and they transfer unchanged across paraphrases.

### 3. Ambiguity linting

The normalizer runs an explicit lint pass over: unquantified qualifiers
("popular", "recent", "large", "high coverage"), undefined authorities
("the tests" — which tests?), and unbounded scopes ("GitHub" — language?
timeframe?). Each lint hit resolves to a documented default (from the
defaults registry, recorded in the canonical spec) or, when no safe default
exists, a rejection diagnostic — never an invisible choice.

Diagnostics are structured and typed (e.g. `MISSING_INFO(field, what)`,
`AMBIGUOUS_TERM(field, phrase, candidate readings)`,
`UNPRICEABLE_CORRECTNESS(reason)`, `BUDGET_EXCEEDED(estimate, cap)`), which
makes front-end quality itself measurable: first-try compile rate, and
whether a rejection's diagnostics lead to a successful fix on resubmission.

### 3a. Explicit delegation (v0.2)

A researcher may answer any diagnostic (or pre-empt one) with an explicit
delegation marker instead of information:

```yaml
criteria:
  popular: DELEGATE        # "system decides what popular means here"
```

Delegation is distinct from silence: silence on an unsafe choice → reject;
`DELEGATE` → the system decides, and its choice enters the canonical spec
as a first-class recorded resolution (`popular := stars>=1000
[DELEGATED, rationale: dominant convention in mined-benchmark literature]`).
Determinism and auditability are preserved — the choice is binding and
visible, just made by the system on request.

### 3b. Weak / heuristic collection oracles (v0.2)

Not every benchmark has a strong operational oracle at collection time —
some labels rest on heuristics, semantic understanding, and review. The
format supports this without pretending otherwise: `correctness` (and the
label procedure in `data`) may declare a **heuristic/semantic procedure**
(e.g. LLM-based semantic judgment with stated criteria, optionally + human
review within `scale.human_budget`). The consequences are made explicit
rather than rejected:

- every label carries an **evidence tier**: `executed` (verified by
  running), `authority` (external process), `heuristic+review`,
  `heuristic-only`;
- heuristic tiers REQUIRE a validation anchor: a labeled sample the
  procedure is scored against (seed `examples`, or a budgeted human-labeled
  sample), and the measured agreement ships in the benchmark's metadata;
- the built benchmark can stratify by tier (e.g. a strong-oracle core set +
  a heuristic-labeled extended set), so consumers choose their rigor level.

Non-operational correctness is thus rejected only when it is *circular or
missing*, not when it is honestly heuristic.

### 4. Variance as a measured property (evaluation requirement)

This turns the concern into an experiment for the paper:
- **Paraphrase robustness**: generate/collect K paraphrases of the same
  spec; compile all; measure benchmark equivalence — canonical-spec
  equality, instance-set overlap, verifier agreement on shared instances,
  and rank-correlation of evaluated systems across the variants.
- **Inter-researcher variance**: give N researchers the same one-paragraph
  task brief; each writes a BenchSpec; measure the same equivalence metrics
  and where divergence enters (source spec vs canonical spec vs build).
- Target claim: divergence surviving the canonical layer reflects genuine
  under-specification (the researchers actually wanted different things),
  not phrasing noise.

## Why this is the right complexity point

- Below this (pure one-sentence prompt): the agent must invent the unit of
  instancehood and the correctness decision — the two places the study
  shows domain intelligence is irreducible; silent wrong guesses there
  produce plausible-but-invalid benchmarks.
- Above this (formal DSL, closed vocabularies): effort explodes, and every
  benchmark class we cataloged that didn't fit the vocabulary would need
  tool changes — contradicting the open-vocabulary goal.
- The 5 required fields are exactly the study's sufficiency residue; all 22
  cataloged constructions can be written in this format (spot-checked:
  Mokav, OmniGIRL, FlakyGuard-analog, HornBro, AntMan — the last expressible
  with `correctness` naming human judgment + a budget, or a proxy check).
