"""Declarative configuration schema (pydantic v2).

YAML is the human surface; these models are the validated runtime contract.
Validation happens once, at load time (see config.loader), so missing/!misshaped
keys fail with a clear pydantic error instead of a cryptic mid-run crash.

This module is pure data: it imports only pydantic + typing. Behaviour modules
(agent.budget, agent.stages, oracle.*) import these Specs, never the reverse.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agentcore.agent import gates

# ── models ───────────────────────────────────────────────────────────────────


class ModelSpec(BaseModel):
    name: str
    class_: str = Field("litellm", alias="class")
    kwargs: dict[str, Any] = {}
    temperature: float | None = None
    rotation: list[str] = []  # structurer escalation ladder
    cache_control: bool = False

    model_config = {"populate_by_name": True}


# ── environment ──────────────────────────────────────────────────────────────


class EnvSpec(BaseModel):
    type: Literal["none", "local", "docker", "deferred"] = "docker"
    image: str | None = None
    image_template: str | None = None  # Jinja over instance vars
    env: dict[str, str] = {}
    timeout: int = 30  # default per-command timeout (s)
    resources: dict[str, Any] = {}  # privileged/cap_add/sysctls/cpuset/ulimits/volumes
    read_only: bool = False  # disable mutating builtins
    provision: str | None = None  # path to an env_setup hook / Dockerfile
    workdir: str = "/workspace"
    container_lifetime: str = "2h"
    # How many sandboxes one run may hold live at once. Host memory is the real bound -- 8g
    # per container by default -- so this is spec content with a generous default.
    max_sandboxes: int = gates.MAX_SANDBOXES
    # Rescue-class runs: the patched workbench container IS the run's
    # deliverable -- the orchestrator commits it as the instance's new snapshot AFTER
    # the run ends. The default teardown would reap it first. With
    # this flag the run leaves its sandboxes running and the CALLER owns their cleanup.
    keep_containers: bool = False
    # filenames whose write auto-triggers an image build + container (re)entry,
    # with the build log fed back to the model (LLM-authored-Dockerfile flow).
    build_on_write: list[str] = []
    # user-declared CLI commands the env provisions onto PATH and advertises to the agent.
    # each: {name, description?, usage?, script?: inline body, install?: one-time setup command}.
    commands: list[dict[str, Any]] = []


# ── stages ───────────────────────────────────────────────────────────────────


class StageSpec(BaseModel):
    name: str
    prompt: str | None = None
    tools: list[str] | None = None  # per-stage toolset (None = inherit global)
    # completion: exactly one of the keys below, e.g. {"sentinel": "STAGE_COMPLETED: x"}
    completion: dict[str, str] = {}
    transitions: list[str] | None = None  # allowed next stages (None = linear next)
    # Hook ref (e.g. "hooks:survey_tool_guard") consulted before EVERY tool dispatch in
    # this stage: hook(ctx, name, args) -> "" to allow, or refusal text (NonTerminating
    # nudge). The static tools: allow-list says which tools exist in a stage; this says
    # whether THIS call is admissible NOW given run state. A model may
    # acknowledge in words ("I need to record") and defer in deeds (run another search).
    # Prompts and dashboards inform; only a gate compels.
    tool_guard: str | None = None
    foreach: str | None = None  # WorkingSet collection to fan out over


# ── budget ───────────────────────────────────────────────────────────────────


class Meter(BaseModel):
    kind: Literal["steps", "wallclock", "cost_usd", "cpu_seconds", "disk_bytes", "test_runs"]
    limit: float
    charge_on: Literal["tool_call", "cycle", "stage"] = "cycle"
    on_exhaust: Literal["abort", "finish_cycle", "fail_with_lesson"] = "fail_with_lesson"
    scope: Literal["agent", "background_process"] = "agent"


class BudgetSpec(BaseModel):
    meters: list[Meter] = [Meter(kind="steps", limit=40, charge_on="tool_call")]
    thresholds: list[dict[str, Any]] = []  # [{at: 0.6, action: force_state, target: ...}]
    max_attempts: int = 1


# ── oracle ───────────────────────────────────────────────────────────────────


class SubmissionSpec(BaseModel):
    command: str | None = None  # run in-env at submit; capture stdout
    aggregate_file: str | None = None  # benchmark-shaped output (preds.json/SARIF)
    key: str | None = None  # instance key for the aggregate


class OracleSpec(BaseModel):
    kind: Literal["none", "sentinel", "deterministic", "llm_judge", "vector", "external"] = "sentinel"
    proxy: str | None = None  # hook/builtin run after each execute (in-env)
    submission: SubmissionSpec | None = None
    grade: str | None = None  # in-process judge OR external join
    aggregate: Literal["all", "weighted", "none"] = "all"
    target: dict[str, Any] = {}  # e.g. {coverage: 0.85}; passed to the proxy
    weights: dict[str, float] = {}  # for aggregate=weighted


# ── attempts / repetition / policy ──────────────────────────────────────────


class SamplingSpec(BaseModel):
    n: int = 1
    select: Literal["majority", "judge", "best_metric", "first_success"] = "first_success"


class AttemptsSpec(BaseModel):
    enabled: bool = False
    carryover: bool = True
    container_policy: Literal["keep", "reset", "llm"] = "reset"
    verify_in_clean_env: bool = False
    sampling: SamplingSpec | None = None


class ToolOutputSpec(BaseModel):
    """How tool output is compressed into agent memory (per agent, per-tool overrides)."""

    strategy: Literal["heuristic", "llm_summary", "raw"] = "heuristic"
    # heuristic policy; None => fall back to the tool's @tool-declared policy, else head_tail
    truncation: Literal["head_tail", "first_error", "latest_n_lines", "none"] | None = None
    threshold_chars: int = 16000
    head_chars: int = 8000
    middle_chars: int = 2000
    tail_chars: int = 8000
    n_lines: int = 80
    summary_model: str = "summarizer"  # named model used when strategy=llm_summary
    summary_prompt: str | None = None  # optional prompt-template name; else a built-in instruction
    max_summary_chars: int = 1200
    per_tool: dict[str, dict[str, Any]] = {}  # tool_name -> partial overrides of the fields above


class ContextSpec(BaseModel):
    """How the model's prompt is assembled each cycle.

    - compact:    rebuild [system, user] every cycle from the action_log (bounded, lossy,
                  no growing conversation). ExecutionAgent style; default.
    - transcript: keep and send a growing chat history (system, user, assistant, observation, ...),
                  bounded by `window` / `max_tokens`. Standard multi-turn chat.
    - hybrid:     transcript WITHIN a stage; on stage transition reset+reseed with the
                  system prompt + completed-stage summaries. analysis_agent style.
    """

    mode: Literal["compact", "transcript", "hybrid"] = "compact"
    window: int = 20  # compact: action_log entries rendered; transcript: messages kept
    # token-budget eviction for transcript/hybrid (~chars/4 estimate). Defaulted to a safe
    # bound so a growing transcript can't blow the model's context window; set null to disable.
    max_tokens: int | None = 100_000
    reset_on_stage: bool | None = None  # transcript/hybrid: reset at stage boundary (None => hybrid:true)
    summarize_on_stage_exit: bool = False  # collapse the finished stage into a summary on reset
    observation_role: Literal["user", "tool"] = "user"  # role used for tool observations in transcript modes
    # COMPACT MODE ONLY: end the prompt with the previous cycle's exchange as real turns --
    # [system, fresh user, assistant, user("output of your last action")]. Compact mode
    # otherwise rebuilds [system, user] from scratch, so the model's own last reply comes back
    # third-person inside a forty-entry log and reads like something someone else did, and
    # the model tends to repeat a mistake it just made. On by default because this applies to
    # every agent using compact mode.
    echo_last_turn: bool = True


class RepetitionSpec(BaseModel):
    enabled: bool = True
    window: int = 12  # how many recent actions to compare
    # How many identical actions within that window before the next one is refused. A low value is too eager, because
    # the block does not decay: a refused call is never appended to the history, so the entries
    # that triggered it only leave the window once the model succeeds at other work. A model
    # that believes the blocked action is the right one keeps retrying it, succeeds at nothing,
    # and the entries never move.
    #
    # 5 within 12 is a real loop rather than a coincidence, and the things that catch a genuine
    # loop are better than this one anyway: `_strike` blocks a candidate after 8 refusals at one
    # state, `tick_current` defers after 100 cycles on one candidate, the stall detector fires
    # at 5 frozen cycles, and the refusal-storm breaker ends the run at 60 consecutive refusals.
    # All of those name the candidate and say what to do instead; this one cannot.
    threshold: int = 5
    # How many CYCLES an entry stays eligible to count against a repeat. Without this the block
    # never decays: a refused call is not appended to the history, so the entries that triggered
    # the block only leave the window when the model succeeds at something ELSE -- and a model
    # convinced the blocked action is the right one succeeds at nothing, so the block is
    # permanent. A block must be a nudge that expires, not a state with no exit.
    # 0 disables the decay (count by action position only).
    decay_cycles: int = 20
    classes: dict[str, int] = {}  # named no-op classes -> threshold
    exempt_tools: list[str] = []  # tools allowed to repeat (idempotent reads)
    reset_on_stage_change: bool = True


class CapSpec(BaseModel):
    """How many delivered instances may share one value of some field.

    Declared per task rather than hardcoded, for the same reason the states are: what counts as
    "the same source" differs. boca records `upstream_url`, defects4j `project`, swtbench `repo`.
    A cap that assumed one name would silently do nothing on the other two.

    Several caps can apply at once, on different dimensions -- repository, language, whatever a
    task records -- and a candidate must satisfy all of them.
    """

    # What this dimension is called, used in the refusal message: "repository", "language".
    dimension: str = "source"
    # Field names to read the value from, first non-empty wins. Several because tasks record the
    # same thing under different names at different lifecycle stages.
    keys: list[str] = []
    # Maximum delivered instances sharing one value. Absent means "use the default limit"
    # (diversity.DEFAULT_LIMIT): a cap block with no `limit:` line is still a cap. An explicit
    # `limit: 0` disables it -- writing the zero out is a decision, defaulting into one is not.
    # (It defaulted to 0 once, and the most natural declaration -- dimension and keys, nothing
    # else -- read back as configured while enforcing nothing.)
    limit: int | None = None
    # Treat values as clone URLs and reduce them to owner/name, so that .git suffixes, git:// vs
    # https:// and trailing slashes do not read as different sources.
    urls: bool = True


class WorklistStateSpec(BaseModel):
    # extra="forbid": every key here IS a control -- a typo (`discrimnates:`,
    # `re_derives:`) under the default ignore-unknown made the declared control silently
    # stop existing, with the spec loading clean and nothing anywhere saying so.
    model_config = {"extra": "forbid"}

    name: str
    # Fields that must be present (and non-empty) on a candidate for it to REACH this state.
    # These are preconditions computed by the harness, not a checklist the model self-reports:
    # a state whose requires are unmet is unreachable, so steps cannot be skipped or claimed.
    requires: list[str] = []
    # Short, imperative instruction shown when a candidate sits in the PREVIOUS state -- i.e.
    # "what to do next". Kept to a couple of sentences on purpose: the per-turn prompt this
    # replaces was far too long for a model to follow reliably.
    instruction: str = ""
    # Names of prompt template vars to surface ONLY at this step. Long admissibility criteria
    # re-sent in every system prompt of every cycle would be relevant to exactly one step. Heavy context belongs where it is used.
    context: list[str] = []
    # Optional "module:function" the HARNESS runs to verify the recorded evidence before the
    # candidate may enter this state. `requires` only asks whether a field is present, which a
    # model satisfies by writing sha_resolved="ok" -- a claim, not a fact. A validator makes the
    # harness check the claim (resolve the SHA against a real clone, parse the test report for
    # tests_run>=1 and an assertion failure). Signature: fn(fields, ctx) -> (ok: bool, detail: str).
    # Runs once, when `requires` first become satisfied; the verdict is stored on the candidate.
    # Named `verify` rather than `validate` because pydantic BaseModel already owns `validate`.
    verify: str | None = None
    # Fields that must be present AND truthy. `requires` alone treats False as present
    # (`_absent(False)` is False), so a model recording `is_bug_fix: false` -- correctly
    # concluding the candidate is NOT admissible -- was PROMOTED to the next state instead of
    # rejected. Listing a field here makes an explicit false value route to REJECTED with the
    # field named, which is what the model actually meant.
    requires_true: list[str] = []
    # A state with `requires` but no `verify:` is a state where the model grades its own
    # homework -- the recurring bug class of this project (four instances in one day, one of
    # them reintroduced by the fix for another). Config load REFUSES such a state unless it
    # carries `self_graded: true`: an explicit, greppable admission that the claim is taken on
    # the model's word, so the gap is a decision on record rather than an omission nobody sees.
    self_graded: bool = False
    # Fields this state's `verify` COMPUTES and writes onto the candidate. Declared, because
    # `requires` cannot express it: `requires` is checked BEFORE `verify` runs, so a field the
    # validator produces can never satisfy its own state's precondition -- the candidate simply
    # never advances. The alternative is to require the field anyway and let the model type a
    # placeholder for the validator to overwrite. Harmless for a label; not harmless for the
    # commit hashes the constructor needs (`parent_commit`, `base_sha`, `upstream_ref`),
    # because asking a model to type a hash it has not read invites fabrication.
    #
    # A derive is NOT a precondition: the candidate still has to reach the state on its own
    # evidence. It is a promise to whoever consumes the delivered record that this field will
    # be there, which is what lets the sampler/collector conformance check see a field that
    # never appears in `requires`.
    derives: list[str] = []
    # QUANTITIES THE HARNESS COMPUTES FROM RECORDED OBSERVATIONS.
    #
    #     computes:
    #       median_runtime: {from: timings,               agg: median}
    #       cv:             {from: timings,               agg: cv}
    #       speedup:        {from: [baseline_ms, tuned_ms], agg: ratio}
    #
    # `derives:` above is a promise that this state's VALIDATOR writes a field. This is the
    # other half: arithmetic the harness can do itself, declared so that no agent implements a
    # median in whatever shell its container happens to have (a shell without `asort` or `bc`
    # makes that expensive). The general form is any
    # field computable from observations the agent already holds: medians, CVs, ratios,
    # speedups, percentages, pass rates, sizes.
    #
    # Unlike `derives:`, a computed field MAY appear in `requires` -- deriving happens in
    # update(), before the state machine reads the fields -- so a lifecycle can still refuse to
    # advance until the underlying observations exist. The agent's own value always wins if it
    # recorded one; the harness records its value beside it and flags the disagreement rather
    # than overwriting, because a summary that disagrees with its own sample is a finding.
    # See agentcore/agent/derive.py for the aggregations and the shapes (a list gives a scalar,
    # an object of lists gives an object of scalars keyed the same way).
    computes: dict[str, dict] = {}
    # Which of the spec's criteria this state's gate enforces, each as a distinctive substring
    # of the criterion's text (case-insensitive, must match exactly one entry of the derived
    # instance's criteria_list). This is the map that was always missing: criteria lived in the
    # spec, gates lived here, and nothing connected them -- so a collector could enforce
    # two-thirds of its own spec with every test green, and the gaps were found by a human
    # reading the two side by side. agentcore.agent.coverage audits the mapping; matching by
    # substring means a criterion whose TEXT drifts breaks the match loudly instead of silently
    # unhooking its gate.
    enforces: list[str] = []
    # A NEGATIVE CONTROL, declared rather than hand-written: prove this candidate's oracle can
    # tell a right answer from a wrong one.
    #
    #     discriminates: {clean: output_digests, perturbed: corrupted_digest,
    #                     describe: corruption_probe}
    #
    # The harness requires all three fields, then requires that `perturbed` DIFFER from `clean`
    # (from every element, when clean is a list). An oracle that reports the same value after a
    # deliberate break is not an oracle, and this is the check that catches the failure mode
    # cheap checks cannot: three "identical" digests that are each the SHA-256 of the empty
    # string are perfectly stable, perfectly worthless, and pass every shape check. Corrupt a program that prints nothing and it still prints nothing, so
    # the probe fails loudly where the digests agreed.
    #
    # Declared, because the point is that no task needs to write this in Python. It is also the
    # groundwork for verifiers the constructor writes itself: a check the harness has watched
    # reject a broken instance is one whose verdicts mean something, and a check that has never
    # been seen to fail is indistinguishable from `return True`.
    discriminates: dict[str, str] = {}
    # MEASUREMENT ORDERING ANNOTATIONS. Enforcing a variance ceiling on a quantity before the
    # instruction that says how to influence it judges a candidate on a rule it has not been
    # given. The distinction the lifecycle needs is between WRITING a quantity down
    # and JUDGING it, and prose cannot carry it reliably, so it is declared:
    #
    #     records: [median_ms, cv]   # this state writes these down; NO acceptance
    #                                # threshold applies to them here
    #     proves:  [cv]              # the spec's threshold for these binds HERE
    #
    # The audit (agentcore.lifecycle_audit) enforces: a field is proved at most once, never
    # before its last recording, never at a state that also records it, and a state whose
    # validator applies a spec threshold to a field must not declare that field as merely
    # recorded. The runtime ignores both keys on purpose -- they change what the AUDIT
    # accepts, not what the loop does, so an annotation can never alter live behaviour.
    records: list[str] = []
    proves: list[str] = []
    # THE RE-DERIVATION CONTROL, sibling of `discriminates`. A runtime recorded without the
    # command that produced it cannot be re-measured, and a certifier would re-run a
    # different (smaller) input than the one reported. The declaration names, per recorded field, the FIELD
    # holding the command that reproduces it, and the harness runs that command itself in the
    # candidate's sandbox when the state's evidence is checked -- before any validator, like
    # the discrimination control:
    #
    #     rederives:
    #       output_digests:    {cmd: run_cmd, mode: digest, normalize: output_normalizer}
    #       largest_median_ms: {cmd: largest_run_cmd, mode: observe,
    #                           on_input: largest_input_path}
    #       cv:                none
    #
    # `on_input:` (optional) names the field holding the input path this command must actually
    # run on. Without it the harness re-runs whatever was recorded and checks only the value;
    # with it the harness ALSO refuses a command that never mentions that path, and substitutes
    # `{the_field_name}` before re-running if the command templates it.
    #
    # Without it an instance could record
    #     costliest_input_path      = /workspace/inputs_generated/plaintext_giant.bin
    #     default_costliest_run_cmd = /workspace/variants/default/test.elf   <- no argument
    # -- byte-identical to its `default_initial_run_cmd` -- and record a digest and a runtime
    # as measurements "on the costliest input" while nothing passed the file to anything.
    # No gate would relate the command to the path.
    #
    # DECLARED, NOT INFERRED. Pairing `*_run_cmd` with `*_input_path` by suffix would invent a
    # naming convention inside the harness and bind every future benchmark to it. A spec that
    # says nothing gets no such check; the harness offers an ability, not an opinion.
    #
    # `normalize:` (optional, digest mode) names the
    # field holding an OUTPUT NORMALIZER script the builder wrote for programs whose output
    # differs only cosmetically between runs (a timestamp header, reordered lines). The
    # HARNESS pipes the command's output through it before hashing -- the builder judges,
    # the harness executes, the script ships with the instance, and the discrimination
    # control is the swallow-guard: clean and corrupted digests both flow through the same
    # script, so one that eats the answer makes them equal and the control refuses. Raw and
    # normalized digests are both recorded with line counts, always. An instance with a
    # plainly deterministic output simply records no normalizer field.
    #
    # `digest`: sha256 of the command's output must equal the recorded value (exact -- for
    # outputs a DETERMINISTIC-style state has already proven stable). `observe`: the command
    # must run (exit 0); the harness's OWN wall-clock and output digest are recorded into the
    # state's evidence -- measured, not judged, the same stance the certifier takes -- so a
    # consumer holds a harness-derived number next to the agent's. `none` is the explicit
    # exemption: a quantity that is a statistic over many runs (cv) has no single reproducing
    # command, and writing `none` puts that decision on record where the audit can see it --
    # absence, by contrast, is refused (audit: underived_measurements). NOTE: YAML reads a
    # bare `none` as the STRING "none" (its null is `null`/`~`), so both spellings are
    # accepted as the exemption -- refusing the one authors naturally write would be a
    # papercut with no safety value.
    rederives: dict[str, dict | str | None] = {}


class KnownFromSpec(BaseModel):
    """Earlier runs or pools of THIS campaign whose registrations the door refuses again.

    `paths`: run directories (their agent_state.json worklist) and/or artifact pools (their
    instances.json). `keys`: the dimension that counts, in the caps' vocabulary -- a program
    is `[repo_url]` however it is spelt (diversity.value_of normalises). A path that does not
    exist contributes nothing and is not an error: a run that never happened registered nothing.
    """

    model_config = {"extra": "forbid"}

    paths: list[str] = []
    keys: list[str] = []


class WorklistSpec(BaseModel):
    """Per-candidate lifecycle, declared per task.

    See agentcore/agent/worklist.py for the rationale. Declared rather than hardcoded because
    the tasks genuinely differ: defects4j/swtbench share a fail->pass shape, boca_flags is
    build -> deterministic -> rebuild-stable -> timed, and mokav includes a witness search.
    """

    model_config = {"extra": "forbid"}  # see WorklistStateSpec: a typoed key is a lost control

    enabled: bool = False
    # Fields that identify a candidate. The worklist is keyed on these, so duplicates are
    # structurally impossible.
    key: list[str] = []
    # DECLARED NORMAL FORMS for key fields. A candidate registered once with an absolute
    # `source_path` and once with a relative one is one file with two identities. The generic
    # case/slash/.git normalization in key_of cannot express path shape, and hardcoding
    # "source_path means a path" in the harness would be a task preference in code. So the
    # SPEC declares it, per field, from the closed vocabulary the harness implements
    # (agentcore.agent.worklist.KEY_NORMALIZERS: git_url, abs_path, full_sha, case_fold, none):
    #
    #     normalize: {upstream_url: git_url, source_path: abs_path, program: none}
    #
    # A normalizer may refuse a value with no computable canonical form (a relative path
    # cannot be made absolute from outside the sandbox); worklist_add then refuses AT THE
    # DOOR with the shape to use, so a would-be silent duplicate becomes an actionable
    # message. Opt-in per field: an undeclared field keeps today's generic treatment, so
    # existing identities do not shift. A name outside the vocabulary is a loud config error --
    # a silently skipped normalizer would be a duplicate defect wearing a declaration.
    normalize: dict[str, str] = {}
    states: list[WorklistStateSpec] = []
    # States that take a candidate out of play. The FIRST is the success state (what gets
    # delivered); REJECTED and BLOCKED are always terminal in addition to these.
    terminal: list[str] = []
    # Working-set key that receives the delivered candidates when a stage in `publish_on` is
    # entered. Without this the worklist is a scoreboard the deliverable never reads: every gate
    # and validator protects `Worklist.delivered()`, while `final_sample` -- the thing the judge
    # and the CLI actually consume -- was retyped by the model from its action log, which is
    # exactly where the duplicates, missing fields and invented SHAs came from.
    publish_to: str = ""
    publish_on: list[str] = []
    # When the FIRST candidate reaches the terminal state, the harness writes
    # {"passed": true, "candidate": <key>} here. That makes "the pipeline is proven" a derived
    # fact -- one candidate actually completed every verified step -- rather than a self-report.
    # `record` is refused on this key only when an agent sets it, so agents without a worklist
    # keep using record legitimately.
    derive_calibration_to: str = ""
    # When a candidate reaches the success state, the action log older than this many entries
    # is folded into ONE summary entry (verified steps + refusal lessons, built from the
    # candidate's own record). The action log can dominate the prompt,
    # most of it full outputs of work on candidates already finished -- the next candidate
    # needs the end result and the obstacles, not fifteen pip logs. 0 disables folding.
    fold_history_keep: int = 8
    # Per-dimension limits on how much one source may contribute. Empty means no cap.
    # Without a cap a run may deliver every instance from one repository, because further
    # instances from an already-cloned repo in an already-built container are nearly free. See agentcore/agent/diversity.py.
    caps: list[CapSpec] = []
    # WHAT EARLIER RUNS ALREADY REGISTERED. A run has no memory of the one before it, so a
    # target of N DISTINCT programs needs this. `known_from` names earlier
    # runs / pools and the dimension that counts (the caps' vocabulary); a candidate whose
    # value on that dimension is already known is refused at the door, naming the earlier
    # run. An operational overlay at launch (`--overrides`), never a spec edit.
    known_from: KnownFromSpec | None = None
    # THE GATES A TASK MAY TUNE. A spec can override each of these. The defaults are the
    # constants in agentcore/agent/gates.py, the one home for the numbers.
    stuck_defer: int = gates.STUCK_DEFER  # refusals at one state before a deferral
    stuck_block: int = gates.STUCK_BLOCK  # refusals at one state before termination
    systemic_at: int = gates.SYSTEMIC_AT  # distinct candidates on one refusal shape
    slow_nudge: int = gates.SLOW_NUDGE  # cycles one candidate may hold the slot
    slow_defer: int = gates.SLOW_DEFER  # ...before it is deferred for others
    cap_default_limit: int = gates.CAP_DEFAULT_LIMIT  # a `caps:` entry that declares none

    @model_validator(mode="after")
    def _fill_cap_defaults(self) -> "WorklistSpec":
        # A cap that declares no `limit:` takes the SPEC's default, not the engine's, so the
        # number a task is held to is written where the task can see it and change it.
        for cap in self.caps:
            if getattr(cap, "limit", None) is None:
                cap.limit = int(self.cap_default_limit)
        return self


class CollectorSpec(BaseModel):
    """The collector as an evolving source PACKAGE rather than one script.

    Piling everything into a single file that is rewritten many times has two costs: every
    edit re-emits the whole file, so writes carry fresh syntax errors (caught only after
    writing), and each rewrite can silently drop working code. It is also the wrong task
    size -- editing a 40-line module is reliable, rewriting 500 lines is not.

    `max_module_lines` is the lever: a write over the limit is refused with "split it", which
    makes small modules the path of least resistance rather than a style request.

    The shipped benchmark CLI belongs in the same tree: every spec's `interface:` field requires
    one, and it is otherwise an afterthought written at the end, if at all.
    """

    enabled: bool = False
    root: str = "/work/collector"
    # Suggested module -> purpose. Scaffolded as stubs so the agent fills in structure that
    # already exists instead of inventing a layout mid-run.
    modules: dict[str, str] = {}
    max_module_lines: int = 150
    # A module may not run against real candidates until its own test passes. Cheap, mechanical,
    # and aimed at syntax errors and the absence of self-testing.
    require_tests: bool = True
    # Modules that must EXIST and have a passing test before mining at scale is allowed.
    # This is how a good intention becomes a precondition: an agent may propose an LLM-based
    # filter in its brainstorm and then never execute it; nothing else holds it to its own plan.
    required_before_scale: list[str] = []

    # ── the toolchain: DEFAULTS, not rules ──────────────────────────────────────────────────
    # These were hardcoded into the tools: only `.py` files could be written, `python3 -m
    # py_compile` decided whether a file was valid, and `collector_run` unlocked only on
    # `python3 -m pytest` exiting 0. A benchmark whose collector is Go, Rust or a shell pipeline
    # could not merely be awkward to build -- it could never reach a passing test and therefore
    # never unlock the run step at all. The harness was choosing the language.
    #
    # What the harness legitimately owns is the WORKFLOW: a file that does not pass its own
    # check never lands, a module that has never passed a test never runs at scale, and modules
    # stay small. What runs those checks is the task's business, declared here or chosen by the
    # agent at runtime with `collector_configure`.
    #
    # Placeholders: {path} in check_cmd, {target} in test_cmd, {module} and {args} in run_cmd.
    # An empty check_cmd means no syntax gate (the file lands unchecked, and the tool says so).
    # Defaults to Python because the commands below do; a task writing its collector in
    # anything else declares its own, and [] means "any suffix". These must agree: an
    # empty list with a python3 run_cmd left bare module names with no suffix to resolve.
    extensions: list[str] = [".py"]  # allowed file suffixes; empty = any
    check_cmd: str = "python3 -m py_compile {path}"
    test_cmd: str = "python3 -m pytest {target} -q"
    run_cmd: str = "python3 {module} {args}"
    # How a module name maps to its test file. {name} is the module's basename.
    test_path: str = "tests/test_{name}"
    # Empty files the skeleton needs beyond the modules themselves. Python wants `__init__.py`
    # so the tree is importable; a Go or Rust collector wants neither, and hardcoding them was
    # one more way the harness assumed a language. Declared, so a task that needs none says so.
    scaffold_files: list[str] = ["__init__.py", "tests/__init__.py"]


class StallSpec(BaseModel):
    """Detect a run that is spending cycles without changing anything observable.

    RepetitionSpec compares actions byte-for-byte, which misses the failure mode actually
    a real failure mode: a model that keeps calling the SAME tool for the SAME purpose while
    varying one argument (for example a new working-set key each time), so no two calls are
    ever identical and repetition never fires.

    The robust signal is not similarity between actions -- it is the ABSENCE OF PROGRESS.
    Each cycle a fingerprint is computed from `watch` (dotted paths into the working set) plus
    the current stage and the count of real side effects. If the fingerprint is unchanged for
    `nudge_after` cycles the agent is told exactly which values are frozen; if it stays frozen
    for `force_after`, the stage is forced forward. Deliberately never aborts the run -- budget
    meters remain the only thing that ends it.

    `watch` is per-task on purpose: `final_sample`/`rejected_candidates` are a convention of the
    collector agents, not of agentcore, so the loop must not hardcode them. Any agent declares
    whatever "progress" means for its own task; an empty `watch` still detects stalls via stage
    and side-effect count alone.
    """

    enabled: bool = True
    watch: list[str] = []  # dotted working-set paths whose change means progress
    nudge_after: int = 5  # frozen cycles before telling the model what's stuck
    force_after: int = 12  # frozen cycles before forcing the next stage
    reset_on_stage_change: bool = True
    # Tools that only write to the working set, so calling them is NOT evidence of progress by
    # itself. Without this the detector is defeated by the exact behaviour it exists to catch:
    # every `record` appends to the action log, so spamming new keys would look like activity.
    # Only the watched paths can vouch for a bookkeeping call. Anything else (running a command,
    # writing a file) changes the world and counts on its own.
    # Tools that only mutate run state, so calling them is NOT evidence of progress. This is a
    # DENY-list of state-only tools, not just the single name `record`: the fingerprint counted
    # every other tool, so a model looping on `worklist_record` bumped the count every cycle and
    # the streak never grew -- the detector would be blind to exactly the loop this design
    # makes most likely.
    # Refusal-storm breaker (loop.py): end the run after this many CONSECUTIVE refused
    # cycles -- each is a paid LLM call that changed nothing. Consecutive, not cumulative:
    # healthy runs intersperse refusals with dispatched work; a temp-0 storm never breaks
    # itself. 0 disables.
    refusal_streak_limit: int = 60
    bookkeeping_tools: list[str] = [
        "record",
        "worklist_record",
        "worklist_add",
        "worklist_status",
        "worklist_drop",
        "worklist_defer",
        "collector_test",
    ]


class SourcesSpec(BaseModel):
    """The list of PLACES the agent works through, one level above the candidate worklist.

    A task that mines a corpus decides where to look before it decides what to take. The
    worklist handles the second half; without this, a recorded pool passes a stage gate and
    then falls out of the prompt (the context window covers only recent cycles), and nothing
    joins leads back to the place they came from.

    Everything task-specific is here rather than in the harness. `key` names the working-set
    field the agent records into; `noun` is whatever this task calls a place to look
    (repository, package, corpus directory); identity and per-place limits are borrowed from
    `worklist.caps`, so a lead attributes to its source through the same normalization the
    diversity gate already uses and the two can never disagree.

    `verify` is the entry gate: a `module:function` returning (ok, detail) for one recorded
    entry, run by the harness before the entry is admitted. This is where "is this actually a
    software project" belongs. Without it the gate can only count.
    """

    model_config = {"extra": "forbid"}  # see WorklistStateSpec: a typoed key is a lost control

    enabled: bool = False
    key: str = "source_pool"  # working-set field the agent records its list into
    noun: str = "source"  # what this task calls a place to look
    batch_size: int = 10  # how many places a bare sources_fetch() hands out
    # Entries NAMED in the standing prompt view, whatever the pool size. The layered view
    # usually names far fewer -- only the current batch is actionable -- but a large list must
    # not be able to push the prompt over on its own. 100 is the ceiling.
    view_cap: int = 100
    verify: str | None = None  # "module:function" -> (ok, detail) per entry
    # What an entry must carry. `requires` is presence only. `evidence` must ALSO differ
    # between entries: an evidence field records something OBSERVED about one place, and
    # thirty entries repeating one sentence is proof nobody looked. Which fields fall on
    # which side is the task's call; that the distinction exists is the harness's.
    requires: list[str] = []
    evidence: list[str] = []
    # The field whose distinct values measure spread. Matched against the declared caps by
    # FIELD NAME, so "org" finds the cap keyed on ["org"] whatever its dimension is labelled.
    spread_by: str | None = None
    # Fields the spec's exclusion roster is matched against. Naming them here is what keeps
    # the roster from being hardcoded to a repository URL and a project name.
    exclusion_fields: list[str] = []
    # Measurable admissibility clauses: [{name, observe, require}]. One declaration drives
    # the required fields, the gate's per-entry check, and the prompt's own description of
    # what to record. Grammar and rationale: agentcore/agent/admissibility.py.
    admissibility: list[dict] = []


# ── top-level agent + run ────────────────────────────────────────────────────


class AgentSpec(BaseModel):
    name: str
    description: str = ""
    loop: Literal["standard", "oracle"] = "standard"
    # json: main model emits the command JSON directly (structurer is a fallback on parse failure).
    # bash_block: a single ```bash fenced command.
    # freeform: SPLIT request — main model answers in free-form prose, then the `structurer`
    #           model converts that into the command every cycle (analysis_agent's two-phase style).
    action_protocol: Literal["json", "bash_block", "freeform"] = "json"
    models: dict[str, ModelSpec]
    env: EnvSpec = EnvSpec()
    tools: list[str] = []
    tool_modules: list[str] = []  # extra modules to register (e.g. "agentcore.tools.se")
    prompts: dict[str, str] = {}
    prerender: str | None = None
    stages: list[StageSpec] = []
    working_set_schema: str | None = None
    instance_schema: str | None = None
    instance_provider: str = "jsonl"
    oracle: OracleSpec = OracleSpec()
    budgets: BudgetSpec = BudgetSpec()
    attempts: AttemptsSpec = AttemptsSpec()
    repetition: RepetitionSpec = RepetitionSpec()
    stall: StallSpec = StallSpec()
    worklist: WorklistSpec = WorklistSpec()
    sources: SourcesSpec = SourcesSpec()
    collector: CollectorSpec = CollectorSpec()
    tool_output: ToolOutputSpec = ToolOutputSpec()
    context: ContextSpec = ContextSpec()
    policy: dict[str, Any] = {}
    side_effects: dict[str, Any] = {}
    hooks: dict[str, str] = {}
    repo_context: dict[str, Any] = {}  # opt-in repo-context add-on (clone/CI-discovery/web-search)
    env_watch: dict[str, Any] = {}  # opt-in EnvWatcher (per-cycle env-change signal)

    @model_validator(mode="after")
    def _check(self) -> "AgentSpec":
        if "main" not in self.models:
            raise ValueError("models must include a 'main' entry")
        stage_names = [s.name for s in self.stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError(f"duplicate stage names in {stage_names}")
        for s in self.stages:
            if s.transitions:
                for t in s.transitions:
                    if t not in stage_names:
                        raise ValueError(f"stage {s.name!r} transitions to unknown stage {t!r}")
        # budgets.thresholds is an untyped list[dict] (see BudgetSpec) precisely so authors
        # can add new threshold actions without a schema change -- but a `target` (currently
        # only meaningful for action=force_state) is always a stage name, so it gets the same
        # existence check as transitions above. Without this, a stale/typo'd target (e.g. left
        # over from a stage rename) doesn't error at load time -- it silently degrades at
        # runtime to a caught ValueError in loop.py's _apply_budget_thresholds, logged as a
        # repeated, easy-to-miss nudge instead of ever actually transitioning.
        for th in self.budgets.thresholds:
            if not isinstance(th, dict):
                continue
            target = th.get("target")
            if target is not None and target not in stage_names:
                raise ValueError(
                    f"budgets.thresholds target {target!r} is not a known stage name (stages: {stage_names})"
                )
        return self


class RunSpec(BaseModel):
    """The fully-resolved run configuration (agent + CLI/env overrides)."""

    agent: AgentSpec
    instances: str | None = None  # path to instances.jsonl (or provider arg)
    workspace_root: str = "./agent_workspace"
    output_root: str = "./runs"
    parallelism: int = 1
    isolation: Literal["in_process", "subprocess"] = "in_process"
