"""Ask a model to write the JUDGMENT sections of one construction stage's instruction; the
harness writes the factual sections and assembles the instruction from the config.

The split is by who knows what. The model writes three sections: where the builder is (in
the benchmark's own terms, bounded by what the earlier stages established), the goal of the
stage, and the traps for this kind of program. The harness writes the rest from the config
(what to write, how it is checked, when the stage is done, the example of a passing report),
so those sections cannot be wrong. Only the model's three sections are stored, on the config
as `_stage_sections`; the instruction is assembled from them and the CURRENT config every
time it is rendered (agentcore.recipes.stage_instruction), so an edited number or predicate
reaches the builder.

The model is not asked for the whole prompt: it could invent files and checks and describe
harness behaviour that does not exist, and a presence check cannot catch a confident false
sentence. The validator here refuses a section for an invented script, a
later stage's file, a sentence that asks the builder to record something, or a sentence that
says what the harness will do.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD = os.path.abspath(os.path.join(HERE, ".."))
if SCAFFOLD not in os.sys.path:
    os.sys.path.insert(0, SCAFFOLD)

from agentcore import recipes  # noqa: E402

ASK = """
WHAT WE WANT FROM YOU

Write three short pieces of text for this stage. They will be shown to the builder agent, in
this order, above the factual sections the system writes itself. Answer with ONE JSON object:

{
  "where_you_are": "one paragraph, 60 to 120 words. Tell the builder what it is doing overall
                    (an instance of THIS benchmark, as scripts that rebuild from nothing) and
                    what the earlier stages established. Use exactly the facts listed above
                    under 'what earlier stages established', nothing more. For the first
                    stage, say that nothing exists yet.",
  "goal":          "two to five sentences, 50 to 120 words. Say what the builder must achieve
                    in this stage, in the benchmark's own terms, and why it matters for the
                    benchmark's verdict or measurement.",
  "traps":         ["three to six sentences, one per bullet. Each names a mistake a builder
                     makes on THIS kind of program at this stage, and what to do instead. Be
                     concrete: this benchmark's build systems, outputs, inputs and tools."]
}

HOW TO WRITE IT

Address the builder as "you". Plain sentences, concrete, no marketing, no bold, no emoji.
Explain as you would to a capable student who has not seen this benchmark before: say what
the builder is trying to get right before saying what to avoid.

Name only the scripts listed for this stage. Never name a later stage's file, and never a
script that is not in the family's list.

The system writes the file list, the checks and the completion rule itself. Do not describe
what the system or the harness will check, run, measure, record or verify; if you find
yourself writing "the harness will ...", drop the sentence.

The builder writes scripts and records nothing. Where the benchmark description says a
runtime, a median, an output or an exception is "recorded", the system records it from the
scripts' runs. Never ask the builder to record, write down, print or report a number or a
reason, in any form: not in a file, not on a script's output, not in a note. There is no
exception path for the builder: a candidate that cannot meet a check is dropped with its
reason, by the builder, through the tool it is given for that.

The builder may change any script written in an earlier stage when the current stage needs
it (a different command in the run script, a bigger input in the generator). Never tell the
builder that a script is fixed or must not be modified.

Answer with the JSON object only.
"""


def _stage_facts(contract: dict, i: int) -> str:
    c = contract
    stages = recipes.stages_for(c)
    st = stages[i]
    does = c.get("_does") or {}
    lines = [f"Stage {i + 1} of {len(stages)}: {st['name']}"]
    if st.get("goal"):
        lines.append(f"The stage's intent, from the family design: {st['goal']}")
    lines.append("Files this stage writes:")
    for f in st["files"]:
        if f.endswith(".sh") and f[:-3] in c["verbs"]:
            args = " ".join(f"<{a}>" for a in c["verbs"][f[:-3]]["args"])
            lines.append(f"  {f} {args}".rstrip() + (f"   -- {does[f[:-3]]}" if does.get(f[:-3]) else ""))
        else:
            lines.append(f"  {f}")
    if not st["files"]:
        lines.append("  (none new; this stage changes what earlier stages wrote)")
    earlier_files = [f for s2 in stages[:i] for f in s2["files"] if f.endswith(".sh")]
    if earlier_files:
        lines.append(
            "Scripts from earlier stages the builder may change in this stage when it needs to: "
            + ", ".join(earlier_files)
        )
    earlier = [r for s2 in stages[:i] for r in s2["rows"]]
    if earlier:
        lines.append(
            "What earlier stages established, EXACTLY this and nothing more: "
            + "; ".join(recipes.row_label(r, c) for r in earlier)
        )
    else:
        lines.append("This is the first stage: nothing exists yet.")
    # The split the planner is held to (benchcompiler.contract_prompt): what arrives with the
    # candidate against what the builder has to go and find. Without it the first stage's text
    # names a commit or a test as though it were given, and the finding work goes unwritten.
    given = list(c.get("_identity_fields") or [])
    establish = [f for f in (c.get("_instance_fields") or []) if f not in given and f != "builder_notes"]
    if given and establish:
        lines.append("What the candidate ARRIVES WITH, and nothing else: " + ", ".join(given))
        lines.append(
            "What the builder must FIND for itself before instance.json can be written: "
            + ", ".join(establish)
            + (
                ". That search is this stage's real work: write the goal around it, and "
                "never say or imply that any of these is already in hand."
                if i == 0
                else ". An earlier stage found these; they are in hand now."
            )
        )
    later = [f for s2 in stages[i + 1 :] for f in s2["files"]]
    if later:
        lines.append("Files of LATER stages (never mention them): " + ", ".join(later))
    lines.append("All scripts of this family: " + ", ".join(f"{v}.sh" for v in c["verbs"]) + ", setup.sh")
    lines.append(
        "This stage's checks (the system describes them itself; you only need to know what they are): "
        + "; ".join(recipes.row_label(r, c) for r in st["rows"])
    )
    o = c["observable"]
    checks = c.get("checks") or [o]
    lines.append(
        "Two machines: the builder's own, and the fresh container bench_check builds from the "
        "Dockerfile. /work/repo is in the fresh one; on the builder's it exists only if the "
        "builder ran setup.sh there. Never write that the builder already has the source, or "
        "that an earlier stage left anything on its machine."
    )
    lines.append(
        "Family facts: the versions are "
        + ", ".join(c["variants"])
        + "; the result is "
        + " and ".join(
            f"{ch['verb']}.sh's "
            + (
                "exit code"
                if ch["reader"] == "exit-code"
                else "JUnit report on stdout"
                if ch["reader"] == "junit-xml"
                else "stdout"
            )
            for ch in checks
        )
    )
    if c.get("measure"):
        m = c["measure"]
        lines.append(
            f"  runtime band {m['floor_ms']:g}..{m['ceiling_ms']:g} ms on the costliest input, "
            f"{m['ratio_of'][0]} at least {m['min_ratio']:g}x slower than {m['ratio_of'][1]}, "
            f"{m['repetitions']} runs each, all measured by the system"
        )
        lines.append(
            f"  How inputs reach the program: {m['inputs']}.sh writes the input files into a "
            f"directory; the system runs {o['verb']}.sh once per input file, passing that file "
            f"as <input>, and picks the costliest itself. {o['verb']}.sh never chooses an input."
        )
    return "\n".join(lines)


def build_stage_prompt_request(
    contract: dict, i: int, raw: dict, feedback: str = "", previous: str | None = None
) -> str:
    spec_text = "\n".join(f"{k}: {v}" for k, v in raw.items())
    retry = ""
    if feedback:
        retry = (
            f"\n\nYOUR PREVIOUS ANSWER WAS REFUSED. Each item below is a rule you broke; fix every one "
            f"and keep everything else:\n{feedback}\n\nPrevious answer:\n{(previous or '')[:4000]}"
        )
    return f"""THE STORY

We are building a benchmark automatically from a written description. For each candidate
project, a builder agent (an LLM) writes a small directory of scripts: a Dockerfile, a setup
script, and a few named scripts, so that the benchmark instance rebuilds itself from nothing.
The system then replays those scripts in a fresh container and derives every fact itself.

The builder works in stages, one at a time. Each stage has its own instruction and is closed
by fixed checks. The system writes the factual parts of each stage's instruction (the files
to write, the checks, when the stage is done). You write the parts that need judgment about
THIS benchmark: where the builder is, what the stage is for, and the traps.

THE BENCHMARK, IN THE RESEARCHER'S OWN WORDS
(This describes what a FINISHED instance must satisfy. Wherever it says something is
"recorded", the system records it; the builder writes scripts only.)
{spec_text}

THE STAGE AND ITS FACTS
{_stage_facts(contract, i)}
{ASK}{retry}
"""


# Phrases that ask the builder to record a number or a reason. A NEGATED use ("do not record
# the runtime") is the advice we want and passes.
_RECORDING = (
    "record the",
    "records the",
    "recording the",
    "write down",
    "report the median",
    "exception field",
    "achieved runtime",
    "output the achieved",
    "and a reason",
    "record a reason",
    "print the runtime",
    "print the median",
)
_NEGATIONS = ("not ", "never ", "don't ", "do not ", "no ", "nothing", "without", "instead of")

# A sentence that freezes a script the builder wrote earlier: the system's own line in the
# last stage says "you need a different command, not a bigger file", and a trap saying
# run.sh "must remain unchanged" contradicts it.
_FROZEN = re.compile(
    r"(must remain unchanged|must not be (modified|changed)|do not modify|"
    r"are fixed from earlier|is fixed from earlier|never (modify|change|edit))",
    re.I,
)

# A sentence that says what the harness or the system does. The model does not know that, and
# boca's first draw said the harness "will detect the exception from the measured runtimes"
# (no such path exists) and "will check that the input files are not identical to any known
# external file" (no such check exists).
_HARNESS_CLAIM = re.compile(
    r"\b(the\s+)?(?<!build )(?<!file )(?<!file)(?<!operating )(?<!type )(?<!init )"
    r"(harness|system|checker)\b[^.;]{0,80}?\b(will|checks?|verif\w*|measures?|detects?|"
    r"records?|re-?runs?|compares?|confirms?|validates?|rejects?|requires?|builds?|runs?|calls?|"
    r"applies|installs?|reads?)\b",
    re.I,
)


def _asks_to_record(text: str) -> list[str]:
    low = text.lower()
    hits = []
    for phrase in _RECORDING:
        for m in re.finditer(re.escape(phrase), low):
            before = low[max(0, m.start() - 40) : m.start()]
            if not any(neg in before for neg in _NEGATIONS):
                hits.append(phrase)
                break
    return hits


def harness_claims(text: str) -> list[str]:
    """Sentences that describe what the harness or the system does."""
    out = []
    for sentence in re.split(r"(?<=[.;!?])\s+", text):
        if _HARNESS_CLAIM.search(sentence):
            out.append(sentence.strip()[:160])
    return out


def validate_stage_sections(answer, contract: dict, i: int) -> str:
    """'' when the three sections are usable; else the misses, named."""
    c = contract
    stages = recipes.stages_for(c)
    st = stages[i]
    misses = []
    if not isinstance(answer, dict):
        return "the answer is not a JSON object"
    where = str(answer.get("where_you_are") or "")
    goal = str(answer.get("goal") or "")
    traps = answer.get("traps")
    if not 40 <= len(where.split()) <= 160:
        misses.append(f"where_you_are has {len(where.split())} words; write 60 to 120")
    if not 30 <= len(goal.split()) <= 160:
        misses.append(f"goal has {len(goal.split())} words; write 50 to 120")
    if (
        not isinstance(traps, list)
        or not 3 <= len(traps) <= 6
        or not all(isinstance(t, str) and t.strip() for t in traps)
    ):
        misses.append("traps must be a list of three to six non-empty sentences")
        traps = [str(t) for t in (traps or []) if isinstance(t, str)]
    text = " ".join([where, goal, *traps])
    known = {f"{v}.sh" for v in c["verbs"]} | {"setup.sh"}
    for name in sorted(set(re.findall(r"(?<![\w/.-])([A-Za-z_][\w-]*\.sh)\b", text))):
        if name not in known:
            misses.append(f"{name} is not a script of this family (the scripts are {sorted(known)})")
    later = {f for s2 in stages[i + 1 :] for f in s2["files"]} - set(st["files"])
    for f in sorted(later):
        if re.search(r"(?<![\w/])" + re.escape(f) + r"(?!\w)", text):
            misses.append(f"{f} belongs to a later stage and must not be mentioned")
    return "; ".join(misses)


def notes_on_stage_sections(answer, contract: dict, i: int) -> list[str]:
    """Sentences worth a human's look, NOT refusals (the harness cannot judge wording
    reliably). Logged beside the accepted text."""
    if not isinstance(answer, dict):
        return []
    text = " ".join(
        [
            str(answer.get("where_you_are") or ""),
            str(answer.get("goal") or ""),
            *[str(t) for t in (answer.get("traps") or []) if isinstance(t, str)],
        ]
    )
    notes = []
    for phrase in _asks_to_record(text):
        notes.append(f"the phrase {phrase!r} asks the builder to record something (the system records)")
    for sentence in harness_claims(text):
        notes.append(f"this sentence describes what the harness or the system does: {sentence!r}")
    for word in sorted(
        set(
            m.group(0).lower()
            for m in re.finditer(r"(?<!commit )(?<!commit_)\b(digests?|hash(?:es|ing)?)\b", text, re.I)
        )
    ):
        notes.append(f"the word {word!r} (the checks compare the program's output)")
    for sentence in re.split(r"(?<=[.;!?])\s+", text):
        if _FROZEN.search(sentence) and re.search(r"\b[a-z_]+\.sh\b|\bscripts?\b", sentence, re.I):
            notes.append(
                f"this sentence calls a script fixed, but the builder may change any earlier "
                f"script: {sentence.strip()[:160]!r}"
            )
    return notes


def assemble_stage_prompt(contract: dict, i: int, sections: dict) -> str:
    """The full instruction: the model's judgment sections in the system's factual frame,
    rendered by the same function the worker uses at run time."""
    stages = recipes.stages_for(contract)
    secs = list(contract.get("_stage_sections") or [None] * len(stages))
    secs += [None] * (len(stages) - len(secs))
    secs[i] = {
        "where_you_are": sections.get("where_you_are"),
        "goal": sections.get("goal"),
        "traps": list(sections.get("traps") or []),
    }
    return recipes.stage_instruction(stages[i], dict(contract, _stage_sections=secs))


def draw_stage_prompt(contract: dict, i: int, raw: dict, model: str, rounds: int = 4) -> tuple[dict | None, list[str]]:
    """Draw one stage's sections; returns (sections, log)."""
    from benchcompiler.llm import call_llm_json

    feedback, previous, log = "", None, []
    for r in range(rounds):
        answer = call_llm_json(
            model,
            build_stage_prompt_request(contract, i, raw, feedback, previous),
            temperature=0,
            site="stage_prompt",
            max_tokens=4000,
        )
        why = validate_stage_sections(answer, contract, i)
        if not why:
            sections = {
                "where_you_are": str(answer["where_you_are"]).strip(),
                "goal": str(answer["goal"]).strip(),
                "traps": [str(t).strip() for t in answer["traps"]],
            }
            for n in notes_on_stage_sections(answer, contract, i):
                log.append(f"stage {i + 1} round {r + 1}: noted, not refused -- {n}")
            log.append(f"stage {i + 1} round {r + 1}: accepted")
            return sections, log
        log.append(f"stage {i + 1} round {r + 1}: refused -- {why}")
        feedback, previous = why, (json.dumps(answer)[:4000] if answer is not None else None)
    return None, log


def draw_stage_prompts(contract: dict, raw: dict, model: str, rounds: int = 4) -> tuple[list[dict | None], list[str]]:
    """Draw every stage's sections; on success they are stored on the contract as
    `_stage_sections` and the assembled instructions are returned for the record."""
    n = len(recipes.stages_for(contract))
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda i: draw_stage_prompt(contract, i, raw, model, rounds=rounds), range(n)))
    sections = [s for s, _ in results]
    log = [ln for _, lg in results for ln in lg]
    if all(sections):
        contract["_stage_sections"] = sections
    return sections, log
