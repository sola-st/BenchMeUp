"""The drawn family config, checked against the spec's own sentences before any worker runs.

A generated config can be right in shape and wrong in meaning: variants that check out two
commits but never apply the test the spec says is applied to both snapshots, or a variant that
is a plain checkout where the spec defines a pairing of post-commit code with a pre-commit
comment. The schema checker accepts such configs.

This is the check that reads the meaning. A second model pass takes the spec and the drawn
config and must trace every sentence of the spec that says WHAT IS BUILT, WHAT IS APPLIED
BEFORE JUDGING, WHAT IS COMPARED OR READ, and WHAT THE BREAK CORRUPTS to the config block that
realizes it, or name the sentence that nothing realizes. The verdict feeds the drawer's next
round as the schema checker's refusals do. Task-blind: it knows nothing about tests, patches
or comments, only that the spec's sentences must have a home in the config.
"""

from __future__ import annotations

import json

from benchcompiler.llm import call_llm_json, strip_code_fence

ANSWER = """
{
  "claims": [
    {"claim": "<the spec's own words, one clause: what is built / applied before judging / compared or read / corrupted>",
     "realized_by": ["<block path in the config, e.g. variants.prefix.setup[1] (apply_patch test_patch)>", "..."],
     "realized": true | false,
     "gap": "<empty when realized; else what the config lacks, in one sentence>"}
  ],
  "verdict": "accept" | "refuse",
  "fix": "<empty when accepted; else the exact change to the config, naming blocks and verbs>"
}
"""

RULES = """
HOW TO AUDIT
1. Quote the spec. Every claim is a clause of the spec's own text about one of four things:
   (a) which sides (variants) are built and from what;
   (b) what is applied to a side BEFORE the verdict is read (a test, a patch, a submission,
       a mode, a pairing of one version's code with another version's text);
   (c) what is compared or read for the verdict, and on which side it must pass or fail;
   (d) what the deliberate break corrupts.
   List every such clause; do not stop at the first.
2. Trace each claim to the config. A claim is realized only by a concrete block: a variant's
   setup step, the observable and its reader, the oracle block, the negative control, the
   solver_input. A verb's `does` sentence realizes something only where a step runs the
   verb: a verb no variant runs before the verdict does not act before the verdict.
3. The one trap that matters most: something the spec applies to a side before judging
   (a test to both snapshots, a submission on top of the base, a text from one commit onto
   the code of another) that the config applies only in `solver_input`, or not at all. The
   harness reads the verdict on the variants AS THEIR SETUP BUILDS THEM; `solver_input` is a
   separate probe. If the spec judges WITH the thing applied, the variant's setup must apply
   it; if the spec judges WITHOUT it (the submission is the fix being judged), the setup must
   not. Say which the spec means, and check the config does that.
   THE RULE FOR "SUBMITTED" SENTENCES. A spec sentence like "apply the submitted test to the
   pre-fix snapshot: it must fail; apply it to the post-fix snapshot: it must pass" is about
   the solver's future submission. The configuration realizes it with the GOLD artifact
   standing in for the submission: when a setup step applies the gold file on BOTH sides the
   sentence names, the oracle says fail on one and pass on the other, and `solver_input`
   names the verb that applies such a file, the sentence is REALIZED. Write, for example:
     "realized_by": ["variants.prefix.setup[1] (apply_test gold_test.patch)",
                     "variants.postfix.setup[1] (apply_test gold_test.patch)", "oracle", "solver_input"]
   Never ask for a second solver_input, for the submission on every variant, or for the
   solver's own file in the setup: `solver_input` is one block, on one side, by design. It
   is a GAP only when the gold file is applied on neither side, on one side only, or by no
   setup step.
4. A pairing the spec defines ("post-commit code with pre-commit comment") is realized only
   by setup steps that build that pairing, not by checking out one commit and naming the
   variant after the pairing.
5. `select` must name the test the spec reads; when it reads an instance field, the field is
   written in braces.
5b. One claim per identity predicate, in these words: "predicate <its run text> holds for
   every candidate the definition admits". It is realized by "predicates" when it holds for
   every candidate the definition admits, in the clone setup.sh makes; it is a gap, with
   the fix, when the definition admits a candidate it rejects (see the predicate rule
   below).
6. Refuse when any claim is unrealized. The `fix` names the block to change and what to put
   there, in the config's own vocabulary (verbs, variants, args). Accept only when every
   claim is realized; then `fix` is empty.
7. Answer with ONE JSON object and nothing else.
"""


def build_audit_prompt(raw: dict, contract: dict) -> str:
    raw_text = "\n".join(f"{k}: {v}" for k, v in raw.items())
    shown = {
        k: contract.get(k)
        for k in (
            "verbs",
            "variants",
            "observable",
            "checks",
            "oracle",
            "negative_control",
            "solver_input",
            "measure",
            "predicates",
        )
    }
    shown["verbs"] = {
        name: {**(v or {}), "does": (contract.get("_does") or {}).get(name, "")}
        for name, v in (contract.get("verbs") or {}).items()
    }
    shown["instance_fields"] = contract.get("_instance_fields") or []
    return f"""You audit a benchmark family's configuration against the researcher's specification.

The configuration is a small JSON object that declares the scripts every instance ships
(verbs), the sides the harness builds (variants, each a list of setup steps), which script's
output is the verdict (observable), how sides must relate (oracle), what deliberately corrupts
the result (negative_control), and how a solver's submission is applied (solver_input). A
harness builds every variant exactly by its setup steps, runs the observable on each, and reads
the oracle. Nothing else happens: what the setup steps do not do is not done.

WHAT THE HARNESS DOES BY ITSELF, FROM THE BLOCKS (never a gap in the configuration):
- builds every variant by its setup steps, in a fresh container, and fails a variant whose
  step exits nonzero;
- runs the observable (one check, or every check of a list, read together) at least three
  times per input and requires the same output each time
  (determinism), and requires the verdict to depend on the input when there are several;
- reads the oracle: with reference-equivalence, every variant's verdict must equal the first
  variant's on every input; with fail-to-pass, the selected test must fail on `fails_on` and
  pass on `passes_on`;
- runs the negative control on a copy of its variant and requires the verdict to change;
- applies each `solver_input` probe value on a copy and requires the two results to differ;
- when `measure` is set: generates the inputs, times the runs, and enforces the floor, the
  ceiling and the ratio between the two named variants;
- runs the identity predicates over instance.json.
Claims about any of these are realized by the harness: mark them realized_by "harness".
Admissibility rules (stars, language, age, forks) and the search for candidates belong to
another step and are never a gap here. Recording, logging and reporting are the harness's
too: a configuration never needs a block for them.

TWO MORE SCOPES THAT ARE NOT GAPS:
- instance.json carries identity and locations only; gold texts, gold patches and reference
  files are written by a verb (the `does` sentence says so) under /bench or the variant. A
  claim that the instance "ships" or "records" a text is realized by that verb; never ask
  for it in instance_fields.
- How a SOLVER is scored at evaluation time (its submission applied to each side, judged
  there) is the consumer running these same verbs later. In the configuration it is
  realized by `solver_input` (the verb that applies a submission, with two probe values)
  together with the oracle. Do not ask for the solver's submission to be applied to every
  variant in the setup; ask only that the gold artifact the oracle needs is applied where
  the spec applies it.
  Concretely: a spec sentence of the form "apply the SUBMITTED test (patch, answer) to side
  A: it must fail; apply it to side B: it must pass" is realized in full by (1) the GOLD
  test (patch, answer) applied by a setup step on side A and on side B, (2) the oracle
  saying it fails on A and passes on B, and (3) `solver_input` naming the verb that applies
  such a test. Mark it realized_by those three. It is a GAP only when the gold artifact is
  applied on neither side, on one side only, or by no setup step.

AN IDENTITY PREDICATE MUST HOLD FOR EVERY CANDIDATE THE DEFINITION ADMITS. The predicates
run in the clone setup.sh makes, at the commit it checks out, BEFORE any verb runs. A
predicate that demands there a file the definition says the change adds or modifies (a test
the PR adds, a file the fix creates) rejects the family's normal case: refuse it, quoting the
predicate and the sentence of the definition it contradicts, and say what to check instead
(the file at the commit that has it: `git cat-file -e {{commit}}:{{path}}`). A predicate
that demands a file only a verb produces (`test -f /work/repo/{{program_name}}` when the
program is the executable the BUILD makes) rejects every candidate: refuse it the same way,
and say that the stage running that verb proves the field instead. A predicate that turns a
non-path field into a path by guessing (`test -f /work/repo/{{program_name}}.c`: nothing says
the program's source is one file of that name at the root) is a guess, not a predicate: refuse
it, and say that only a field that IS a path in the repository at the commit gets a file
predicate.

A verb's `does` sentence realizes what the verb does WHEN A SETUP STEP RUNS IT (an install
folded into a checkout verb counts, if a variant runs that verb). It does not realize an
action on a side where no step runs it.

The specification is the only source of truth. Your job: for every sentence of the spec that
says what is built, what is applied before judging, what is compared or read, and what the
break corrupts, find the configuration block that does it, or say that none does. Confine
yourself to those four kinds of claim; a claim outside them is not a claim.

=== THE SPEC (the researcher's own words) ===
{raw_text}

=== THE CONFIGURATION TO AUDIT ===
{json.dumps(shown, indent=1)}

=== YOUR ANSWER, ONE JSON OBJECT ===
{ANSWER}
{RULES}
"""


def parse_audit_answer(text: str) -> tuple[dict | None, str]:
    try:
        raw = json.loads(strip_code_fence(text))
    except Exception as e:  # noqa: BLE001
        return None, f"not a JSON object: {e}"
    if not isinstance(raw, dict) or raw.get("verdict") not in ("accept", "refuse"):
        return None, "the answer has no verdict of accept or refuse"
    claims = raw.get("claims")
    if not isinstance(claims, list) or not claims:
        return None, "the answer lists no claims; the spec has sentences about what is built and judged"
    unrealized = [c for c in claims if isinstance(c, dict) and not c.get("realized")]
    if raw["verdict"] == "accept" and unrealized:
        return None, "the answer accepts while listing an unrealized claim; an unrealized claim is a refusal"
    if raw["verdict"] == "refuse" and not str(raw.get("fix") or "").strip():
        return None, "the answer refuses without a fix; name the block to change and what to put there"
    return raw, ""


def audit_contract(raw: dict, contract: dict, model: str) -> tuple[bool, str, dict | None]:
    """(accepted, why-or-fix, the audit). A malformed audit answer is retried once; a second
    malformed answer counts as a refusal that says so, never as an acceptance."""
    last = ""
    for _ in range(2):
        # 32000, like the drawer: with a small limit a reasoning model can spend every token
        # reasoning and answer nothing (finish reason "length").
        result = call_llm_json(
            model, build_audit_prompt(raw, contract), temperature=0, site="contract.audit", max_tokens=32000
        )
        text = result if isinstance(result, str) else json.dumps(result)
        audit, why = parse_audit_answer(text)
        if audit is not None:
            if audit["verdict"] == "accept":
                return True, "", audit
            gaps = "; ".join(
                f"{c.get('claim')!s} -> {c.get('gap')!s}"
                for c in audit["claims"]
                if isinstance(c, dict) and not c.get("realized")
            )
            return False, f"UNREALIZED IN THE CONFIG: {gaps}\nFIX: {audit.get('fix')}", audit
        last = why
    return False, f"the audit could not be read ({last}); the config is not accepted without an audit", None


def audit_table(audit: dict | None) -> str:
    """One line per claim, for logs and reports."""
    if not audit:
        return "(no audit)"
    lines = []
    for c in audit.get("claims") or []:
        if not isinstance(c, dict):
            continue
        mark = "ok  " if c.get("realized") else "GAP "
        by = ", ".join(str(x) for x in (c.get("realized_by") or [])) or "-"
        lines.append(
            f"{mark} {str(c.get('claim'))[:110]}  <- {by[:90]}"
            + (f"  | {c.get('gap')}" if not c.get("realized") else "")
        )
    lines.append(f"verdict: {audit.get('verdict')}" + (f"  fix: {audit.get('fix')}" if audit.get("fix") else ""))
    return "\n".join(lines)
