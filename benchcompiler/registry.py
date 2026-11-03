"""Documented defaults and named patterns the formalizer draws on.

These exist so ambiguity resolution is consistent and auditable across
specs, rather than the LLM inventing a fresh interpretation each call. Grown
from the project's literature study (study/schema.md, study/analysis.md).
Editable — add an entry here rather than letting the model improvise one.
"""

# Common vague terms -> a resolvable default, IF one is safe. An entry that
# says a term is NOT safely defaultable tells the formalizer to reject
# (AMBIGUOUS_TERM) rather than guess when a spec uses that term without
# quantifying it.
TERM_DEFAULTS = {
    "popular": "stars >= 1000 (or top-N by download/usage rank if the source has no star concept); state which was used",
    "recent": "created or last-active within 12 months of the population snapshot date",
    "active": "at least 1 commit within the last 6 months of the population snapshot date",
    "well-maintained": "has CI configured AND >=2 commits/month over the last 6 months",
    "reproducible": "identical outcome across 3 clean-environment reruns",
    "large project": "NOT SAFELY DEFAULTABLE — must be quantified (LOC range, file count, or commit count)",
    "small project": "NOT SAFELY DEFAULTABLE — must be quantified",
    "high coverage": "NOT SAFELY DEFAULTABLE — must be quantified (e.g. >=80% statement coverage)",
    "high quality": "NOT SAFELY DEFAULTABLE — restate as a concrete, checkable property",
    "correct": "NOT SAFELY DEFAULTABLE on its own — this is exactly what `correctness` must operationalize, not describe circularly",
}

# Named oracle strategies observed across the project's 22-paper catalog.
# `other` is the escape hatch for genuinely new patterns — they must still
# be operational (a precise decision procedure), not just labeled "other".
ORACLE_PATTERNS = {
    "fail-to-pass-tests": "tests that fail before a change and pass after it; nothing previously-passing regresses",
    "execution-replay": "re-execute a real process (e.g. CI) on both the original and candidate state; trust the observed transition, not recorded history",
    "external-authority": "harvest a verdict from an existing authoritative process (judge platform, CVE assignment, maintainer PR acceptance) — no manual labeling needed",
    "injection": "inject a known defect/label into a known-good artifact so ground truth is true by construction",
    "rerun-labeling": "execute N times under controlled variation (order, environment); label from the distribution of observed outcomes",
    "human-annotation": "domain experts label instances against a stated protocol; requires a priced human budget and, ideally, an inter-annotator agreement target",
    "llm-judge": "an LLM scores correctness; requires validation against a human-labeled sample, with the measured agreement reported in the benchmark's metadata",
    "reference-equivalence": "compare candidate behavior against a reference implementation over a generated/augmented input set",
    "other": "a genuinely new pattern not covered above — must still be operational (state the decision procedure precisely, not just a label)",
}

# Which of the patterns above establish ground truth by CONTRASTING two or more states of the
# artifact -- before and after a fix, clean and injected, candidate and reference. Those are the
# ones an `artifact_variants` contract can express, because a variant IS a state of the artifact
# and the certifier re-executes `rebuild-<name>` / `check-<name>` per state.
#
# The discriminator is contrast, NOT execution. `rerun-labeling` executes the artifact N times
# and belongs OUT: one artifact, many runs, ground truth read off the distribution -- there is no
# second state to compare against, so demanding two named variants would force an author to
# invent one. `llm-judge`, `human-annotation` and `external-authority` do not execute the
# artifact at all.
#
# Kept here rather than in the verifier so the vocabulary and its consequences stay in one file:
# the previous arrangement had the registry advertising four patterns the verifier forbade.
VARIANT_ORACLE_PATTERNS = frozenset(
    {
        "fail-to-pass-tests",  # the buggy state and the fixed state
        "execution-replay",  # the original state and the candidate state
        "injection",  # the known-good artifact and the injected one
        "reference-equivalence",  # the candidate implementation and the reference
    }
)

# Default quality gates applied when the spec's `quality` field is omitted
# or thin. Drawn from what the rigorous entries in the catalog actually did.
DEFAULT_QUALITY_GATES = [
    "reproducibility: re-verify a sample of instances across 3 clean reruns",
    "corruption sensitivity: a deliberately corrupted instance/label must be rejected by the pipeline's own gates",
    "deduplication: near-duplicate instances removed",
]

# Fields where DELEGATE is a legitimate answer to a whole field, not just a
# sub-item — i.e. fields with a safe system-wide default. task/instance/
# data/correctness are NOT here: per the project's RQ5 sufficiency analysis,
# the unit of instancehood and the operational correctness condition are
# irreducible and may not be delegated wholesale (sub-items within `data`
# or `criteria`, like a specific popularity threshold, MAY be delegated).
DELEGABLE_WHOLE_FIELDS = {"interface", "quality", "references", "constraints"}
