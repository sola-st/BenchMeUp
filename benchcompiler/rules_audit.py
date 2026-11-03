"""A second model reviews the search rules against the specification.

The formalizer turns the researcher's specification into search rules (the canonical spec)
and checks them mechanically. This module adds the review the approach describes: a second
model reads the specification and the proposed rules and says whether the rules omit a
requirement the specification states or leave an important ambiguity unresolved (a threshold
the specification only names, a source it does not pin down). Its feedback goes back to the
generating model, which revises the rules; `formalize` repeats this until the judge accepts
or the round limit is reached.

The judge compares the rules with the specification only. Its approval does not establish
that the resulting criteria are objectively correct.
"""

from __future__ import annotations

import json

from benchcompiler.llm import call_llm_json, strip_code_fence

ANSWER = """
{
  "omissions": [
    {"spec_clause": "<the specification's own words, one clause>",
     "gap": "<what the rules leave out of it, in one sentence>"}
  ],
  "ambiguities": [
    {"term": "<a word or condition the rules still leave open>",
     "gap": "<why an agent would have to guess, in one sentence>"}
  ],
  "verdict": "accept" | "refuse",
  "fix": "<empty when accepted; else the exact change to the rules, naming the fields>"
}
"""

INSTRUCTIONS = """
HOW TO REVIEW
1. Read the specification and list every requirement it states about what a candidate is, where
   candidates may be found, which candidates are kept or excluded, and what is recorded about
   each one.
2. Check that each requirement has a home in the rules. A requirement with none is an omission.
3. Check that the rules settle every term the specification leaves vague. Search agents work
   independently, so a rule that says "popular" or "recent" without a number lets two agents
   apply two different policies. A term left open is an ambiguity.
4. Do not judge whether the criteria are good. Judge only whether the rules say what the
   specification says and leave no decision to the agents that the specification leaves open.
5. Accept when there are no omissions and no ambiguities that matter. Otherwise refuse and give
   one fix that names the fields to change.
"""


def build_judge_prompt(raw: dict, rules: dict) -> str:
    spec_text = "\n".join(f"{key}: {value}" for key, value in raw.items())
    return f"""You review the search rules generated for a benchmark against the researcher's
specification, before any search agent uses them.

=== THE SPECIFICATION (the researcher's own words) ===
{spec_text}

=== THE SEARCH RULES ===
{json.dumps(rules, indent=1, default=str)}
{INSTRUCTIONS}
Answer with one JSON object of this shape:
{ANSWER}
"""


def parse_judge_answer(text: str) -> tuple[dict | None, str]:
    try:
        answer = json.loads(strip_code_fence(text))
    except Exception as exc:  # noqa: BLE001
        return None, f"not a JSON object: {exc}"
    if not isinstance(answer, dict) or answer.get("verdict") not in ("accept", "refuse"):
        return None, "the answer has no verdict of accept or refuse"
    open_items = list(answer.get("omissions") or []) + list(answer.get("ambiguities") or [])
    if answer["verdict"] == "accept" and open_items:
        return None, "the answer accepts while listing an omission or an ambiguity"
    if answer["verdict"] == "refuse" and not str(answer.get("fix") or "").strip():
        return None, "the answer refuses without a fix; name the field to change and what to put there"
    return answer, ""


def judge_rules(raw: dict, rules: dict, model: str) -> tuple[bool, str]:
    """(accepted, feedback). A malformed answer is retried once; a second malformed answer is
    a refusal that says so, never an acceptance."""
    last = ""
    for _ in range(2):
        result = call_llm_json(
            model, build_judge_prompt(raw, rules), temperature=0, site="rules.judge", max_tokens=32000
        )
        text = result if isinstance(result, str) else json.dumps(result)
        answer, why = parse_judge_answer(text)
        if answer is not None:
            if answer["verdict"] == "accept":
                return True, ""
            gaps = [
                f"{item.get('spec_clause')!s} -> {item.get('gap')!s}"
                for item in answer.get("omissions") or []
                if isinstance(item, dict)
            ]
            gaps += [
                f"{item.get('term')!s}: {item.get('gap')!s}"
                for item in answer.get("ambiguities") or []
                if isinstance(item, dict)
            ]
            return False, "THE RULES DO NOT SETTLE: " + "; ".join(gaps) + f"\nFIX: {answer.get('fix')}"
        last = why
    return False, f"the review could not be read ({last}); the rules are not accepted without a review"
