"""The registered sites of the pipeline stages that are not agent runs.

Registered into `agentcore.observability.sites.SITES` on import, so the one registry covers
the whole pipeline while the agentcore tests, which cannot import benchcompiler, keep
checking their own half. `benchcompiler/tests/test_offline_stages_record_their_llm_exchanges.py`
checks that every `where` here imports.
"""

from agentcore.observability.sites import SITES, Site

STAGE_SITES = {
    "formalizer.formalize": Site(
        "benchcompiler.formalizer:_formalize_once", "one formalization round: raw spec plus feedback -> reply"
    ),
    "formalizer.verdict": Site(
        "benchcompiler.formalizer:_formalize_once",
        "what the round concluded: ACCEPT, REJECT, or retry with the model's own blockers",
        classes=("offline-decision-untraced",),
    ),
    "formalizer.propose": Site(
        "benchcompiler.formalizer:propose_oracle_strategies",
        "alternative oracle strategies for an unpriced correctness mechanism",
    ),
    "designer.contract": Site(
        "benchcompiler.designer:design_contract",
        "one contract round: canonical spec plus feedback -> measure/observables/parameters",
    ),
    "designer.session": Site("benchcompiler.designer:design_session", "one consumer-session round"),
    "designer.lifecycle": Site("benchcompiler.designer:design_lifecycle", "one lifecycle round"),
    "designer.grade": Site(
        "benchcompiler.designer:design_contract",
        "the grader's problems with a round's reply, and whether the loop stopped",
        classes=("offline-decision-untraced",),
    ),
    "verifier.propose": Site(
        "benchcompiler.verifier_loop:model_proposer",
        "one repair round: the instance record plus what happened -> a recipe",
    ),
    "preflight.launch_gate": Site(
        "benchcompiler.bench_cli:cmd_fanout",
        "the mechanical gate before a paid launch: pass or refuse, with every problem",
        classes=("paid-launch-unrecorded",),
    ),
    # bench certify: a record per verb execution, and the last word.
    "certify.start": Site("benchcompiler.bench_cli:run_consumer_sweep", "the certification sweep opened its stage log"),
    "certify.verb": Site(
        "benchcompiler.bench_cli:run_consumer_sweep",
        "one execution of the artifact's own verb: instance, verb, exit, seconds",
        recurring=True,
    ),
    "certify.summary": Site(
        "benchcompiler.bench_cli:run_consumer_sweep", "the sweep's verdict: instances fully ok and with failures"
    ),
    "certify.end": Site(
        "benchcompiler.bench_cli:run_consumer_sweep", "the stage log closed; an error here means the sweep raised"
    ),
    "llm.call": Site("benchcompiler.llm:call_llm_json", "a model exchange whose caller named no site"),
}

SITES.update(STAGE_SITES)
