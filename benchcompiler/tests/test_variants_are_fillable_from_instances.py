"""Every placeholder in a declared check_cmd must be a field the collector actually records.

For every task that declares a variant contract, each `{placeholder}` in each `check_cmd` must be a
field the collector's worklist requires or derives; a placeholder nobody records is a command that
cannot run.
"""

from __future__ import annotations

import json
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = os.path.join(HERE, "..", "..", "agents")

COLLECTORS = ["defects4j_repair_collector", "swtbench_repro_collector", "boca_flags_collector"]

# Supplied by the artifact runtime rather than by an instance record: every artifact has a
# working copy, so a contract may reference it without any collector recording it.
RUNTIME_PROVIDED = {"workdir", "work_dir", "root", "instance_id", "id", "variant"}

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _spec(agent):
    path = os.path.join(AGENTS, agent, "instances", "instances.jsonl")
    return json.loads(open(path).read().splitlines()[0])


def _recorded_fields(agent):
    """Everything an instance of this collector can carry: required, derived, or the key."""
    with open(os.path.join(AGENTS, agent, "agent.yaml")) as fh:
        w = (yaml.safe_load(fh) or {}).get("worklist") or {}
    out = set(w.get("key") or [])
    for st in w.get("states") or []:
        out |= set(st.get("requires") or [])
        out |= set(st.get("derives") or [])
    return out


def test_the_check_would_catch_a_placeholder_nobody_records():
    """The guard must be able to fail, or it checks nothing."""
    fields = {"triggering_test"} | RUNTIME_PROVIDED
    cmd = "run --test {triggering_test} --seed {random_seed}"
    missing = [p for p in _PLACEHOLDER.findall(cmd) if p not in fields]
    assert missing == ["random_seed"]
