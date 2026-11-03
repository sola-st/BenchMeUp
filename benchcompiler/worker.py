"""Emit a WORKER from the spec, its canonical and a drawn family config.

The pipeline: the user writes the spec; the system draws the canonical, the family config
and the staged plan; the system searches for candidates; workers do instance construction
through the stages; the system verifies the constructed instances. This module is the step
that turns "config and staged plan" into a worker directory the runner can launch:

    <out>/
      agent.yaml               the worker: one loop, one prompt, the registration state and
                               one state per construction stage (generated from the config)
      contract.json            the family config, the model's stage plan and stage sections
      prompts/                 (shared: agents/_shared/worker_system.j2, worker_cycle.j2)
      instances/instances.jsonl  what the worker is told about the benchmark and its candidates
      instances/schema.py      the instance schema

Identity comes from the canonical's search block (`sampling.lead_key`, its normalizers and
caps), never from the model: the search identifies a candidate by those fields, the worker
registers it by them, and the config's instance.json carries them under the same names.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD = os.path.abspath(os.path.join(HERE, ".."))
if SCAFFOLD not in sys.path:
    sys.path.insert(0, SCAFFOLD)

from agentcore import recipes  # noqa: E402
from agentcore.tools import recipes_tools as rt  # noqa: E402

SCHEMA_PY = '''"""Instance schema for a worker: what it is told about the benchmark and its candidates."""

from __future__ import annotations

from pydantic import BaseModel


class WorkerTask(BaseModel):
    benchmark_name: str
    task_description: str
    candidate_definition: str
    source_requirements: str
    criteria: str
    sample_size: int = 10
    criteria_list: list[str] = []
    quality_gates_list: list[str] = []
    instance_format: str = ""
    examples: list[dict] = []
    admissibility: list[dict] = []
    identity_field: str = ""
    identity_fields: list[str] = []       # what the search hands over
    fields_to_establish: list[str] = []   # instance.json fields the worker establishes itself
    candidates: list[dict] = []   # the pool the system hands to the worker, one at a time
'''


def _canon(canonical: dict) -> dict:
    return canonical.get("canonical_spec") if isinstance(canonical.get("canonical_spec"), dict) else canonical


def identity_from_canonical(canonical: dict) -> tuple[list[str], dict, list[dict]]:
    """(key fields, normalizers, caps) from the canonical's search block."""
    c = _canon(canonical)
    s = c.get("sampling") or {}
    key = [str(k) for k in (s.get("lead_key") or [])]
    if not key:
        raise ValueError(
            "the canonical's sampling block declares no lead_key; the search step "
            "must say what identifies a candidate before a worker can register one"
        )
    normalize = {str(k): str(v) for k, v in (s.get("normalize") or {}).items() if str(k) in key}
    caps = []
    for cap in s.get("caps") or []:
        if isinstance(cap, dict) and cap.get("keys") and set(map(str, cap["keys"])) <= set(key):
            caps.append(
                {
                    "dimension": str(cap.get("dimension") or "spread"),
                    "keys": [str(k) for k in cap["keys"]],
                    "limit": cap.get("limit"),
                }
            )
    if not caps:
        # One program once, by its first identity field: what the boca campaign used when the
        # canonical's cap keys (org, repo) were not fields a candidate carries.
        caps.append({"dimension": "repository", "keys": [key[0]], "limit": 1})
    return key, normalize, caps


def _text(v) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v or "")


def worker_instance(
    raw_spec: dict, canonical: dict, contract: dict, key: list[str], benchmark_name: str, sample_size: int
) -> dict:
    c = _canon(canonical)
    crit = c.get("criteria") or []
    crit_list = [str(x) for x in crit] if isinstance(crit, list) else [str(crit)]
    files = ["Dockerfile", "setup.sh", "instance.json"] + [f"{v}.sh" for v in contract["verbs"]]
    fields = list(contract.get("_instance_fields") or key)
    return {
        "benchmark_name": benchmark_name,
        "task_description": _text(raw_spec.get("task") or c.get("task")).strip(),
        "candidate_definition": _text(c.get("candidate_definition") or raw_spec.get("instance")).strip(),
        # The spec's own data paragraph: where candidates come from and what is not a source.
        # The search step's handover appends its leads to this text.
        "source_requirements": _text(raw_spec.get("data") or c.get("data_source")).strip(),
        "criteria": " ".join(crit_list),
        "criteria_list": crit_list,
        "quality_gates_list": [str(x) for x in (c.get("quality_gates") or [])]
        if isinstance(c.get("quality_gates"), list)
        else [],
        "sample_size": int(sample_size),
        "instance_format": (
            "a directory of scripts under /bench: " + ", ".join(files) + f"; instance.json carries {', '.join(fields)}"
        ),
        "examples": [x for x in (c.get("examples") or []) if isinstance(x, dict)],
        "admissibility": [x for x in (c.get("admissibility") or []) if isinstance(x, dict)],
        "identity_field": key[0],
        # The search hands the identity; every other instance.json field is the worker's to
        # establish, and the brief says so.
        "identity_fields": list(key),
        "fields_to_establish": [f for f in fields if f not in key],
        # Whether the identity names ONLY where to look, or also WHAT to look at. sonar and
        # comment are identified by the repository alone, so a noted commit that does not hold
        # up is one lead among many and the worker keeps looking in that repository. boca, d4j
        # and swt carry the commit IN the identity: there, a different commit is a different
        # candidate, and swapping it delivers an instance under an identity that no longer
        # describes it.
        "identity_is_repository_only": len(key) == 1,
    }


def registration_instruction(key: list[str], raw_spec: dict) -> str:
    fields = ", ".join(key)
    return (
        f"The system hands you this candidate; its identity is {fields}, exactly as the search "
        f"step recorded it. Your instance.json carries these same values."
    )


def render_agent_yaml(
    name: str,
    description: str,
    key: list[str],
    normalize: dict,
    caps: list[dict],
    contract: dict,
    raw_spec: dict,
    *,
    model: str,
    budget_usd: float = 2.0,
    wallclock_s: int = 21600,
) -> str:
    stage_names = rt.stage_state_names(contract)
    stages = recipes.stages_for(contract)
    # The instruction text is NOT baked in: the worker renders every stage's instruction from
    # contract.json at run time (recipes_tools.sync_stage_instructions), so an edit to the
    # config reaches the prompt. The placeholder is what a reader of agent.yaml sees.
    stage_states = [
        {
            "name": sname,
            "requires": [f"stage_{i + 1}_delivered"],
            "instruction": f"(stage {i + 1} of {len(stages)}: {st['name']} -- rendered from contract.json at run time)",
            "verify": f"agentcore.tools.recipes_tools:stage_gate_{i + 1}",
        }
        for i, (st, sname) in enumerate(zip(stages, stage_names))
    ]
    head = {
        "name": name,
        "description": description,
        "loop": "standard",
        "action_protocol": "json",
        "models": {"main": {"name": model, "temperature": 0.3, "kwargs": {"max_tokens": 32000}}},
        # THE WORKER'S MACHINE IS ITS DOCKERFILE: the environment starts on the host workspace
        # and, the moment bench/Dockerfile is written, becomes a container built from it
        # (build_on_write), with the workspace carried in and /bench linked to bench/.
        "env": {
            "type": "deferred",
            "workdir": "/workspace",
            "build_on_write": ["Dockerfile"],
            "timeout": 1800,
            "container_lifetime": "28800s",
            "resources": {"mem_limit": "8g"},
        },
        "tools": [
            "write_to_file",
            "str_replace",
            "read_file",
            "linux_terminal",
            "bench_check",
            "bench_deliver",
            "drop_candidate",
        ],
        "tool_modules": ["agentcore.tools.recipes_tools"],
        "prompts": {"system": "../_shared/worker_system.j2", "cycle": "../_shared/worker_cycle.j2"},
        "prerender": "agentcore.tools.recipes_tools:prerender",
        "worklist": {
            "enabled": True,
            "key": key,
            "normalize": normalize,
            "caps": caps,
            "terminal": [stage_names[-1]],
            "publish_to": "final_sample",
            "publish_on": [],
            "derive_calibration_to": "calibration_result",
            "states": [
                {
                    "name": "REGISTERED",
                    "requires": list(key),
                    "instruction": registration_instruction(key, raw_spec),
                    "verify": "agentcore.agent.collector_gates:candidate_is_admissible",
                }
            ]
            + stage_states,
        },
    }
    tail = {
        "instance_schema": "instances.schema:WorkerTask",
        "tool_output": {"strategy": "heuristic", "truncation": "head_tail"},
        "oracle": {"kind": "sentinel", "grade": "agentcore.tools.recipes_tools:finish_gate"},
        "budgets": {
            "meters": [
                {"kind": "cost_usd", "limit": budget_usd, "charge_on": "cycle"},
                {"kind": "wallclock", "limit": wallclock_s, "on_exhaust": "finish_cycle"},
            ]
        },
        "attempts": {"enabled": False},
        "repetition": {"exempt_tools": ["linux_terminal", "container_exec", "bench_check"]},
    }
    comment = (
        "# A WORKER does instance construction and nothing else: it takes candidates one at a\n"
        "# time, builds each as a directory of scripts through the stages the family config\n"
        "# declares, and the system checks every stage in a fresh container. One loop, one prompt.\n"
        "# The run ends when the target is met or the budget is spent; the system writes the run\n"
        "# summary from the record (`bench recipes-summary <run_dir>`).\n"
        "#\n"
        "# GENERATED by `bench worker` from the spec, the canonical and contract.json. The stage\n"
        "# states are rendered from contract.json; regenerate rather than edit.\n\n"
    )
    return comment + yaml.safe_dump({**head, **tail}, sort_keys=False, width=100, default_flow_style=False)


def emit_worker(
    spec_path: str,
    canonical_path: str,
    out_dir: str,
    *,
    model: str,
    sample_size: int = 10,
    name: str | None = None,
    draw: bool = True,
    contract: dict | None = None,
    log=print,
    rounds: int = 6,
    contract_json: str | None = None,
) -> dict:
    """Draw (or take) the family config and the stage sections, then write the worker.

    The accepted config is written to <out_dir>/contract.accepted.json before the stage text
    is drawn, and `contract_json` takes such a file back so that only the stage text is drawn,
    so a failure in the stage text does not discard accepted config rounds."""
    from benchcompiler.contract_prompt import draw_contract
    from benchcompiler.stage_prompt import draw_stage_prompts

    with open(spec_path, encoding="utf-8") as f:
        raw_spec = yaml.safe_load(f) or {}
    with open(canonical_path, encoding="utf-8") as f:
        canonical = json.load(f)
    key, normalize, caps = identity_from_canonical(canonical)
    benchmark_name = str(raw_spec.get("name") or os.path.splitext(os.path.basename(spec_path))[0])
    name = name or f"{benchmark_name}_worker"

    kept_path = os.path.join(out_dir, "contract.accepted.json")
    if contract is None and contract_json:
        with open(contract_json, encoding="utf-8") as f:
            contract = json.load(f)
        log(f"family config taken from {contract_json}")
    if contract is None:
        if not draw:
            raise ValueError("no contract given and --no-draw set")
        log(f"drawing the family config with {model} (identity {key})")
        contract, clog = draw_contract(raw_spec, model, rounds=rounds, identity_fields=key)
        if contract is not None:
            # what the search hands over, kept on the config so the stage text is written
            # knowing which instance.json fields are a given and which are the builder's work
            contract["_identity_fields"] = list(key)
        for ln in clog:
            log("  " + ln)
        if contract is None:
            raise RuntimeError("the family config could not be drawn; see the log above")
        missing = [k for k in key if k not in (contract.get("_instance_fields") or [])]
        if missing:
            raise RuntimeError(f"the drawn config's instance_fields lack the identity fields {missing}")
        os.makedirs(out_dir, exist_ok=True)
        with open(kept_path, "w", encoding="utf-8") as f:
            json.dump(contract, f, indent=1)
        log(f"accepted config kept at {kept_path}")
    contract.setdefault("_identity_fields", list(key))
    if not contract.get("_stage_sections"):
        log("drawing the stage sections")
        sections, slog = draw_stage_prompts(contract, raw_spec, model, rounds=max(4, rounds))
        for ln in slog:
            log("  " + ln)
        if not all(sections):
            raise RuntimeError(
                f"a stage's sections could not be drawn; see the log above. The accepted "
                f"config is kept at {kept_path}: pass it back with --contract-json to "
                f"draw only the stage text"
            )

    os.makedirs(os.path.join(out_dir, "instances"), exist_ok=True)
    with open(spec_path, "rb") as f:
        spec_sha = hashlib.sha256(f.read()).hexdigest()
    record = dict(contract)
    record["_drawn"] = {
        "model": model,
        "spec": os.path.abspath(spec_path),
        "spec_sha256": spec_sha,
        "canonical": os.path.abspath(canonical_path),
        "identity": key,
        "ts": time.time(),
    }
    with open(os.path.join(out_dir, "contract.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    description = _text(raw_spec.get("task")).strip().replace("\n", " ")
    with open(os.path.join(out_dir, "agent.yaml"), "w", encoding="utf-8") as f:
        f.write(render_agent_yaml(name, description, key, normalize, caps, contract, raw_spec, model=model))
    with open(os.path.join(out_dir, "instances", "schema.py"), "w", encoding="utf-8") as f:
        f.write(SCHEMA_PY)
    inst = worker_instance(raw_spec, canonical, contract, key, benchmark_name, sample_size)
    with open(os.path.join(out_dir, "instances", "instances.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(inst) + "\n")
    if os.path.isfile(kept_path):
        os.remove(kept_path)
    stages = recipes.stages_for(contract)
    log(f"worker written to {out_dir}: key {key}, {len(stages)} stages: " + "; ".join(s["name"] for s in stages))
    return {"out": out_dir, "key": key, "stages": [s["name"] for s in stages], "name": name}
