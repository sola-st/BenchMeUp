"""The worker's tools: `bench_check`, `bench_deliver`, `drop_candidate`, and the hand-off.

A worker does instance construction and nothing else. The system hands it one candidate at
a time from the pool in its instance (`candidates`); the worker writes the instance files
into `bench/` in its workspace, checks them stage by stage, and either delivers the candidate
or drops it with a reason; the system then hands the next one. The run ends when the pool is
used up, the target is met, or the budget is spent.

  bench_check      exports bench/ from the worker's machine and replays it in a fresh
                   container built from the worker's own Dockerfile (agentcore.recipes)
                   through the current stage's checks, and shows the report. The worker's
                   machine itself is built from that Dockerfile the moment it is written
                   (env.build_on_write), so it works in the environment it ships.
  bench_deliver    closes the stage. If bench/ is unchanged since the last passing check of
                   this stage, nothing is re-run: the stage gate accepts the cached report.
  drop_candidate   ends the candidate with the reason, and hands the next one.

Nothing the worker typed is read by a check; only what its files do from a blank machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time

from agentcore import recipes
from agentcore.tools.context import ToolContext
from agentcore.tools.registry import tool
from agentcore.tools.result import ToolResult

CHECKS_PER_STAGE = int(os.environ.get("AGENTCORE_RECIPE_CHECKS", "8") or 8)
CONTRACT_FILE = "contract.yaml"
SANDBOX_LIFETIME = os.environ.get("AGENTCORE_SANDBOX_LIFETIME", "8h")
_SCRATCH = "_recipe_checks"
_PASSES = "_recipe_passes"
_POOL = "_recipe_pool"


# ── plumbing ────────────────────────────────────────────────────────────────────────────


def _agent_scratch(ctx) -> dict:
    return getattr(getattr(ctx, "agent", None), "_scratch", None) or {}


def _agent_dir(ctx) -> str:
    return str(_agent_scratch(ctx).get("agent_dir") or os.getcwd())


def _out_dir(ctx) -> str:
    return str(_agent_scratch(ctx).get("out_dir") or os.getcwd())


def bench_dir(ctx) -> str:
    """The instance directory the worker writes, `bench/` of its workspace, as a host path.

    Before the worker's Dockerfile is written, the workspace is a host directory. After it,
    the worker's machine is the container built from that Dockerfile and its files live
    there; `bench/` is exported from it into a temporary host directory for the checks, so
    what is checked is exactly what the worker's machine holds."""
    sc = _agent_scratch(ctx)
    if sc.get("bench_dir"):
        return str(sc["bench_dir"])
    env = getattr(ctx, "env", None)
    container = getattr(env, "_container", None)
    if container is not None:
        import io
        import tarfile
        import tempfile

        base = getattr(getattr(env, "spec", None), "workdir", None) or "/workspace"
        tmp = tempfile.mkdtemp(prefix="bench_export_")
        try:
            stream, _ = container.get_archive(f"{base}/bench")
            buf = io.BytesIO()
            for chunk in stream:
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf, mode="r|*") as tar:
                tar.extractall(tmp, filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(tmp)
        except Exception:  # noqa: BLE001 - no bench/ on the machine yet: the caller reports it
            pass
        return os.path.join(tmp, "bench")
    ws = getattr(env, "workspace", None) or sc.get("workspace") or os.getcwd()
    return os.path.join(str(ws), "bench")


def archive_bench(ctx, slug: str) -> None:
    """Set the finished candidate's bench/ aside on the worker's machine (bench_done/<slug>/),
    so the next candidate starts from an empty bench/ and cannot inherit its files."""
    env = getattr(ctx, "env", None)
    base = getattr(getattr(env, "spec", None), "workdir", None) or "/workspace"
    container = getattr(env, "_container", None)
    if container is not None:
        from agentcore.environments import docker_helpers as dh

        dh.exec_in_container(
            container,
            f"mkdir -p '{base}/bench_done' && "
            f"([ -d '{base}/bench' ] && mv '{base}/bench' '{base}/bench_done/{slug}' || true) && "
            f"mkdir -p '{base}/bench'",
        )
        return
    ws = getattr(env, "workspace", None) or _agent_scratch(ctx).get("workspace")
    if ws and os.path.isdir(os.path.join(ws, "bench")):
        os.makedirs(os.path.join(ws, "bench_done"), exist_ok=True)
        shutil.move(os.path.join(ws, "bench"), os.path.join(ws, "bench_done", slug))


def load_family_contract(agent_dir: str) -> dict:
    """The family config beside the worker's agent.yaml: `contract.json` (drawn, with the
    model's stage plan and stage sections) when present, else the hand-written `contract.yaml`."""
    drawn = os.path.join(agent_dir, "contract.json")
    if os.path.isfile(drawn):
        with open(drawn, encoding="utf-8") as f:
            raw = json.load(f)
        core = {
            k: raw.get(k)
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
        # The field list goes through validation with the core: a `select` that names a field
        # is resolved there (a select without braces would otherwise stay literal if the
        # list were attached only after validation).
        core["instance_fields"] = raw.get("_instance_fields") or raw.get("instance_fields")
        c = recipes.validate_contract(core, where=drawn)
        for k in ("_does", "_instance_fields", "_stages", "_stage_sections", "_drawn"):
            if raw.get(k):
                c[k] = raw[k]
        return c
    return recipes.load_contract(os.path.join(agent_dir, CONTRACT_FILE))


def contract_of(ctx) -> dict:
    cache = ctx.scratch.setdefault("_recipe_contract", {})
    path = _agent_dir(ctx)
    if cache.get("path") != path or "contract" not in cache:
        cache["contract"] = load_family_contract(path)
        cache["path"] = path
    return cache["contract"]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:120] or "candidate"


def _key_fields(ctx) -> list[str]:
    spec = getattr(getattr(ctx, "agent", None), "spec", None)
    wl = getattr(spec, "worklist", None)
    return list(getattr(wl, "key", None) or [])


def _key_slug(ctx, fields: dict) -> str:
    keys = _key_fields(ctx)
    parts = [str(fields.get(k) or "") for k in keys] or [json.dumps(fields, sort_keys=True)]
    return _slug("|".join(parts)) + "-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def _persist(ctx, slug: str, inst_dir: str | None, report: recipes.Report, kind: str) -> str:
    """Keep the report (and, on submit, the instance directory) under the run directory."""
    base = os.path.join(_out_dir(ctx), "recipes", slug)
    os.makedirs(base, exist_ok=True)
    n = len([f for f in os.listdir(base) if f.startswith(kind)]) + 1
    path = os.path.join(base, f"{kind}_{n:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**report.to_dict(), "ts": time.time()}, f, indent=1)
    if inst_dir and os.path.isdir(inst_dir):
        # bench/ as it stood at this check, on the host: when the worker's machine dies and
        # is recreated from its image (files gone), the prompt hook restores it from here.
        last = os.path.join(base, "bench_last")
        shutil.rmtree(last, ignore_errors=True)
        shutil.copytree(inst_dir, last)
    if inst_dir and report.ok and kind == "submit":
        dst = os.path.join(base, "instance")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(inst_dir, dst)
        with open(os.path.join(base, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=1)
    return base


def replay_dir(ctx, *, log=None, stage: int | None = None) -> tuple[recipes.Report, str]:
    """Replay bench/ of the workspace; through `stage` (0-based) when the worker has stages."""
    inst = bench_dir(ctx)
    if not os.path.isdir(inst):
        raise RuntimeError(
            "there is no bench/ directory on your machine yet: write bench/Dockerfile, "
            "bench/setup.sh and bench/instance.json with write_to_file, then check"
        )
    c = contract_of(ctx)
    until, files = None, None
    if stage is not None:
        files, until = recipes.stage_scope(c, stage)
    n = len(recipes.stages_for(c))
    label = "bench/ of your workspace, replayed in a fresh container" + (
        f" -- stage {stage + 1} of {n}" if stage is not None else ""
    )
    rep = recipes.Replay(inst, c, log=log, label=label, until=until, required_files=files).run()
    return rep, inst


STAGE_PREFIX = "STAGE_"


def stage_state_names(contract: dict) -> list[str]:
    """One worklist state per stage: STAGE_1_INSTALL_COMPILER_CLONE, STAGE_2_WRITE_BUILD_SCRIPT, ..."""
    out = []
    for i, st in enumerate(recipes.stages_for(contract)):
        slug = re.sub(r"[^A-Z0-9]+", "_", st["name"].upper()).strip("_")
        words = [w for w in slug.split("_") if w not in ("THE", "A", "AN", "AND", "OF", "INTO", "ON", "IT", "TO")]
        short = ""
        for w in words:
            if len(short) + len(w) + 1 > 28:
                break
            short = f"{short}_{w}" if short else w
        out.append(f"{STAGE_PREFIX}{i + 1}_{short or slug[:28].rstrip('_')}")
    return out


def _target_stage(ctx, wl, cur) -> int | None:
    """The stage the candidate is working on: the first STAGE_ state after its current state.
    None when the worker has no stage states (single DONE gate) or the candidate is past
    the last stage."""
    names = [st.name for st in wl.spec.states]
    stage_states = [n for n in names if n.startswith(STAGE_PREFIX)]
    if not stage_states:
        return None
    state = cur[1].get("state")
    later = [n for n in stage_states if state not in names or names.index(n) > names.index(state)]
    if not later:
        return None
    return stage_states.index(later[0])


def sync_stage_instructions(ctx) -> None:
    """The stage states' instructions come from the LIVE config, rendered now, never from the
    text baked into agent.yaml at generation time, so the prompt never shows a command the
    check no longer runs."""
    spec = getattr(getattr(ctx, "agent", None), "spec", None)
    wl = getattr(spec, "worklist", None)
    if wl is None or ctx.scratch.get("_recipe_instructions_synced"):
        return
    try:
        apply_stage_instructions(wl, contract_of(ctx))
        ctx.scratch["_recipe_instructions_synced"] = True
    except Exception:  # noqa: BLE001 - the baked text stays when the config cannot be read
        pass


def apply_stage_instructions(worklist_spec, contract: dict) -> None:
    """Set every stage state's instruction from the config, rendered now."""
    stages = recipes.stages_for(contract)
    by_name = {n: st for n, st in zip(stage_state_names(contract), stages)}
    for st in worklist_spec.states:
        if st.name in by_name:
            st.instruction = recipes.stage_instruction(by_name[st.name], contract)


def _wl_and_current(ctx):
    from agentcore.tools.worklist_tools import _wl

    sync_stage_instructions(ctx)
    wl = _wl(ctx)
    if wl is None:
        return None, None, ToolResult.fail("This worker has no candidate list configured.")
    cur = wl.current()
    if cur is None:
        key = hand_next(ctx, wl)
        cur = wl.current() if key else None
    if cur is None:
        return wl, None, ToolResult.fail("No candidate is in hand: the pool is used up.")
    return wl, cur, None


# ── the hand-off: the system gives the worker its next candidate ────────────────────────


def _pool(ctx) -> list[dict]:
    inst = getattr(ctx, "instance", None) or {}
    return [c for c in (inst.get("candidates") or []) if isinstance(c, dict)]


def hand_next(ctx, wl) -> str | None:
    """Register the next candidate of the pool that is not yet in the list. Returns its key,
    or None when the pool is used up. A candidate the list refuses (a duplicate, an excluded
    project) is recorded and skipped."""
    refused = ctx.scratch.setdefault(_POOL, {})
    for entry in _pool(ctx):
        k = wl.key_of(entry)
        if not k or k in wl.store.get("candidates", {}) or k in refused:
            continue
        ok, detail = wl.add(entry)
        if ok:
            return k
        refused[k] = " ".join(str(x) for x in (detail if isinstance(detail, (tuple, list)) else [detail]))
    return None


def pool_exhausted(ctx, wl) -> bool:
    refused = ctx.scratch.get(_POOL, {}) or {}
    for entry in _pool(ctx):
        k = wl.key_of(entry)
        if k and k not in wl.store.get("candidates", {}) and k not in refused:
            return False
    return wl.current() is None


def _after_candidate(ctx, wl, head: str) -> ToolResult:
    """What follows a delivery or a drop: the target check, then the next candidate, or the
    end of the run when the pool is used up."""
    inst = getattr(ctx, "instance", None) or {}
    target = int(inst.get("sample_size") or 0)
    delivered = len(wl.delivered())
    if target and delivered >= target:
        return ToolResult(
            display_text=f"{head}\n\nTarget met: {delivered} of {target} delivered. The run ends here.",
            finished=True,
            structured={"success_declared": True, "delivered": delivered},
        )
    nxt = hand_next(ctx, wl)
    if nxt:
        return ToolResult(
            display_text=f"{head}\n\nNext candidate: {nxt}. Its first stage is in the prompt; bench/ on "
            f"your machine is empty again (the previous files are under bench_done/).",
            structured={"next": nxt, "delivered": delivered},
        )
    return ToolResult(
        display_text=f"{head}\n\nNo candidate left in the pool ({delivered} delivered"
        + (f" of {target} wanted" if target else "")
        + "). The run ends here.",
        finished=True,
        structured={"success_declared": delivered > 0, "delivered": delivered},
    )


# ── the tools ───────────────────────────────────────────────────────────────────────────


def _passes(ctx) -> dict:
    return ctx.scratch.setdefault(_PASSES, {})


def _remember_pass(ctx, slug: str, stage: int | None, fp: str, rep: recipes.Report) -> None:
    _passes(ctx)[f"{slug}@{stage if stage is not None else 'all'}"] = {"fp": fp, "report": rep}


def _cached_pass(ctx, slug: str, stage: int | None, fp: str) -> recipes.Report | None:
    hit = _passes(ctx).get(f"{slug}@{stage if stage is not None else 'all'}")
    if hit and hit.get("fp") == fp and getattr(hit.get("report"), "ok", False):
        return hit["report"]
    return None


@tool("bench_check", idempotent_ok=True)
def bench_check(ctx: ToolContext) -> ToolResult:
    """Check the current stage of the candidate you are working on: the files in your bench/
    directory are replayed in a fresh container built from your Dockerfile, and every check of
    the stage is reported.

    A failing check shows its command, its exit code, its error output and, where the check has
    a bound, the measured numbers against it: fix the file it names and check again. When every
    check passes, the reply lists any observations (SEEN) for you to judge: fix what needs
    fixing, or say in instance.json why it is fine, then call bench_deliver. A HARNESS FAULT
    means the checking container failed, not your files: run bench_check again. You can check
    as often as you need."""
    wl, cur, err = _wl_and_current(ctx)
    if err:
        return err
    stage = _target_stage(ctx, wl, cur)
    slug = _key_slug(ctx, cur[1].get("fields") or {})
    counts = ctx.scratch.setdefault(_SCRATCH, {})
    ckey = f"{slug}@{stage if stage is not None else 'all'}"
    counts[ckey] = int(counts.get(ckey, 0)) + 1  # counted for the record; no cap
    try:
        rep, inst = replay_dir(ctx, stage=stage)
    except RuntimeError as e:
        return ToolResult.fail(str(e))
    except recipes.ContractError as e:
        return ToolResult.fail(f"HARNESS/SPEC FAULT: the family config cannot be read: {e}")
    _persist(ctx, slug, inst, rep, "check")
    fp = recipes.fingerprint(inst)
    if rep.ok:
        _remember_pass(ctx, slug, stage, fp, rep)
    n = len(recipes.stages_for(contract_of(ctx)))
    last = stage is None or stage == n - 1
    # The run's budget stop leaves one cycle to deliver a clean final check (loop.should_finish).
    if rep.ok and not rep.observations and last:
        ctx.scratch["deliverable"] = {"slug": slug, "fp": fp}
    else:
        ctx.scratch.pop("deliverable", None)
    if rep.ok and not rep.observations:
        head = (
            "THE SCRIPTS RAN AND EVERY OBSERVATION IS CLEAN -- bench_deliver delivers this candidate."
            if last
            else f"STAGE {stage + 1} of {n}: THE SCRIPTS RAN AND EVERY OBSERVATION IS CLEAN -- bench_deliver "
            f"closes it and gives you the next stage."
        )
    elif rep.ok:
        seen = ", ".join(f"{recipes.row_label(r.name, contract_of(ctx))} [{r.name}]" for r in rep.observations)
        head = (
            f"THE SCRIPTS RAN. {len(rep.observations)} observation(s) to judge (SEEN below): {seen}. "
            f"Fix what needs fixing and check again, or bench_deliver when you are satisfied and have "
            f"said why in instance.json."
        )
    elif _harness_origin(rep):
        # The replay container died under the harness (e.g. exit 137 mid-row).
        # Not the candidate's failure: it does not
        # count against the cap, and the worker is told to simply check again.
        counts[ckey] -= 1
        head = (
            f"HARNESS FAULT at {recipes.row_label(rep.first_failure.name, contract_of(ctx))} "
            f"[{rep.first_failure.name}]: the container running the checks died under the harness. "
            f"This is not your files' failure and does not count as a check; run bench_check again."
        )
    else:
        hard = next(r for r in rep.rows if r.ok is False and r.hard)
        head = (
            f"THE SCRIPTS DID NOT RUN: {recipes.row_label(hard.name, contract_of(ctx))} "
            f"[{hard.name}]. Fix the file the report names and check again."
        )
    return ToolResult(display_text=f"{head}\n{rep.text()}", structured=rep.to_dict())


@tool("bench_deliver", idempotent_ok=True)
def bench_deliver(ctx: ToolContext) -> ToolResult:
    """Close the current stage of the candidate you are working on.

    If bench/ is unchanged since your last passing bench_check of this stage, nothing runs
    again; otherwise the checks run now and must pass. Closing a stage gives you the
    instructions for the next one; closing the last stage delivers the instance, and the system
    gives you the next candidate. When it is not accepted, the reply says why."""
    wl, cur, err = _wl_and_current(ctx)
    if err:
        return err
    ctx.scratch.pop("deliverable", None)
    key = cur[0]
    stage = _target_stage(ctx, wl, cur)
    slug = _key_slug(ctx, cur[1].get("fields") or {})
    inst = bench_dir(ctx)
    if not os.path.isdir(inst):
        return ToolResult.fail(f"there is no bench/ directory in your workspace yet ({inst})")
    fp = recipes.fingerprint(inst)
    if _cached_pass(ctx, slug, stage, fp) is None:
        try:
            rep, inst = replay_dir(ctx, stage=stage)
        except RuntimeError as e:
            return ToolResult.fail(str(e))
        except recipes.ContractError as e:
            return ToolResult.fail(f"HARNESS/SPEC FAULT: the family config cannot be read: {e}")
        _persist(ctx, slug, inst, rep, "check")
        if not rep.ok:
            return ToolResult(
                display_text=f"NOT ACCEPTED for {key}: the scripts did not run on the blank machine.\n{rep.text()}",
                structured={"candidate": key, "advanced": False, **rep.to_dict()},
            )
        _remember_pass(ctx, slug, stage, fp, rep)
    token = "submitted" if stage is None else f"stage_{stage + 1}_delivered"
    before = cur[1].get("state")
    c = wl.update(key, {token: fp})
    key = next((k for k, cand in wl.store["candidates"].items() if cand is c), key)
    state = (c or {}).get("state")
    if state in (wl.spec.terminal or []):
        archive_bench(ctx, slug)
        return _after_candidate(ctx, wl, f"DONE: {key} passed every check and is delivered.")
    if stage is not None and state != before:
        con = contract_of(ctx)
        nxt = _target_stage(ctx, wl, (key, c))
        n = len(recipes.stages_for(con))
        text = recipes.stage_instruction(recipes.stages_for(con)[nxt], con) if nxt is not None else ""
        return ToolResult(
            display_text=f"STAGE {stage + 1} of {n} CLOSED for {key}.\n\n{text}",
            structured={"candidate": key, "state": state, "advanced": True},
        )
    refused = wl.validation_error(key) or "(no verdict recorded)"
    return ToolResult(
        display_text=f"NOT ACCEPTED for {key}.\n{refused}",
        structured={"candidate": key, "state": state, "advanced": False, "refused": refused},
    )


@tool(
    "drop_candidate",
    params={
        "reason": "why this candidate cannot be made into an instance, in one or two sentences that quote "
        "what you saw (the check that failed and what it said, or what the program turned out "
        "to be)"
    },
    aliases={"reason": ["why", "explanation"]},
)
def drop_candidate(ctx: ToolContext, reason: str) -> ToolResult:
    """Give up on the candidate you are working on, with the reason.

    The reason is required: which check could not be passed and what it said, or what the
    program turned out to be. It is kept in the run log, your bench/ files are archived, and
    the system gives you the next candidate."""
    from agentcore.agent.worklist import REJECTED

    wl, cur, err = _wl_and_current(ctx)
    if err:
        return err
    if not str(reason or "").strip():
        return ToolResult.fail(
            "Give the reason: which check could not be passed and what it said, or what the program turned out to be."
        )
    ctx.scratch.pop("deliverable", None)
    key = cur[0]
    # A RETYPED VALUE PROVES NOTHING: a hash with swapped characters is a different hash,
    # and "absent" proved against it says nothing about the candidate.
    from agentcore.tools.misquote import misquoted_value

    retyped = misquoted_value(str(reason), cur[1].get("fields") or {})
    if retyped:
        return ToolResult.fail(f"NOT DROPPED. {retyped}")
    wl.terminate(key, REJECTED, str(reason).strip())
    archive_bench(ctx, _key_slug(ctx, cur[1].get("fields") or {}))
    return _after_candidate(ctx, wl, f"{key} dropped: {str(reason).strip()[:300]}")


# ── the gates: one per stage, the last one keeps the instance ───────────────────────────


def _harness_origin(rep: recipes.Report) -> bool:
    """Any row reports the harness's own failure (a container the exec layer could not reach),
    not a property of the candidate's files. Read over every row, since the failing row is
    not necessarily the first."""
    return recipes.harness_fault(rep)


def _gate_report(ctx, fields: dict, index: int | None) -> tuple[recipes.Report | None, str, str]:
    """(report, instance dir, error). Uses the cached passing report when bench/ is unchanged
    since it, else replays now."""
    slug = _key_slug(ctx, fields)
    inst = bench_dir(ctx)
    token = "submitted" if index is None else f"stage_{index + 1}_delivered"
    fp = str(fields.get(token) or "")
    if os.path.isdir(inst) and fp and fp == recipes.fingerprint(inst):
        cached = _cached_pass(ctx, slug, index, fp)
        if cached is not None:
            return cached, inst, ""
    try:
        rep, inst = replay_dir(ctx, stage=index)
    except RuntimeError as e:
        return None, inst, str(e)
    except recipes.ContractError as e:
        return None, inst, f"HARNESS/SPEC FAULT: the family config cannot be read: {e}"
    return rep, inst, ""


def submit_gate(fields: dict, ctx) -> tuple[bool, str]:
    """`verify:` for a worker with a single terminal state (no stage states)."""
    rep, inst, err = _gate_report(ctx, fields, None)
    if rep is None:
        return False, err
    slug = _key_slug(ctx, fields)
    where = _persist(ctx, slug, inst, rep, "submit")
    if rep.ok:
        return True, f"every check passed; instance kept at {where}\n{rep.text()}"
    if _harness_origin(rep):
        return False, f"HARNESS FAULT at row {rep.first_failure.name}: {rep.text()}"
    return False, rep.text()


def _stage_gate(index: int):
    def gate(fields: dict, ctx) -> tuple[bool, str]:
        c = contract_of(ctx)
        n = len(recipes.stages_for(c))
        rep, inst, err = _gate_report(ctx, fields, index)
        if rep is None:
            return False, err
        slug = _key_slug(ctx, fields)
        kind = "submit" if index == n - 1 else f"stage{index + 1}"
        where = _persist(ctx, slug, inst, rep, kind)
        if not rep.ok and _harness_origin(rep):
            return False, f"HARNESS FAULT at row {rep.first_failure.name}: {rep.text()}"
        if rep.ok:
            mine = set(recipes.stages_for(c)[index]["rows"])
            fields[f"_stage_{index + 1}_summary"] = "\n".join(
                f"{recipes.row_label(r.name, c)}: {'' if r.ok else 'SEEN, accepted by the builder: '}"
                f"{r.detail.splitlines()[0][:110]}"
                for r in rep.rows
                if r.ok is not None and r.name in mine
            )
            return True, (
                f"stage {index + 1} of {n} passed"
                + (f"; instance kept at {where}" if index == n - 1 else "")
                + "\n"
                + rep.text()
            )
        return False, rep.text()

    gate.__name__ = f"stage_gate_{index + 1}"
    return gate


def stage_progress_text(fields: dict, contract: dict) -> str:
    """The 'what earlier stages established' block for a candidate: each closed stage's name
    and its checks' one-liners, from the summaries the gates kept."""
    stages = recipes.stages_for(contract)
    out = []
    for i, st in enumerate(stages):
        summary = fields.get(f"_stage_{i + 1}_summary")
        if not summary:
            break
        out.append(f"[stage {i + 1}: {st['name']}] done")
        out.extend("  " + ln for ln in str(summary).splitlines())
    return "\n".join(out)


def restore_bench_from(ctx, bench_last: str) -> str | None:
    """If the worker's machine exists but its bench/ is gone (the container died and was
    recreated from its image), put back the copy kept at the last check and say so.
    Returns the note for the prompt, or None when nothing was needed."""
    env = getattr(ctx, "env", None)
    container = getattr(env, "_container", None)
    if container is None or not os.path.isdir(bench_last):
        return None
    from agentcore.environments import docker_helpers as dh

    base = getattr(getattr(env, "spec", None), "workdir", None) or "/workspace"
    code, out = dh.exec_in_container(container, f"ls -A '{base}/bench' 2>/dev/null | wc -l")
    try:
        n = int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        n = -1
    if n != 0 and code == 0:
        return None
    container.put_archive(base, recipes._tar_of(bench_last, "bench"))
    dh.exec_in_container(container, f"ln -sfn '{base}/bench' /bench")
    files = sorted(os.listdir(bench_last))
    return (
        f"Your machine died and was recreated from its image. bench/ was restored from your last "
        f"check ({', '.join(files)}); everything else on the machine (/work, /tmp, installed "
        f"packages) is gone: run setup.sh again before building or running anything."
    )


def prerender(ctx) -> dict:
    """Before each prompt: hand the worker its next candidate when none is in hand, and
    render what the earlier stages of the candidate in hand established."""
    out: dict = {}
    try:
        # `_wl_and_current` returns three values (worklist, current, error).
        wl0, cur0, _err = _wl_and_current(ctx)
        if cur0:
            slug = _key_slug(ctx, cur0[1].get("fields") or {})
            note = restore_bench_from(ctx, os.path.join(_out_dir(ctx), "recipes", slug, "bench_last"))
            if note:
                out["machine_note"] = note
    except Exception:  # noqa: BLE001 - a restore must never break a cycle
        pass
    try:
        from agentcore.tools.worklist_tools import _wl

        synced_before = bool(ctx.scratch.get("_recipe_instructions_synced"))
        sync_stage_instructions(ctx)
        wl = _wl(ctx)
        if wl is not None and (
            (wl.current() is None and hand_next(ctx, wl)) or not synced_before or wl.current() is not None
        ):
            # The loop rendered the candidate block BEFORE this hook ran, from an empty list;
            # render it again so this prompt already shows the candidate just handed over.
            from agentcore.agent.worklist_render import render_text, render_worklist

            agent = ctx.agent
            wv = render_worklist(
                agent.spec, agent.working_set, target=(getattr(agent, "instance", None) or {}).get("sample_size")
            )
            names = list(agent.stages.tools_allow(agent.spec.tools) or agent.registry.names())
            out.update(wv)
            out["worklist_text"] = render_text(wv, {"tool_names": names})
        cur = wl.current() if wl else None
        if cur:
            out["recipes_progress"] = stage_progress_text(cur[1].get("fields") or {}, contract_of(ctx))
    except Exception:  # noqa: BLE001 - a prompt variable must never break a cycle
        pass
    return out


def finish_gate(ctx):
    """The run may end when the target is met, the pool is used up, or the budget is spent.
    The worker reports nothing: the system writes the summary from the record
    (benchcompiler.recipes_pool.summary)."""
    from agentcore.agent.quota import quota_not_met
    from agentcore.oracle.base import OracleResult
    from agentcore.tools.worklist_tools import _wl

    wl = _wl(ctx)
    if wl is not None and pool_exhausted(ctx, wl):
        n = len(wl.delivered())
        return OracleResult(verdict="ACCEPT", success=n > 0, reason=f"pool used up; {n} delivered")
    short = quota_not_met(ctx)
    if short:
        return OracleResult(verdict="REJECT_CONTINUE", success=False, reason=short, message=short)
    return OracleResult(verdict="ACCEPT", success=True, reason="target met or budget spent")


def __getattr__(name: str):
    if name.startswith("stage_gate_"):
        try:
            return _stage_gate(int(name[len("stage_gate_") :]) - 1)
        except ValueError:
            pass
    raise AttributeError(name)


def render_stage_states(contract: dict, indent: str = "    ") -> str:
    """The worklist `states:` entries for a worker, as YAML text, one per stage: each
    requires the sandbox-free stage token, carries the stage's instruction, and is verified
    by its stage gate. The last one is the terminal state."""
    import yaml

    stages = recipes.stages_for(contract)
    names = stage_state_names(contract)
    entries = []
    for i, (st, name) in enumerate(zip(stages, names)):
        entries.append(
            {
                "name": name,
                "requires": [f"stage_{i + 1}_delivered"],
                "instruction": recipes.stage_instruction(st, contract),
                "verify": f"agentcore.tools.recipes_tools:stage_gate_{i + 1}",
            }
        )
    text = yaml.safe_dump(entries, sort_keys=False, width=100, default_flow_style=False)
    return "\n".join((indent + ln) if ln.strip() else ln for ln in text.splitlines())
