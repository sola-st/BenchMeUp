"""Merge and certify pools of RECIPE instances (phase 4 of the recipes plan).

A recipe instance is a directory (Dockerfile, setup.sh, instance.json, the family's verbs). A
worker keeps each delivered one under `<run>/recipes/<slug>/instance` with the report that
accepted it beside it. This module:

  merge(run_dirs, out, contract)   copies every accepted instance into `out/<slug>/`, dedups on
                                   instance.json, writes instances.json, the contract, a README
                                   and a small `bench` CLI whose verbs ARE the instance's scripts
  certify(pool)                    replays every instance with the same engine construction used
                                   (agentcore.recipes), in a fresh container from its own
                                   Dockerfile, and writes certify.json

No image is shipped. A pool certifies from Dockerfiles alone; the engine's image and setup
snapshot caches are a speed-up, never a requirement.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCAFFOLD = os.path.abspath(os.path.join(HERE, ".."))
if SCAFFOLD not in sys.path:
    sys.path.insert(0, SCAFFOLD)

from agentcore import recipes  # noqa: E402


def _accepted(run_dir: str) -> list[tuple[str, str, dict]]:
    """(slug, instance_dir, report) for every accepted instance under a run directory."""
    out = []
    base = os.path.join(run_dir, "recipes")
    if not os.path.isdir(base):
        return out
    for slug in sorted(os.listdir(base)):
        d = os.path.join(base, slug)
        inst, rep = os.path.join(d, "instance"), os.path.join(d, "report.json")
        if os.path.isdir(inst) and os.path.isfile(rep):
            with open(rep, encoding="utf-8") as f:
                report = json.load(f)
            if report.get("ok"):
                out.append((slug, inst, report))
    return out


def _identity(inst_dir: str) -> dict:
    with open(os.path.join(inst_dir, "instance.json"), encoding="utf-8") as f:
        return json.load(f)


def _summary(report: dict) -> dict:
    rows = {r["name"]: r for r in report.get("rows", [])}
    s = {"ok": report.get("ok"), "elapsed_s": report.get("elapsed_s")}
    if "separates" in rows and rows["separates"].get("evidence"):
        s["ratio"] = rows["separates"]["evidence"].get("ratio")
        s["medians_ms"] = rows["separates"]["evidence"].get("medians")
    if "in_band" in rows and rows["in_band"].get("evidence"):
        s["costliest"] = rows["in_band"]["evidence"].get("costliest")
    return s


def load_any_contract(path: str) -> dict:
    """A family config as `contract.yaml` (hand-written) or `contract.json` (drawn, beside a
    worker's agent.yaml; only the engine's blocks are read)."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
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
        return recipes.validate_contract(core, where=path)
    return recipes.load_contract(path)


def merge(run_dirs: list[str], out: str, contract_path: str, name: str = "benchmark") -> dict:
    contract = load_any_contract(contract_path)
    os.makedirs(out, exist_ok=True)
    seen: dict[str, str] = {}
    instances, dropped = [], []
    for rd in run_dirs:
        run = os.path.basename(os.path.normpath(rd))
        for slug, inst_dir, report in _accepted(rd):
            ident = _identity(inst_dir)
            key = json.dumps(ident, sort_keys=True)
            if key in seen:
                dropped.append({"slug": slug, "run": run, "duplicate_of": seen[key]})
                continue
            iid = slug.rsplit("-", 1)[0] + "-" + hashlib.sha256(key.encode()).hexdigest()[:8]
            dst = os.path.join(out, iid)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(inst_dir, dst)
            seen[key] = iid
            instances.append(
                {
                    "id": iid,
                    "run": run,
                    "fields": ident,
                    "construction": _summary(report),
                    "fingerprint": recipes.fingerprint(dst),
                }
            )
    import yaml as _yaml

    with open(os.path.join(out, "contract.yaml"), "w", encoding="utf-8") as f:
        _yaml.safe_dump(
            {
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
            },
            f,
            sort_keys=False,
        )
    with open(os.path.join(out, "instances.json"), "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=1)
    manifest = {
        "name": name,
        "instances": len(instances),
        "dropped_duplicates": dropped,
        "verbs": {v: spec["args"] for v, spec in contract["verbs"].items()},
        "empty": not instances,
        "written": time.time(),
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    _write_cli(out)
    _write_readme(out, name, contract, instances)
    return manifest


def summary(run_dir: str) -> dict:
    """The run summary, from the record: what a worker delivered, what it dropped and why,
    where every candidate stopped, what it spent. Read from agent_state.json; the model
    reports nothing."""
    with open(os.path.join(run_dir, "agent_state.json"), encoding="utf-8") as f:
        state = json.load(f)
    ws = state.get("working_set") or {}
    wl = (ws.get("worklist") or {}).get("candidates") or {}
    terminal = set((ws.get("worklist") or {}).get("terminal") or [])
    delivered, dropped, by_state = [], [], {}
    for key, c in wl.items():
        st = str(c.get("state") or "?")
        by_state[st] = by_state.get(st, 0) + 1
        if st in terminal:
            delivered.append({"key": key})
        elif st in ("REJECTED", "BLOCKED"):
            hist = [h for h in (c.get("history") or []) if isinstance(h, str)]
            reason = next((h.split(":", 1)[1].strip() for h in reversed(hist) if h.startswith(f"{st}:")), "")
            dropped.append({"key": key, "state": st, "reason": reason[:300]})
    budget = state.get("budget") or {}
    res_path = os.path.join(run_dir, "result.json")
    cost = None
    if os.path.isfile(res_path):
        with open(res_path, encoding="utf-8") as f:
            cost = (json.load(f) or {}).get("cost")
    out = {
        "run": os.path.abspath(run_dir),
        "cycles": state.get("cycle"),
        "finished": state.get("finished"),
        "success": state.get("success"),
        "delivered": delivered,
        "dropped": dropped,
        "in_flight": len(wl) - len(delivered) - len(dropped),
        "by_state": by_state,
        "budget": budget,
        "cost_usd": cost,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def summary_text(s: dict) -> str:
    lines = [
        f"run {s['run']}: {len(s['delivered'])} delivered, {len(s['dropped'])} dropped, "
        f"{s['in_flight']} unfinished, {s.get('cycles')} cycles"
        + (f", ${s['cost_usd']:.2f}" if isinstance(s.get("cost_usd"), (int, float)) else "")
    ]
    for d in s["delivered"]:
        lines.append(f"  delivered  {d['key']}")
    for d in s["dropped"]:
        lines.append(f"  {d['state'].lower():9s}  {d['key']}: {d['reason'] or '(no reason recorded)'}")
    lines.append("  by state: " + ", ".join(f"{k} {v}" for k, v in sorted(s["by_state"].items())))
    return "\n".join(lines)


def certify(pool: str, *, limit: int | None = None, log=None) -> dict:
    contract = recipes.load_contract(os.path.join(pool, "contract.yaml"))
    with open(os.path.join(pool, "instances.json"), encoding="utf-8") as f:
        instances = json.load(f)
    results = []
    for inst in instances[: limit or None]:
        d = os.path.join(pool, inst["id"])
        fp = recipes.fingerprint(d)
        if fp != inst.get("fingerprint"):
            results.append({"id": inst["id"], "ok": False, "detail": "the instance directory changed since merge"})
            continue
        rep = recipes.Replay(d, contract, log=(lambda s, i=inst["id"]: log(f"{i}: {s}")) if log else None).run()
        results.append({"id": inst["id"], "ok": rep.ok, "report": rep.to_dict(), "summary": _summary(rep.to_dict())})
    summary = {
        "certified": sum(1 for r in results if r["ok"]),
        "checked": len(results),
        "failed": [r["id"] for r in results if not r["ok"]],
        "ts": time.time(),
    }
    with open(os.path.join(pool, "certify.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    return summary


_CLI = r'''#!/usr/bin/env python3
"""bench: the pool's command interface. Every verb is the instance's own script.

  bench list                      instances
  bench verbs                     the family's verbs and their arguments
  bench start <id>                build the instance container (Dockerfile) and run setup.sh
  bench <verb> <id> [args...]     run <verb>.sh with the given arguments inside /work
  bench exec <id> -- <cmd>        a shell command inside the instance container
  bench stop <id>                 remove the instance container

Needs only the docker CLI. `bench replay <id>` runs the family's gate rows when the harness
(agentcore.recipes) is importable; a consumer without it reads certify.json instead.
"""
import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    with open(os.path.join(HERE, "instances.json")) as f:
        return {i["id"]: i for i in json.load(f)}


def _manifest():
    with open(os.path.join(HERE, "manifest.json")) as f:
        return json.load(f)


def _run(argv, **kw):
    return subprocess.run(argv, text=True, errors="replace", **kw)


def _cname(iid):
    return "benchpool_" + hashlib.sha256((HERE + iid).encode()).hexdigest()[:12]


def _image(iid):
    d = os.path.join(HERE, iid)
    with open(os.path.join(d, "Dockerfile"), "rb") as f:
        tag = "benchpool/img:" + hashlib.sha256(f.read()).hexdigest()[:16]
    if _run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
        print(f"building {tag} from {iid}/Dockerfile", file=sys.stderr)
        # BENCH_BUILD_ARGS: a JSON object of --build-arg values, so a Dockerfile ARG with a vendor
        # default (a download URL) can be pointed at a local mirror without editing the instance.
        args = []
        try:
            for k, v in (json.loads(os.environ.get("BENCH_BUILD_ARGS") or "{}") or {}).items():
                args += ["--build-arg", f"{k}={v}"]
        except ValueError:
            pass
        r = _run(["docker", "build", "-q", "-t", tag, *args, "-f", os.path.join(d, "Dockerfile"), d])
        if r.returncode != 0:
            sys.exit(r.returncode)
    return tag


def _running(cname):
    r = _run(["docker", "inspect", "-f", "{{.State.Running}}", cname], capture_output=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def cmd_start(iid):
    cname = _cname(iid)
    if _running(cname):
        return cname
    _run(["docker", "rm", "-f", cname], capture_output=True)
    tag = _image(iid)
    r = _run(["docker", "run", "-d", "--name", cname, "-w", "/work", tag, "sh", "-c", "mkdir -p /work && sleep 86400"],
             capture_output=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); sys.exit(1)
    _run(["docker", "cp", os.path.join(HERE, iid) + "/.", f"{cname}:/bench"])
    _run(["docker", "exec", cname, "sh", "-c", "chmod +x /bench/*.sh"])
    r = _run(["docker", "exec", "-w", "/bench", cname, "sh", "/bench/setup.sh"])
    if r.returncode != 0:
        print(f"setup.sh exited {r.returncode}", file=sys.stderr); sys.exit(r.returncode)
    return cname


def cmd_verb(verb, iid, args):
    verbs = _manifest()["verbs"]
    if verb not in verbs:
        print(f"no verb {verb!r}; verbs: {list(verbs)}", file=sys.stderr); sys.exit(2)
    if len(args) != len(verbs[verb]):
        print(f"{verb} takes {verbs[verb]}", file=sys.stderr); sys.exit(2)
    cname = cmd_start(iid)
    r = _run(["docker", "exec", "-w", "/work", cname, "sh", f"/bench/{verb}.sh", *args])
    sys.exit(r.returncode)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    cmd, rest = argv[0], argv[1:]
    inst = _load()
    if cmd == "list":
        for iid, i in inst.items():
            print(iid, json.dumps(i["fields"]))
        return 0
    if cmd == "verbs":
        for v, a in _manifest()["verbs"].items():
            print(f"{v} {' '.join('<'+x+'>' for x in a)}")
        return 0
    if not rest or rest[0] not in inst:
        print(f"unknown instance; bench list shows {len(inst)}", file=sys.stderr); return 2
    iid = rest[0]
    if cmd == "start":
        print(cmd_start(iid)); return 0
    if cmd == "stop":
        _run(["docker", "rm", "-f", _cname(iid)], capture_output=True); return 0
    if cmd == "exec":
        cname = cmd_start(iid)
        tail = rest[1:]
        if tail and tail[0] == "--":
            tail = tail[1:]
        return _run(["docker", "exec", "-w", "/work", cname, "sh", "-c", " ".join(tail)]).returncode
    if cmd == "replay":
        try:
            sys.path.insert(0, os.environ.get("BENCH_SCAFFOLD", ""))
            from agentcore import recipes
        except Exception:
            print("agentcore.recipes is not importable here; read certify.json", file=sys.stderr); return 2
        c = recipes.load_contract(os.path.join(HERE, "contract.yaml"))
        rep = recipes.Replay(os.path.join(HERE, iid), c, log=lambda s: print(s, file=sys.stderr)).run()
        print(rep.text()); return 0 if rep.ok else 1
    cmd_verb(cmd, iid, rest[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
'''


def _write_cli(out: str) -> None:
    path = os.path.join(out, "bench")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_CLI)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_readme(out: str, name: str, contract: dict, instances: list[dict]) -> None:
    verbs = "\n".join(
        f"  ./bench {v} <id> {' '.join('<' + a + '>' for a in s['args'])}" for v, s in contract["verbs"].items()
    )
    rows = "\n".join(f"  {i['id']}  {json.dumps(i['fields'])}" for i in instances) or "  (none)"
    text = f"""# {name}

{len(instances)} instance(s). Each is a directory of scripts that rebuilds itself from nothing:
`Dockerfile` (toolchain), `setup.sh` (clone at the commit), `instance.json` (identity) and the
family's verbs. The verbs are the interface; `./bench` only runs them inside the instance's
container. No image ships with this pool: `./bench start <id>` builds it from the Dockerfile.

Verbs:
{verbs}

Variants (argument sets the harness used):
{json.dumps(contract["variants"], indent=2)}

Oracle: {json.dumps(contract["oracle"])}
Measure: {json.dumps(contract.get("measure"))}

Instances:
{rows}

`certify.json`, when present, is the harness's replay of every instance from its own Dockerfile
in a fresh container: the ten gate rows and their evidence. `./bench replay <id>` re-runs them
where the harness is importable.
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)
