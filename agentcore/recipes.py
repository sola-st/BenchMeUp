"""Replay an instance's own scripts in a fresh container and derive every value from the runs.

This is the replay engine. An instance is a
directory:

    Dockerfile        toolchain and dependencies; builds the instance container
    setup.sh          run once inside it: clone at the commit, prepare /work
    <verb>.sh ...     the family's verbs, declared in the family CONTRACT; stdout is the observable
    instance.json     identity fields and per-instance constants the contract may name
    notes.md          prose; never read here

The harness never reads a value a model typed. It builds the image, runs setup once and
snapshots it, starts a fresh container from the snapshot for every replay, copies the instance
directory to /bench, runs the verbs with arguments bound from the contract, and takes exit
codes, stdout and wall-clock from those runs. The rows in ROWS are the whole gate
set; a failing row reports the command, cwd, exit, stderr tail, the derived values and the
bound. Rows run in cost order and the replay stops at the first failure.

Two argument names are reserved and bound by the harness: `dir` (a variant's directory under
/work/variants) and `input` (a path the contract's inputs verb produced). Everything else
comes from a variant's setup step or the solver's probe value.

Nothing here knows a family. The differences between task families are the contract
(see benchcompiler/examples/*.contract.yaml), and this module's procedure is the same for
both.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import tarfile
import time
from dataclasses import dataclass, field
from typing import Callable

from agentcore.wallclock import NOW_MS_SH

ROWS = [
    "replays",
    "identity",
    "variants_build",
    "inputs",
    "stable",
    "input_sensitive",
    "oracle",
    "can_fail",
    "solver_input",
    "in_band",
    "separates",
]
RESERVED_ARGS = ("dir", "input")
ORACLE_KINDS = ("reference-equivalence", "fail-to-pass")
NC_EXPECT = ("observable_changes", "check_fails")
BENCH = "/bench"
WORK = "/work"
VARIANTS = "/work/variants"
INPUTS = "/work/inputs"
OBS = "/tmp/bench_obs"
_SHA = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    """The family contract cannot be read. The message names the field and the fix."""


# ── the contract ────────────────────────────────────────────────────────────────────────


def load_contract(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return validate_contract(raw, where=path)


def validate_contract(c: dict, where: str = "contract") -> dict:
    """Return the contract normalized, or raise ContractError naming the defect and the fix."""

    def bad(msg):
        raise ContractError(f"{where}: {msg}")

    if not isinstance(c, dict):
        bad("the contract must be a mapping")
    verbs = c.get("verbs")
    if not isinstance(verbs, dict) or not verbs:
        bad("`verbs` must be a non-empty mapping of verb name -> {args: [...]}")
    norm_verbs = {}
    for name, spec in verbs.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(name)):
            bad(f"verb {name!r}: names are lowercase identifiers (the file is <verb>.sh)")
        args = list((spec or {}).get("args") or [])
        if any(not isinstance(a, str) or not a for a in args):
            bad(f"verb {name!r}: `args` must be a list of names")
        norm_verbs[str(name)] = {"args": args}
    variants = c.get("variants")
    if not isinstance(variants, dict) or not variants:
        bad("`variants` must be a non-empty mapping of variant name -> {setup: [steps]}")
    norm_variants = {}
    for vname, vspec in variants.items():
        steps = list((vspec or {}).get("setup") or [])
        if not steps:
            bad(f"variant {vname!r}: `setup` must list at least one step {{verb, args}}")
        norm_steps = []
        for st in steps:
            if not isinstance(st, dict) or "verb" not in st:
                bad(f"variant {vname!r}: each setup step is a mapping with `verb`")
            v = str(st["verb"])
            if v not in norm_verbs:
                bad(f"variant {vname!r}: setup names verb {v!r}, which `verbs` does not declare")
            sargs = dict(st.get("args") or {})
            for a in sargs:
                if a in RESERVED_ARGS:
                    bad(
                        f"variant {vname!r}: step {v!r} binds reserved argument {a!r}; the "
                        f"harness binds `dir` and `input`"
                    )
                if a not in norm_verbs[v]["args"]:
                    bad(
                        f"variant {vname!r}: step {v!r} binds {a!r}, not an argument of {v!r} ({norm_verbs[v]['args']})"
                    )
                if re.search(r"<\s*solver|\{\s*solver", str(sargs[a]), re.I):
                    bad(
                        f"variant {vname!r}: step {v!r} binds {a!r} to {sargs[a]!r}, a solver "
                        f"placeholder; a variant is built by the harness alone, and the solver's "
                        f"submission enters only through `solver_input`"
                    )
            norm_steps.append({"verb": v, "args": sargs})
        norm_variants[str(vname)] = {"setup": norm_steps}
    obs = c.get("observable")
    # ANY NUMBER OF CHECKS: `observable` is one check {verb, compare,
    # reader, select} or a list of them; a stored config may also carry the list as
    # `checks`. The harness runs every check on every observation and reads them together.
    # `observable` stays the first check for the code that names one; `checks` is the list.
    raw_checks = (
        c["checks"] if isinstance(c.get("checks"), list) and c["checks"] else obs if isinstance(obs, list) else [obs]
    )
    if not raw_checks:
        bad("`observable` must be a check {verb, compare, reader, select} or a non-empty list of them")
    checks = []
    for i, ob in enumerate(raw_checks):
        at = f"observable[{i}]" if len(raw_checks) > 1 else "observable"
        if not isinstance(ob, dict) or str(ob.get("verb") or "") not in norm_verbs:
            bad(f"`{at}.verb` must name a declared verb")
        compare = str(ob.get("compare") or "exact")
        if compare != "exact" and not re.fullmatch(r"numeric:[0-9.eE+-]+", compare):
            bad(f"`{at}.compare` is `exact` or `numeric:<tolerance>`")
        reader = str(ob["reader"]) if ob.get("reader") else None
        if reader not in (None, "junit-xml", "exit-code"):
            bad(
                f"`{at}.reader` is null, `junit-xml` (a framework report on stdout, with "
                f"`select`) or `exit-code` (the verb's exit status: 0 passes, else fails)"
            )
        ch = {
            "verb": str(ob["verb"]),
            "compare": compare,
            "reader": reader,
            "select": (str(ob["select"]) if ob.get("select") else None),
        }
        obs_args = set(norm_verbs[ch["verb"]]["args"])
        if not obs_args <= {"dir", "input"}:
            bad(
                f"`{at}.verb` {ch['verb']!r} takes {sorted(obs_args)}; the harness can "
                f"bind only `dir` and `input` when it runs a check. Give the verb `dir` only "
                f"(and `input` when `measure` provides inputs): the test to run, the file to check, "
                f"the mode, are read from /bench/instance.json or from a file a setup step wrote, "
                f"not passed as arguments. A submission enters through `solver_input`, never through "
                f"a check's arguments"
            )
        if ch["reader"] == "junit-xml" and not ch["select"]:
            bad(
                f"`{at}.reader: junit-xml` needs `{at}.select`: the test the harness reads from the "
                f"report, written as the instance.json field that names it, in braces (for "
                f'example "{{test_name}}"), with that field listed in `instance_fields`'
            )
        checks.append(ch)
    if len({ch["verb"] for ch in checks}) != len(checks):
        bad("every check in `observable` names a different verb")
    norm_obs = checks[0]
    oracle = c.get("oracle")
    if not isinstance(oracle, dict) or str(oracle.get("kind") or "") not in ORACLE_KINDS:
        bad(f"`oracle.kind` must be one of {list(ORACLE_KINDS)}")
    norm_oracle = {"kind": str(oracle["kind"])}
    if norm_oracle["kind"] == "reference-equivalence" and len(norm_variants) < 2:
        bad(
            "a reference-equivalence oracle compares the observable ACROSS variants and needs at "
            "least two; with one side there is nothing to agree with (use fail-to-pass with an "
            "exit-code check when the verdict is a check script's exit status)"
        )
    if norm_oracle["kind"] == "fail-to-pass":
        for k in ("fails_on", "passes_on"):
            if str(oracle.get(k) or "") not in norm_variants:
                bad(f"`oracle.{k}` must name a declared variant for a fail-to-pass oracle")
            norm_oracle[k] = str(oracle[k])
        if not any(ch["reader"] for ch in checks):
            bad(
                "a fail-to-pass oracle needs a check with a `reader`: `exit-code` when the verb is a "
                "check that exits 0/nonzero, or `junit-xml` with `select` when it prints a report"
            )
    nc = c.get("negative_control")
    norm_nc = None
    if nc:
        if not isinstance(nc, dict) or str(nc.get("verb") or "") not in norm_verbs:
            bad("`negative_control.verb` must name a declared verb (or the block is null)")
        if str(nc.get("variant") or "") not in norm_variants:
            bad("`negative_control.variant` must name a declared variant")
        expect = str(nc.get("expect") or "observable_changes")
        if expect not in NC_EXPECT:
            bad(f"`negative_control.expect` is one of {list(NC_EXPECT)}")
        then = nc.get("then")
        if expect == "check_fails" and str(then or "") not in norm_verbs:
            bad("`negative_control.then` must name the check verb when expect is check_fails")
        if then is not None and (not isinstance(then, str) or then not in norm_verbs):
            # A {verb, args} structure here would make the sweep below raise a KeyError on
            # norm_verbs[str(then)]; refuse it with an actionable message instead.
            bad(
                f"`negative_control.then` must be the NAME of a declared verb (a string, one of "
                f"{sorted(norm_verbs)}), not {then!r}; it is only used when expect is "
                f"check_fails, so drop it otherwise"
            )
        for v_ in [str(nc["verb"])] + ([str(then)] if then else []):
            extra = set(norm_verbs[v_]["args"]) - {"dir"}
            if extra:
                bad(
                    f"`negative_control` verb {v_!r} takes {sorted(extra)} beyond `dir`; the "
                    f"harness runs it with the variant directory only"
                )
        norm_nc = {
            "verb": str(nc["verb"]),
            "variant": str(nc["variant"]),
            "expect": expect,
            "then": (str(then) if then else None),
        }
    elif norm_oracle["kind"] != "fail-to-pass":
        bad("`negative_control` is required unless the oracle is fail-to-pass (where the failing variant is the break)")
    si = c.get("solver_input")
    norm_si = None
    if si:
        if not isinstance(si, dict) or str(si.get("verb") or "") not in norm_verbs:
            bad("`solver_input.verb` must name a declared verb")
        arg = str(si.get("arg") or "")
        if arg not in norm_verbs[str(si["verb"])]["args"] or arg in RESERVED_ARGS:
            bad(f"`solver_input.arg` must be a non-reserved argument of {si['verb']!r}")
        pv = list(si.get("probe_values") or [])
        if len(pv) != 2:
            bad("`solver_input.probe_values` must hold exactly two values")
        if str(si.get("variant") or "") not in norm_variants:
            bad("`solver_input.variant` must name a declared variant")
        in_setup = any(st["verb"] == str(si["verb"]) for st in norm_variants[str(si["variant"])]["setup"])
        extra = set(norm_verbs[str(si["verb"])]["args"]) - {"dir", arg}
        if extra and not in_setup:
            bad(
                f"`solver_input.verb` {si['verb']!r} takes {sorted(extra)} beyond `dir` and "
                f"{arg!r}; nothing binds them when the harness applies a probe value"
            )
        norm_si = {
            "verb": str(si["verb"]),
            "arg": arg,
            "variant": str(si["variant"]),
            "probe_values": [str(v) for v in pv],
        }
    m = c.get("measure")
    norm_m = None
    if m:
        if not isinstance(m, dict) or str(m.get("inputs") or "") not in norm_verbs:
            bad("`measure.inputs` must name the verb that writes inputs into `dir`")
        try:
            norm_m = {
                "inputs": str(m["inputs"]),
                "repetitions": int(m.get("repetitions") or 5),
                "floor_ms": float(m["floor_ms"]),
                "ceiling_ms": float(m["ceiling_ms"]),
                "min_ratio": float(m["min_ratio"]),
            }
        except (KeyError, TypeError, ValueError) as e:
            bad(f"`measure` needs numeric floor_ms, ceiling_ms, min_ratio and repetitions ({e})")
        ro = list(m.get("ratio_of") or [])
        if len(ro) != 2 or any(str(v) not in norm_variants for v in ro):
            bad("`measure.ratio_of` must name two declared variants [slow, fast]")
        norm_m["ratio_of"] = [str(v) for v in ro]
        if norm_m["repetitions"] < 2:
            bad("`measure.repetitions` must be at least 2")
        if set(norm_verbs[norm_m["inputs"]]["args"]) - {"dir"}:
            bad(
                f"`measure.inputs` verb {norm_m['inputs']!r} takes more than `dir`; the harness "
                f"runs it with the inputs directory only"
            )
    preds = []
    for p in list(c.get("predicates") or []):
        if not isinstance(p, dict) or not p.get("run") or not p.get("require"):
            bad("each predicate is {run: <command>, require: <rule>}")
        if not _require_readable(str(p["require"])):
            bad(
                f"predicate require {p['require']!r} is not readable; use `exit zero`, "
                f"`exit nonzero`, `output is non-empty`, `output is a git sha` or "
                f"`output matches <regex>`"
            )
        # GIT'S PEEL SYNTAX IS NOT A FIELD: `{commit_hash}^{commit}` would have format_map
        # look up a field named `commit`, and every candidate would fail the identity row.
        # `^{commit}`, `^{tree}`, `^{object}`
        # and `^{tag}` after a caret are git's own spelling; no family identifies a candidate
        # by such a field. Doubled here so the render puts git's spelling back.
        run = re.sub(r"\^\{(commit|tree|object|tag)\}", r"^{{\1}}", str(p["run"]))
        preds.append({"run": run, "require": str(p["require"])})
    used = {ch["verb"] for ch in checks} | {st["verb"] for v in norm_variants.values() for st in v["setup"]}
    if norm_nc:
        used |= {norm_nc["verb"]} | ({norm_nc["then"]} if norm_nc["then"] else set())
    if norm_si:
        used.add(norm_si["verb"])
    if norm_m:
        used.add(norm_m["inputs"])
    unused = sorted(set(norm_verbs) - used)
    if unused:
        bad(f"verb(s) {unused} are declared but no block names them; remove them or use them")
    out = {
        "verbs": norm_verbs,
        "variants": norm_variants,
        "observable": norm_obs,
        "checks": checks,
        "oracle": norm_oracle,
        "negative_control": norm_nc,
        "solver_input": norm_si,
        "measure": norm_m,
        "predicates": preds,
    }
    # Optional, carried through: the instance.json fields (the identity the worklist keys on)
    # and a stage plan; a plan in the file is validated like a drawn one.
    if c.get("instance_fields"):
        out["_instance_fields"] = [str(f) for f in c["instance_fields"]]
        # A FIELD'S NAME READS THE FIELD: `select: triggering_test_ids` without braces would
        # make the oracle look for a test named that. The notation is resolvable, so it is resolved, not refused.
        for ch in checks:
            sel = ch.get("select") or ""
            if sel in out["_instance_fields"]:
                ch["select"] = "{" + sel + "}"
    if c.get("does"):
        out["_does"] = {str(k): str(v) for k, v in dict(c["does"]).items()}
    if c.get("stages"):
        out["_stages"] = validate_stage_plan(out, c["stages"])
    return out


def _require_readable(rule: str) -> bool:
    r = rule.strip().lower()
    return r in ("exit zero", "exit nonzero", "output is non-empty", "output is a git sha") or r.startswith(
        "output matches "
    )


def rows_for(contract: dict) -> list[str]:
    """The rows a replay under this contract runs, in order."""
    rows = list(ROWS)
    if not contract.get("measure"):
        rows = [r for r in rows if r not in ("inputs", "in_band", "separates")]
    if not contract.get("solver_input"):
        rows.remove("solver_input")
    return rows


# ── the report ──────────────────────────────────────────────────────────────────────────


@dataclass
class Row:
    name: str
    ok: bool | None = None  # None: not run (an earlier HARD row failed)
    detail: str = ""
    evidence: dict = field(default_factory=dict)
    hard: bool = False  # a failed hard row blocks: the scripts did not run


@dataclass
class Report:
    instance_dir: str
    rows: list[Row]
    elapsed_s: float = 0.0
    contract: dict | None = None

    @property
    def ok(self) -> bool:
        """The scripts ran on the blank machine: no HARD row failed. Every other row is an
        observation for the builder to judge."""
        return not any(r.ok is False and r.hard for r in self.rows)

    @property
    def first_failure(self) -> Row | None:
        for r in self.rows:
            if r.ok is False:
                return r
        return None

    @property
    def observations(self) -> list:
        """Rows that saw something to judge, in order (not hard failures)."""
        return [r for r in self.rows if r.ok is False and not r.hard]

    @property
    def first_observation(self):
        return next(iter(self.observations), None)

    def text(self) -> str:
        out = [f"recipe report: {self.instance_dir}  ({self.elapsed_s:.0f} s)"]
        for i, r in enumerate(self.rows, 1):
            mark = "ok " if r.ok else (("FAIL" if r.hard else "SEEN") if r.ok is False else "-- ")
            first, _, rest = r.detail.partition("\n")
            label = f"{row_label(r.name, self.contract)} [{r.name}]"
            out.append(f"{i:2d} {label:<52} {mark}  {first}")
            for ln in rest.splitlines():
                out.append(f"{'':60}{ln}")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "instance_dir": self.instance_dir,
            "ok": self.ok,
            "elapsed_s": self.elapsed_s,
            "rows": [
                {"name": r.name, "ok": r.ok, "hard": bool(r.hard), "detail": r.detail, "evidence": r.evidence}
                for r in self.rows
            ],
        }


def row_label(name: str, contract: dict | None = None) -> str:
    """What a row checks, in plain words, for reports the builder and a reader see. The row
    ids stay as they are in code and evidence."""
    c = contract or {}
    if name == "oracle":
        k = (c.get("oracle") or {}).get("kind")
        if k == "fail-to-pass":
            o = c["oracle"]
            return f"fails on {o['fails_on']}, passes on {o['passes_on']}"
        return "the versions give the same output"
    if name == "separates" and c.get("measure"):
        slow, fast = c["measure"]["ratio_of"]
        return f"{slow} is slower than {fast} by the margin"
    return {
        "replays": "container builds, setup runs",
        "identity": "commit and files exist",
        "variants_build": "both versions build",
        "inputs": "inputs are generated",
        "stable": "same output every run",
        "input_sensitive": "output depends on the input",
        "can_fail": "the break is detected",
        "solver_input": "a submission changes the result",
        "in_band": "runtime inside the band",
        "separates": "the versions separate",
    }.get(name, name)


class _Stop(Exception):
    """A row saw something; carries the detail and evidence. `hard` when the scripts did not
    run at all (the image, setup.sh, a version's setup or a script itself failed): only that
    blocks a stage; everything else is an observation the builder judges."""

    def __init__(self, detail: str, evidence: dict | None = None, hard: bool = False):
        super().__init__(detail)
        self.detail = detail
        self.evidence = evidence or {}
        self.hard = hard


#: Rows whose failure means the scripts did not run on the blank machine.
HARD_ROWS = {"replays", "variants_build"}
#: What the exec layer says when the container itself is unreachable. Not the candidate's
#: failure and not an observation to judge: it blocks, so no delivery rests on it.
HARNESS_FAULT = "HARNESS ERROR"


def _unreachable(exc: BaseException) -> bool:
    """A docker or exec-layer failure: the machine, not the candidate."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(w in name for w in ("docker", "apierror", "notfound", "connection")) or any(
        w in text
        for w in (
            "is not running",
            "no such container",
            "cannot connect to the docker",
            "connection aborted",
            "container is gone",
        )
    )


def is_hard_failure(row: str, detail: str, *, raised_hard: bool) -> bool:
    """Whether a failed row blocks the stage.

    THE RAISING SITE DECIDES, NEVER THE WORDING. The detail carries the candidate's own
    output (the oracle row appends what the check printed, can_fail what the break printed,
    stable a diff of two runs), so a log line saying "build.sh exited 1" inside that output
    must not turn an observation the builder is meant to judge into a blocking failure."""
    return bool(raised_hard or row in HARD_ROWS or HARNESS_FAULT in (detail or ""))


def harness_fault(report) -> bool:
    """The report holds a failure of the harness's own (a container the exec layer could not
    reach), rather than anything about the candidate's files."""
    return any(r.ok is False and HARNESS_FAULT in str(r.detail or "") for r in report.rows)


# ── the replay ──────────────────────────────────────────────────────────────────────────


@dataclass
class Exec:
    argv: list
    cwd: str
    exit: int
    elapsed_ms: int | None = None
    digest: str | None = None
    nbytes: int | None = None
    stderr_tail: str = ""
    out_path: str | None = None
    raw: str = ""
    checks: list = field(default_factory=list)  # per check: verb, exit, digest, nbytes, out_path

    def where(self) -> str:
        return f"`{' '.join(shlex.quote(a) for a in self.argv)}` in {self.cwd}"


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fingerprint(instance_dir: str) -> str:
    """One hash over every file in the instance directory (names and contents)."""
    h = hashlib.sha256()
    for root, _dirs, files in os.walk(instance_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            h.update(os.path.relpath(p, instance_dir).encode())
            h.update(b"\0")
            h.update(_file_hash(p).encode())
    return h.hexdigest()


def _tar_of(instance_dir: str, arcname: str = "bench") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for root, _dirs, files in os.walk(instance_dir):
            for fn in sorted(files):
                p = os.path.join(root, fn)
                rel = os.path.join(arcname, os.path.relpath(p, instance_dir))
                info = tar.gettarinfo(p, arcname=rel)
                info.mode = 0o755 if fn.endswith(".sh") else 0o644
                info.uid = info.gid = 0
                with open(p, "rb") as fh:
                    tar.addfile(info, fh)
    return buf.getvalue()


def _numeric_equal(a: str, b: str, tol: float) -> tuple[bool, str]:
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False, f"token counts differ ({len(ta)} vs {len(tb)})"
    for i, (x, y) in enumerate(zip(ta, tb)):
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            if x != y:
                return False, f"token {i} differs: {x!r} vs {y!r}"
            continue
        if abs(fx - fy) / max(abs(fx), abs(fy), 1.0) > tol:
            return False, f"token {i} differs beyond {tol:g}: {x} vs {y}"
    return True, "numerically equal"


def _unescape(s: str) -> str:
    import html

    return html.unescape(s)


def _junit_outcome(xml_text: str, select: str) -> tuple[str, str]:
    """('pass'|'fail'|'absent', detail) for the test `select` in a JUnit XML report."""
    if not select:
        return "absent", "no test selected"

    # NOTATION IS RESOLVED: a test may be written Class#method (Defects4J spelling),
    # path::Class::test (pytest ids) or Class.method (as the report says it).
    # Separators are normalized on both sides before comparing.
    def norm(s: str) -> str:
        # JUnit 5 reports write method() and method(String)[1]. The argument list and the invocation index are dropped; every invocation of a
        # parameterized method then matches the method, and they are aggregated below.
        s = re.sub(r"\([^()]*\)(\[\d+\])?$", "", s.strip())
        s = re.sub(r"#|::|/", ".", s).strip(".")
        return re.sub(r"\.py(?=\.|$)", "", s)  # a pytest path id: tests/x/test_y.py::t -> tests.x.test_y.t

    sel = norm(select)
    seen: list[str] = []
    by_class: list[tuple[str, bool]] = []
    matched: list[tuple[str, bool]] = []
    errored_modules: list[tuple[str, str]] = []  # (module, first error line)
    for m in re.finditer(r"<testcase\b([^>]*?)(/>|>(.*?)</testcase>)", xml_text, re.S):
        attrs, inner = m.group(1), m.group(3) or ""
        name = re.search(r'\bname="([^"]*)"', attrs)
        cls = re.search(r'\bclassname="([^"]*)"', attrs)
        full = f"{cls.group(1) if cls else ''}.{name.group(1) if name else ''}".strip(".")
        seen.append(full)
        nfull = norm(full)
        if sel == nfull or sel == norm(name.group(1) if name else "") or nfull.endswith("." + sel):
            failed = bool(re.search(r"<(failure|error)\b", inner))
            matched.append((full, failed))
            continue
        ncls = norm(cls.group(1)) if cls else ""
        if ncls and (ncls == sel or ncls.endswith("." + sel)):
            by_class.append((full, bool(re.search(r"<(failure|error)\b", inner))))
        err = re.search(r"<error\b[^>]*>(.*?)</error>", inner, re.S)
        if err and not (cls and cls.group(1)):
            # A MODULE THAT FAILS TO COLLECT FAILS EVERY TEST IN IT: when the test module
            # cannot import a name the fix adds, pytest writes one erroring testcase named after the module (classname="") and none for
            # its tests; the selected test is not absent, it failed before it could run.
            lines = [ln.strip() for ln in _unescape(err.group(1)).splitlines() if ln.strip()]
            said = next((ln for ln in lines if ln.startswith("E ")), lines[-1] if lines else "")
            errored_modules.append((nfull, said))
    if matched:
        if len(matched) == 1:
            full, failed = matched[0]
            return ("fail" if failed else "pass"), f"{full}: {'failed' if failed else 'passed'}"
        kinds = {f for _, f in matched}
        if len(kinds) == 1:
            failed = kinds.pop()
            return ("fail" if failed else "pass"), (
                f"{len(matched)} invocation(s) of {select}: all {'failed' if failed else 'passed'}"
            )
        names = ", ".join(f"{n} {'failed' if f else 'passed'}" for n, f in matched[:6])
        return "mixed", f"the invocations of {select} disagree: {names}"
    for mod, said in errored_modules:
        if sel.startswith(mod + "."):
            return "fail", f"{select}: its module {mod} failed to collect ({said})"
    if by_class:
        # A CLASS SELECTS ITS TESTS: a select may name a class, not a method. Every test of the class must share the outcome; a mixed class is neither.
        kinds = {f for _, f in by_class}
        if len(kinds) == 1:
            failed = kinds.pop()
            return ("fail" if failed else "pass"), (
                f"{len(by_class)} test(s) of {select}: all {'failed' if failed else 'passed'}"
            )
        names = ", ".join(f"{n} {'failed' if f else 'passed'}" for n, f in by_class[:6])
        return "mixed", f"the tests of {select} disagree: {names}"
    why = f"{select!r} not in the report ({len(seen)} testcases: {seen[:5]})"
    if not seen:
        # NO TESTCASE AT ALL: a runner that printed the framework's console text instead of
        # the XML would make "0 testcases" look like an uncollectable test.
        why += (
            ". The output holds no <testcase> at all: either the framework's XML report was not "
            "printed (print it on stdout, not the console text) or the runner collected nothing. "
            "A test module that fails to import on this side counts as its tests failing once "
            "the report is printed: pytest writes that error as a testcase."
        )
    if errored_modules and "." not in sel:
        mod, said = errored_modules[0]
        path = mod.replace(".", "/") + ".py"
        why += (
            f". The module {mod} failed to collect ({said}); a module that fails to import fails "
            f"every test in it, but a bare test name cannot be placed in a module. If {select} "
            f"lives there, select it with its module path, {path}::{select}, and the error "
            f"counts as its failure."
        )
    return "absent", why


#: How much of a script's stderr the report quotes: the first STDERR_HEAD and the last
#: STDERR_TAIL characters. The tail alone hid a check.sh's "Starting SonarQube..." and its first
#: error behind a long scanner log.
STDERR_HEAD, STDERR_TAIL = 400, 600


def stderr_excerpt_sh(path: str) -> str:
    """Shell that prints the stderr file at `path` whole when short, else its head and tail
    with the number of bytes left out between them."""
    q = shlex.quote(path)
    keep = STDERR_HEAD + STDERR_TAIL
    return (
        f'__n=$(wc -c <{q} 2>/dev/null || echo 0); if [ "$__n" -gt {keep} ]; then '
        f"head -c {STDERR_HEAD} {q}; printf '\\n... [%s bytes left out] ...\\n' $((__n-{keep})); "
        f"tail -c {STDERR_TAIL} {q}; else cat {q} 2>/dev/null; fi"
    )


class Replay:
    """Replay one instance directory under one family contract.

    `dh` is the docker helper module (agentcore.environments.docker_helpers); tests pass a
    fake. `timeout` bounds every single exec. `lane` toggles the measurement lane for timed
    runs. `keep` leaves the replay container running for inspection.
    """

    def __init__(
        self,
        instance_dir: str,
        contract: dict,
        *,
        dh=None,
        timeout: int = 1800,
        lifetime: str = "3h",
        lane: bool = True,
        keep: bool = False,
        tag_prefix: str = "benchrecipe",
        log: Callable[[str], None] | None = None,
        label: str = "",
        until: str | None = None,
        required_files: set | None = None,
    ):
        self.dir = os.path.abspath(instance_dir)
        self.label = label or self.dir
        # A STAGED check stops after `until` (a row name) and demands only the files that
        # stage needs: the environment stage has no verbs yet, and row 1 must not ask for them.
        self.until = until
        self.required_files = set(required_files) if required_files else None
        self.c = contract
        if dh is None:
            from agentcore.environments import docker_helpers as dh
        self.dh = dh
        self.timeout = timeout
        self.lifetime = lifetime
        self.lane = lane
        self.keep = keep
        self.prefix = tag_prefix
        self.log = log or (lambda s: None)
        self.container = None
        self.inputs: list[str] = []
        self.primary = (
            contract["measure"]["ratio_of"][0] if contract.get("measure") else next(iter(contract["variants"]))
        )
        self.reps = contract["measure"]["repetitions"] if contract.get("measure") else 3
        # AN OBSERVABLE RUN HAS ITS OWN CEILING. A run.sh that starts a server which never
        # returns would hold the host-wide measurement lane for the whole exec timeout while
        # other workers queue behind it. A run is
        # allowed ten times the contract's runtime ceiling (100 s for a 10 s ceiling), and
        # ten minutes when the family declares no measure; past that it is a failed row that
        # names the ceiling, not a silent wait.
        m = contract.get("measure")
        self.observe_timeout = max(60, int(10 * m["ceiling_ms"] / 1000)) if m else 600
        self._obs_n = 0
        self.instance: dict = {}
        # OPAQUE VARIANT DIRECTORIES. A run script that keys on its directory name
        # (`case "$1" in *default*) sleep 1;; esac`) passed every packaging control when the
        # variants lived under their own names. Here each variant is materialized under a
        # random name, so the builder's scripts cannot tell the sides apart; the map is in
        # the report's evidence for a human reader.
        self.vdir: dict[str, str] = {}

    def _vdir(self, variant: str) -> str:
        if variant not in self.vdir:
            import secrets

            self.vdir[variant] = f"{VARIANTS}/v_{secrets.token_hex(4)}"
        return self.vdir[variant]

    def _scratch_dir(self) -> str:
        import secrets

        return f"{VARIANTS}/s_{secrets.token_hex(4)}"

    # ── files and images ────────────────────────────────────────────────────────

    def _read(self, name: str) -> str:
        with open(os.path.join(self.dir, name), encoding="utf-8", errors="replace") as f:
            return f.read()

    def image_tag(self) -> str:
        return f"{self.prefix}/img:{hashlib.sha256(self._read('Dockerfile').encode()).hexdigest()[:16]}"

    def snapshot_tag(self) -> str:
        h = hashlib.sha256()
        for name in ("Dockerfile", "setup.sh", "instance.json"):
            h.update(self._read(name).encode())
        return f"{self.prefix}/setup:{h.hexdigest()[:16]}"

    def ensure_image(self) -> str:
        tag = self.image_tag()
        if not self.dh.image_exists(tag):
            self.log(f"building image {tag}")
            self.dh.build_image_from_dockerfile(self._read("Dockerfile"), tag=tag)
        return tag

    def _start(self, image: str):
        """A blank machine that can still reach GitHub as somebody.

        setup.sh clones the project, and this container is built from the Dockerfile alone --
        nothing of the builder's own machine is in it, credentials included. Started with no
        environment every clone would go out anonymous: GitHub throttles that, then answers
        401, and git stops to ask for a username nobody is there to type. The builder's own
        machine has the token, so the identical script would work when the builder runs it and
        fail when the harness re-runs it, which reads as a builder error and is not.

        The token alone is not enough: plain `git clone https://github.com/...` authenticates
        through ~/.netrc, which is what the manifest writes. Both are needed, and this is what
        the search's own check container has done all along (tools/search_tools._run_check).
        """
        c = self.dh.start_container(
            image,
            self.lifetime,
            workdir=WORK,
            env=self.dh.llm_key_passthrough_env({}),
            labels={"com.benchcompiler.recipe": os.path.basename(self.dir)},
        )
        try:
            self.dh.exec_in_container(c, self.dh.tokens_manifest_cmd(), timeout=60, capture_limit=0)
        except Exception as e:  # noqa: BLE001 - a missing credential must not kill the replay
            self.log(f"credentials manifest not written ({e}); clones will be anonymous")
        c.put_archive("/", _tar_of(self.dir))
        return c

    def ensure_snapshot(self) -> tuple[str, Exec | None]:
        tag = self.snapshot_tag()
        if self.dh.image_exists(tag):
            return tag, None
        image = self.ensure_image()
        self.log(f"running setup.sh for snapshot {tag}")
        c = self._start(image)
        try:
            ex = self._exec(c, [*self._interp("setup.sh"), f"{BENCH}/setup.sh"], cwd=BENCH)
            if ex.exit != 0:
                raise _Stop(
                    f"setup.sh exited {ex.exit}\n{ex.where()}\n{_failure_excerpt(ex.raw)}", {"setup": ex.__dict__}
                )
            self.dh.commit_container(c, repository=tag.split(":")[0], tag=tag.split(":")[1])
            return tag, ex
        finally:
            _remove(c)

    # ── execution ─────────────────────────────────────────────────────────────────

    def _exec(self, container, argv: list[str], cwd: str = WORK, timeout: int | None = None) -> Exec:
        cmd = " ".join(shlex.quote(a) for a in argv)
        t0 = time.monotonic()
        code, out = self.dh.exec_in_container(
            container, cmd, timeout=timeout or self.timeout, workdir=cwd, capture_limit=0
        )
        return Exec(argv=argv, cwd=cwd, exit=int(code), raw=str(out), elapsed_ms=int((time.monotonic() - t0) * 1000))

    def _fill(self, value) -> str:
        """`{field}` in a contract value is read from instance.json (a variant's commit, a test's
        name). Unknown fields are refused by name; `{{` and `}}` are literal braces."""
        text = str(value)
        if "{" not in text:
            return text
        try:
            return text.format_map(_Quoted(self.instance, quote=False))
        except KeyError as e:
            raise _Stop(
                f"the contract names {{{e.args[0]}}}, which instance.json does not carry "
                f"(it has {sorted(self.instance)})"
            )

    def _interp(self, script: str) -> list[str]:
        """The interpreter a script's first line names, else sh. A `#!/bin/bash` script
        using bash syntax must not run under dash; the notation is resolvable, so it is
        resolved."""
        try:
            first = self._read(script).split("\n", 1)[0]
        except (OSError, AttributeError):  # no such file, or a Replay built without a directory
            return ["sh"]
        if first.startswith("#!"):
            words = first[2:].strip().split()
            if words:
                return words
        return ["sh"]

    def _verb_argv(self, verb: str, binding: dict) -> list[str]:
        spec = self.c["verbs"][verb]
        argv = [*self._interp(f"{verb}.sh"), f"{BENCH}/{verb}.sh"]
        for a in spec["args"]:
            if a not in binding:
                raise _Stop(f"verb {verb!r} needs argument {a!r} and nothing binds it (bound: {sorted(binding)})")
            argv.append(self._fill(binding[a]))
        return argv

    def _run_verb(self, verb: str, binding: dict, cwd: str = WORK) -> Exec:
        return self._exec(self.container, self._verb_argv(verb, binding), cwd=cwd)

    def _observe(self, variant_dir: str, input_path: str | None, *, timed: bool, label: str = "") -> Exec:
        """Run the observable verb once; stdout digested and kept, wall-clock inside the
        container around a shell handed the command whole."""
        checks = self._checks()
        verb = self._obs_name()
        binding = {"dir": variant_dir}
        if input_path is not None:
            binding["input"] = input_path
        argvs = [self._verb_argv(ch["verb"], binding) for ch in checks]
        argv = argvs[0] if len(argvs) == 1 else [f"{ch['verb']}.sh" for ch in checks] + [variant_dir]
        self._obs_n += 1
        tag = f"{self._obs_n:04d}"
        # EVERY CHECK, IN ORDER: each check's stdout goes to its own
        # file and is recorded (exit, digest, bytes); the observation's output is the single
        # check's stdout, or the checks' outputs one after another under a header line.
        outs = [f"{OBS}/{tag}.{i}.out" for i in range(len(checks))]
        errs = [f"{OBS}/{tag}.{i}.err" for i in range(len(checks))]
        # ONLY THE CHECKS ARE TIMED: the per-check digest and byte count run outside the
        # __t0..__t1 window, so a sha256sum and a fork do not enter every timed observation.
        runs = "".join(
            f"{' '.join(shlex.quote(a) for a in av)} >{oi} 2>{ei}; __rc{i}=$?; "
            for i, (av, oi, ei) in enumerate(zip(argvs, outs, errs))
        )
        records = "".join(
            f'echo "CHECK {i} $__rc{i} $(sha256sum {oi} | cut -c1-64) $(wc -c <{oi})"; ' for i, oi in enumerate(outs)
        )
        if len(checks) == 1:
            o, e = outs[0], errs[0]
            combine = "__rc=$__rc0; "
        else:
            o, e = f"{OBS}/{tag}.out", f"{OBS}/{tag}.err"
            parts = "; ".join(f"echo '=== {ch['verb']}.sh ==='; cat {oi}" for ch, oi in zip(checks, outs))
            rcs = " ".join(f"$__rc{i}" for i in range(len(checks)))
            combine = (
                f"{{ {parts}; }} >{o}; cat {' '.join(errs)} >{e}; __rc=0; "
                f'for __r in {rcs}; do [ "$__rc" -eq 0 ] && __rc=$__r; done; '
            )
        wrapper = (
            f"mkdir -p {OBS} && cd {WORK} && __t0=$({NOW_MS_SH}); {runs}"
            f'__t1=$({NOW_MS_SH}); {records}{combine}echo "ELAPSED $((__t1-__t0))"; '
            f'echo "DIGEST $(sha256sum {o} | cut -c1-64)"; echo "BYTES $(wc -c <{o})"; '
            f"echo STDERR; {stderr_excerpt_sh(e)}; exit $__rc"
        )
        from agentcore import measurement_lane

        if timed and self.lane:
            with measurement_lane.hold(f"recipe {label or verb}", container=self.container.id):
                code, out = self.dh.exec_in_container(
                    self.container, wrapper, timeout=self.observe_timeout, workdir=WORK, capture_limit=0
                )
        else:
            code, out = self.dh.exec_in_container(
                self.container, wrapper, timeout=self.observe_timeout, workdir=WORK, capture_limit=0
            )
        ex = Exec(argv=argv, cwd=WORK, exit=int(code), raw=str(out), out_path=o)
        if int(code) == 124:
            raise _Stop(
                f"{verb} did not finish within {self.observe_timeout} s on "
                f"{os.path.basename(input_path or '')} (ten times the runtime ceiling"
                f"{' the contract declares' if self.c.get('measure') else ''}); a run that "
                f"waits on a server, a terminal or a network peer never returns\n{ex.where()}",
                {"exec": _plain(ex), "timeout_s": self.observe_timeout},
                hard=True,
            )
        m = re.search(r"^ELAPSED (\d+)", str(out), re.M)
        d = re.search(r"^DIGEST ([0-9a-f]{64})", str(out), re.M)
        b = re.search(r"^BYTES (\d+)", str(out), re.M)
        ex.elapsed_ms = int(m.group(1)) if m else None
        ex.digest = d.group(1) if d else None
        ex.nbytes = int(b.group(1)) if b else None
        ex.stderr_tail = (
            str(out).split("STDERR", 1)[1].strip()[-(STDERR_HEAD + STDERR_TAIL + 80) :] if "STDERR" in str(out) else ""
        )
        recs = {int(mm.group(1)): mm for mm in re.finditer(r"^CHECK (\d+) (\d+) ([0-9a-f]{64}) (\d+)", str(out), re.M)}
        ex.checks = [
            {
                "verb": ch["verb"],
                "exit": int(recs[i].group(2)) if i in recs else ex.exit,
                "digest": recs[i].group(3) if i in recs else None,
                "nbytes": int(recs[i].group(4)) if i in recs else None,
                "out_path": outs[i],
            }
            for i, ch in enumerate(checks)
        ]
        # a check whose reader is not its exit status crashed if it exited nonzero
        crashed = next(
            (rec for ch, rec in zip(checks, ex.checks) if rec["exit"] != 0 and ch["reader"] != "exit-code"), None
        )
        if crashed is not None:
            verb, o = crashed["verb"], crashed["out_path"]
            ex.exit = crashed["exit"]
            # An empty stderr says nothing; the program may have said why on stdout -- unless
            # stdout is a binary stream (a gzip body would fill a report with noise).
            head = ""
            if not ex.stderr_tail:
                raw_head = self.dh.get_file(self.container, o)[:400]
                printable = sum(ch.isprintable() or ch in "\n\t" for ch in raw_head)
                if raw_head.strip() and printable >= 0.9 * len(raw_head):
                    head = raw_head.strip()
                elif ex.nbytes:
                    head = f"({ex.nbytes} bytes, not text)"
            # THE RULE AND THE FIX, NOT ONLY THE NUMBER: "exited 1" alone teaches nothing.
            reads = "the test report it prints" if self._check_of(verb)["reader"] == "junit-xml" else "what it prints"
            on = f" on {os.path.basename(input_path)}" if input_path else ""
            # NOTHING PRINTED IS SAID HERE TOO: with both streams empty the row would show only
            # "stderr: (empty)" and nothing to act on.
            silent = "" if (head or ex.stderr_tail) else f"\n{_failure_excerpt('')}"
            raise _Stop(
                f"{verb}.sh exited {ex.exit}{on}"
                f"\n{ex.where()}\nstderr: {ex.stderr_tail or '(empty)'}"
                + (f"\nstdout: {head}" if head else "")
                + silent
                + f"\nThe exit status of {verb}.sh is not the verdict: the harness reads "
                f"{reads}, and a nonzero exit is a crash. Print the verdict and exit 0 "
                f"whatever it is.",
                {"exec": _plain(ex)},
                hard=True,
            )
        if ex.digest is None:
            raise _Stop(
                f"the observation wrapper printed no output record; `date`, `sha256sum` and "
                f"`wc` must exist in the container\n{_failure_excerpt(ex.raw)}"
            )
        return ex

    def _read_out(self, ex: Exec, limit: int = 200_000) -> str:
        return self.dh.get_file(self.container, ex.out_path)[:limit]

    def _diff(self, a: Exec, b: Exec) -> str:
        code, out = self.dh.exec_in_container(
            self.container, f"diff {a.out_path} {b.out_path} | head -30", timeout=60, capture_limit=0
        )
        return str(out).strip()

    def _tree_sig(self, d: str) -> str:
        """The digest of every file under `d`; "" when there is no such directory (the shell's
        error is not returned as the signature)."""
        code, out = self.dh.exec_in_container(
            self.container,
            f"cd {shlex.quote(d)} && find . -type f -exec sha256sum {{}} + | sort -k2 | sha256sum | cut -c1-64",
            timeout=300,
            capture_limit=0,
        )
        if int(code) != 0:
            return ""
        last = str(out).strip().splitlines()[-1] if str(out).strip() else ""
        return last if re.fullmatch(r"[0-9a-f]{64}", last) else ""

    def _equal(self, a: Exec, b: Exec) -> tuple[bool, str]:
        cmp = next((ch["compare"] for ch in self._checks() if ch["compare"] != "exact"), "exact")
        if cmp == "exact":
            return (a.digest == b.digest), ("same digest" if a.digest == b.digest else "digests differ")
        tol = float(cmp.split(":", 1)[1])
        return _numeric_equal(self._read_out(a), self._read_out(b), tol)

    # ── variants ──────────────────────────────────────────────────────────────────

    def _materialize(self, variant: str, into: str, override: dict | None = None) -> list[Exec]:
        # THE PARENT OF A PATH THE HARNESS HANDS OUT EXISTS: a verb doing
        # `cp -r /work/repo "$1"` needs /work/variants to exist. The
        # variant directory itself is the first verb's to create; its parent is ours.
        self.dh.exec_in_container(self.container, f"mkdir -p {VARIANTS}", timeout=60, capture_limit=0)
        execs = []
        for st in self.c["variants"][variant]["setup"]:
            binding = {"dir": into, **st["args"]}
            if override and st["verb"] == override.get("verb"):
                binding[override["arg"]] = override["value"]
            ex = self._run_verb(st["verb"], binding)
            execs.append(ex)
            if ex.exit != 0:
                raise _Stop(
                    f"{st['verb']}.sh exited {ex.exit} while building variant "
                    f"{variant!r}\n{ex.where()}\n{_failure_excerpt(ex.raw)}",
                    {"exec": _plain(ex), "variant": variant},
                    hard=True,
                )
        return execs

    # ── rows ──────────────────────────────────────────────────────────────────────

    def run(self) -> Report:
        t0 = time.monotonic()
        names = rows_for(self.c)
        if self.until:
            if self.until not in names:
                raise ContractError(f"no row named {self.until!r}; rows are {names}")
            names = names[: names.index(self.until) + 1]
        rows = [Row(name) for name in names]
        state: dict = {}
        try:
            for row in rows:
                fn = getattr(self, f"_row_{row.name}")
                try:
                    row.detail, row.evidence = fn(state)
                    row.ok = True
                    self.log(f"{row.name}: ok -- {row.detail.splitlines()[0]}")
                except _Stop as s:
                    # AN OBSERVATION, NOT A REFUSAL: the report goes on
                    # to the next row; only a hard failure (the scripts did not run) stops it.
                    row.ok = False
                    row.detail, row.evidence = s.detail, s.evidence
                    row.hard = is_hard_failure(row.name, s.detail, raised_hard=bool(s.hard))
                    self.log(f"{row.name}: {'FAIL' if row.hard else 'SEEN'} -- {s.detail.splitlines()[0]}")
                    if row.hard:
                        break
                except Exception as e:  # noqa: BLE001 - a row that cannot be observed is reported, not raised
                    row.ok = False
                    text = f"{type(e).__name__}: {e}"
                    if HARNESS_FAULT in text or _unreachable(e):
                        # NOT AN OBSERVATION: a replay container the exec layer cannot reach
                        # says nothing about the candidate's files, and "the scripts ran"
                        # would be reported for a check that never ran.
                        row.detail = f"{HARNESS_FAULT}: the row could not be run ({text[:200]})"
                        row.hard = True
                        self.log(f"{row.name}: FAIL -- {row.detail}")
                        break
                    row.detail = (
                        f"could not be observed ({text[:200]}); an earlier observation "
                        f"this one builds on may be the reason"
                    )
                    self.log(f"{row.name}: SEEN -- {row.detail}")
        finally:
            if self.container is not None and not self.keep:
                _remove(self.container)
                self.container = None
        return Report(self.label, rows, elapsed_s=time.monotonic() - t0, contract=self.c)

    def _row_replays(self, state) -> tuple[str, dict]:
        wanted = (
            sorted(self.required_files | set(BASE_FILES))
            if self.required_files is not None
            else ["Dockerfile", "setup.sh", "instance.json"] + [f"{v}.sh" for v in self.c["verbs"]]
        )
        missing = [n for n in wanted if not os.path.isfile(os.path.join(self.dir, n))]
        if missing:
            raise _Stop(
                f"the instance directory lacks {missing}"
                + (
                    ""
                    if self.required_files is not None
                    else "; every declared verb is a file <verb>.sh beside Dockerfile, setup.sh and instance.json"
                )
            )
        try:
            self.instance = json.loads(self._read("instance.json"))
            assert isinstance(self.instance, dict)
        except Exception as e:  # noqa: BLE001
            raise _Stop(f"instance.json is not a JSON object: {e}")
        try:
            image = self.ensure_image()
        except Exception as e:  # noqa: BLE001
            raise _Stop(f"the Dockerfile does not build\n{str(e)[-1500:]}")
        snap, setup_ex = self.ensure_snapshot()
        self.container = self._start(snap)
        ev = {"image": image, "snapshot": snap, "setup": (_plain(setup_ex) if setup_ex else "cached")}
        return (f"image {image}; setup.sh {'ran in %d ms' % setup_ex.elapsed_ms if setup_ex else 'cached'}", ev)

    def _row_identity(self, state) -> tuple[str, dict]:
        ev = {"predicates": []}
        # A FIELD LEFT AS A PLACEHOLDER IS NOT ESTABLISHED: an instance may carry rule_id "?",
        # line 0 and an empty enclosing declaration while every check passes because none
        # reads those fields. The fields the contract names are the
        # instance's identity; an unfilled one is refused by name.
        # ONLY THE IDENTITY FIELDS: a placeholder elsewhere is the
        # worker's business as long as the thing can be located; the harness does not know
        # which other fields matter.
        identity = [str(f) for f in ((self.c.get("_drawn") or {}).get("identity") or [])]
        hollow = {f: self.instance.get(f) for f in identity if _is_placeholder(self.instance.get(f))}
        if hollow:
            ev["placeholders"] = hollow
            shown = ", ".join(f"{k} = {v!r}" for k, v in hollow.items())
            # THE MESSAGE SAYS HOW: at the first stage no script exists yet that could derive
            # the value, so the message names what to write and where.
            raise _Stop(
                f"instance.json leaves {shown}: these fields identify the instance, and a "
                f"placeholder is not a value. Write the values into bench/instance.json and "
                f"check again; if they cannot be established after looking, drop the candidate "
                f"saying what you looked at.",
                ev,
            )
        for p in self.c["predicates"]:
            try:
                cmd = p["run"].format_map(_Quoted(self.instance))
            except KeyError as e:
                raise _Stop(
                    f"predicate `{p['run']}` names {{{e.args[0]}}}, which instance.json "
                    f"does not carry (it has {sorted(self.instance)})"
                )
            code, out = self.dh.exec_in_container(self.container, cmd, timeout=600, workdir=WORK, capture_limit=0)
            out = str(out).strip()
            ok, why = _judge(p["require"], int(code), out)
            ev["predicates"].append({"run": cmd, "exit": int(code), "ok": ok, "output": out[-300:]})
            if not ok:
                raise _Stop(
                    f"predicate failed: `{cmd}` -> exit {code}; require {p['require']}: {why}\n{_tail(out)}", ev
                )
        return f"{len(ev['predicates'])} predicate(s) hold", ev

    def _row_variants_build(self, state) -> tuple[str, dict]:
        ev = {"variants": {}, "inputs": []}
        parts = []
        for v in self.c["variants"]:
            d = self._vdir(v)
            execs = self._materialize(v, d)
            tree = self._tree_sig(d)
            ev["variants"][v] = {"dir": d, "steps": [_plain(e) for e in execs], "tree": tree}
            if not tree:
                # A SETUP THAT LEAVES NO DIRECTORY BUILT NOTHING: verbs that work in /work/repo
                # exit 0, and without this check the shell's "can't cd" would pass as the
                # tree signature.
                steps = ", ".join(f"{st['verb']}.sh" for st in self.c["variants"][v]["setup"])
                raise _Stop(
                    f"the setup of variant {v!r} ({steps}) left no files at {d}. Every verb "
                    f"works in the directory it is given as its first argument, never in "
                    f"/work/repo: /work/repo is the clone setup.sh made, and each variant is "
                    f"a separate copy of it built into its own directory, so that the harness "
                    f"can compare and break them one at a time. Make {steps.split(', ')[0]} "
                    f"create <dir> (copy or clone the repository into it) and every later "
                    f"verb work inside <dir>.",
                    ev,
                )
            parts.append(f"{v} {sum(e.elapsed_ms or 0 for e in execs)} ms")
        if not self.c.get("measure"):
            self.inputs = [None]
        return ", ".join(parts), ev

    def _row_inputs(self, state) -> tuple[str, dict]:
        """Measure families only: the inputs verb writes the files every later row runs on.
        Its own row, so a generator that wrote nothing is not reported as a build failure."""
        m = self.c["measure"]
        ex = self._run_verb(m["inputs"], {"dir": INPUTS})
        ev = {"exec": _plain(ex), "inputs": []}
        if ex.exit != 0:
            raise _Stop(f"{m['inputs']}.sh exited {ex.exit}\n{ex.where()}\n{_failure_excerpt(ex.raw)}", ev)
        code, out = self.dh.exec_in_container(
            self.container, f"find {INPUTS} -type f -printf '%s %p\\n' | sort -n", timeout=60, capture_limit=0
        )
        files = [ln.split(" ", 1) for ln in str(out).strip().splitlines() if " " in ln]
        self.inputs = [p for _s, p in files]
        ev["inputs"] = [{"path": p, "bytes": int(s)} for s, p in files]
        if not self.inputs:
            raise _Stop(
                f"{m['inputs']}.sh exited 0 but wrote no files under {INPUTS}\n{ex.where()}\n"
                f"{_failure_excerpt(ex.raw)}",
                ev,
            )
        sizes = ", ".join(f"{os.path.basename(p)} {int(s)} B" for s, p in files)
        return f"{len(self.inputs)} input(s): {sizes}", ev

    def _primary_runs(self, state) -> dict:
        """reps observations of the primary variant on every input; cached in state."""
        if "primary" in state:
            return state["primary"]
        runs: dict = {}
        d = self._vdir(self.primary)
        timed = bool(self.c.get("measure"))
        for inp in self.inputs:
            runs[inp] = [
                self._observe(d, inp, timed=timed, label=f"{self.primary} {os.path.basename(inp or '')}")
                for _ in range(self.reps)
            ]
        state["primary"] = runs
        return runs

    def _checks(self) -> list[dict]:
        """Every check the config declares (any number)."""
        # Normalized here, so a config read from disk without validate_contract (a test, an
        # older stored contract) cannot raise KeyError deep in a row.
        return [
            {
                "verb": ch["verb"],
                "compare": ch.get("compare") or "exact",
                "reader": ch.get("reader"),
                "select": ch.get("select"),
            }
            for ch in (self.c.get("checks") or [self.c["observable"]])
        ]

    def _check_of(self, verb: str) -> dict:
        return next((ch for ch in self._checks() if ch["verb"] == verb), self._checks()[0])

    def _obs_name(self) -> str:
        """The check scripts, named: "check.sh", or "run_tests.sh and check_all.sh"."""
        names = [f"{ch['verb']}.sh" for ch in self._checks()]
        return names[0] if len(names) == 1 else " and ".join(names)

    def _verdict(self, ex: Exec) -> str:
        """What the oracle reads from one observation: the output's digest with no reader,
        the exit status with `exit-code`, the selected test's outcome with `junit-xml`; with
        several checks, each check's reading, named. The stability and sensitivity rows
        compare the raw output, not this (the agent's script masks what varies)."""
        checks = self._checks()
        if len(checks) == 1:
            reader = checks[0]["reader"]
            if reader == "exit-code":
                return f"exit {ex.exit}"
            if reader == "junit-xml":
                return self._read_outcome(ex)[0]
            return ex.digest or ""
        parts = []
        for ch, rec in zip(checks, ex.checks or []):
            if ch["reader"] == "exit-code":
                parts.append(f"{ch['verb']}: exit {rec['exit']}")
            elif ch["reader"] == "junit-xml":
                parts.append(f"{ch['verb']}: {self._outcome_of(ch, rec)[0]}")
            else:
                parts.append(f"{ch['verb']}: {rec.get('digest') or ''}")
        return "; ".join(parts)

    def _select_of(self, ch: dict) -> str | list[str]:
        """The test a check selects, with `{field}` references read from instance.json.
        A select that is exactly one `{field}` whose value is a list names every test in it
        (e.g. a `triggering_test_ids`): the oracle then wants the same outcome
        from each of them."""
        raw = ch.get("select") or ""
        m = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", raw)
        if m and isinstance(self.instance.get(m.group(1)), list):
            return [str(x) for x in self.instance[m.group(1)]]
        return raw.format_map(_Quoted(self.instance, quote=False))

    def _select(self) -> str | list[str]:
        return self._select_of(self._checks()[0])

    def _select_label(self) -> str:
        labels = []
        for ch in self._checks():
            # a check names what it selects, whether or not it declares a reader: the label is
            # for the builder's messages, not for the reading
            if not (ch["reader"] or ch["select"]):
                continue
            sel = self._select_of(ch)
            if isinstance(sel, list):
                field = (ch.get("select") or "").strip("{}")
                labels.append(", ".join(sel) if sel else f"{field} (an empty list)")
            else:
                # with an exit-code reader nothing is selected: the verdict is the check's own
                # (otherwise the message would read "on violating,  should fail and passed")
                labels.append(sel or f"{ch['verb']}.sh")
        return " and ".join(labels) if labels else self._obs_name()

    @property
    def _what_is_read(self) -> str:
        return "verdict" if any(ch["reader"] for ch in self._checks()) else "output"

    def _row_stable(self, state) -> tuple[str, dict]:
        runs = self._primary_runs(state)
        ev = {"variant": self.primary, "per_input": {}}
        for inp, execs in runs.items():
            digests = [e.digest for e in execs]
            # THE RAW OUTPUT, BYTE FOR BYTE: what varies between runs
            # (a timestamp, a duration, a path) is the agent's to strip or mask in its script.
            verdicts = list(digests)
            ev["per_input"][inp or "-"] = {
                "digests": digests,
                "verdicts": [self._verdict(e) for e in execs],
                "elapsed_ms": [e.elapsed_ms for e in execs],
                "bytes": execs[0].nbytes,
            }
            # AN EXIT-CODE CHECK MAY PRINT NOTHING: its verdict is the exit status.
            if execs[0].nbytes == 0 and not any(ch["reader"] == "exit-code" for ch in self._checks()):
                # NO INPUT IS NOT AN INPUT NAMED "-". The reader's channel is named so the fix is plain.
                reader = next((ch["reader"] for ch in self._checks() if ch["reader"]), "") or ""
                on = f" on {os.path.basename(inp)}" if inp else ""
                reads = (
                    f" The {reader} reader reads the report from stdout: the script must "
                    f"print it there (a report written to a file is printed with cat)."
                    if reader and reader != "exit-code"
                    else ""
                )
                silent = (
                    ""
                    if execs[0].stderr_tail
                    else " stderr is empty too: if the script silences it (2>/dev/null, -q), let it "
                    "through, so the report shows why nothing was printed."
                )
                if re.search(r"<testsuite|<testcase", execs[0].stderr_tail or ""):
                    # THE REPORT WENT TO STDERR: say so, not only that stdout was empty.
                    silent = (
                        " The report is on stderr (stderr below shows <testsuite>): print it on "
                        "stdout instead, or redirect it there."
                    )
                raise _Stop(
                    f"{self._obs_name()} printed nothing{on}; an empty "
                    f"observable proves nothing.{reads}{silent}"
                    f"\n{execs[0].where()}\nstderr: {execs[0].stderr_tail or '(empty)'}",
                    ev,
                )
            distinct = sorted(set(verdicts))
            if len(distinct) > 1:
                a = execs[0]
                b = next(e for e in execs if e.digest != verdicts[0])
                on = f" on {os.path.basename(inp)}" if inp else ""
                raise _Stop(
                    f"{self._obs_name()}{on} "
                    f"gave {len(distinct)} distinct outputs in {len(execs)} runs. The output must "
                    f"be identical byte for byte from run to run: whatever varies (a timestamp, "
                    f"a duration, a temporary path, a run id) is yours to strip or mask in "
                    f"{self._obs_name()} before it prints\n{a.where()}\ndiff of "
                    f"two runs:\n{self._diff(a, b)}",
                    ev,
                )
        return f"{len(runs)} input(s), {self.reps} runs each, one output per input", ev

    def _row_input_sensitive(self, state) -> tuple[str, dict]:
        runs = self._primary_runs(state)
        digests = {inp: execs[0].digest for inp, execs in runs.items()}
        ev = {"digests": {os.path.basename(k or "-"): v for k, v in digests.items()}}
        if len(digests) < 2:
            return "one input; sensitivity is judged by the oracle and can_fail rows", ev
        if len(set(digests.values())) < 2:
            raise _Stop(
                f"the same output on every input ({len(digests)} inputs): the observable "
                f"does not depend on the input, so a wrong answer could not be told from "
                f"a right one\n{next(iter(runs.values()))[0].where()}",
                ev,
            )
        return f"{len(set(digests.values()))} distinct outputs over {len(digests)} inputs", ev

    def _row_oracle(self, state) -> tuple[str, dict]:
        kind = self.c["oracle"]["kind"]
        if kind == "reference-equivalence":
            runs = self._primary_runs(state)
            ev = {"kind": kind, "per_variant": {}}
            for v in self.c["variants"]:
                if v == self.primary:
                    continue
                d = self._vdir(v)
                ev["per_variant"][v] = {}
                for inp in self.inputs:
                    ex = self._observe(d, inp, timed=False, label=v)
                    same, why = self._equal(runs[inp][0], ex)
                    ev["per_variant"][v][os.path.basename(inp or "-")] = {"digest": ex.digest, "same": same}
                    if not same:
                        on = f" on {os.path.basename(inp)}" if inp else ""
                        raise _Stop(
                            f"{v} differs from {self.primary}{on}: "
                            f"{why}\n{ex.where()}\ndiff:\n{self._diff(runs[inp][0], ex)}",
                            ev,
                        )
            n = sum(len(x) for x in ev["per_variant"].values())
            return f"every variant matches {self.primary} on every input ({n} comparisons)", ev
        # fail-to-pass
        o = self.c["oracle"]
        label = self._select_label()
        ev = {"kind": kind, "select": [self._select_of(ch) for ch in self._checks() if ch["reader"]]}
        outcomes = {}
        for v, want in ((o["fails_on"], "fail"), (o["passes_on"], "pass")):
            ex = self._observe(self._vdir(v), self.inputs[0], timed=False, label=v)
            got, why = self._read_outcome(ex)
            outcomes[v] = {"got": got, "why": why}
            if got != want:
                if got == "mixed":
                    raise _Stop(
                        f"on {v}, every one of {label} should {want} and they disagree: {why}\n{ex.where()}",
                        {**ev, "outcomes": outcomes},
                    )
                # WHAT THE RUN SAID: an absent test with "0 testcases" and no excerpt gives no
                # hint; the answer is usually in the run's own stderr.
                said = ""
                # an exit-code verdict says only "exit 0"; what the check printed is the
                # reason
                if got == "absent" or any(ch["reader"] == "exit-code" for ch in self._checks()):
                    out_text = self._read_out(ex, limit=200_000).strip()
                    # a traceback says why at its END; keep both ends of a long output
                    shown = out_text if len(out_text) <= 700 else f"{out_text[:200]}\n[...]\n{out_text[-500:]}"
                    said = f"\nstderr: {ex.stderr_tail or '(empty)'}\nstdout: {shown if shown else '(empty)'}"
                raise _Stop(
                    f"on {v}, {label} should {want} and {got == 'absent' and 'is absent' or got + 'ed'}: "
                    f"{why}\n{ex.where()}{said}",
                    {**ev, "outcomes": outcomes},
                )
        return f"{label} fails on {o['fails_on']} and passes on {o['passes_on']}", {**ev, "outcomes": outcomes}

    def _outcome_of(self, ch: dict, rec: dict) -> tuple[str, str]:
        """One check's reading: pass/fail from its exit status, or the selected test's
        outcome from the report it printed."""
        reader = ch["reader"]
        if reader == "exit-code":
            return ("pass" if int(rec["exit"]) == 0 else "fail"), f"exit {rec['exit']}"
        text = self.dh.get_file(self.container, rec["out_path"])[:5_000_000]
        if reader == "junit-xml":
            select = self._select_of(ch)
            if isinstance(select, list):
                if not select:
                    field = (ch.get("select") or "").strip("{}")
                    return "absent", (
                        f"{field} is an empty list in instance.json: name the test(s) the report must show"
                    )
                got = {s: _junit_outcome(text, s) for s in select}
                kinds = {o for o, _ in got.values()}
                why = "; ".join(w for _, w in got.values())
                if len(kinds) == 1:
                    return kinds.pop(), why
                if "absent" in kinds:
                    return "absent", why
                return "mixed", why
            return _junit_outcome(text, select)
        raise _Stop(f"no reader named {reader!r}; the contract may declare exit-code or junit-xml")

    def _read_outcome(self, ex: Exec, select=None) -> tuple[str, str]:
        """The observation's outcome over every check with a reader: pass when all pass;
        fail when any fails; absent or mixed when any check reads so. `why` names each check."""
        checks = self._checks()
        recs = ex.checks or [{"verb": checks[0]["verb"], "exit": ex.exit, "out_path": ex.out_path}]
        results = [(ch["verb"], *self._outcome_of(ch, rec)) for ch, rec in zip(checks, recs) if ch["reader"]]
        if not results:
            raise _Stop("no check has a reader; the contract may declare exit-code or junit-xml")
        if len(results) == 1:
            return results[0][1], results[0][2]
        kinds = {o for _, o, _ in results}
        why = "; ".join(f"{v}.sh: {w}" for v, _, w in results)
        for k in ("mixed", "absent", "fail"):
            if k in kinds:
                return k, why
        return "pass", why

    def _row_can_fail(self, state) -> tuple[str, dict]:
        nc = self.c.get("negative_control")
        if nc is None:
            return "covered by the oracle row: the failing variant is the break", {"via": "oracle"}
        src = self._vdir(nc["variant"])
        dst = self._scratch_dir()
        code, out = self.dh.exec_in_container(
            self.container, f"rm -rf {dst} && cp -a {src} {dst}", timeout=600, capture_limit=0
        )
        if int(code) != 0:
            raise _Stop(f"could not copy {src} to {dst}: {_tail(out)}")
        ex = self._run_verb(nc["verb"], {"dir": dst})
        ev = {"break": _plain(ex)}
        if ex.exit != 0:
            raise _Stop(f"{nc['verb']}.sh exited {ex.exit}\n{ex.where()}\n{_failure_excerpt(ex.raw)}", ev)
        inp = self.inputs[0]
        if nc["expect"] == "check_fails":
            chk = self._run_verb(nc["then"], {"dir": dst})
            ev["check"] = _plain(chk)
            if chk.exit == 0:
                raise _Stop(
                    f"after {nc['verb']}.sh, {nc['then']}.sh still exits 0: the check "
                    f"cannot fail\n{chk.where()}\n{_failure_excerpt(chk.raw)}",
                    ev,
                )
            return f"after {nc['verb']}.sh, {nc['then']}.sh exits {chk.exit}", ev
        clean = (
            self._primary_runs(state)[inp][0]
            if nc["variant"] == self.primary
            else self._observe(src, inp, timed=False, label="clean")
        )
        try:
            broken = self._observe(dst, inp, timed=False, label="broken")
        except _Stop as s:
            ev["broken"] = s.evidence
            return f"after {nc['verb']}.sh the observable verb fails ({s.detail.splitlines()[0]})", ev
        ev["clean_digest"], ev["broken_digest"] = clean.digest, broken.digest
        if any(ch["reader"] for ch in self._checks()):
            same = self._verdict(clean) == self._verdict(broken)
        else:
            same, _why = self._equal(clean, broken)
        if same:
            # WHICH HALF FAILED: "broke nothing the observable sees" leaves open whether the
            # break edited nothing or edited what the check never reads. The tree signature tells the two apart.
            touched = self._tree_sig(src) != self._tree_sig(dst)
            ev["break_changed_files"] = touched
            half = (
                f"{nc['verb']}.sh changed files in the copy, but the {self._what_is_read} of "
                f"{self._obs_name()} is the same: it does not read what the break changed. "
                f"If the program was installed as a copy (pip install ., make install), the run reads the "
                f"copy, not the tree: install in place (pip install -e .) or rebuild after the break"
                if touched
                else f"{nc['verb']}.sh changed no file in the copy: it broke nothing"
            )
            on = f" on {os.path.basename(inp)}" if inp else ""  # no input is not an input named "-"
            raise _Stop(
                f"after {nc['verb']}.sh the {self._what_is_read} is unchanged{on}: {half}\n{ex.where()}"
                f"\n{nc['verb']}.sh said: {_tail(ex.raw) or '(nothing)'}",
                ev,
            )
        return f"after {nc['verb']}.sh the output changes", ev

    def _stand_in(self, si: dict) -> tuple[str, str] | None:  # kept for the older callers
        p = probe_placement(self.c)
        return (p["verb"], p["arg"]) if p["kind"] == "replace" else None

    def _stand_in_doc(self, si: dict) -> tuple[str, str] | None:
        """The setup step of the solver_input's variant that already applies the first probe
        value (the gold): (verb, arg). THE SUBMISSION STANDS WHERE THE GOLD STANDS:
        if a setup step applies gold_test.patch and solver_input names a different verb,
        running that verb on top of the built variant would apply the gold patch a second
        time and fail. When the gold is a setup argument, each probe replaces it there."""
        first = str(si["probe_values"][0])
        for st in self.c["variants"][si["variant"]]["setup"]:
            for a, v in st["args"].items():
                if str(v) == first:
                    return st["verb"], a
        return None

    def _row_solver_input(self, state) -> tuple[str, dict]:
        si = self.c["solver_input"]
        place = probe_placement(self.c)
        stand_in = (place["verb"], place["arg"]) if place["kind"] == "replace" else None
        probes = []
        for i, val in enumerate(si["probe_values"]):
            d = self._scratch_dir()
            if stand_in:
                self._materialize(si["variant"], d, override={"verb": stand_in[0], "arg": stand_in[1], "value": val})
            else:
                code, out = self.dh.exec_in_container(
                    self.container, f"rm -rf {d} && cp -a {self._vdir(si['variant'])} {d}", timeout=600, capture_limit=0
                )
                ex = self._run_verb(si["verb"], {"dir": d, si["arg"]: val})
                if ex.exit != 0:
                    raise _Stop(
                        f"{si['verb']}.sh exited {ex.exit} with {si['arg']}={val!r}\n"
                        f"{ex.where()}\n{_failure_excerpt(ex.raw)}"
                    )
            probes.append({"value": val, "dir": d, "tree": self._tree_sig(d)})
        ev = {"probes": probes, "placement": place}
        if stand_in:
            ev["stands_in"] = {
                "verb": stand_in[0],
                "arg": stand_in[1],
                "note": f"each probe replaces the gold in setup step {stand_in[0]} {stand_in[1]}",
            }
        # The two answers must give different results (the gold is not required to get the
        # accepted verdict).
        inp = self.inputs[0]
        obs = [self._observe(p["dir"], inp, timed=False, label=f"probe {p['value']}") for p in probes]
        ev["observables"] = [o.digest for o in obs]
        v_gold, v_wrong = self._verdict(obs[0]), self._verdict(obs[1])
        ev["verdicts"] = {"gold": v_gold, "wrong": v_wrong}
        if v_wrong != v_gold:
            return (
                f"{si['arg']}={probes[0]['value']!r} gives {v_gold!r}; {probes[1]['value']!r} gives {v_wrong!r}",
                ev,
            )
        if probes[0]["tree"] != probes[1]["tree"]:
            return (f"{si['arg']}={probes[0]['value']!r} and {probes[1]['value']!r} build different trees", ev)
        same, _ = self._equal(obs[0], obs[1])
        if not same:
            return f"{si['arg']} changes the observable", ev
        if self.c.get("measure"):
            costliest = state.get("costliest") or self.inputs[-1]
            a, b = [], []
            for _ in range(self.reps):
                a.append(self._observe(probes[0]["dir"], costliest, timed=True, label="probe a").elapsed_ms)
                b.append(self._observe(probes[1]["dir"], costliest, timed=True, label="probe b").elapsed_ms)
            ev["timings"] = {"a": a, "b": b}
            # DISJOINT RANGES ARE NOT ENOUGH. Two identical -O2 builds gave five samples each
            # within 2 ms and the ranges happened not to overlap (found by the mutant test).
            # The medians must also differ by the contract's own margin, the same number the
            # ratio row holds the real variants to.
            ma, mb = sorted(a)[len(a) // 2], sorted(b)[len(b) // 2]
            margin = self.c["measure"]["min_ratio"]
            far = (max(ma, mb) / max(min(ma, mb), 1)) >= margin
            if far and (max(a) < min(b) or max(b) < min(a)):
                return (f"{si['arg']} moves the timing beyond its spread and margin ({sorted(a)} vs {sorted(b)})", ev)
        raise _Stop(
            f"{si['arg']}={probes[0]['value']!r} and {probes[1]['value']!r} produce the same "
            f"tree and the same observable: the solver's input never reaches the result",
            ev,
        )

    def _row_in_band(self, state) -> tuple[str, dict]:
        m = self.c["measure"]
        runs = self._primary_runs(state)
        curve = []
        for inp in self.inputs:
            ms = sorted(e.elapsed_ms for e in runs[inp])
            curve.append({"input": os.path.basename(inp), "median_ms": ms[len(ms) // 2], "samples": ms})
        costliest = max(curve, key=lambda r: r["median_ms"])
        state["costliest"] = next(p for p in self.inputs if os.path.basename(p) == costliest["input"])
        ev = {"curve": curve, "costliest": costliest, "floor_ms": m["floor_ms"], "ceiling_ms": m["ceiling_ms"]}
        shown = ", ".join(f"{r['input']} {r['median_ms']} ms" for r in curve)
        if not (m["floor_ms"] <= costliest["median_ms"] <= m["ceiling_ms"]):
            raise _Stop(
                f"costliest input {costliest['input']} runs {costliest['median_ms']} ms on "
                f"{self.primary}; the band is {m['floor_ms']:g}..{m['ceiling_ms']:g} ms\n"
                f"curve: {shown}",
                ev,
            )
        return f"costliest {costliest['input']} {costliest['median_ms']} ms; curve: {shown}", ev

    def _row_separates(self, state) -> tuple[str, dict]:
        m = self.c["measure"]
        slow, fast = m["ratio_of"]
        inp = state.get("costliest") or self.inputs[-1]
        a, b = [], []
        for _ in range(self.reps):
            a.append(self._observe(self._vdir(slow), inp, timed=True, label=slow).elapsed_ms)
            b.append(self._observe(self._vdir(fast), inp, timed=True, label=fast).elapsed_ms)
        ma, mb = sorted(a)[len(a) // 2], sorted(b)[len(b) // 2]
        ratio = (ma / mb) if mb else float("inf")
        ev = {
            "input": os.path.basename(inp),
            slow: sorted(a),
            fast: sorted(b),
            "medians": {slow: ma, fast: mb},
            "ratio": ratio,
            "min_ratio": m["min_ratio"],
        }
        if ratio < m["min_ratio"]:
            raise _Stop(
                f"{slow} {ma} ms / {fast} {mb} ms = {ratio:.2f}x on {os.path.basename(inp)}; "
                f"the contract requires {m['min_ratio']:g}x" + (" (the fast side is SLOWER)" if ratio < 1 else ""),
                ev,
            )
        return f"{slow} {ma} ms / {fast} {mb} ms = {ratio:.2f}x on {os.path.basename(inp)}", ev


# ── helpers ─────────────────────────────────────────────────────────────────────────────


class _Quoted(dict):
    """`{field}` from instance.json, shell-quoted (predicates) or plain (select)."""

    def __init__(self, fields: dict, quote: bool = True):
        super().__init__(fields)
        self._quote = quote

    def __getitem__(self, k):
        v = super().__getitem__(k)
        return shlex.quote(str(v)) if self._quote else str(v)


def _judge(rule: str, code: int, out: str) -> tuple[bool, str]:
    r = rule.strip().lower()
    if r == "exit zero":
        return code == 0, f"exit {code}"
    if r == "exit nonzero":
        return code != 0, f"exit {code}"
    if r == "output is non-empty":
        return bool(out.strip()), ("non-empty" if out.strip() else "empty output")
    if r == "output is a git sha":
        last = out.strip().splitlines()[-1].strip() if out.strip() else ""
        return bool(_SHA.match(last)), (f"{last[:50]!r} is not a 40-hex sha" if not _SHA.match(last) else "sha")
    if r.startswith("output matches "):
        pat = rule.strip()[len("output matches ") :]
        ok = re.search(pat, out, re.M) is not None
        return ok, ("matches" if ok else f"no match for /{pat}/")
    return False, f"unreadable rule {rule!r}"


_ERRORISH = re.compile(
    r"(error|fatal|\*\*\*|not found|No such file|cannot |Permission denied|undefined reference|Traceback)", re.I
)


def _failure_excerpt(text: str, tail_n: int = 20, err_n: int = 12) -> str:
    """The tail of a failed command's output, preceded by the error-looking lines that came
    before it. A parallel make prints its error and then twenty progress lines from other
    jobs; a 30-line tail then shows only the progress."""
    lines = [ln for ln in str(text).rstrip().splitlines()]
    if not any(ln.strip() for ln in lines):
        # NOTHING PRINTED IS SAID: a row saying "exited 1" over a blank gives nothing to act on.
        return (
            "it printed nothing on stdout or stderr. Either the script silences its commands' "
            "output (`> /dev/null 2>&1`, `-q`) or a `set -e` line failed without a message (a "
            "`cd` to a missing directory, a `test`). Let the output through, or run the script "
            "with `bash -x` on your machine to see which line fails."
        )
    tail = lines[-tail_n:]
    before = lines[:-tail_n] if len(lines) > tail_n else []
    errs = [ln for ln in before if _ERRORISH.search(ln)][-err_n:]
    if errs:
        return "\n".join(["(error-looking lines before the tail:)", *errs, "(tail:)", *tail])
    return "\n".join(tail)


def _tail(text: str, n: int = 30) -> str:
    lines = str(text).rstrip().splitlines()
    return "\n".join(lines[-n:])


def _plain(ex: Exec | None) -> dict:
    if ex is None:
        return {}
    return {
        "argv": ex.argv,
        "cwd": ex.cwd,
        "exit": ex.exit,
        "elapsed_ms": ex.elapsed_ms,
        "digest": ex.digest,
        "bytes": ex.nbytes,
        "stderr_tail": ex.stderr_tail,
        "tail": _tail(ex.raw, 12),
    }


def _remove(container) -> None:
    try:
        container.remove(force=True)
    except Exception:  # noqa: BLE001
        pass


# ── the builder's stages, derived from the contract ─────────────────────────────────────


_PLACEHOLDERS = {
    "",
    "?",
    "??",
    "unknown",
    "tbd",
    "todo",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "placeholder",
    "missing",
    "fixme",
    "xxx",
}


def _is_placeholder(v) -> bool:
    """None, an empty or whitespace string, a placeholder word, or `<like this>`. Numbers and
    lists are values (a line number may be 0 only by the same mistake, but the harness does
    not know which fields are line numbers)."""
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s in _PLACEHOLDERS or (s.startswith("<") and s.endswith(">"))
    return False


def _fields_named(contract: dict) -> list[str]:
    """Every `{field}` the contract reads from instance.json, in first-seen order."""
    seen: list[str] = []

    def walk(v):
        if isinstance(v, str):
            for m in re.finditer(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})", v):
                if m.group(1) not in seen:
                    seen.append(m.group(1))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk({k: contract.get(k) for k in ("variants", "observable", "checks", "solver_input", "predicates")})
    for f in contract.get("_instance_fields") or []:
        if f not in seen:
            seen.append(f)
    return seen


def stages_for(contract: dict) -> list[dict]:
    """The builder's per-candidate stages, grouped by which scripts the rows exercise.

    Nothing here names a family: stage 2 is whichever verbs the variants' setup steps name,
    stage 3 is the observable verb plus the inputs and break verbs when declared, stage 4 is
    the solver verb when it is not already written and the runtime rows when declared. A
    family with no measure and a solver verb already in setup has three stages.
    """
    c = contract
    if c.get("_stages"):
        return [dict(st) for st in c["_stages"]]
    rows = rows_for(c)
    setup_verbs: list[str] = []
    for v in c["variants"].values():
        for st in v["setup"]:
            if st["verb"] not in setup_verbs:
                setup_verbs.append(st["verb"])
    obs_verbs = [ch["verb"] for ch in (c.get("checks") or [c["observable"]])]
    if c.get("measure"):
        obs_verbs.append(c["measure"]["inputs"])
    nc = c.get("negative_control")
    if nc:
        for v in (nc["verb"], nc.get("then")):
            if v and v not in obs_verbs:
                obs_verbs.append(v)
    written = set(setup_verbs) | set(obs_verbs)
    variant_names = " and ".join(c["variants"])
    obs_verb = " and ".join(f"{ch['verb']}" for ch in (c.get("checks") or [c["observable"]]))
    stages = [
        {
            "name": "set up the toolchain and clone the code",
            "files": ["Dockerfile", "setup.sh", "instance.json"],
            "verbs": [],
            "rows": [r for r in ("replays", "identity") if r in rows],
        },
        {
            "name": f"build the {variant_names} versions",
            "files": [f"{v}.sh" for v in setup_verbs],
            "verbs": setup_verbs,
            "rows": ["variants_build"],
        },
        {
            "name": f"make {obs_verb}.sh produce the result and break it on purpose",
            "files": [f"{v}.sh" for v in obs_verbs if v not in setup_verbs],
            "verbs": [v for v in obs_verbs if v not in setup_verbs],
            "rows": [r for r in ("inputs", "stable", "input_sensitive", "oracle", "can_fail") if r in rows],
        },
    ]
    last_rows = [r for r in ("solver_input", "in_band", "separates") if r in rows]
    if last_rows:
        si = c.get("solver_input")
        new_verbs = [si["verb"]] if si and si["verb"] not in written else []
        name = (
            "apply a submission and grow the workload into the band"
            if c.get("measure")
            else "apply a submission and show it changes the result"
        )
        stages.append({"name": name, "files": [f"{v}.sh" for v in new_verbs], "verbs": new_verbs, "rows": last_rows})
    return stages


def stage_scope(contract: dict, index: int) -> tuple[set, str]:
    """(files required through stage `index`, the last row of that stage), for a staged check."""
    stages = stages_for(contract)
    if not 0 <= index < len(stages):
        raise ContractError(f"no stage {index}; there are {len(stages)}")
    files: set = set()
    for st in stages[: index + 1]:
        files |= set(st["files"])
    rows = stages[index]["rows"]
    return files, rows[-1]


def stage_instruction(stage: dict, contract: dict) -> str:
    """One stage's instruction for the builder, structured the way a staged agent prompt reads
    best: goal, what to write, how it is checked, traps, the exact completion signal, and an
    example of a passing report. Every family-specific word comes from the contract; the
    model's `goal` paragraph, when the plan is drawn, is the traps section."""
    c = contract
    does = c.get("_does") or {}
    stages = stages_for(c)
    n = len(stages)
    idx = next((i for i, st in enumerate(stages) if st["name"] == stage["name"]), 0)
    # The model's judgment sections for this stage (benchcompiler.stage_prompt), when the
    # config carries them: where you are, the goal, the traps. Everything else below is
    # rendered from the config as it is NOW, so an edited predicate or number reaches the
    # builder; a stored prompt text did not.
    secs = c.get("_stage_sections") or []
    sections = secs[idx] if idx < len(secs) and isinstance(secs[idx], dict) else None
    rows = list(stage.get("rows") or [])
    rowset = set(rows)
    kind = (
        "environment"
        if "replays" in rowset
        else "variants"
        if rowset == {"variants_build"}
        else "observable"
        if rowset & {"stable", "oracle", "can_fail", "inputs", "input_sensitive"}
        else "last"
    )
    o = c["observable"]
    m = c.get("measure")
    nc = c.get("negative_control")
    si = c.get("solver_input")

    def verb_line(v: str) -> str:
        args = " ".join(f"<{a}>" for a in c["verbs"][v]["args"])
        d = does.get(v) or ""
        return f"  {v}.sh {args}".rstrip() + (f"\n      {d}" if d else "")

    out: list[str] = [f"STAGE {idx + 1} of {n}: {stage['name']}", ""]

    if sections and str(sections.get("where_you_are") or "").strip():
        out.append("Where you are")
        out.append("=============")
        out.append(str(sections["where_you_are"]).strip())
        out.append("")

    # Goal
    out.append("Goal of this stage")
    out.append("==================")
    if sections and str(sections.get("goal") or "").strip():
        out.append(str(sections["goal"]).strip())
    elif stage.get("goal"):
        out.append(stage["goal"])
    elif kind == "environment":
        out.append(
            "Get a container that can build and check this program, and the source at the exact "
            "commit inside it. Nothing else yet: no build, no run scripts."
        )
    elif kind == "variants":
        out.append(
            "Make the build script(s) work so that every version the benchmark compares builds "
            "in the fresh container from the scripts alone."
        )
    elif kind == "observable":
        out.append(
            "Make the program produce its result on demand, on generated inputs, and make a "
            "deliberate break that the result reveals."
        )
    else:
        out.append(
            "Show that a solver's submission reaches the result"
            + (", and grow the workload until the runtime numbers in the spec hold." if m else ".")
        )
    out.append("")

    # What to write
    out.append("What to write into bench/")
    out.append("=========================")
    files = list(stage.get("files") or [])
    if not files:
        out.append("  (no new file; this stage edits what earlier stages wrote)")
    for f in files:
        if f == "Dockerfile":
            out.append("  Dockerfile     FROM a base image; install the toolchain and every dependency the build")
            out.append("                 and the checks need. No clone, no build, no source in the image.")
        elif f == "setup.sh":
            out.append(
                "  setup.sh       runs once inside that container (your bench/ is at /bench there), with the network: clone the"
            )
            out.append(
                "                 repository at the exact commit into /work/repo and leave /work ready. No build."
            )
        elif f == "instance.json":
            out.append("  instance.json  a JSON object with " + ", ".join(_fields_named(c)) + ".")
        elif f.endswith(".sh") and f[:-3] in c["verbs"]:
            out.append(verb_line(f[:-3]))
        else:
            out.append(f"  {f}")
    if kind == "variants":
        out.append(
            "A build script's output is not read: the check looks at its exit code (0 = built) and at "
            "what it leaves in <dir>. Print what you like, or nothing."
        )
    if kind == "observable" and m:
        out.append(
            f"{m['inputs']}.sh writes the input files directly into <dir> (no subdirectory); what it prints "
            f"is not read, only those files count."
        )
    # EVERY STAGE SHOWS HOW THE SIDES ARE BUILT: a stage must not judge a candidate on a side
    # whose setup (for example applying the test patch) is only described at a later stage.
    out.append("The sides the checks build, each into its own randomly named directory <dir>, in this order:")
    for vname, v in c["variants"].items():
        steps = "; then ".join(
            f"{st['verb']}.sh <dir>" + "".join(f" {val}" for val in st["args"].values()) for st in v["setup"]
        )
        out.append(f"  {vname}: {steps}")
    out.append(
        "What a setup step applies is in the tree when the check runs, whatever the checkout alone "
        "lacks: a candidate is not dropped for a file or a change that a later step of its side "
        "supplies."
    )
    if kind == "observable":
        checks = c.get("checks") or [o]
        if len(checks) > 1:
            out.append(
                f"This family has {len(checks)} check scripts, run one after another on every observation: "
                + ", ".join(f"{ch['verb']}.sh" for ch in checks)
                + ". They are read together."
            )
        for ch in checks:
            if ch["reader"] == "exit-code":
                out.append(f"The verdict of {ch['verb']}.sh is its EXIT CODE: 0 passes, anything else fails.")
            elif ch["reader"] == "junit-xml":
                out.append(
                    f"{ch['verb']}.sh prints the test framework's own JUnit XML report on stdout, unmodified; "
                    f"the harness reads test {ch['select']} from it. It exits 0 whether the test passes "
                    f"or fails: its exit status is not the verdict, and a nonzero exit is a crash."
                )
            else:
                out.append(
                    f"{ch['verb']}.sh prints the program's result and nothing else: no timings, no logs, no "
                    f"progress lines (send those to stderr). It exits 0 whatever the result: its exit "
                    f"status is not the verdict, and a nonzero exit is a crash."
                )
        if m:
            out.append(
                f"{m['inputs']}.sh <dir> writes a few inputs of rising size. Any size that exercises the "
                f"program is fine here; reaching the runtime band is the last stage's job, and an input "
                f"already in the band is kept, not shrunk."
            )
        if nc:
            out.append(
                f"{nc['verb']}.sh <dir> corrupts the {nc['variant']} version's computation so that the "
                f"result changes; a break the result cannot reveal is refused."
            )
    if kind == "last":
        if si:
            out.append(
                f"The solver submits <{si['arg']}> to {si['verb']}.sh on the {si['variant']} version. The "
                f"harness applies {si['probe_values'][0]!r} and {si['probe_values'][1]!r} and requires a "
                f"different result; if a probe value names a file under /bench, write it into bench/."
            )
        if m:
            slow, fast = m["ratio_of"]
            out.append(
                f"Grow {m['inputs']}.sh's largest input until {slow}'s median over {m['repetitions']} runs "
                f"is between {m['floor_ms']:g} and {m['ceiling_ms']:g} ms, and {slow} / {fast} is at least "
                f"{m['min_ratio']:g}x."
            )
    out.append("")

    # How it is checked
    out.append("How it is checked")
    out.append("=================")
    out.append("Your own machine and the checker's are two machines. bench_check builds a FRESH container")
    out.append("from your Dockerfile and runs setup.sh in it, so /work/repo is there. On YOUR machine it")
    out.append("exists only if you ran setup.sh yourself (`bash /bench/setup.sh`): a stage that passed its")
    out.append("checks proves the scripts work in the fresh container, not that anything is left on yours.")
    out.append("bench_check copies bench/ from your machine, builds a FRESH container from your Dockerfile,")
    out.append("runs setup.sh, and runs your scripts there, and REPORTS what it saw. Only what those files")
    out.append("do on that machine counts. The report has one line per observation below: `ok`, or `SEEN`")
    out.append("with what was observed. A SEEN line is yours to judge (fix the script, or decide it is fine")
    out.append("and say why in instance.json); it never blocks you. Only FAIL blocks, and FAIL means the")
    out.append("scripts did not run: the image does not build, setup.sh fails, or a script the stage runs")
    out.append("exits nonzero where its exit status is not the verdict (a version's setup, the break, the")
    out.append("script that applies an answer).")
    out.append("This stage's observations, in order:")
    for r in rows:
        out.append(f"  - {row_label(r, c)} [{r}]")
    if kind == "environment" and c["predicates"]:
        out.append("The identity commands, run with your instance.json values:")
        out.extend(f"  {p['run']}   -> require: {p['require']}" for p in c["predicates"])
    if kind == "observable" and c["oracle"]["kind"] == "reference-equivalence":
        tol = f" (numeric tolerance {o['compare'].split(':')[1]})" if o["compare"] != "exact" else ""
        # SAID PLAINLY: "the versions must agree" with variants named inconsistent and
        # consistent reads as a contradiction; "output" means the verdict the check prints, and
        # the break, not the variant, is what must change it.
        names = " and ".join(c["variants"])
        nc = c.get("negative_control")
        out.append(
            f"What {o['verb']}.sh prints is the verdict, and it must be the SAME on {names}: each "
            f"variant is checked against its own expectation, so a correct {o['verb']}.sh prints "
            f"the same verdict on every variant{tol}."
        )
        if nc:
            out.append(
                f"The only thing that changes the verdict is {nc['verb']}.sh: after it, {o['verb']}.sh "
                f"must print something different. A {o['verb']}.sh that prints different verdicts on "
                f"{names} fails the oracle row."
            )
    if kind == "observable" and c["oracle"]["kind"] == "fail-to-pass":
        oc = c["oracle"]
        out.append(
            f"The check{'s, read together,' if len(c.get('checks') or [o]) > 1 else ''} must FAIL on "
            f"{oc['fails_on']} and PASS on {oc['passes_on']}."
        )
    if kind == "last" and m:
        out.append("The report shows the runtime of every input; a flat curve means the program is not doing the")
        out.append("work, and you need a different command, not a bigger file.")
    out.append("")

    # Traps
    traps = {
        "environment": [
            "A package that exists in your sandbox but not in the Dockerfile fails here.",
            "A setup.sh that builds or that needs a file from your sandbox fails here.",
        ],
        "variants": [
            "A build that ignores the arguments it is given produces the same program for every version.",
            "A script that assumes its own directory instead of using <dir> breaks under the random names.",
        ],
        "observable": [
            "Output with a timestamp, a pid or a path is never stable.",
            "A program that prints a constant does not depend on its input.",
            "A break that only removes a file leaves the result unchanged.",
        ],
        "last": ["A workload dominated by startup or I/O does not separate; change the command, not the size."],
    }[kind]
    if sections and isinstance(sections.get("traps"), list) and sections["traps"]:
        traps = [str(t).strip() for t in sections["traps"] if str(t).strip()]
    out.append("Traps")
    out.append("=====")
    out.extend(f"  - {t}" for t in traps)
    out.append("")

    # Done when
    out.append("Done when")
    out.append("=========")
    if stage.get("done_when"):
        out.append(stage["done_when"])
    out.append("What you verified yourself, and every SEEN line you accepted with the reason, goes into")
    out.append('bench/instance.json under "builder_notes": it travels with the instance and is what a human')
    out.append("reads to see why you judged it sound.")
    out.append(
        "bench_check reports the scripts ran (no FAIL) and you are satisfied with every SEEN line. Then call bench_deliver"
        + (
            ": the candidate is delivered."
            if idx == n - 1
            else ": the stage closes and you get the next stage's instruction. Do not write the next stage's "
            "files before that."
        )
    )
    out.append("")

    # Example
    out.append("Example of a passing report for this stage")
    out.append("==========================================")
    start = rows_for(c).index(rows[0]) + 1 if rows else 1
    for k, r in enumerate(rows):
        out.append(f"  {start + k:2d} {row_label(r, c)} [{r}]".ljust(56) + "ok")
    return "\n".join(out)


# ── a stage plan proposed by the model, validated against the contract ─────────────────

BASE_FILES = ("Dockerfile", "setup.sh", "instance.json")


def probe_placement(contract: dict) -> dict:
    """Where the solver_input row puts each probe value.

    `replace`: a setup step of the row's variant already applies THE GOLD (the first probe
    value is that step's argument), so each probe replaces it there -- applying the gold a
    second time on top would fail (a test patch, build flags).
    `on_top`: no setup step applies the gold, so the variant is materialized as it stands and
    the verb runs on a copy of it. A step that uses the SAME VERB for something else (a setup
    that applies the commit's own test with apply_patch, while the answer is a fix patch) is
    left alone: matching by verb would replace the test patch and the row would prove
    nothing."""
    si = contract.get("solver_input")
    if not si:
        return {"kind": "none"}
    gold = str((si.get("probe_values") or [""])[0])
    for st in contract["variants"][si["variant"]]["setup"]:
        for a, v in (st.get("args") or {}).items():
            if str(v) == gold:
                return {"kind": "replace", "verb": st["verb"], "arg": a}
    return {"kind": "on_top", "verb": si["verb"], "arg": si["arg"]}


def _row_needs(contract: dict) -> dict[str, set]:
    """Which files each row needs before it can run. Computed from the contract, so the stage
    a row closes at is whichever stage completes that set."""
    c = contract
    base = set(BASE_FILES)
    setup_files = {f"{st['verb']}.sh" for v in c["variants"].values() for st in v["setup"]}
    obs = {f"{ch['verb']}.sh" for ch in (c.get("checks") or [c["observable"]])}
    inputs = {f"{c['measure']['inputs']}.sh"} if c.get("measure") else set()
    nc = c.get("negative_control")
    nc_files = ({f"{nc['verb']}.sh"} | ({f"{nc['then']}.sh"} if nc.get("then") else set())) if nc else set()
    si = c.get("solver_input")
    si_files = {f"{si['verb']}.sh"} if si else set()
    needs = {
        "replays": base,
        "identity": base,
        "variants_build": base | setup_files,
        "inputs": base | setup_files | inputs,
        "stable": base | setup_files | inputs | obs,
        "input_sensitive": base | setup_files | inputs | obs,
        "oracle": base | setup_files | inputs | obs,
        "can_fail": base | setup_files | inputs | obs | nc_files,
        "solver_input": base | setup_files | inputs | obs | si_files,
        "in_band": base | setup_files | inputs | obs,
        "separates": base | setup_files | inputs | obs,
    }
    return {r: needs[r] for r in rows_for(c)}


def _bench_files_applied(contract: dict) -> list[tuple[str, str]]:
    """(where, /bench/<file>) for every literal /bench path a block hands to a verb: variant
    setup arguments, solver_input probe values, the negative control's `then` step. A value
    in braces is an instance field, resolved at replay, and is not a file here."""
    out: list[tuple[str, str]] = []

    def take(where: str, value) -> None:
        if isinstance(value, str) and value.startswith("/bench/") and "{" not in value:
            out.append((where, value))
        elif isinstance(value, (list, tuple)):
            for v in value:
                take(where, v)
        elif isinstance(value, dict):
            for v in value.values():
                take(where, v)

    for vname, v in (contract.get("variants") or {}).items():
        for st in v.get("setup") or []:
            take(f"variant {vname!r} ({st.get('verb')})", st.get("args") or {})
    si = contract.get("solver_input") or {}
    take("solver_input's probe_values", si.get("probe_values") or [])
    nc = contract.get("negative_control") or {}
    if isinstance(nc.get("then"), dict):
        take("the negative control's `then` step", nc["then"].get("args") or {})
    return out


def validate_stage_plan(contract: dict, stages) -> list[dict]:
    """Check a proposed stage plan and attach to each stage the rows it closes.

    Every instance file appears in exactly one stage; the first stage holds the Dockerfile,
    setup.sh and instance.json; every row is closed by some stage (the plan writes the files
    a row needs in an order the checker can follow); each stage has a goal sentence. Raises
    ContractError naming the defect.
    """

    def bad(msg):
        raise ContractError(f"stages: {msg}")

    if not isinstance(stages, list) or not 2 <= len(stages) <= 6:
        bad("declare 2 to 6 stages, each {name, files, goal, done_when}")
    all_files = set(BASE_FILES) | {f"{v}.sh" for v in contract["verbs"]}
    seen: dict[str, str] = {}
    out = []
    for i, st in enumerate(stages):
        if not isinstance(st, dict) or not str(st.get("name") or "").strip():
            bad(f"stage {i + 1} has no name")
        files = [str(f) for f in (st.get("files") or [])]
        for f in files:
            # A data file the instance ships (a gold patch, a comment file, a reference
            # answer) is welcome in any stage; what must appear exactly once are the base
            # files and the verbs. A `.sh` that is not a declared verb is a verb nobody
            # declared, and is refused.
            if f.endswith(".sh") and f not in all_files:
                bad(
                    f"stage {st['name']!r} writes {f!r}, which is not a declared verb "
                    f"(the verbs are {sorted(contract['verbs'])}); declare it or drop it"
                )
            if f in seen:
                bad(f"{f!r} is written in both {seen[f]!r} and {st['name']!r}; each file belongs to one stage")
            seen[f] = str(st["name"])
        if i == 0 and not set(BASE_FILES) <= set(files):
            bad(f"the first stage must write {list(BASE_FILES)} (the environment comes first)")
        if not str(st.get("goal") or "").strip():
            bad(f"stage {st['name']!r} has no `goal` sentence")
        if len(str(st["name"]).replace("_", " ").split()) < 2:
            bad(
                f"stage {st['name']!r}: name a stage by what the builder DOES, in a short phrase "
                f'("build both versions", "write the check"), not a one-word label'
            )
        out.append(
            {
                "name": str(st["name"]),
                "files": files,
                "goal": str(st["goal"]).strip(),
                "done_when": str(st.get("done_when") or "").strip(),
                "verbs": [f[:-3] for f in files if f.endswith(".sh") and f != "setup.sh"],
            }
        )
    missing = all_files - set(seen)
    if missing:
        bad(f"no stage writes {sorted(missing)}; every verb and the three base files belong to one stage")
    # A FILE THE CONFIG APPLIES IS WRITTEN BY A STAGE: a patch the config applies, reverts or
    # probes under /bench must be listed by some stage, or no worker is told to write it.
    for where, path in _bench_files_applied(contract):
        name = path[len("/bench/") :]
        if name and name not in seen:
            bad(
                f"no stage writes {name!r}, which {where} applies as {path}; add {name!r} to the "
                f"`files` of the stage where the builder can derive it (a gold patch: the stage that "
                f"clones, from the commits; an empty patch: the first stage) and say in that stage's "
                f"`goal` what the file holds"
            )
    order = {st["name"]: i for i, st in enumerate(out)}
    # THE IDENTITY ROW RUNS IN THE FIRST STAGE: a predicate demanding a /bench file written
    # in a later stage would fail every candidate at stage 1. A predicate reads the clone
    # and what the first stage wrote; a /bench file a later stage writes is not identity.
    for p in contract.get("predicates") or []:
        for path in re.findall(r"/bench/[A-Za-z0-9_.\-]+", str(p.get("run") or "")):
            name = path[len("/bench/") :]
            if name in seen and order[seen[name]] > 0:
                bad(
                    f"predicate {p['run']!r} reads {path}, which stage {seen[name]!r} writes; the "
                    f"identity predicates run in the first stage, on the fresh clone and the files "
                    f"it wrote. Drop that predicate (the plan already requires a stage to write "
                    f"{name!r}), or write {name!r} in the first stage if it can be derived there"
                )
            if name not in seen:
                bad(
                    f"predicate {p['run']!r} reads {path}, which no stage writes; drop it or add "
                    f"{name!r} to the first stage's `files`"
                )
    # The observable is written in a LATER stage than the variants' scripts: a plan that
    # lumps building and running into one stage is the one-shot writing the stages exist to
    # prevent.
    setup_files = {f"{st['verb']}.sh" for v in contract["variants"].values() for st in v["setup"]}
    last_setup = max((order[seen[f]] for f in setup_files if f in seen), default=0)
    for ch in contract.get("checks") or [contract["observable"]]:
        obs_file = f"{ch['verb']}.sh"
        if obs_file in seen and obs_file not in setup_files and order[seen[obs_file]] <= last_setup:
            bad(
                f"{obs_file} (a check) is written in stage {seen[obs_file]!r}, the same stage as or "
                f"before the variants' scripts {sorted(setup_files)}; put the variants' scripts in their own "
                f"stage, closed by the build check, and the checks in a later one"
            )
    needs = _row_needs(contract)
    # The runtime rows close at the LAST stage whatever the files say: growing the workload
    # writes no new script. The solver row does too when the solver's script was already
    # written for a variant (e.g. flags that go through build.sh).
    si = contract.get("solver_input")
    last_rows = {"in_band", "separates"}
    if si and any(st["verb"] == si["verb"] for v in contract["variants"].values() for st in v["setup"]):
        last_rows.add("solver_input")
    written: set = set()
    closed: set = set()
    for i, st in enumerate(out):
        written |= set(st["files"])
        final = i == len(out) - 1
        st["rows"] = [
            r for r in rows_for(contract) if r not in closed and needs[r] <= written and (final or r not in last_rows)
        ]
        closed |= set(st["rows"])
    if closed != set(rows_for(contract)):
        bad(f"rows {sorted(set(rows_for(contract)) - closed)} are never closed")
    empty = [st["name"] for st in out if not st["rows"]]
    if empty:
        bad(f"stage(s) {empty} close no check; merge them into the next stage")
    return out
