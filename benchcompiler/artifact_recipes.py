"""Package delivered recipe instances into a drivable benchmark. No model, at any point.

`bench merge` reads `working_set.final_sample` out of agent_state.json and gates its
mechanical interface path on `variants_root` -- both of which belong to the collector/sampler
pipeline. The recipes pipeline writes somewhere else entirely: `recipes/<candidate>/instance/`
holding the scripts, beside a `report.json` holding the harness's own verdict.

Nothing here asks a model anything. The interface was decided once, at formalization and
staging, and lives in the family's contract: the verbs and their arguments, the variants and
the exact setup sequence for each, the checks and their readers, the oracle kind, and the
solver input with two reference answers. This module renders that decision; it does not
re-derive it. Same instances plus same contract produce a byte-identical artifact.

What is written:

    <out>/
      contract.json          the family's interface, copied verbatim
      manifest.json          what was packaged, from where, and each instance's report
      instances/<id>/        the instance directory as delivered (Dockerfile, setup.sh, verbs)
      bench                  the interpreter -- reads contract.json, knows nothing else
      README.md              rendered from the contract

The interpreter ships standalone: plain Python over the `docker` CLI, with no agentcore
import, so a third party needs this directory and docker and nothing of ours.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import time

#: paths inside the container, matching agentcore.recipes so a packaged instance behaves the
#: way it did when the harness proved it
BENCH = "/bench"
WORK = "/work"
VARIANTS = "/work/variants"

REQUIRED = ("Dockerfile", "setup.sh", "instance.json")


def collect(run_dirs: list[str]) -> list[dict]:
    """Every delivered instance under the given run directories, newest wins on a repeat.

    Delivered means the three markers the harness leaves together: report.json (its verdict),
    a submit_*.json (the builder handed it over) and instance/ (the scripts themselves). A
    directory with only some of those is work in flight, not a delivery.
    """
    found: dict[str, dict] = {}
    for run in run_dirs:
        for cand in sorted(glob.glob(os.path.join(run, "**", "recipes", "*"), recursive=True)):
            if not os.path.isdir(cand) or os.path.basename(cand) == "bench_last":
                continue
            inst = os.path.join(cand, "instance")
            rep = os.path.join(cand, "report.json")
            if not (os.path.isdir(inst) and os.path.exists(rep) and glob.glob(os.path.join(cand, "submit_*.json"))):
                continue
            missing = [f for f in REQUIRED if not os.path.exists(os.path.join(inst, f))]
            if missing:
                continue
            try:
                with open(rep, encoding="utf-8") as fh:
                    report = json.load(fh)
                with open(os.path.join(inst, "instance.json"), encoding="utf-8") as fh:
                    fields = json.load(fh)
            except (OSError, ValueError):
                continue
            key = os.path.basename(cand)
            prev = found.get(key)
            # The same candidate can be built twice (an earlier epoch stopped and a later one
            # rebuilt it); both directories are valid. The newer report wins, and the manifest
            # records that it happened.
            if prev and os.path.getmtime(prev["report_path"]) >= os.path.getmtime(rep):
                prev["duplicates"] += 1
                continue
            found[key] = {
                "id": key,
                "dir": inst,
                "report_path": rep,
                "report": report,
                "fields": fields,
                "run": run,
                "duplicates": (prev or {}).get("duplicates", 0),
            }
    return [found[k] for k in sorted(found)]


def _digest(path: str) -> str:
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if not os.path.isfile(p):
            continue
        h.update(name.encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def consumer_interface(contract: dict) -> dict:
    """The verbs a consumer drives, read straight off the contract.

    Every entry names scripts the instance ships and arguments the contract declares. Nothing
    is guessed: if a family's `materialize` is three steps it is because its variant says so.
    """
    checks = contract.get("checks") or [contract["observable"]]
    si = contract.get("solver_input") or {}
    iface = {
        "prepare": {"description": "build the image and run setup.sh once, then snapshot it"},
        "materialize": {
            v: [{"verb": s["verb"], "args": dict(s.get("args") or {})} for s in spec["setup"]]
            for v, spec in contract["variants"].items()
        },
        "check": [
            {"verb": c["verb"], "reader": c.get("reader"), "select": c.get("select"), "compare": c.get("compare")}
            for c in checks
        ],
        "oracle": contract["oracle"],
    }
    if si:
        # HOW THE SUBMISSION REACHES THE INSTANCE. Two ways, and the difference is decided by
        # the GOLD VALUE, not by the verb -- the rule agentcore.recipes.probe_placement uses.
        #   replace: a setup step of that variant already applies the gold, so the submission
        #            takes its place (e.g. compiler flags, a test patch)
        #   on_top:  no setup step applies the gold, so the variant is built as it stands and
        #            the verb runs afterwards
        # Matching by VERB instead goes wrong when a buggy side applies the commit's own test
        # with apply_patch while the answer is a fix patch: a verb match would replace the test.
        gold = str((si.get("probe_values") or [""])[0])
        placement, step_verb, step_arg = "on_top", si["verb"], si["arg"]
        for st in contract["variants"][si["variant"]]["setup"]:
            for a, v in (st.get("args") or {}).items():
                if str(v) == gold:
                    placement, step_verb, step_arg = "replace", st["verb"], a
                    break
            if placement == "replace":
                break
        iface["submit"] = {
            "verb": step_verb,
            "arg": step_arg,
            "variant": si["variant"],
            "placement": placement,
            "reference_answers": list(si.get("probe_values") or []),
        }
    m = contract.get("measure") or {}
    if m:
        iface["measure"] = m
    # THE CHECK VERB MAY NEED AN INPUT, AND SOMETHING HAS TO MAKE IT. boca's `run` takes
    # <dir> <input> and its inputs come from `gen_inputs`, declared as measure.inputs. A first
    # version of this interpreter skipped any argument called `input`, so run.sh was handed an
    # empty path, printed nothing, and `judge` compared 0 bytes against 0 bytes and called the
    # instance sound. A pass on an empty observable is the vacuous verdict this project exists
    # to avoid, so the generator is part of the interface.
    if m.get("inputs"):
        iface["inputs"] = {"verb": m["inputs"], "dir": "/work/inputs"}
    return iface


def write_benchmark(
    run_dirs: list[str], contract_path: str, out_dir: str, *, name: str = "benchmark", log=print
) -> dict:
    with open(contract_path, encoding="utf-8") as fh:
        contract = json.load(fh)
    instances = collect(run_dirs)
    if not instances:
        raise SystemExit(f"no delivered instances under {run_dirs}")

    os.makedirs(os.path.join(out_dir, "instances"), exist_ok=True)
    packaged = []
    for rec in instances:
        dst = os.path.join(out_dir, "instances", rec["id"])
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(rec["dir"], dst)
        packaged.append(
            {
                "id": rec["id"],
                "fields": rec["fields"],
                "files": sorted(os.listdir(dst)),
                "digest": _digest(dst),
                "harness_verdict_ok": bool(rec["report"].get("ok")),
                "rows": {r.get("name"): r.get("ok") for r in (rec["report"].get("rows") or [])},
                "from_run": rec["run"],
                "rebuilt_from_an_earlier_epoch": rec["duplicates"] > 0,
            }
        )

    iface = consumer_interface(contract)
    with open(os.path.join(out_dir, "contract.json"), "w", encoding="utf-8") as fh:
        json.dump({"interface": iface, "contract": contract}, fh, indent=1)

    manifest = {
        "name": name,
        "instances": packaged,
        "count": len(packaged),
        "runs": list(run_dirs),
        "contract": os.path.abspath(contract_path),
        "packaged_at": time.time(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    bench = os.path.join(out_dir, "bench")
    with open(bench, "w", encoding="utf-8") as fh:
        fh.write(INTERPRETER)
    os.chmod(bench, 0o755)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme(name, contract, iface, packaged))

    log(f"{name}: {len(packaged)} instance(s) -> {out_dir}")
    log(f"  variants: {', '.join(iface['materialize'])}")
    log(f"  checks:   {', '.join(c['verb'] + '.sh' for c in iface['check'])}")
    log(f"  oracle:   {iface['oracle']['kind']}")
    log(f"  try:      {out_dir}/bench list")
    return manifest


def readme(name: str, contract: dict, iface: dict, packaged: list) -> str:
    o = iface["oracle"]
    if o["kind"] == "fail-to-pass":
        rule = f"An instance holds when the check FAILS on `{o['fails_on']}` and PASSES on `{o['passes_on']}`."
    else:
        rule = "An instance holds when every variant produces the identical observable."
    lines = [
        f"# {name}",
        "",
        f"{len(packaged)} instances. Every one was built by an agent from a real "
        f"repository and verified by the harness before delivery; `manifest.json` "
        f"carries each one's verdict row by row.",
        "",
        "## The rule",
        "",
        rule,
        "",
        "## Driving it",
        "",
        "```",
        "./bench list",
        "./bench prepare <id>",
    ]
    for v in iface["materialize"]:
        lines.append(f"./bench materialize <id> {v}")
    lines.append("./bench check <id> <variant>")
    if iface.get("submit"):
        lines.append("./bench submit <id> <answer>")
    if iface.get("measure"):
        lines.append("./bench measure <id>")
    lines += [
        "```",
        "",
        "## What each verb runs",
        "",
        "| verb | scripts, in order |",
        "|---|---|",
        "| `prepare` | `docker build` on the instance's Dockerfile, then `setup.sh` |",
    ]
    for v, steps in iface["materialize"].items():
        seq = " ; ".join(s["verb"] + ".sh " + " ".join(str(a) for a in s["args"].values()) for s in steps)
        lines.append(f"| `materialize {v}` | `{seq}` |")
    for c in iface["check"]:
        how = c["reader"] or "raw output, compared byte for byte"
        sel = f", selecting `{c['select']}`" if c.get("select") else ""
        lines.append(f"| `check` | `{c['verb']}.sh` read as {how}{sel} |")
    if iface.get("submit"):
        s = iface["submit"]
        lines.append(f"| `submit` | `{s['verb']}.sh` on the `{s['variant']}` side as its `{s['arg']}` |")
        lines += [
            "",
            "Reference answers shipped with every instance: "
            + ", ".join(f"`{p}`" for p in s["reference_answers"])
            + ". The first should pass, the second should not.",
        ]
    lines += [
        "",
        "## Provenance",
        "",
        "`contract.json` holds the interface this CLI interprets. It was derived from "
        "the benchmark's specification during formalization, not written by hand and "
        "not inferred at packaging time -- so this artifact is reproducible: the same "
        "instances and the same contract produce the same directory.",
    ]
    return "\n".join(lines) + "\n"


INTERPRETER = r'''#!/usr/bin/env python3
"""Drive one benchmark instance. Reads contract.json; contains no task logic of its own.

Needs python3 and the docker CLI. Nothing else, and nothing from the project that built it.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH, WORK, VARIANTS = "/bench", "/work", "/work/variants"


def load():
    with open(os.path.join(HERE, "contract.json")) as fh:
        doc = json.load(fh)
    with open(os.path.join(HERE, "manifest.json")) as fh:
        man = json.load(fh)
    return doc["interface"], doc["contract"], man


def idir(i):
    return os.path.join(HERE, "instances", i)


def fields(i):
    with open(os.path.join(idir(i), "instance.json")) as fh:
        return json.load(fh)


def flat(val):
    """A field as text a shell or a selector can use (a list becomes space-separated words)."""
    if isinstance(val, (list, tuple)):
        return " ".join(flat(x) for x in val)
    if isinstance(val, dict):
        return " ".join(flat(x) for x in val.values())
    return str(val)


def fill(v, f):
    s = str(v)
    for k, val in f.items():
        s = s.replace("{%s}" % k, flat(val))
    return s


def interp(i, script):
    """The interpreter the script's own shebang names, else sh."""
    p = os.path.join(idir(i), script)
    try:
        first = open(p, errors="replace").readline()
    except OSError:
        return ["sh"]
    return first[2:].strip().split() if first.startswith("#!") else ["sh"]


def tag(i):
    return "bench-%s" % "".join(c if c.isalnum() else "-" for c in i.lower())[:100]


def cname(i):
    return "benchrun-%s" % "".join(c if c.isalnum() else "-" for c in i.lower())[:80]


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, errors="replace", **kw)


def dexec(i, argv, cwd=WORK, timeout=3600):
    return sh(["docker", "exec", "-w", cwd, cname(i)] + argv, timeout=timeout)


def prepare(i):
    d = idir(i)
    print("building the image (Dockerfile only, no scripts in it)")
    # the Dockerfile is built on its own, exactly as the harness built it
    import tempfile, shutil as _sh
    with tempfile.TemporaryDirectory() as t:
        _sh.copy(os.path.join(d, "Dockerfile"), os.path.join(t, "Dockerfile"))
        r = sh(["docker", "build", "-q", "-t", tag(i), t])
    if r.returncode:
        print(r.stderr[-3000:]); return 1
    sh(["docker", "rm", "-f", cname(i)])
    r = sh(["docker", "run", "-d", "--name", cname(i), "-w", WORK, tag(i), "sleep", "infinity"])
    if r.returncode:
        print(r.stderr[-2000:]); return 1
    sh(["docker", "cp", d + "/.", cname(i) + ":" + BENCH])
    print("running setup.sh")
    r = dexec(i, interp(i, "setup.sh") + [BENCH + "/setup.sh"], cwd=BENCH)
    sys.stdout.write(r.stdout[-2000:]); sys.stderr.write(r.stderr[-2000:])
    print("prepare:", "ok" if r.returncode == 0 else "FAILED exit %d" % r.returncode)
    return r.returncode


def vdir(v):
    return VARIANTS + "/" + v


def fresh(v):
    """A directory no earlier build used, for `judge` and `submit`.

    The harness never rebuilds a variant where one already stands: a submission is built in a
    scratch directory. Rebuilding in place made a checkout done with `git worktree add` exit 128
    ("already exists") the second time, and a correct answer read as "it did not build"."""
    return "%s/%s.%d" % (VARIANTS, v, int(time.time() * 1000000))


def outcome(verdicts):
    """One side's outcome over all of its checks, by the harness's rule (agentcore.recipes
    _read_outcome): mixed or absent if any check reads so, else fail if ANY check fails, else
    pass. A check that passes on a side (such as the whole suite) does not override a failing one."""
    vs = [v for _, v in verdicts]
    for k in ("mixed", "absent", "fail"):
        if k in vs:
            return k
    return "pass" if vs and all(v == "pass" for v in vs) else next((v for v in vs if v != "pass"), "absent")


def materialize(i, variant, override=None, into=None):
    iface, contract, _ = load()
    if variant not in iface["materialize"]:
        print("no such variant:", variant, "have:", ", ".join(iface["materialize"])); return 2
    f = fields(i)
    dexec(i, ["mkdir", "-p", VARIANTS], timeout=60)
    for step in iface["materialize"][variant]:
        verb, args = step["verb"], dict(step["args"])
        if override and override[0] == verb:
            args[override[1]] = override[2]
        argv = interp(i, verb + ".sh") + [BENCH + "/" + verb + ".sh"]
        binding = {"dir": into or vdir(variant), **args}
        for a in contract["verbs"][verb]["args"]:
            if a not in binding:
                print("verb %s needs %s and nothing binds it" % (verb, a)); return 2
            argv.append(fill(binding[a], f))
        r = dexec(i, argv)
        if r.returncode:
            print("%s.sh exited %d building %r" % (verb, r.returncode, variant))
            sys.stderr.write((r.stdout + r.stderr)[-3000:]); return r.returncode
    print("materialized %s at %s" % (variant, into or vdir(variant)))
    return 0


def read_junit(text, select):
    """Outcome of the selected test(s) in a JUnit report: 'pass', 'fail', or 'absent'.

    Parsed as XML, so a <failure> nested in a <testcase> is always seen.
    """
    import re
    import xml.etree.ElementTree as ET
    wanted = [w.strip() for w in str(select).replace(",", " ").split() if w.strip()]

    def leaf(tok):
        """The test's own name inside a selector, whatever notation named it.

        A pytest nodeid (`tests/a/b_test.py::TheClass::the_test`) and a JUnit report
        (classname="tests.a.b_test.TheClass" name="the_test") spell one test differently, so the
        leaf name is compared; JUnit's `name` is that leaf for pytest, Maven Surefire and Gradle.
        """
        return tok.replace("::", "#").replace("/", ".").split("#")[-1].strip()

    leaves = {leaf(w) for w in wanted if leaf(w)}

    cases = []
    try:
        # several <testsuite> roots are wrapped in one element; an XML declaration is dropped first
        body = text[text.find("<"):]
        body = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", body)
        if body.count("<testsuite") > 1:
            body = "<all>" + body + "</all>"
        root = ET.fromstring(body)
        for tc in root.iter("testcase"):
            bad = bool(list(tc.iter("failure")) or list(tc.iter("error")))
            cases.append((tc.get("classname") or "", tc.get("name") or "", bad))
    except (ET.ParseError, ValueError):
        # an unparseable report is not a passing one
        if "<testcase" not in text:
            return "absent"
        for m in re.finditer(r"<testcase\b[^>]*?(/>|>.*?</testcase>)", text, re.S):
            block = m.group(0)
            nm = re.search(r'name="([^"]*)"', block)
            cl = re.search(r'classname="([^"]*)"', block)
            cases.append((cl.group(1) if cl else "", nm.group(1) if nm else "",
                          "<failure" in block or "<error" in block))

    seen = failed = False
    for cls, name, bad in cases:
        ident = "%s#%s" % (cls, name)
        if wanted:
            hit = (name in leaves
                   or any(w in ident or w in name for w in wanted)
                   or leaf(ident) in leaves)
            if not hit:
                continue
        seen = True
        failed = failed or bad
    return "absent" if not seen else ("fail" if failed else "pass")


def ensure_inputs(i):
    """Run the declared input generator once, and return the input files it made.

    Without this the check verb is handed an empty path: boca's run.sh then prints nothing and
    two empty observables compare equal, which reads as a pass and means nothing.
    """
    iface, contract, _ = load()
    spec = iface.get("inputs")
    if not spec:
        return []
    d = spec["dir"]
    r = dexec(i, ["sh", "-c", "ls -1 %s 2>/dev/null | head -50" % d], timeout=60)
    names = [x for x in r.stdout.split() if x]
    if not names:
        dexec(i, ["mkdir", "-p", d], timeout=60)
        verb = spec["verb"]
        argv = interp(i, verb + ".sh") + [BENCH + "/" + verb + ".sh", d]
        g = dexec(i, argv)
        if g.returncode:
            print("%s.sh exited %d generating inputs" % (verb, g.returncode))
            sys.stderr.write((g.stdout + g.stderr)[-2000:])
            return []
        r = dexec(i, ["sh", "-c", "ls -1 %s 2>/dev/null | head -50" % d], timeout=60)
        names = [x for x in r.stdout.split() if x]
    return [d + "/" + n for n in names]


def check(i, variant, quiet=False, into=None):
    iface, contract, _ = load()
    f = fields(i)
    verdicts = []
    inputs = ensure_inputs(i)
    for c in iface["check"]:
        verb = c["verb"]
        needs_input = "input" in contract["verbs"][verb]["args"]
        if needs_input and not inputs:
            print("  %s.sh needs an input and none could be generated" % verb)
            verdicts.append((verb, "NO INPUT")); continue
        argv = interp(i, verb + ".sh") + [BENCH + "/" + verb + ".sh"]
        binding = {"dir": into or vdir(variant)}
        if needs_input:
            binding["input"] = inputs[-1]      # the costliest one the generator made
        for a in contract["verbs"][verb]["args"]:
            if a not in binding:
                binding[a] = ""
            argv.append(fill(binding[a], f))
        r = dexec(i, argv)
        if c["reader"] == "exit-code":
            v = "pass" if r.returncode == 0 else "fail"
        elif c["reader"] == "junit-xml":
            v = read_junit(r.stdout, fill(c.get("select") or "", f))
        else:
            v = r.stdout            # raw: the observable itself, compared across variants
        verdicts.append((verb, v))
        if not quiet:
            shown = v if c["reader"] else ("%d bytes" % len(r.stdout))
            print("  %-14s %s" % (verb + ".sh", shown))
    return verdicts


def judge(i):
    iface, _, _ = load()
    o = iface["oracle"]
    print("instance:", i)
    if o["kind"] == "fail-to-pass":
        out = {}
        for side in (o["fails_on"], o["passes_on"]):
            d = fresh(side)
            if materialize(i, side, into=d):
                return 1
            print("check on %s:" % side)
            out[side] = check(i, side, into=d)
        ok = outcome(out[o["fails_on"]]) == "fail" and outcome(out[o["passes_on"]]) == "pass"
        print("VERDICT:", "holds" if ok else "does NOT hold",
              "(must fail on %s, pass on %s)" % (o["fails_on"], o["passes_on"]))
        return 0 if ok else 1
    ref = None
    for v in iface["materialize"]:
        d = fresh(v)
        if materialize(i, v, into=d):
            return 1
        got = check(i, v, quiet=True, into=d)
        text = "".join(str(x) for _, x in got)
        print("  %-12s %d bytes of output" % (v, len(text)))
        if not text.strip():
            # two variants that print nothing would compare equal; that is not a verdict
            print("VERDICT: cannot judge -- %s produced no observable output" % v)
            return 2
        if ref is None:
            ref = text
        elif text != ref:
            print("VERDICT: does NOT hold -- %s differs from the reference" % v); return 1
    print("VERDICT: holds (every variant produced the identical observable)")
    return 0


def submit(i, answer):
    """Judge a submission the way THIS benchmark's oracle says, not one way for all of them.

    A first version asked whether every check came back "pass". That is the fail-to-pass
    reading. Under reference-equivalence the reader is null and a check returns the observable
    itself, so nothing was ever equal to "pass" and a correct answer (-O3 on boca) was
    rejected alongside a broken one -- a judge that fails everything looks identical to a
    strict one and is worth nothing.
    """
    iface, _, _ = load()
    s = iface.get("submit")
    if not s:
        print("this benchmark declares no solver input"); return 2
    o = iface["oracle"]

    reference = None
    if o["kind"] != "fail-to-pass":
        # capture what the untouched side produces BEFORE the submission replaces its argument
        ref_variant = o.get("reference") or s["variant"]
        d = fresh(ref_variant)
        if materialize(i, ref_variant, into=d):
            return 1
        reference = "".join(str(x) for _, x in check(i, ref_variant, quiet=True, into=d))
        if not reference.strip():
            print("cannot judge: the reference side produced no output"); return 2
        print("reference (%s): %d bytes" % (ref_variant, len(reference)))

    place = s.get("placement", "replace")
    print("applying %r through %s.sh on the %s side (%s)"
          % (answer, s["verb"], s["variant"], place))
    sub = fresh(s["variant"])
    if place == "replace":
        if materialize(i, s["variant"], override=(s["verb"], s["arg"], answer), into=sub):
            print("SUBMISSION: rejected (it did not build)"); return 1
    else:
        if materialize(i, s["variant"], into=sub):
            print("SUBMISSION: rejected (the variant did not build)"); return 1
        _, contract, _ = load()
        f = fields(i)
        argv = interp(i, s["verb"] + ".sh") + [BENCH + "/" + s["verb"] + ".sh"]
        binding = {"dir": sub, s["arg"]: answer}
        for a in contract["verbs"][s["verb"]]["args"]:
            if a not in binding:
                print("verb %s needs %s and nothing binds it" % (s["verb"], a)); return 2
            argv.append(fill(binding[a], f))
        r = dexec(i, argv)
        if r.returncode:
            print("%s.sh exited %d applying the submission" % (s["verb"], r.returncode))
            sys.stderr.write((r.stdout + r.stderr)[-2000:])
            print("SUBMISSION: rejected (it could not be applied)"); return 1
    got = check(i, s["variant"], into=sub)

    if o["kind"] == "fail-to-pass":
        ok = outcome(got) == "pass"
        why = "the check passes where it used to fail" if ok else "the check does not pass"
    else:
        text = "".join(str(x) for _, x in got)
        ok = bool(text.strip()) and text == reference
        why = ("the output is identical to the reference" if ok
               else "no output at all" if not text.strip()
               else "the output differs from the reference")
    print("SUBMISSION:", "accepted" if ok else "rejected", "--", why)
    if ok and iface.get("measure"):
        print("note: this benchmark also scores runtime; `measure` is the timed comparison "
              "and is not part of this verdict")
    return 0 if ok else 1


def main():
    iface, contract, man = load()
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("verbs: list | show <id> | prepare <id> | materialize <id> <variant> | "
              "check <id> <variant> | judge <id>" + (" | submit <id> <answer>" if iface.get("submit") else ""))
        print("variants:", ", ".join(iface["materialize"]))
        if iface.get("submit"):
            print("reference answers:", ", ".join(str(x) for x in iface["submit"]["reference_answers"]))
        return 0
    cmd = a[0]
    if cmd == "list":
        # the whole id is printed: it is what every other verb takes
        for r in man["instances"]:
            print("%s  %s" % ("ok" if r["harness_verdict_ok"] else " ?", r["id"]))
        print("%d instances" % man["count"]); return 0
    if cmd == "verbs":
        print(json.dumps(iface, indent=1)); return 0
    if len(a) < 2:
        print("need an instance id"); return 2
    i = a[1]
    if not os.path.isdir(idir(i)):
        print("no such instance:", i); return 2
    if cmd == "show":
        print(json.dumps(fields(i), indent=1)); return 0
    if cmd == "prepare":
        return prepare(i)
    if cmd == "materialize":
        return materialize(i, a[2])
    if cmd == "check":
        check(i, a[2]); return 0
    if cmd == "judge":
        return judge(i)
    if cmd == "submit":
        return submit(i, a[2])
    print("unknown verb:", cmd); return 2


if __name__ == "__main__":
    sys.exit(main())
'''
