"""Structured result reports: the harness reads what a tool RECORDED, by declared kind.

The framework must not know any task's oracle (tests/test_worklist.py pins this), so
nothing here is keyed on a build system or a language. A contract declares a report
KIND as data; this module implements the readers for the kinds the harness supports
and dispatches on the string. Adding a kind is adding a reader, not teaching the
framework about a task.

A check that runs a test framework must be judged by what the framework recorded --
Surefire and Gradle write JUnit XML (target/surefire-reports/TEST-*.xml,
build/test-results/**/*.xml), pytest writes it with --junitxml. The XML carries
tests/failures/errors counts and every test's name, and it separates <failure>
(an assertion held or broke: a real verdict) from <error> (the harness or
environment broke: NO verdict). Exit codes and log regexes are not verdicts: an exit code can be constant, and a
missing-tool message can match a log pattern.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class TestResults:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    failed_names: list = field(default_factory=list)
    error_names: list = field(default_factory=list)
    suites: int = 0

    @property
    def conclusive(self) -> bool:
        """A verdict exists: at least one test really ran and none died on
        infrastructure. <error> entries mean the run proves nothing either way."""
        return self.tests > 0 and self.errors == 0

    def describe(self) -> str:
        names = ", ".join(self.failed_names[:5]) or "-"
        errs = ", ".join(self.error_names[:5]) or "-"
        return (
            f"{self.tests} test(s) ran across {self.suites} suite(s): "
            f"{self.failures} failed ({names}), {self.errors} errored ({errs}), "
            f"{self.skipped} skipped"
        )


def parse_junit(xml_text: str) -> TestResults | None:
    """One JUnit XML document -> TestResults; None if it is not JUnit XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # report files arrive via `cat` with log lines around them; trim to the
        # last closing suite tag and retry before giving up
        for tag in ("</testsuites>", "</testsuite>"):
            end = xml_text.rfind(tag)
            if end >= 0:
                try:
                    root = ET.fromstring(xml_text[: end + len(tag)])
                    break
                except ET.ParseError:
                    continue
        else:
            return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites and root.tag != "testsuites":
        return None
    r = TestResults(suites=len(suites))
    for s in suites:
        for case in s.iter("testcase"):
            r.tests += 1
            name = f"{case.get('classname', '')}#{case.get('name', '')}".strip("#")
            if case.find("failure") is not None:
                r.failures += 1
                r.failed_names.append(name)
            elif case.find("error") is not None:
                r.errors += 1
                r.error_names.append(name)
            elif case.find("skipped") is not None:
                r.skipped += 1
    return r


def _parse_multidoc_chunk(body: str) -> TestResults | None:
    """Several XML documents that landed in ONE chunk -- because only the first file
    carried an <?xml prolog for the line-based splitter to cut on.

    Surefire writes its report with trailing XML comments after the root close, and its
    junitreports/ twin ships no prolog, so a cat over both is two <testsuite> roots in one
    chunk. ET refuses ('junk after document element') and the trim-to-last-close retry still
    holds two roots. Comments and prologs are stripped (legal only at document edges), everything is wrapped in a
    synthetic root, and every testsuite in the forest is counted."""
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"<\?xml[^>]*\?>", "", body)
    body = re.sub(r"<!DOCTYPE[^>]*>", "", body)
    for tag in ("</testsuites>", "</testsuite>"):
        end = body.rfind(tag)
        if end >= 0:
            body = body[: end + len(tag)]
            break
    else:
        return None
    first = body.find("<")
    if first < 0:
        return None
    try:
        root = ET.fromstring("<benchstream>" + body[first:] + "</benchstream>")
    except ET.ParseError:
        return None
    suites = list(root.iter("testsuite"))
    if not suites:
        return None
    r = TestResults(suites=len(suites))
    for s in suites:
        for case in s.iter("testcase"):
            r.tests += 1
            name = f"{case.get('classname', '')}#{case.get('name', '')}".strip("#")
            if case.find("failure") is not None:
                r.failures += 1
                r.failed_names.append(name)
            elif case.find("error") is not None:
                r.errors += 1
                r.error_names.append(name)
            elif case.find("skipped") is not None:
                r.skipped += 1
    return r


def parse_junit_stream(text: str) -> TestResults | None:
    """Concatenated XML documents (a `cat` over report files) -> merged TestResults.

    Splits on XML prologs / root openings so `cat target/surefire-reports/TEST-*.xml`
    parses as a whole; returns None when no document parses."""
    chunks, current = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("<?xml") and current:
            chunks.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    merged: TestResults | None = None
    for ch in chunks:
        start = ch.find("<")
        if start < 0:
            continue
        r = parse_junit(ch[start:])
        if r is None:
            r = _parse_multidoc_chunk(ch[start:])
        if r is None:
            continue
        if merged is None:
            merged = r
        else:
            merged.tests += r.tests
            merged.failures += r.failures
            merged.errors += r.errors
            merged.skipped += r.skipped
            merged.failed_names += r.failed_names
            merged.error_names += r.error_names
            merged.suites += r.suites
    return merged


# ── report DISCOVERY: composed from DECLARED patterns, never known here ──────
#
# WHERE a build tool writes its reports is an ecosystem convention, so it is not hardcoded
# here: the spec DECLARES its discovery patterns (`report_discovery` in the
# instance -- {kind: [find patterns]}); this module keeps only the mechanics: how a
# find is composed, the freshness marker, the flood bound, the truncation marker,
# and the parsing of the declared FORMAT. A contract that declares a report kind
# without discovery patterns fails closed at the call site.

# A pattern is spec content but lands in a shell word: keep it to glob characters.
_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9_*?./\[\]-]{1,120}$")


def find_expression(patterns) -> str:
    """The find(1) name/path expression for the DECLARED patterns, or "" when none
    are declared or any is unsafe to place in a shell word. A pattern containing a
    '/' matches the whole path (-path); a bare filename pattern matches the name."""
    pats = [str(p).strip() for p in (patterns or []) if str(p).strip()]
    if not pats or any(not _SAFE_PATTERN.match(p) for p in pats):
        return ""
    terms = []
    for p in pats:
        op = "-path" if "/" in p else "-name"
        pat = p if (op == "-name" or p.startswith(("*", "/", "."))) else "*" + p
        terms.append(f"{op} '{pat}'")
    return "\\( " + " -o ".join(terms) + " \\)"


def compose_dump_command(patterns) -> str:
    """The command a gate appends after a check run to surface the reports the spec
    declares, wherever the build tool put them. "" when no patterns are declared.

    The 50-file bound stays (a pathological tree must not flood the reader) but it
    does not bind in silence: a multi-module build with more reports than that would
    have its counts merged from a subset and read as the whole. The marker
    is an XML comment so it cannot disturb the verdict it is attached to. Only
    files NEWER than /tmp/.bench_check_started -- touched just before the check --
    can witness the run."""
    expr = find_expression(patterns)
    if not expr:
        return ""
    return (
        f"find . -maxdepth 12 {expr} "
        "-newer /tmp/.bench_check_started 2>/dev/null > /tmp/.bench_reports.list; "
        "head -50 /tmp/.bench_reports.list | xargs cat 2>/dev/null; "
        "_n=$(wc -l < /tmp/.bench_reports.list); "
        '[ "$_n" -gt 50 ] && printf \'\\n<!-- BENCH_REPORTS_TRUNCATED: %s report files '
        'found, 50 read -->\\n\' "$_n"; true'
    )


def compose_stale_command(patterns) -> str:
    """The SAME declared patterns without the freshness filter, each hit with its
    mtime: the diagnostic a no-report refusal attaches so its reader can tell a
    discovery miss from a runner miss."""
    expr = find_expression(patterns)
    if not expr:
        return ""
    return f"find . -maxdepth 12 {expr} -printf '%TF %TT %p\\n' 2>/dev/null | head -8; true"


def discovery_patterns(instance, kind: str) -> list:
    """The declared discovery patterns for this report kind, from the instance's
    `report_discovery` block. [] when the instance declares none -- the caller
    fails closed with a message naming the missing declaration."""
    block = (instance or {}).get("report_discovery") or {}
    if not isinstance(block, dict):
        return []
    pats = block.get(str(kind or "").strip().lower())
    return [str(p) for p in pats] if isinstance(pats, list) else []


# ── the declared-kind front door ─────────────────────────────────────────────


def _read_xml_report(text: str):
    return parse_junit_stream(text)


# Kind names are CONTRACT DATA (string keys), never behaviour the framework assumes.
_READERS = {"junit-xml": _read_xml_report}


def supported_kinds() -> list:
    return sorted(_READERS)


def parse_report(kind: str, text: str):
    """Parsed results for a declared report kind, or None if the kind is unknown or
    the text carries no such report."""
    reader = _READERS.get(str(kind or "").strip().lower())
    return reader(text or "") if reader else None


def dump_command(kind: str, patterns=None) -> str:
    """The command that surfaces this kind of report after a run, composed from the
    DECLARED discovery patterns; "" when the kind is unknown or none are declared
    (distinguish at the call site via `supported_kinds()`)."""
    if str(kind or "").strip().lower() not in _READERS:
        return ""
    return compose_dump_command(patterns)


# FIX THE RUN, NEVER THE REPORT. A check script that ends with a sed rewriting <error> into
# <failure> and errors="1" into errors="0" in its own TEST-*.xml would launder an
# environment error into "bug reproduced". Where a variant is judged on its report, a check
# that edits that report is forging the evidence. The verifier stage applies the same
# patterns (benchcompiler/verifier_contract).
def _report_file_regex(patterns) -> str:
    """The file-name vocabulary of the DECLARED discovery patterns, as one regex
    alternation: each pattern's path components (globs translated), so the same
    declaration that finds the reports also protects them. "" when none declared."""
    alts = []
    for p in patterns or []:
        for comp in str(p or "").split("/"):
            comp = comp.strip()
            # Any all-star component ("*", "**") contributes no vocabulary. Skipping only
            # the exact "*" let "**" through as `[^\s'"]*[^\s'"]*`, which matches the EMPTY
            # string -- so the alternation matched anywhere and every ordinary redirect
            # ("mvn test > /tmp/build.log", even a bare "2>&1") was accused of forging the
            # report. This is the ONE home for the rule; benchcompiler imports it.
            if not comp or set(comp) <= {"*"}:
                continue
            rx = re.escape(comp).replace(r"\*", r"[^\s'\"]*").replace(r"\?", r"[^\s'\"]")
            if rx not in alts:
                alts.append(rx)
    return "(" + "|".join(alts) + ")" if alts else ""


#: A model told that cases "ended in an error rather than a verdict" may append a python
#: one-liner that rewrites `<error>` to `<failure>`, sets failures="1" and zeroes errors="N"
#: across its own TEST-*.xml. That is `cp /bin/true` aimed at the report instead of the binary.
#:
#: THIS IS A BLOCKLIST AND BLOCKLISTS ARE WEAK. A rewrite phrased another way gets past it.
#: It is worth having because it names a known attack and makes it cost a round; the
#: defence that actually works is structural and needs no list -- the harness reads the
#: reports and decides.
# Format-tier tampering shapes: reaching for the attributes the junit-xml verdict
# rule reads, whatever the file is called. These stay whether or not discovery
# patterns are declared -- they are properties of the FORMAT, not of any ecosystem.
_ATTRIBUTE_EDITS = (
    r"(failures|errors)=\\?['\"][0-9]",
    r"<\s*/?\s*error\s*[ >].{0,80}<\s*/?\s*failure",
)


def report_tampering(command: str, patterns=None) -> str:
    """Why this check edits the report it is judged on, or "". Never raises.

    FIX THE RUN, NEVER THE REPORT: where a variant is judged on its report, a check
    that edits that report (for example a sed rewriting <error> into <failure> in its
    own TEST-*.xml) is forging the evidence.
    The file-name vocabulary comes from the DECLARED discovery patterns; the
    attribute shapes are format-tier and always on."""
    text = str(command or "")
    report_file = _report_file_regex(patterns)
    shapes = list(_ATTRIBUTE_EDITS)
    if report_file:
        shapes = [
            # sed -i / perl -pi over a report path
            rf"(sed\s+-i|perl\s+-[a-z]*i[a-z]*\s)[^;&|]*{report_file}",
            # sed -i handed report files by find -exec (the path is in
            # the find, the sed only sees `{{}}`)
            rf"find[^;&|]*{report_file}[^;&|]*-exec[^;&|]*(sed\s+-i|perl\s+-[a-z]*i[a-z]*\s)",
            # python open(..., 'w') / .write over a report path
            rf"open\([^)]*['\"]w['\"][^)]*\)[^;]*{report_file}",
            rf"{report_file}[^;]*open\([^)]*['\"]w['\"]",
            # a redirect that lands ON a report file
            rf">\s*[^\s;&|]*{report_file}",
        ] + shapes
    for pattern in shapes:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return (
                f"rewrites the test report it is judged on ({m.group(0)[:70]!r}). "
                f"Copy a report wherever you like; changing what it says is forging "
                f"the evidence, and the harness reads the report to decide"
            )
    return ""
