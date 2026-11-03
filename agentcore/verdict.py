"""What is this check's verdict? Asked once, here.

MIRROR of `benchcompiler/verdict.py` -- the two packages deliberately do not import each
other, the same way reports.py mirrors junitxml.py. The rule and its cases live in
`benchcompiler/verdict_cases.py`, and
`benchcompiler/tests/test_the_verdict_rule_is_one_rule.py` runs BOTH implementations against
every case, so a divergence between the runtime that builds instances and the packager that
ships them is a red test rather than a pool that fails a gate it already passed.

WHY THIS EXISTS. The question must have one answer: a check whose contract says its REPORT
decides must not be judged on its EXIT CODE. A buggy variant may declare `expect: pass`
(its script exits 0 when it observes the bug) alongside `expect_failures: 1` (its report
shows the triggering test failing), so scripts legitimately write `mvn test … || true` --
and a reader taking the exit code as the answer would read a constant.

THE RULE.
  * A contract naming a report kind (`results:`) is judged on its REPORT and nothing else.
    No report, or a report with no cases, is not a verdict. Cases that ERRORED are not
    verdicts either. `expect_failures` is a FLOOR: a variant asked to show the defect must
    show at least that many failing cases; a variant asked for a clean run must show none.
  * A contract naming no report kind is judged on its EXIT CODE, polarity from `expect:`.
  * The channel the contract did not name never decides. When the two point different ways
    the basis says so.
"""

from __future__ import annotations

REPORT_KINDS = ("junit-xml",)


def report_kind_of(contract) -> str:
    """The report kind this contract declares, or "" when it declares none."""
    kind = str((contract or {}).get("results") or "").strip().lower()
    return kind if kind in REPORT_KINDS else ""


def want_failures_of(contract) -> int:
    """How many failing cases this variant's report must carry, as a FLOOR.

    It is not an exact count: an instance whose bug trips more than one test is valid.
    """
    declared = (contract or {}).get("expect_failures")
    if str(declared).lstrip("-").isdigit():
        return int(declared)
    return 1 if str((contract or {}).get("expect") or "").strip().lower() == "fail" else 0


def wants_zero_exit(contract) -> bool:
    """Whether this check's SCRIPT is expected to exit 0. Not the report's question."""
    return str((contract or {}).get("expect") or "pass").strip().lower() != "fail"


def _parse(kind: str, text: str):
    from agentcore import reports

    return reports.parse_report(kind, text or "")


def verdict_of(contract, exit_code: int, report_text: str = "") -> tuple[bool, str]:
    """(ok, basis) for one check, judged on the channel its contract declares."""
    kind = report_kind_of(contract)
    if not kind:
        want_zero = wants_zero_exit(contract)
        got_zero = int(exit_code) == 0
        ok = got_zero == want_zero
        return ok, (
            f"exit {exit_code} (this check declares no report kind, so its exit "
            f"status is the verdict; it must exit "
            f"{'zero' if want_zero else 'nonzero'})"
        )

    res = _parse(kind, report_text)
    if res is None or res.tests == 0:
        return False, (
            f"no {kind} verdict: the run wrote no usable test report, so nothing "
            f"ran that could decide this. Exit {exit_code} is not a verdict about "
            f"a test run -- a report that does not exist is the commonest way a "
            f"test that never ran reads as a test that failed."
        )
    if res.errors > 0:
        return False, (
            f"INCONCLUSIVE: {res.errors} case(s) ended in an error rather than a "
            f"verdict ({', '.join(res.error_names[:5])}). An error is not a "
            f"failing case; fix the environment so the check genuinely runs."
        )

    want = want_failures_of(contract)
    said = res.describe()
    disagrees = (int(exit_code) == 0) != wants_zero_exit(contract)
    note = (
        f" -- the check exited {exit_code}, which is not the channel this variant is judged on (`results: {kind}`)"
        if disagrees
        else ""
    )

    if want >= 1 and res.failures < want:
        return False, (
            f"this variant must show the defect: its report has to carry at least "
            f"{want} failing case(s) and shows {res.failures} ({said}){note}"
        )
    if want == 0 and res.failures > 0:
        return False, (
            f"expected a clean report (expect_failures: 0) but it shows failures: "
            f"{', '.join(res.failed_names[:5])} ({said}){note}"
        )
    return True, (f"its {kind} report satisfies the contract (expect_failures: {want}, a floor): {said}{note}")
