"""Read the evidence the model recorded, however it chose to write it.

WHY THIS IS SHARED. Per-collector copies of "turn this field into a list" and "read the report
at this path" drift apart and each refuses correct answers in its own way, so one implementation
is shared by all collectors.

THE RULE THESE HELPERS ENCODE. Refuse only what is genuinely unresolvable. A validator's job is
to establish that a claim is TRUE, never that it is spelled canonically. Every refusal on record
that turned out to be expensive was the second thing wearing the costume of the first:

    e806e6b                                 a valid git prefix, refused for being short
    <sha>^                                  git's own notation for a parent, refused
    '["src/main/A.java"]'                   a correct list, refused for its quoting
    target/surefire-reports/TEST-*.xml      a correct location, refused for being relative

None of those weaken the guarantee, because the guarantee is enforced by going and LOOKING --
resolving against the real repository, reading the real file. A fabrication still fails there.

SAFETY IS NOT RELAXED. `read_reports` keeps the injection guard that the absolute-path rule was
really protecting: the charset excludes every shell metacharacter, and the path reaches `sh` as
a positional argument rather than being spliced into the command text. Only globbing is left
unquoted, which is the point.
"""

from __future__ import annotations

import html
import json
import os
import posixpath
import re
import shlex

# The charset of _SAFE_PATH plus `*`, and no leading-slash requirement (relative paths are
# resolved against the clone first). Spaces, ; $ ` ' " & | are all excluded.
SAFE_REPORT = re.compile(r"^[A-Za-z0-9_.*/-]{1,220}$")


def as_paths(value) -> list[str]:
    """A list of strings, however the model expressed it.

    A model may send `"src_files": "[\\"src/main/A.java\\"]"` -- a STRING
    containing JSON -- and a plain whitespace split turns that into `['["src/main/A.java"]']`,
    which then matches nothing in the real diff. Refusing that is refusing a correct answer for
    its packaging, and since repeated refusals feed the strike counter, it blocks good candidates.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(as_paths(item))
        return out
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return as_paths(json.loads(text))
        except Exception:  # noqa: BLE001 - not JSON after all; fall through to plain splitting
            text = text.strip("[]")
    return [p for p in (s.strip().strip("\"'") for s in re.split(r"[\s,]+", text)) if p]


def read_reports(runner, sandbox: str, report: str, clone: str = "") -> tuple[int, str, str]:
    """(code, text, resolved) -- the report(s) named, in any of the shapes that mean the same.

    `runner` is a callable (sandbox, command) -> (returncode, output), so this works against a
    per-candidate container or a plain subprocess in tests without knowing which it is.

    All of these are correct answers to "where is the report":

        /w/repo/target/surefire-reports/TEST-FooTest.xml    an absolute file
        target/surefire-reports/TEST-FooTest.xml            relative to the clone -- which is how
                                                            surefire prints its own output
        /w/repo/target/surefire-reports/                    the directory
        /w/repo/target/surefire-reports/TEST-*.xml          a glob, for when the generated
                                                            filename was never on screen

    Refusing at the oracle is the most expensive place to refuse anything: by the time it looks,
    the candidate has already been cloned, built, and run twice.
    """
    resolved = (report or "").strip()
    if resolved and not resolved.startswith("/") and clone:
        resolved = posixpath.join(clone, resolved)
    if not resolved.startswith("/") or not SAFE_REPORT.match(resolved):
        return 2, "", resolved
    # $1 unquoted so a glob expands; `if` rather than `[ -f ] && cat` because with the latter the
    # branch that does not apply is the loop's last command, so sh exits 1 and a perfectly
    # readable report reads as "could not read".
    # `$1/*`, not `$1/*.xml`: an XML-only glob would make JUnit/surefire the only readable
    # report format, leaving no path for a suite emitting TAP, JSON or a plain text
    # summary. What this function owns is READING what the caller pointed at; deciding what the bytes
    # mean is the validator's job.
    cmd = 'sh -c \'for f in $1 $1/*; do if [ -f "$f" ]; then cat -- "$f"; fi; done\' _ ' + shlex.quote(resolved)
    code, out = runner(sandbox, cmd)
    return code, out, resolved


def split_junit_documents(text: str) -> list[str]:
    """Several concatenated XML documents -> one per report.

    A directory or glob of surefire reports is a legitimate answer, and parsing only the first
    would undercount the tests that ran.
    """
    chunks = [c for c in re.split(r"(?=<\?xml)", text) if c.strip()]
    return chunks or ([text] if text.strip() else [])


# ── the tracker, as the harness observed it ─────────────────────────────────────────────────

# Recognised tracker URL shapes, rewritten to the API endpoint that answers in JSON. GitHub's
# issues endpoint answers for pull requests too (and tells them apart: a PR carries a
# `pull_request` key). Anything else is fetched as-is and read by its HTML <title>.
_GH_ITEM_URL = re.compile(r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)/(?:issues|pull)/(\d+)")
_JIRA_BROWSE_URL = re.compile(r"^(https?://[^\s/]+(?:/[^\s/]+)*?)/browse/([A-Z][A-Z0-9]{1,19}-\d{1,6})")


def is_pull_request_url(url: str) -> bool:
    """Does this URL's SHAPE alone say pull request? No fetch, no network, no cost.

    Here rather than in a validator because the grammar above is already this module's, and
    three copies of a URL regex is how a rule drifts. It answers only for the shape
    `_GH_ITEM_URL` recognises -- a JIRA browse link, a GitLab URL or anything else gets False
    and is judged by the live fetch, which is the only thing that can judge it.

    WHY IT EXISTS. `read_tracker_item` tells a PR from an issue after fetching,
    and `lead_is_corroborated` refuses it -- but that runs at DELIVERED, once a clone exists.
    The path segment says `/pull/`, so nothing has to be fetched (or cloned) to know.
    """
    m = _GH_ITEM_URL.match(str(url or "").strip())
    if not m:
        return False
    return "/pull/" in m.group(0).lower()


def _read_tracker_item(url: str, runner) -> tuple[dict | None, str]:
    """Fetch the tracker item a model-written URL points at, and say what it IS.

    A URL-shaped field proves nothing: `issue_url` sits in a state's `requires`, so it is
    always present and always URL-shaped, and two collectors independently grew a branch that
    accepted the SHAPE as evidence -- which accepted every commit (confirmed on a feature, a
    refactor and a version bump). Whatever the criterion -- "is a bug", "is a reported issue"
    -- it must be decided from what the TRACKER says, so this helper does the fetching and
    normalising once, and each validator applies its own predicate.

    Returns (info, detail). info is None when the item could not be read -- the check has NOT
    run and the caller must say so rather than word it as a judgment. Otherwise info is:

        {"exists": bool,          False: the tracker answered "no such item"
         "kind":  "issue" | "pull" | "jira" | "page",
         "type":  str,            JIRA issue type ("Bug", "Improvement", ...), else ""
         "title": str, "labels": [str], "body": str}

    `runner(cmd) -> (code, output)` runs in the candidate's own sandbox, which has network --
    it cloned the repository from the same forge.
    """
    gh, jira = _GH_ITEM_URL.match(url), _JIRA_BROWSE_URL.match(url)
    if gh:
        fetch = f"https://api.github.com/repos/{gh[1]}/{gh[2]}/issues/{gh[3]}"
    elif jira:
        fetch = f"{jira[1]}/rest/api/2/issue/{jira[2]}?fields=summary,issuetype,labels"
    else:
        fetch = url
    q = shlex.quote(fetch)
    # AUTHENTICATE, on GitHub only.
    #
    # A bare `curl -sL` would spend GitHub's ANONYMOUS 60-requests-per-hour bucket on every
    # DELIVERED validation while the run's own shell holds a token, producing spurious
    # "could not reach the tracker" refusals.
    #
    # Only api.github.com. `fetch` is derived from a model-written URL and the fallback branch
    # passes that URL through verbatim, so an unconditional header would hand a GitHub
    # credential to whatever host a candidate names. And `$GITHUB_TOKEN` is expanded by the
    # sandbox shell rather than interpolated here: the command string is echoed into the
    # trajectory, which gets rendered and shared. Double quotes, because single ones send the
    # literal text.
    auth = ""
    if fetch.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"):
        auth = '-H "Authorization: Bearer $GITHUB_TOKEN" '
    wget_auth = '--header="Authorization: Bearer $GITHUB_TOKEN" ' if auth else ""
    code, out = runner(f"curl -sL {auth}--max-time 25 {q} 2>/dev/null || wget -qO- {wget_auth}-T 25 {q} 2>/dev/null")
    text = (out or "").strip()
    if code != 0 or not text:
        return None, f"could not fetch {fetch}"

    # JSON only when the answer really IS JSON. `"{" in text` would match any HTML page containing a
    # brace -- `<style>b{}</style>` parses as the empty object.
    doc = None
    stripped_text = text.lstrip()
    if stripped_text.startswith("{"):
        try:
            doc = json.loads(text[text.index("{") : text.rindex("}") + 1])
        except Exception:  # noqa: BLE001
            doc = None
    if not isinstance(doc, dict) or not doc:
        doc = None
    if isinstance(doc, dict) and (jira or "fields" in doc or "errorMessages" in doc):
        if doc.get("errorMessages"):
            msgs = "; ".join(map(str, doc["errorMessages"]))[:120]
            if "does not exist" in msgs.lower():
                return {"exists": False, "kind": "jira", "type": "", "title": "", "labels": [], "body": ""}, msgs
            return None, f"the tracker answered: {msgs}"
        f = doc.get("fields") or {}
        return {
            "exists": True,
            "kind": "jira",
            "type": str((f.get("issuetype") or {}).get("name") or ""),
            "title": str(f.get("summary") or ""),
            "labels": [str(x) for x in (f.get("labels") or [])],
            "body": "",
        }, "read from the tracker"
    if isinstance(doc, dict) and (gh or "title" in doc or "message" in doc):
        title = str(doc.get("title") or "")
        if not title:
            msg = str(doc.get("message") or "")
            if "rate limit" in msg.lower():
                return None, "the tracker is rate-limiting this address right now"
            if "not found" in msg.lower():
                return {
                    "exists": False,
                    "kind": "issue",
                    "type": "",
                    "title": "",
                    "labels": [],
                    "body": "",
                }, "the tracker has no item at that URL"
            return None, f"the tracker answered: {msg[:2000]!r}"
        labels = [str(l.get("name") or "").lower() for l in (doc.get("labels") or []) if isinstance(l, dict)]
        return {
            "exists": True,
            "kind": "pull" if "pull_request" in doc else "issue",
            "type": "",
            "title": title,
            "labels": labels,
            "body": str(doc.get("body") or "")[:4000],
        }, "read from the tracker"
    # A tracker this helper does not have an API shape for. GitHub and JIRA were the only two
    # that could ever produce a usable answer, and validators branch on `kind` -- so a project
    # on GitLab, Bugzilla or a mailing-list tracker was not merely unverified, it was
    # unprovable by construction. Everything below hands back CONTENT instead of a shrug, so a
    # validator's own predicate has something to run against.
    if isinstance(doc, dict):
        title = str(doc.get("title") or doc.get("summary") or doc.get("name") or "")
        labels = [str(x).lower() for x in (doc.get("labels") or doc.get("tags") or []) if isinstance(x, str)]
        return {
            "exists": True,
            "kind": "json",
            "type": str(doc.get("type") or ""),
            "title": title,
            "labels": labels,
            "body": text[:4000],
            "raw": text[:4000],
        }, "read as JSON from an unrecognised tracker"
    m = re.search(r"<title>(.{0,300}?)</title>", text, re.IGNORECASE | re.DOTALL)
    if m:
        # The page BODY too, not just its title. A title-only answer is why the fallback path
        # was nearly useless: "is this a bug" cannot be decided from eight words when the
        # tracker puts the type, the labels and the description in the page.
        stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        body = " ".join(html.unescape(stripped).split())[:4000]
        return {
            "exists": True,
            "kind": "page",
            "type": "",
            "title": m.group(1).strip(),
            "labels": [],
            "body": body,
            "raw": text[:4000],
        }, "read from the page"
    if text:
        return {
            "exists": True,
            "kind": "text",
            "type": "",
            "title": "",
            "labels": [],
            "body": text[:4000],
            "raw": text[:4000],
        }, "read as plain text"
    return None, "the tracker returned nothing this check could read"


# ── the environment, as the harness observed it ─────────────────────────────────────────────

SANDBOX_RECIPES = "sandbox_recipes"


# One tracker item, read once per run: the DELIVERED gate refetches on every
# validation attempt and a candidate is validated once per recording attempt.
_TRACKER_CACHE: dict[str, tuple[dict, str]] = {}


def clear_tracker_cache() -> None:
    """Drop the per-run tracker cache. For tests, and for anything reusing a process."""
    _TRACKER_CACHE.clear()


def read_tracker_item(url: str, runner) -> tuple[dict | None, str]:
    """`_read_tracker_item`, asked at most once per URL per run.

    A FAILED read is deliberately NOT cached. "TRANSIENT: could not reach the tracker" means the
    check has not run and the caller is explicitly told to defer and retry; remembering that
    would turn one network blip into a permanent verdict for the rest of the run -- the opposite
    of what the wording promises. `exists: False` IS cached: the tracker answering "no such
    item" is a real answer, and asking again cannot change it.
    """
    hit = _TRACKER_CACHE.get(url)
    if hit is not None:
        return hit
    info, detail = _read_tracker_item(url, runner)
    if info is not None:
        _TRACKER_CACHE[url] = (info, detail)
    return info, detail


def actual_recipe(ctx, sandbox: str) -> dict | None:
    """The image and setup commands this sandbox was really created from, or None.

    container_create already stores exactly this, so there is no reason to ask the model to
    describe its own environment in prose and then check the prose against a regex. That check
    passes things like:

        "ubuntu:22.04, OpenJDK 11, Apache Maven 3.6.3 with -Denforcer.skip=true"

    which is well-formed and NOT REPRODUCIBLE -- ubuntu:22.04 has no Maven. The agent installed
    it inside the sandbox and recorded where it started rather than how it got there. Every task
    hit this: defects4j named a base image without its apt step, boca named `make src/yes`
    without ./bootstrap && ./configure, swtbench named a python version without the build
    toolchain pandas needs.
    """
    ws = getattr(ctx, "working_set", None) if ctx is not None else None
    recipes = getattr(ws, SANDBOX_RECIPES, None) if ws is not None else None
    if not isinstance(recipes, dict):
        recipes = getattr(ws, "__dict__", {}).get(SANDBOX_RECIPES) if ws is not None else None
    if not isinstance(recipes, dict):
        return None
    return recipes.get((sandbox or "").strip())


def recipe_text(recipe: dict) -> str:
    """image + the setup that ran, as one line a third party could actually replay."""
    image = str(recipe.get("built_image") or recipe.get("image") or "").strip()
    setup = [str(c).strip() for c in (recipe.get("setup") or []) if str(c).strip()]
    head = image or "(dockerfile)"
    return f"{head}; setup: {' && '.join(setup)}" if setup else head


def env_recipe_from_sandbox(fields: dict, ctx, field: str = "env_recipe") -> tuple[bool, str]:
    """Stamp `field` from the sandbox the candidate actually used. (ok, detail).

    Deliberately AUTHORITATIVE over whatever the model wrote: the harness watched the container
    being created, and the model is recalling it. What the model wrote is kept alongside when it
    adds detail, because a version string it read from `mvn -v` is real information the recipe
    does not carry.
    """
    sandbox = str(fields.get("sandbox") or "").strip()
    if not sandbox:
        return True, "no sandbox recorded; leaving env_recipe as given"
    recipe = actual_recipe(ctx, sandbox)
    if recipe is None:
        return True, f"no stored recipe for sandbox {sandbox!r}; leaving env_recipe as given"
    if not (recipe.get("built_image") or recipe.get("image") or recipe.get("dockerfile")):
        return False, (
            f"sandbox {sandbox!r} was created without a named image, so nobody can "
            f"rebuild this instance. Recreate it with container_create(image=...)."
        )
    stamped = recipe_text(recipe)
    said = str(fields.get(field) or "").strip()
    fields[field] = f"{stamped} | as recorded: {said}" if said and said not in stamped else stamped

    # The recipe alone is not enough and two rounds of data say so: it records what
    # container_create was given, while the agent then apt-installs, ./configures and downloads
    # inside the running container. The provenance block carries the pinned digest, the committed
    # image, and a documentary Dockerfile rebuilt from every command that actually ran.
    try:
        from agentcore.tools import provenance

        ws = getattr(ctx, "working_set", None) if ctx is not None else None
        if ws is not None:
            fields["provenance"] = provenance.provenance_block(ws, sandbox, recipe)
    except Exception:  # noqa: BLE001 - provenance is evidence, never a gate
        pass
    return True, f"env_recipe stamped from the sandbox that was actually used: {stamped[:90]}"
