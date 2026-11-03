"""Per-instance sandboxes — opt-in via `tool_modules: ["agentcore.tools.containers"]`.

Why this exists. `DockerEnvironment` owns exactly ONE container (`self._container`), and every
path that changes it -- `reprovision()`, `build_and_start()`, the `set_environment` tool --
tears the previous one down first. That is right for an agent working a single task and wrong
for a *collection* agent, whose job is to produce many instances that each need their own
environment:

  * defects4j_repair candidates pin different JDKs (a parent revision may not compile under
    the container's default JDK, and a second environment cannot stand beside the first);
  * swtbench_repro candidates need per-project Python versions and conflicting pinned
    dependency sets, installed twice per candidate (pre-fix and post-fix snapshots).

With one shared container the agent must either serialize everything through a single mutable
environment -- where instance N's `pip install` silently invalidates instance N-1's validated
result -- or throw the environment away and lose the previous instance's setup. Neither
produces a trustworthy fail-to-pass witness, because the environment the transition was
observed in no longer exists by the time the sample is delivered.

DECLARATIVE BY DESIGN. A sandbox is defined by a RECIPE (image or Dockerfile, workdir, and an
ordered list of setup commands), which is stored in the WorkingSet -- the one piece of run state
that `observability.state` snapshots every cycle. The live container object lives in
`ctx.scratch`, which is deliberately NOT snapshotted because it cannot be serialized.

That split buys three things at once:

  1. RESUME. `run/resume.py` restores the WorkingSet but starts a fresh environment ("Live
     docker/deferred containers are NOT snapshotted"). Because the recipe survives, a resumed
     run rebuilds each sandbox on first use instead of finding it gone.
  2. CRASH RECOVERY MID-RUN. A container can die to a Docker 409 "container is not running"
     (SIGKILL/137). A sandbox that dies is rebuilt from its recipe on the next exec,
     automatically.
  3. REPRODUCIBILITY. The recipe is part of the delivered record, so a third party can rebuild
     the exact environment an instance was validated in -- which is what makes a fail-to-pass
     claim checkable rather than merely asserted.

Rehydration is LAZY (on next use, not eagerly at resume) so a run that never touches a sandbox
again pays nothing for it.

  container_create   define + start a named sandbox (recipe recorded)
  container_exec     run a command in one sandbox (rebuilds it if it died/was resumed)
  container_list     what exists, live status, and whether it needs rebuilding
  container_remove   dispose of one (or all), forgetting the recipe too
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from agentcore.agent import gates
from agentcore.environments import docker_helpers as dh
from agentcore.tools import provenance
from agentcore.tools.context import ToolContext
from agentcore.tools.registry import tool
from agentcore.tools.result import ToolResult

# live, unserializable container objects (per-run only)
SCRATCH_KEY = "_instance_containers"
# serializable recipes; lives in the WorkingSet so it is snapshotted and survives resume
RECIPE_KEY = "sandbox_recipes"
# What has HAPPENED to each sandbox: created / removed / rebuilt / failed, with the cycle it
# happened on. Two jobs: a model that removes a sandbox and tries to recreate it can have its
# attempts refused as "repetition", because a byte comparison cannot see that the container
# was deleted in between. So the ledger
# (a) carries a GENERATION per sandbox, which makes an action against a rebuilt sandbox a
# different action from the harness's point of view, and (b) is shown to the model, which
# otherwise has to reconstruct the lifecycle of its own containers from scrollback.
EVENT_KEY = "sandbox_events"
MAX_EVENTS = 200
# Monotonic {sandbox: times created}. Separate from the event log because that log is bounded.
GENERATION_KEY = "sandbox_generations"

# Bound the blast radius: an agent mining 15-20 candidates should be recycling sandboxes, not
# accumulating them. Hitting this means it stopped cleaning up, so the error says so rather
# than silently evicting something it may still be validating.
MAX_CONTAINERS = gates.MAX_SANDBOXES  # the fallback when the spec declares no `max_sandboxes`


def live_sandboxes(recipes: dict) -> list[str]:
    """The names in a recipe book whose sandbox exists right now, sorted. A removed sandbox
    keeps its recipe (marked `removed`) so validation can rebuild it; it holds no name, no
    slot and no host memory. THE one definition of "live", for the cap, the teardown and
    the prompt alike."""
    return sorted(str(n) for n, r in (recipes or {}).items() if isinstance(r, dict) and not r.get("removed"))


def _max_sandboxes(ctx: ToolContext) -> int:
    """The live-sandbox ceiling for THIS run: the environment spec's `max_sandboxes`, else the
    fallback above. The bound that actually matters is host memory, which the per-container
    mem_limit already carries."""
    try:
        declared = int(getattr(getattr(ctx.env, "spec", None), "max_sandboxes", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        declared = 0
    return declared if declared > 0 else MAX_CONTAINERS


# Setup is replayed on rebuild, so it must stay bounded -- but bound it by SIZE, not by number
# of list entries. A model may pass its setup as a shell SCRIPT split into lines (hundreds of
# "commands"); a line-count cap would reject it. Passing a script is a reasonable
# reading of "setup", so accept it; total bytes is the honest proxy for "slower to replay than
# the original build" anyway.
MAX_SETUP_BYTES = 20000


def _normalize_setup(setup) -> list:
    """Accept a shell script (str) or a list of commands, and return a list of commands.

    A multi-line string is ONE command -- joining it back into a script is exactly what the
    model meant, and splitting on newlines would break heredocs, loops and continuations.
    """
    if not setup:
        return []
    if isinstance(setup, str):
        return [setup]
    return [c for c in (str(x) for x in setup) if c.strip()]


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


# ── recipe store (WorkingSet: snapshotted) ───────────────────────────────────────────────────


def _recipes(ctx: ToolContext) -> dict[str, Any]:
    ws = ctx.working_set
    recipes = getattr(ws, RECIPE_KEY, None)
    if not isinstance(recipes, dict):
        recipes = {}
        setattr(ws, RECIPE_KEY, recipes)
    return recipes


def _live(ctx: ToolContext) -> dict[str, Any]:
    return ctx.scratch.setdefault(SCRATCH_KEY, {})


# ── sandbox lifecycle ledger (WorkingSet: snapshotted) ───────────────────────────────────────


def events(ws) -> list[dict]:
    """Every sandbox lifecycle event this run, oldest first."""
    log = getattr(ws, EVENT_KEY, None)
    if not isinstance(log, list):
        log = []
        setattr(ws, EVENT_KEY, log)
    return log


def record_event(ctx: ToolContext, event: str, name: str, detail: str = "") -> None:
    """Append one lifecycle fact. Failures are recorded too, by explicit request: a create that
    FAILED is exactly the thing a model needs to see, and it is invisible in a ledger that only
    lists what currently exists."""
    log = events(ctx.working_set)
    if len(log) >= MAX_EVENTS:
        del log[: len(log) - MAX_EVENTS + 1]
    log.append(
        {
            "cycle": int(getattr(getattr(ctx, "agent", None), "cycle", 0) or 0),
            "event": str(event),
            "sandbox": str(name),
            "detail": str(detail)[:2000],
        }
    )
    if event == "created":
        # Counted in its own monotonic tally, NOT by scanning the log. The log is bounded, so
        # counting "created" entries in it made a long-lived sandbox's generation go DOWN once
        # its early events were trimmed -- and generation is part of the repetition key, so a
        # fresh action would then collide with an old one and be refused as a repeat.
        counts = getattr(ctx.working_set, GENERATION_KEY, None)
        if not isinstance(counts, dict):
            counts = {}
            setattr(ctx.working_set, GENERATION_KEY, counts)
        counts[str(name)] = int(counts.get(str(name), 0)) + 1


def generation(ws, name: str) -> int:
    """How many times this sandbox has been (re)created. 0 if never.

    The number that makes `container_exec(X, cmd)` after a rebuild a DIFFERENT action from the
    same call before it -- see agentcore/agent/loop.py::_repetition_key. Monotonic by
    construction; see record_event.
    """
    counts = getattr(ws, GENERATION_KEY, None)
    if isinstance(counts, dict) and name in counts:
        return int(counts.get(name) or 0)
    # Snapshots written before the tally existed: fall back to counting, which is right until
    # the log is trimmed and never worse than what it replaced.
    return sum(1 for e in events(ws) if e.get("sandbox") == name and e.get("event") == "created")


def ledger_text(ws, limit: int = 12) -> str:
    """The ledger as the model should read it: newest last, one line each."""
    log = events(ws)[-limit:]
    if not log:
        return ""
    return "\n".join(
        f"  cycle {e.get('cycle')}: {e.get('event')} {e.get('sandbox')}"
        + (f" -- {e.get('detail')}" if e.get("detail") else "")
        for e in log
    )


def _bad_name(name: str) -> str | None:
    if not _NAME_RE.match(name or ""):
        return (
            f"Invalid container name {name!r}. Use 1-63 chars: letters, digits, and "
            f"'_', '.', '-' (must start alphanumeric), e.g. 'commons-io-jdk8'."
        )
    return None


def _readonly(ctx: ToolContext) -> bool:
    return bool(getattr(getattr(ctx.env, "spec", None), "read_only", False))


def _is_running(container) -> bool:
    try:
        container.reload()
        return container.status == "running"
    except Exception:
        return False


def _start_from_recipe(ctx: ToolContext, name: str, recipe: dict) -> tuple[Any, list[str]]:
    """Materialize a sandbox from its recipe. Returns (container, setup_failures)."""
    spec = getattr(ctx.env, "spec", None)
    image = recipe.get("image") or ""
    built = None
    if recipe.get("dockerfile"):
        built = dh.build_image_from_dockerfile(recipe["dockerfile"], f"agentcore-inst-{name}".lower())
        image = built
    workdir = recipe.get("workdir") or "/work"
    container = dh.start_container(
        image,
        lifetime=_effective_lifetime(ctx, recipe.get("lifetime")),
        resources=(getattr(spec, "resources", None) or {}),
        env=dh.llm_key_passthrough_env(getattr(spec, "env", None) or {}),
        workdir=workdir,
        # so a killed or crashed run's containers can be found without guessing what was
        # inside them (see docker_helpers.run_labels). The run dir lives in ctx.scratch --
        # ToolContext has no out_dir attribute.
        labels=dh.run_labels(str((getattr(ctx, "scratch", None) or {}).get("out_dir") or ""), name),
    )
    try:
        dh.exec_in_container(container, f"mkdir -p '{workdir}'")
        # THE CREDENTIAL FILE, BEFORE THE SETUP STEPS. DockerEnvironment writes
        # /root/.netrc and /root/.curlrc into the main machine; a sandbox needs the same
        # files, or every `git clone` in a setup step or a later exec is anonymous.
        dh.exec_in_container(container, dh.tokens_manifest_cmd())
    except Exception:
        pass
    failures = []
    for cmd in recipe.get("setup") or []:
        try:
            code, out = dh.exec_in_container(
                container, cmd, timeout=int(getattr(spec, "timeout", None) or dh.DEFAULT_EXEC_TIMEOUT), workdir=workdir
            )
            if code != 0:
                # The supervisor's story sits at the END of the output and is the part that
                # explains a kill or an unknown status; a 200-char tail would cut it off exactly
                # when it matters ("exit 137" with an apt download line as its whole
                # explanation). Keep the tail AND the whole story.
                story_at = out.find("[supervisor]")
                shown = out[-200:] if story_at < 0 else out[max(0, story_at - 200) : story_at] + out[story_at:]
                failures.append(f"{cmd!r} -> exit {code}: {shown.strip()}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{cmd!r} -> {type(e).__name__}: {e}")
    recipe["built_image"] = built
    return container, failures


def _checkpoint_interval() -> float | None:
    """Seconds between sandbox checkpoints; None = disabled, 0 = every exec (tests)."""
    raw = os.environ.get("AGENTCORE_CHECKPOINT_INTERVAL", "600").strip().lower()
    if raw in ("off", "no", "none", "disabled"):
        return None
    try:
        v = float(raw)
    except ValueError:
        return 600.0
    return None if v < 0 else v


def checkpoint_repo(ctx: ToolContext, name: str) -> str:
    """The image repository for ONE sandbox's checkpoint, namespaced to THIS run.

    `agentcore-ckpt-{name}` alone is host-global, and the sandbox NAME is agent-chosen:
    two concurrent runs mining the same lead both call their sandbox "bcal-candidate",
    so run A's courtesy snapshot displaced run B's and B's next rebuild silently restored
    A's filesystem -- the read-a-different-filesystem class, delivered by the mechanism
    built to prevent work loss. The run's out_dir is the identity that actually differs,
    hashed short so the repo name stays a valid docker reference.
    """
    import hashlib

    out_dir = str((getattr(ctx, "scratch", None) or {}).get("out_dir") or "")
    ns = hashlib.sha256(out_dir.encode()).hexdigest()[:8] if out_dir else "solo"
    return f"agentcore-ckpt-{ns}-{name}".lower()


def _maybe_checkpoint(ctx: ToolContext, name: str, container) -> None:
    """docker-commit the sandbox's live filesystem, interval-gated, ONE image per
    sandbox (the new commit displaces the old id, so checkpointing cannot eat the disk).
    A recipe rebuild restores neither built binaries nor generated inputs, which a
    checkpoint preserves. A checkpoint is a
    courtesy: any failure here must never break the exec that triggered it."""
    interval = _checkpoint_interval()
    if interval is None:
        return
    recipe = _recipes(ctx).get(name)
    if recipe is None:
        return
    now = time.time()
    if interval > 0 and now - float(recipe.get("_ckpt_ts") or 0) < interval:
        return
    repo = checkpoint_repo(ctx, name)
    try:
        client = dh.get_docker_client()
        try:
            old_id = client.images.get(f"{repo}:latest").id
        except Exception:  # noqa: BLE001 - no prior checkpoint
            old_id = None
        img = dh.commit_container(container, repository=repo, tag="latest")
        new_id = getattr(img, "id", None) or str(img)
        if old_id and old_id != new_id:
            try:
                client.images.remove(old_id)
            except Exception:  # noqa: BLE001 - e.g. the running container still uses
                pass  # it right after a restore; it collapses next cycle
        recipe["checkpoint_image"] = f"{repo}:latest"
        recipe["checkpoint_at"] = time.strftime("%H:%M:%S")
        recipe["_ckpt_ts"] = now
        # How much of the recipe's setup the checkpoint already CONTAINS. A restore
        # replays the delta (setup steps recorded after this point) -- without it, a
        # checkpoint taken after `git clone` but before the recipe gained its
        # `git checkout <parent>` step would restore the clone-only tree, and validation
        # would judge candidates on a commit nobody chose.
        recipe["checkpoint_setup_len"] = len(recipe.get("setup") or [])
    except Exception:  # noqa: BLE001
        pass


#: The harness could not reach a sandbox at all -- no recipe, or the daemon refused. NOT 127:
#: that is the shell's "command not found", and conflating them would make the re-derivation
#: control tell agents to install a tool into a container that no longer existed, then charge
#: a strike for not doing it.
SANDBOX_UNREACHABLE = 199
UNREACHABLE_PREFIX = "HARNESS ERROR: "


def _ensure_live(ctx: ToolContext, name: str, revive_removed: bool = False) -> tuple[Any, str]:
    """Return (container, note). Rebuilds from the recipe if the sandbox is missing or dead.

    This is the single place that reconciles "what the recipe says should exist" with "what is
    actually running", so resume, container death and ordinary use all funnel through one path.

    A REMOVED sandbox (container_remove keeps the recipe, flagged) is refused by default: the
    agent decided it was done, and a stray exec must not helpfully resurrect it. VALIDATION
    passes revive_removed=True, because a validator may need a sandbox the agent had already
    removed.
    """
    live = _live(ctx)
    entry = live.get(name)
    if entry is not None and _is_running(entry.get("container")):
        return entry["container"], ""

    recipe = _recipes(ctx).get(name)
    if recipe is None:
        known = ", ".join(sorted(_recipes(ctx))) or "(none)"
        return None, (f"no recipe is recorded for {name!r}; sandboxes this run knows: {known}")
    if recipe.get("removed") and not revive_removed:
        return None, (
            f"sandbox {name!r} was removed; its recipe is kept for validation. "
            f"Recreate it with container_create to work in it again."
        )
    if recipe.get("removed"):
        recipe.pop("removed", None)  # revived for validation: it is live again, say so
    why = "was not running in this process (resumed run?)" if entry is None else "had died"
    if entry is not None:
        # Reap the dead container before replacing it. Without this the exited corpse is
        # dropped from the registry and stays on the host forever -- cleanup_containers only
        # ever sees the replacement. Caught by actually killing a container and checking
        # `docker ps -a` afterwards, not by reading the code.
        try:
            dh.cleanup(entry.get("container"), image=None)
        except Exception:
            pass
    # Checkpoint first: a recipe replay restores the environment but NOT the agent's
    # work (built binaries, generated inputs). The checkpoint image has everything up to its timestamp; the recipe
    # is the fallback when no checkpoint exists or it will not start.
    container, failures, from_ckpt = None, [], False
    ck = recipe.get("checkpoint_image")
    if ck:
        try:
            dh.get_docker_client().images.get(ck)
            spec = getattr(ctx.env, "spec", None)
            container = dh.start_container(
                ck,
                lifetime=_effective_lifetime(ctx, recipe.get("lifetime")),
                resources=(getattr(spec, "resources", None) or {}),
                env=dh.llm_key_passthrough_env(getattr(spec, "env", None) or {}),
                workdir=recipe.get("workdir") or "/work",
                # same ownership labels as a recipe start, so a checkpoint-restored
                # sandbox stays attributable
                labels=dh.run_labels(str((getattr(ctx, "scratch", None) or {}).get("out_dir") or ""), name),
            )
            from_ckpt = True
        except Exception:  # noqa: BLE001 - fall back to the recipe
            container = None
    if container is None:
        container, failures = _start_from_recipe(ctx, name, recipe)
    elif from_ckpt:
        # THE CHECKPOINT REPLAYS ITS DELTA: steps the recipe gained after
        # the checkpoint was taken are not in the image, and skipping them hands
        # validation a sandbox on a state nobody chose. Older checkpoints carry no
        # marker; for those the delta is unknowable and every step is replayed --
        # idempotent steps tolerate it, and a loud failure beats a silent wrong tree.
        done = recipe.get("checkpoint_setup_len")
        steps = list(recipe.get("setup") or [])
        delta = steps[int(done) :] if isinstance(done, int) else steps
        spec = getattr(ctx.env, "spec", None)
        for cmd in delta:
            try:
                code, out = dh.exec_in_container(
                    container,
                    cmd,
                    timeout=int(getattr(spec, "timeout", None) or dh.DEFAULT_EXEC_TIMEOUT),
                    workdir=recipe.get("workdir") or "/work",
                )
                if code != 0:
                    failures.append(f"{cmd!r} -> exit {code}: {out[-200:].strip()}")
            except Exception as e:  # noqa: BLE001
                failures.append(f"{cmd!r} -> {type(e).__name__}: {e}")
    live[name] = {"container": container}
    record_event(
        ctx,
        "created",
        name,
        f"rebuilt automatically ({why})"
        + (", from checkpoint" if from_ckpt else "")
        + (f"; {len(failures)} setup failure(s)" if failures else ""),
    )
    if from_ckpt:
        note = (
            f"[sandbox {name!r} {why}; restored from checkpoint taken at "
            f"{recipe.get('checkpoint_at', '?')} -- your files and builds up to "
            f"then are present; anything after it must be redone]\n"
        )
    else:
        note = (
            f"[sandbox {name!r} {why}; rebuilt from its recorded recipe"
            f"{' with ' + str(len(failures)) + ' setup failure(s): ' + '; '.join(failures) if failures else ''}"
            f" -- only the recipe's setup exists; files created by later commands are gone]\n"
        )
    return container, note


# ── tools ────────────────────────────────────────────────────────────────────────────────────


# idempotent_ok: exempt from the generic repetition check. This tool refuses its OWN
# duplicates with a message that names what to do instead ("a sandbox named X already
# exists ... use container_exec, or container_remove first"), and after a REMOVE the
# identical call is not a duplicate at all -- it is the only way to get the sandbox back.
def _effective_lifetime(ctx: ToolContext, lifetime: str | None) -> str:
    """A sandbox with no explicit lifetime lives as long as the spec says containers live.

    A hardcoded default would silently override the spec's container_lifetime for every
    candidate sandbox, so sandboxes would die and be rebuilt repeatedly, each rebuild
    dropping whatever ad-hoc state the recipe never held."""
    if lifetime:
        return str(lifetime)
    spec = getattr(getattr(ctx, "env", None), "spec", None)
    return str(getattr(spec, "container_lifetime", None) or "2h")


@tool(
    "container_create",
    idempotent_ok=True,
    params={
        "name": "your handle for this sandbox (one candidate = one sandbox); reuse it in every container_exec",
        "image": "docker image to start from (e.g. ubuntu:22.04); mutually exclusive with dockerfile",
        "dockerfile": "inline Dockerfile text, when an image is not enough",
        "workdir": "absolute path commands run in by default (default /work)",
        "lifetime": "how long the sandbox may live, e.g. '6h' (default: the task's setting)",
        "setup": (
            "list of shell commands that recreate this environment from scratch -- "
            "they are REPLAYED whenever the sandbox is rebuilt, so put clones and "
            "installs here, not in ad-hoc execs"
        ),
    },
    aliases={
        "name": ["container", "sandbox", "container_name", "sandbox_name"],
        "image": ["base_image", "docker_image", "from"],
    },
)
def container_create(
    ctx: ToolContext,
    name: str,
    image: str = "",
    dockerfile: str = "",
    workdir: str = "/work",
    lifetime: str = "",
    setup=None,
) -> ToolResult:
    """Start a named, isolated sandbox for one candidate instance.

    Give either `image` (e.g. "maven:3.9-eclipse-temurin-8") or `dockerfile` (raw Dockerfile
    text). `setup` may be a shell script (one string) or a list of commands, run once the
    sandbox is up
    (installs, clones, builds) -- put per-instance setup HERE rather than in ad-hoc
    container_exec calls, because the recipe is recorded and replayed to rebuild this exact
    environment after a resume or a container crash, and it is what lets someone else
    reproduce the instance later.

    The sandbox is independent of the agent's own environment and of every other sandbox, so
    per-instance toolchains cannot collide. Call container_remove when the instance is done.
    """
    if _readonly(ctx):
        return ToolResult.fail("Environment is read-only; container_create is disabled.")
    if err := _bad_name(name):
        return ToolResult.fail(err)
    if not image and not dockerfile:
        return ToolResult.fail(
            "neither `image` nor `dockerfile` was given -- pass exactly one. Use image=... for a "
            'published base image (e.g. "maven:3.9-eclipse-temurin-8"), or dockerfile=... to '
            "build one."
        )
    if image and dockerfile:
        return ToolResult.fail(
            "both `image` and `dockerfile` were given -- pass exactly one, so it is unambiguous "
            "which base the sandbox is built from."
        )
    setup = _normalize_setup(setup)
    total = sum(len(c) for c in setup)
    if total > MAX_SETUP_BYTES:
        return ToolResult.fail(
            f"setup is {total} bytes, over the {MAX_SETUP_BYTES}-byte limit. Write the long part "
            f"to a script file with collector_write or a heredoc, and make setup run that script "
            f"-- the whole of setup is replayed whenever this sandbox is rebuilt."
        )

    recipes = _recipes(ctx)
    if name in recipes and not recipes[name].get("removed"):
        # THE SAME RECIPE AGAIN IS "ENSURE", NOT A CONFLICT. A model may call container_create
        # on a live sandbox to reset or double-check a machine it cannot see in the
        # prompt. An identical recipe is answered with the running sandbox; a DIFFERENT one
        # is still refused, naming what differs, because silently keeping the old machine
        # would hand the candidate an environment it did not ask for.
        have = recipes[name]
        want = {"image": image, "dockerfile": dockerfile or "", "workdir": workdir or "/work", "setup": setup}
        differs = [k for k in want if have.get(k) != want[k]]
        if not differs:
            return ToolResult(
                display_text=(
                    f"Sandbox {name!r} is already running with exactly this recipe "
                    f"(image {have.get('image')!r}, workdir {have.get('workdir')!r}); "
                    f"nothing was created. Run commands in it with container_exec -- "
                    f"if it had died, container_exec rebuilds it from this recipe."
                ),
                structured={"name": name, "existing": True},
            )
        return ToolResult.fail(
            f"A sandbox named {name!r} already exists (image {have.get('image')!r}) with a "
            f"different recipe -- {', '.join(differs)} differ. Use container_exec to run "
            f"commands in the one that exists, or container_remove it first if you really "
            f"want a machine built from this new recipe."
        )
    # Removed recipes keep the record for validation but do not hold the name or count
    # toward the live cap -- a removed sandbox consumes no host resources.
    # THE MESSAGE MUST NAME THE SANDBOXES THE CAP IS ACTUALLY COUNTING. The guard counts
    # LIVE recipes; reporting `len(recipes)` and `sorted(recipes)` would include every
    # already-removed one, and a container_remove on such a name would find nothing to free.
    # Name only what is live, so removing something from the list frees a slot.
    live_names = live_sandboxes(recipes)
    limit = _max_sandboxes(ctx)
    if len(live_names) >= limit:
        return ToolResult.fail(
            f"{len(live_names)} sandboxes already exist (limit {limit}, the environment's "
            f"`max_sandboxes`): {live_names}. Remove the ones whose instances are finished "
            f"with container_remove."
        )

    recipe = {
        "image": image,
        "dockerfile": dockerfile or "",
        "workdir": workdir or "/work",
        "lifetime": _effective_lifetime(ctx, lifetime),
        "setup": setup,
    }
    try:
        container, failures = _start_from_recipe(ctx, name, recipe)
    except Exception as e:  # noqa: BLE001 - surface provisioning failure as a usable nudge
        # A create that FAILED belongs in the ledger too: a record of
        # what currently exists cannot tell a model why the thing it expected is missing.
        record_event(ctx, "create failed", name, f"{type(e).__name__}: {e}")
        return ToolResult.fail(f"Could not start sandbox {name!r}: {type(e).__name__}: {e}")
    record_event(
        ctx,
        "created",
        name,
        f"image {image or 'dockerfile'}" + (f"; {len(failures)} setup failure(s)" if failures else ""),
    )

    # Resolve the tag to a digest now, while we know it was just pulled. `gcc:latest` is a
    # moving pointer -- the same tag six months from now is a different compiler. Locally built
    # images have no RepoDigests, and that is fine: the snapshot covers those.
    try:

        def _probe(cmd):
            import subprocess

            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return r.returncode, (r.stdout or "") + (r.stderr or "")

        digest = provenance.resolve_digest(_probe, recipe.get("built_image") or image)
        if digest:
            recipe["image_digest"] = digest
    except Exception:  # noqa: BLE001 - a missing digest must not stop the sandbox starting
        pass

    # Environment facts, probed once at creation: otherwise each agent rediscovers a
    # missing `time` builtin, missing xxd, or unset TERM on its own. The backend can see all of it the
    # moment the container starts; the facts live in the recipe (rebuilds keep them)
    # and in the create result. Just the needed things, not everything.
    try:
        from agentcore.environments import docker_helpers as _dh_facts

        probe = (
            "echo ___envfacts___; "
            "echo sh-$(readlink -f /bin/sh 2>/dev/null || echo unknown); "
            "for t in bash time xxd script tmux perl python3 gcc make git curl; do "
            "command -v $t >/dev/null 2>&1 && echo have-$t || echo miss-$t; done; "
            "echo term-${TERM:-unset}"
        )
        code, out = _dh_facts.exec_in_container(container, probe, timeout=30)
        if code == 0 and "___envfacts___" in out:
            lines = [l.strip() for l in out.split("___envfacts___", 1)[1].splitlines() if l.strip()]
            sh = next((l[3:] for l in lines if l.startswith("sh-")), "unknown")
            have = [l[5:] for l in lines if l.startswith("have-")]
            miss = [l[5:] for l in lines if l.startswith("miss-")]
            term = next((l[5:] for l in lines if l.startswith("term-")), "unset")
            facts = (
                f"sh->{sh.rsplit('/', 1)[-1]}; available: {', '.join(have) or '-'}; "
                f"NOT available: {', '.join(miss) or '-'}; TERM={term}"
            )
            recipe["env_facts"] = facts
    except Exception:  # noqa: BLE001 - facts are a courtesy, never a blocker
        pass

    recipes[name] = recipe
    _live(ctx)[name] = {"container": container}
    msg = (
        f"Sandbox {name!r} is running on {recipe.get('built_image') or image} "
        f"(workdir {recipe['workdir']}). This machine PERSISTS for this candidate: files "
        f"survive between container_exec calls, and the harness verifies this candidate's "
        f"evidence HERE -- a recorded command must work in this sandbox, not on your main "
        f"machine. Run commands with "
        f"container_exec(name={name!r}, command=...). Active: {sorted(recipes)}"
    )
    if recipe.get("env_facts"):
        msg += f"\nIts environment: {recipe['env_facts']}"
    if failures:
        msg += (
            f"\nWARNING: {len(failures)} setup command(s) failed -- fix them and recreate, "
            f"or this environment will not rebuild correctly later:\n  " + "\n  ".join(failures)
        )
    return ToolResult(
        display_text=msg,
        structured={
            "name": name,
            "image": recipe.get("built_image") or image,
            "workdir": recipe["workdir"],
            "setup_failures": failures,
            "env_facts": recipe.get("env_facts"),
            "active": sorted(recipes),
        },
    )


# first_error, not head_tail: every build and test in this project runs through here,
# and head_tail keeps the prologue and the epilogue while discarding the middle --
# where the compiler error lives.
# first_error extracts the failure regions and falls back to head_tail when a run
# succeeded and there is nothing to extract.
@tool(
    "container_exec",
    truncation="first_error",
    params={
        "name": "which sandbox to run in -- its name as shown to you",
        "command": "shell command; runs at the sandbox's workdir unless cwd is given",
        "cwd": "absolute path inside the sandbox to run in (optional)",
        "timeout": "seconds before the command is killed (optional; default from the task)",
    },
    aliases={"name": ["container", "sandbox", "container_name", "sandbox_name"], "command": ["cmd", "exec", "script"]},
)
def container_exec(ctx: ToolContext, name: str, command: str, cwd: str = "", timeout: int = 0) -> ToolResult:
    """Run a shell command inside ONE named sandbox -- a separate machine with its own filesystem.

    NOT the same environment as `linux_terminal`, and not the same as any other sandbox. Each has
    its own filesystem and nothing is shared. `/workspace/foo` here and `/workspace/foo` under
    linux_terminal are two different directories with the same name; a file written in one is
    invisible in the other, and nothing warns you -- both commands succeed and print plausible
    output. Cloning a repository in one and then running its tests in the other is the single
    most expensive mistake available here: the test really runs, the report really exists, and
    the harness still cannot read it.

    THE RULE: one candidate, one sandbox. Everything for a candidate -- clone, dependency
    install, build, test, reading the report -- happens inside the sandbox recorded on that
    candidate. The harness verifies your evidence in exactly that sandbox.

    Use this for: all per-candidate work. Its isolation is the point: a toolchain installed for
    one candidate cannot break another's.

    If the sandbox died, or this is a resumed run, it is transparently rebuilt from its recipe
    (image + workdir + setup commands) before your command runs, so a crash does not lose it.
    """
    recipes = _recipes(ctx)
    if name not in recipes:
        return ToolResult.fail(
            f"No sandbox named {name!r}. Existing: {sorted(recipes) or '(none)'}. "
            f"Create it first with container_create."
        )
    try:
        container, note = _ensure_live(ctx, name)
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail(f"Sandbox {name!r} could not be (re)started: {type(e).__name__}: {e}")
    if container is None:
        return ToolResult.fail(f"Sandbox {name!r} has no usable recipe; recreate it.")

    spec = getattr(ctx.env, "spec", None)
    eff_timeout = timeout or getattr(spec, "timeout", None) or dh.DEFAULT_EXEC_TIMEOUT
    workdir = cwd or recipes[name].get("workdir") or None
    # docker's exec API requires an ABSOLUTE workdir; a relative cwd is natural for a model to
    # pass but is a hard error here (mirrors DockerEnvironment.execute's handling).
    if workdir and not workdir.startswith("/"):
        base = (recipes[name].get("workdir") or "/").rstrip("/")
        workdir = f"{base}/{workdir.lstrip('./')}"
    try:
        code, out = dh.exec_in_container(container, command, timeout=int(eff_timeout), workdir=workdir)
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail(f"Exec in sandbox {name!r} failed: {type(e).__name__}: {e}")
    # Every command that runs in a sandbox goes into the provenance trace, in order. This is the
    # raw material for reconstructing the environment later: env_recipe records where the
    # container STARTED, and 9 of 15 unverifiable instances across two rounds failed because
    # nothing recorded how it got from there to a working build.
    try:
        provenance.record_exec(ctx.working_set, name, command, out, code)
    except Exception:  # noqa: BLE001 - provenance must never break the command that produced it
        pass
    if code == 0:
        _maybe_checkpoint(ctx, name, container)

    # Name the machine on every output, and say which directory it ran in. `[name]` alone was too
    # quiet: the same repo was cloned here and on the main machine at the SAME path, and nothing
    # in either output distinguished them afterwards. The banner is the reminder placed exactly
    # where the mistake happens.
    where = f" cwd={workdir}" if workdir else ""
    return ToolResult(
        display_text=(
            f"{note}=== RAN ON: SANDBOX {name!r} (this candidate's own machine -- its "
            f"files are NOT on the main machine){where} ===\n"
            f"(exit {code}{shell_error_note(out) if code == 0 else ''}) {out}"
            f"{unshipped_binary_note(command) if code == 0 else ''}"
        ),
        structured={
            "container": name,
            "returncode": code,
            "timed_out": code == 124,
            "rebuilt": bool(note),
            "ran_on": f"sandbox:{name}",
            "workdir": workdir,
        },
    )


# WHO EMITS THESE. Anchored on the PROGRAM that printed the line, not on the phrase, because
# the phrases themselves are ordinary English that a test's own expected output quotes freely --
# `expected: java.io.FileNotFoundException: No such file or directory` is a passing assertion,
# not a failure. A shell or coreutils diagnostic always opens `<prog>: ...` at column 0.
#
# This list is shell vocabulary, not benchmark content, so it is not the hardcoded-task-
# assumption rule: no task can rename `chmod`. It is deliberately short -- what it misses is
# reported as nothing, which is safe.
_EMITTERS = (
    r"(?:/(?:usr/)?bin/)?(?:sh|bash|dash|env|chmod|chown|cp|mv|rm|ln|mkdir|rmdir|"
    r"cat|touch|ls|find|xargs|tar|unzip|git|make|javac|java|python3?|pip3?)"
)
_PHRASES = (
    r"command not found|not found|No such file or directory|Permission denied|"
    r"cannot access|cannot stat|cannot move|cannot remove|cannot create|"
    r"is a directory|Is a directory|No such process"
)
_SHELL_ERROR = re.compile(rf"(?m)^{_EMITTERS}(?::\s*\d+)?:\s*[^\n]*?(?:{_PHRASES})[^\n]*")


# Where a PATH executable lives on any Linux image. An OS fact, like the shell and coreutils
# program names `shell_error_note` keys on -- nothing here knows about build tools or any
# benchmark.
# Where a PATH executable lives on any Linux image. An OS fact, like the shell and coreutils
# program names `shell_error_note` keys on -- nothing here knows about build tools or any
# benchmark.
_PATH_DIRS = ("/usr/local/bin", "/usr/local/sbin", "/usr/bin", "/usr/sbin", "/bin", "/sbin")

# Verbs that put a file somewhere. The DESTINATION is what matters: `cp /usr/bin/mvn /tmp/`
# reads from a PATH directory and creates nothing there, so the test is on the last argument
# rather than on the presence of the path anywhere in the command.
_PUT_VERBS = ("ln", "cp", "mv", "install")
_REDIRECT = re.compile(r">>?\s*(?P<p>(?:%s)/[\w.\-+]+)" % "|".join(_PATH_DIRS))


def _lands_in_path_dir(segment: str) -> str:
    """The PATH location a segment writes to, or "". Destination only."""
    m = _REDIRECT.search(segment)
    if m:
        return m.group("p")
    words = segment.split()
    if not words:
        return ""
    verb = words[0].strip("(){};&|")
    if verb not in _PUT_VERBS:
        return ""
    last = words[-1].strip("'\";")
    for d in _PATH_DIRS:
        if last == d or last.startswith(d + "/"):
            return last.rstrip("/")
    return ""


def unshipped_binary_note(cmd: str) -> str:
    """Say that an executable created at a PATH location will not reach the artifact, or "".

    An instance can name a `build_tool` the shipped environment does not provide, because the
    agent made the missing binary by hand --
    `ln -sf $(which mvn) /usr/bin/maven`, `cat > /usr/local/bin/gradle`. The artifact is built
    from the recipe (base image or Dockerfile, workdir, setup), and there is no commit-on-deliver
    anywhere, so a file installed after container creation exists for the rest of the run,
    satisfies the terminal gate's rehearsal, and is absent from what ships.

    This says so when it happens, rather than leaving it to be discovered after the sandboxes
    are gone.

    Never refuses: making a binary by hand is legitimate while exploring, and whether it belongs
    in the recipe is the agent's call.
    """
    text = str(cmd or "")
    for segment in re.split(r"&&|\|\||;|\n", text):
        where = _lands_in_path_dir(segment.strip())
        if where:
            return (
                f" (NOTE: this puts {where} into the sandbox, and it is not in this "
                f"sandbox's recipe. The packaged benchmark is rebuilt from the recipe -- "
                f"base image and setup steps -- so {where} will NOT exist there, however "
                f"well it works here. If the instance needs it, put it in the sandbox's "
                f"setup, or record a value that works without it.)"
            )
    return ""


def shell_error_note(out: str) -> str:
    """ "" unless the shell reported a failure inside output a command called a success.

    A compound command's exit code belongs to its
    LAST statement, so

        cat > gradlew <<'X' ... X ; mv gradlew gradlew.real ; chmod +x gradlew ; echo done

    destroys a project's launcher, prints `chmod: cannot access ...: No such file or
    directory`, and reports exit 0 because `echo` succeeded.

    The harness was honest -- the error was printed verbatim -- and still misleading, because
    `(exit 0)` is a VERDICT and it was the wrong one. This adds a clause to the verdict. It
    never refuses and never alters the exit code: `ls foo || echo missing` is a legitimate
    probe, and stopping it would cost far more than this saves.
    """
    m = _SHELL_ERROR.search(str(out or ""))
    if not m:
        return ""
    line = " ".join(m.group(0).split())
    return (
        f", but the shell reported an error inside it: {line[:140]} -- "
        f"an exit code belongs to the LAST statement, so an earlier one can fail and the "
        f"command still report success"
    )


@tool("container_list", idempotent_ok=True)
def container_list(ctx: ToolContext) -> ToolResult:
    """List the sandboxes that exist, with their images and whether each is currently live."""
    recipes = _recipes(ctx)
    if not recipes:
        return ToolResult.text("No sandboxes exist. Create one with container_create.")
    live = _live(ctx)
    rows = []
    for name, recipe in sorted(recipes.items()):
        entry = live.get(name)
        running = bool(entry) and _is_running(entry.get("container"))
        rows.append(
            {
                "name": name,
                "image": recipe.get("built_image") or recipe.get("image") or "(dockerfile)",
                "workdir": recipe.get("workdir"),
                "setup_commands": len(recipe.get("setup") or []),
                "status": "running" if running else "needs rebuild (will happen on next exec)",
            }
        )
    listing = "\n".join(
        f"  {r['name']}  image={r['image']}  workdir={r['workdir']}  "
        f"setup={r['setup_commands']} cmd(s)  status={r['status']}"
        for r in rows
    )
    log = ledger_text(ctx.working_set)
    tail = f"\n\nWhat has happened to your sandboxes:\n{log}" if log else ""
    return ToolResult(
        display_text=f"{len(rows)} sandbox(es):\n{listing}{tail}",
        structured={"containers": rows, "events": events(ctx.working_set)[-12:]},
    )


def exec_in_sandbox(
    ctx: ToolContext,
    name: str,
    cmd: str,
    timeout: int = 600,
    revive_removed: bool = False,
    note_sink: list | None = None,
    capture_limit: int | None = None,
) -> tuple[int, str]:
    """(exit, output) in the NAMED sandbox, at its RECIPE's workdir -- the same directory
    container_exec gives the agent.

    This is the ONE path for harness-side re-execution (the rederives control, validator
    re-runs): exec_in_container with no workdir lands at the image's default cwd, while the
    agent works at the recipe's workdir (/work). A command with any relative path would then
    produce a STABLE WRONG output -- an error message hashes as reproducibly as a result."""
    container, note = _ensure_live(ctx, name, revive_removed=revive_removed)
    if container is None:
        # The candidate's machine is gone. Say whose fault that is, in the words the
        # worklist's `_harness_fault` looks for, and with a code that is not the shell's.
        return SANDBOX_UNREACHABLE, (
            f"{UNREACHABLE_PREFIX}the harness could not reach sandbox {name!r} "
            f"({note.strip() or 'it has no usable recipe'}). Nothing was executed, so this "
            f"says nothing about the candidate or its recorded evidence."
        )
    if note and note_sink is not None:
        # The revival is load-bearing context for whoever judges this exec's output: a
        # rebuilt sandbox holds only what the recipe's setup recreates, so digests hashed in
        # a fresh workspace need to be known as such. Not prepended to the output -- digest paths hash that.
        note_sink.append(note.strip())
    workdir = (_recipes(ctx).get(name) or {}).get("workdir") or "/work"
    return dh.exec_in_container(container, cmd, timeout=timeout, workdir=workdir, capture_limit=capture_limit)


# idempotent_ok: same reasoning as container_create -- removing an already-gone sandbox
# gets a specific refusal from the tool itself, which beats a generic repetition nudge.
# The decorator belongs on container_remove, not on exec_in_sandbox, which is
# harness-side (worklist rederives, validators) and is called directly, never by name
# from a model.
@tool(
    "container_remove",
    idempotent_ok=True,
    params={
        "name": "which sandbox to dispose of; its recipe is kept so validation can rebuild it",
        "all": "true to dispose of every sandbox",
    },
    aliases={"name": ["container", "sandbox", "container_name", "sandbox_name"]},
)
def container_remove(ctx: ToolContext, name: str = "", all: bool = False) -> ToolResult:
    """Dispose of a named sandbox (or every sandbox with all=True). The RECIPE is kept.

    Do this as soon as a candidate is fully validated and its evidence recorded -- sandboxes
    hold real disk and memory on the host, and there is a cap on how many can exist at once.

    Removal frees the container but keeps how to rebuild it: container_exec on a removed name
    is refused, the checks of your evidence rebuild it when they need it, and container_create
    may reuse the name."""
    # The recipe is kept, as _run_in's own refusal message promises, so a validator that
    # later needs the sandbox can rebuild it.
    recipes = _recipes(ctx)
    live = _live(ctx)
    if not name and not all:
        return ToolResult.fail("Pass a `name`, or all=True to remove every sandbox.")
    targets = live_sandboxes(recipes) if all else [name]
    removed, missing = [], []
    for t in targets:
        recipe = recipes.get(t)
        entry = live.pop(t, None)
        if (recipe is None or recipe.get("removed")) and entry is None:
            missing.append(t)
            continue
        if recipe is not None:
            recipe["removed"] = True
        try:
            # The container goes; the built image STAYS -- it is what a validation-time
            # revival rebuilds from, and deleting it here made revival impossible.
            dh.cleanup(entry.get("container") if entry else None, image=None)
        except Exception:
            pass  # best-effort: the container is gone either way
        removed.append(t)
        record_event(ctx, "removed", t)
    for t in missing:
        record_event(ctx, "remove failed", t, "no such sandbox")
    active = live_sandboxes(recipes)
    if missing and not removed:
        return ToolResult.fail(f"No sandbox named {missing[0]!r}. Existing: {active or '(none)'}")
    return ToolResult(
        display_text=(f"Removed {removed} (recipes kept for validation). Remaining: {active or '(none)'}"),
        structured={"removed": removed, "remaining": active},
    )


def cleanup_containers(scratch: dict) -> int:
    """Tear down every LIVE sandbox recorded in a run's scratch. Returns how many were removed.

    Called from the run's own teardown so an aborted, crashed or budget-exhausted run cannot
    leak containers onto the host -- exactly the leak that had to be reaped by hand after an
    earlier interrupted batch left 10 orphaned containers running.

    Deliberately does NOT touch the recipes in the WorkingSet: they are the record of what was
    built, they are what a resumed run rebuilds from, and they are part of the delivered
    instance evidence. Killing the process must not erase them.
    """
    reg = (scratch or {}).get(SCRATCH_KEY) or {}
    n = 0
    for _name, entry in list(reg.items()):
        try:
            dh.cleanup(entry.get("container"), image=None)
            n += 1
        except Exception:
            pass
    reg.clear()
    return n
