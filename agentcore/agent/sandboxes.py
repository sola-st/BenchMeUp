"""Sandbox claims, checked at the right moment: existence at discovery, exclusivity at build.

Two distinct guarantees are checked at different moments:

  * EXISTENCE (the recorded name is a real container, not a placeholder) belongs at discovery.

  * EXCLUSIVITY (one candidate, one container) does NOT belong at discovery. Enforcing it
    there would make bulk mining impossible (registering a candidate would demand a container)
    and would forbid the economical thing: reusing one environment for several candidates of
    the same repository.

    Discovery and screening are read-only against a clone (rev-parse, log, diff by hash), so
    one MINING container per repository can serve every candidate from that repository. What
    must be exclusive is the container the instance's evidence is produced in -- builds, test
    runs, timings -- because the image saved at delivery IS the instance's environment, and a
    shared one describes whichever candidate finished first. So exclusivity is enforced by the
    first state that produces such evidence, per collector.

Collectors use thin wrappers over these two functions, so the logic lives in one place.
"""

from __future__ import annotations

import re

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")

# States whose candidate no longer holds a claim on its container.
_DONE = ("BLOCKED", "REJECTED")


def sandbox_exists(fields: dict, ctx) -> tuple[bool, str]:
    """The recorded sandbox is a real, created container -- possibly a shared mining one."""
    name = str(fields.get("sandbox") or "").strip()
    if not name:
        return False, "record `sandbox`: the container this candidate's work happens in"
    if not _SAFE_NAME.match(name):
        return False, "sandbox must be the plain name of a container you created"
    ws = getattr(ctx, "working_set", None) if ctx is not None else None
    recipes = getattr(ws, "sandbox_recipes", None) if ws is not None else None
    if isinstance(recipes, dict) and recipes and name not in recipes:
        return False, (
            f"there is no sandbox named {name!r} -- create it with container_create "
            f"first. Existing: {sorted(recipes)}. A placeholder like 'pending' is not "
            f"a container, and an instance recorded against one cannot be rebuilt."
        )
    return True, f"{name} exists"


def sandbox_is_exclusive(fields: dict, ctx) -> tuple[bool, str]:
    """The recorded sandbox exists AND no other active candidate is using it.

    Run this from the first state that produces instance evidence (build logs, test reports,
    timings): from here on the container's state becomes part of the instance, and the image
    saved at delivery must describe THIS candidate alone.
    """
    ok, why = sandbox_exists(fields, ctx)
    if not ok:
        return False, why
    name = str(fields.get("sandbox") or "").strip()
    ws = getattr(ctx, "working_set", None) if ctx is not None else None
    if ws is not None:
        wl = (getattr(ws, "worklist", None) or {}).get("candidates") or {}
        for other, cand in wl.items():
            # Identity, not key: validators are called as fn(fields, ctx) with no candidate
            # key, and the row being validated is the one holding THIS fields dict.
            if cand.get("fields") is fields:
                continue
            if str((cand.get("fields") or {}).get("sandbox") or "").strip() != name:
                continue
            if cand.get("state") in _DONE:
                continue  # that candidate is done with it; reuse is fine
            return False, (
                f"sandbox {name!r} already belongs to candidate {other[:44]!r}. Sharing a "
                f"container was fine while mining and screening (read-only), but from this "
                f"step on the container's state becomes part of THIS instance -- the image "
                f"saved at delivery is its environment, and a shared one describes whichever "
                f"candidate finished first. Create a fresh container for this candidate "
                f"(container_create) and record it as `sandbox`, then redo this step there."
            )
    return True, f"{name} is this candidate's own container"
