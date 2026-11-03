"""Environment fingerprint + diff — the basis of `env_info` and the EnvWatcher signal.

A fingerprint is a cheap, structured snapshot of the execution environment:
  system   : OS + Python version
  packages : installed Python distributions {name: version} (pip freeze)
  fs       : top-level workspace map {entry: {type, subdirs, files} | {type, size}}

diff_fingerprints turns two snapshots into human-readable change signals
("created dir build/ (3 subdirs, 12 files)", "installed numpy==1.26.4", "pytest 7.0->8.0").

Env-var changes are intentionally NOT tracked: each command runs in a fresh shell, so an
`export` does not persist between calls — there is no stable per-session env to diff.
"""

from __future__ import annotations

import os

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", ".pytest_cache"}


def system_info(ctx) -> dict:
    osr = ctx.env.execute("uname -sr 2>/dev/null").output.strip()
    py = ctx.env.execute("python3 --version 2>&1 || python --version 2>&1").output.strip()
    return {"os": osr, "python": py}


def package_fingerprint(ctx) -> dict:
    res = ctx.env.execute("python3 -m pip freeze 2>/dev/null || pip freeze 2>/dev/null")
    pkgs: dict[str, str] = {}
    for ln in (res.output or "").splitlines():
        if "==" in ln:
            name, _, ver = ln.partition("==")
            pkgs[name.strip().lower()] = ver.strip()
    return pkgs


def fs_fingerprint(root: str, max_walk: int = 50_000) -> dict:
    """Top-level entries of `root`; each dir summarized by its recursive subdir/file counts."""
    out: dict[str, dict] = {}
    if not root or not os.path.isdir(root):
        return out
    for entry in sorted(os.listdir(root)):
        if entry.startswith(".") or entry in _SKIP_DIRS:
            continue
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            subdirs = files = walked = 0
            for _dp, dn, fn in os.walk(full):
                dn[:] = [d for d in dn if d not in _SKIP_DIRS]
                subdirs += len(dn)
                files += len(fn)
                walked += len(dn) + len(fn)
                if walked > max_walk:
                    break
            out[entry] = {"type": "dir", "subdirs": subdirs, "files": files}
        else:
            try:
                out[entry] = {"type": "file", "size": os.path.getsize(full)}
            except OSError:
                out[entry] = {"type": "file", "size": 0}
    return out


def fingerprint(ctx, *, files: bool = True, packages: bool = True, system: bool = True) -> dict:
    fp: dict = {}
    if system:
        fp.update(system_info(ctx))
    if packages:
        fp["packages"] = package_fingerprint(ctx)
    if files:
        ws = getattr(ctx.env, "workspace", None)
        fp["fs"] = fs_fingerprint(ws) if ws else {}
    return fp


def diff_fingerprints(old: dict, new: dict) -> tuple[list[str], dict]:
    """Return (human signals, structured changes) describing what changed old -> new."""
    signals: list[str] = []
    changes: dict = {"fs": [], "packages": [], "system": []}

    for key in ("os", "python"):
        if new.get(key) and old.get(key) != new.get(key):
            signals.append(f"{key}: {old.get(key)!r} -> {new.get(key)!r}")
            changes["system"].append(key)

    oldfs, newfs = old.get("fs", {}), new.get("fs", {})
    for k, e in newfs.items():
        if k not in oldfs:
            if e["type"] == "dir":
                signals.append(f"created dir {k}/ ({e['subdirs']} subdirs, {e['files']} files)")
            else:
                signals.append(f"created file {k}")
            changes["fs"].append({"path": k, "change": "added"})
        else:
            o = oldfs[k]
            if e["type"] == "dir" and (e["files"], e["subdirs"]) != (o.get("files"), o.get("subdirs")):
                df = e["files"] - o.get("files", 0)
                signals.append(f"{k}/ now {e['files']} files ({df:+d})")
                changes["fs"].append({"path": k, "change": "modified"})
            elif e["type"] == "file" and e.get("size") != o.get("size"):
                signals.append(f"modified file {k}")
                changes["fs"].append({"path": k, "change": "modified"})
    for k in oldfs:
        if k not in newfs:
            signals.append(f"removed {k}")
            changes["fs"].append({"path": k, "change": "removed"})

    oldp, newp = old.get("packages", {}), new.get("packages", {})
    for name, ver in newp.items():
        if name not in oldp:
            signals.append(f"installed {name}=={ver}")
            changes["packages"].append({"name": name, "change": "added", "version": ver})
        elif oldp[name] != ver:
            signals.append(f"{name} {oldp[name]}->{ver}")
            changes["packages"].append({"name": name, "change": "upgraded", "version": ver})
    for name in oldp:
        if name not in newp:
            signals.append(f"removed pkg {name}")
            changes["packages"].append({"name": name, "change": "removed"})

    return signals, changes


def summarize(fp: dict) -> str:
    lines = [f"os: {fp.get('os', '?')}", f"python: {fp.get('python', '?')}"]
    pkgs = fp.get("packages")
    if pkgs is not None:
        sample = ", ".join(sorted(pkgs)[:15])
        lines.append(f"packages: {len(pkgs)} installed" + (f" (e.g. {sample})" if sample else ""))
    fs = fp.get("fs") or {}
    if fs:
        lines.append("workspace:")
        for k, e in fs.items():
            if e["type"] == "dir":
                lines.append(f"  {k}/  ({e['subdirs']} subdirs, {e['files']} files)")
            else:
                lines.append(f"  {k}  ({e['size']} bytes)")
    return "\n".join(lines)
