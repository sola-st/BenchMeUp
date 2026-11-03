"""Built-in environments: NoneEnvironment and LocalEnvironment.

DockerEnvironment / DeferredEnvironment live in environments/docker.py and register themselves with the factory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any

from agentcore.config.schema import EnvSpec
from agentcore.protocols import ExecResult


class CommandProvisioner:
    """Mixin: provision user-declared CLI commands (EnvSpec.commands) into the env and advertise
    them. A command is dropped as an executable in a bin dir that is on PATH, or installed via a
    one-time `install` command, so the agent can run it through linux_terminal."""

    def command_specs(self) -> list[dict]:
        """The declared commands, for advertising to the agent (name/description/usage)."""
        out = []
        for c in getattr(self.spec, "commands", None) or []:
            if c.get("name"):
                out.append({"name": c["name"], "description": c.get("description", ""), "usage": c.get("usage", "")})
        return out

    def provision_commands(self) -> list[str]:
        """Write script commands into bin_dir (+x) and run any install commands. Returns names."""
        names: list[str] = []
        for c in getattr(self.spec, "commands", None) or []:
            name = c.get("name")
            if not name:
                continue
            if c.get("script"):
                body = c["script"]
                if not body.startswith("#!"):
                    body = "#!/usr/bin/env bash\n" + body
                self._write_command(name, body)
            if c.get("install"):
                self.execute(c["install"])
            names.append(name)
        return names

    def _write_command(self, name: str, body: str) -> None:  # overridden per env
        raise NotImplementedError


class NoneEnvironment:
    """No execution environment (read-only / no-container tasks like PR review)."""

    def __init__(self, spec: EnvSpec, **_: Any):
        self.spec = spec

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> ExecResult:
        return ExecResult(output="", returncode=126, stderr="This agent has no execution environment (env.type=none).")

    def reprovision(self, spec: EnvSpec) -> None:  # pragma: no cover - inert
        self.spec = spec

    def checkpoint(self) -> str:
        return ""

    def restore(self, handle: str) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def get_template_vars(self) -> dict[str, Any]:
        return {"env_type": "none", "container_present": False}


class LocalEnvironment(CommandProvisioner):
    """Run commands on the host via subprocess, rooted at a workspace dir."""

    def __init__(self, spec: EnvSpec, *, workspace: str | None = None, **_: Any):
        self.spec = spec
        self.workspace = os.path.abspath(workspace or spec.workdir or tempfile.mkdtemp(prefix="agentcore_ws_"))
        os.makedirs(self.workspace, exist_ok=True)
        self.bin_dir = os.path.join(self.workspace, ".agentcore", "bin")
        os.makedirs(self.bin_dir, exist_ok=True)

    def _write_command(self, name: str, body: str) -> None:
        path = os.path.join(self.bin_dir, name)
        with open(path, "w") as f:
            f.write(body)
        os.chmod(path, 0o755)

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> ExecResult:
        wd = os.path.join(self.workspace, cwd) if cwd else self.workspace
        env = {**os.environ, **(self.spec.env or {})}
        env["PATH"] = self.bin_dir + os.pathsep + env.get("PATH", "")  # provisioned commands first
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=wd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout or self.spec.timeout,
            )
            return ExecResult(output=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
        except subprocess.TimeoutExpired as e:
            out = (e.output or "") if isinstance(e.output, str) else (e.output or b"").decode("utf-8", "replace")
            return ExecResult(output=out, returncode=124, stderr="timeout", timed_out=True)

    def reprovision(self, spec: EnvSpec) -> None:
        self.spec = spec

    def checkpoint(self) -> str:
        """Snapshot the workspace by copying it to a temp dir; returns the handle."""
        handle = tempfile.mkdtemp(prefix="agentcore_ckpt_")
        shutil.copytree(self.workspace, os.path.join(handle, "ws"), dirs_exist_ok=True)
        return handle

    def restore(self, handle: str) -> None:
        src = os.path.join(handle, "ws")
        if os.path.isdir(src):
            shutil.rmtree(self.workspace, ignore_errors=True)
            shutil.copytree(src, self.workspace)

    def cleanup(self) -> None:
        pass

    def get_template_vars(self) -> dict[str, Any]:
        return {
            "env_type": "local",
            "workspace": self.workspace,
            "container_present": False,
            "commands": self.command_specs(),
        }
