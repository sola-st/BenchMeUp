"""DockerEnvironment and DeferredEnvironment.

The owned docker backend, lifted+unified from ExecutionAgent and analysis_agent.
Connects to the docker daemon LAZILY (inside methods), so importing this module
only requires the package to be present — never a running daemon. Importing it
registers both environment builders with the factory.

  * DockerEnvironment   — runs commands inside a long-lived container built/started
                          from ``spec.image``. checkpoint()=docker commit -> image id;
                          restore() recreates a container from that image.
  * DeferredEnvironment — starts with NO container; execute() runs on the HOST
                          workspace via subprocess until reprovision()/build_and_start()
                          stands up a container, after which execute() runs in-container.
                          Models "the LLM authors the Dockerfile mid-run, we build it".
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Optional

from agentcore.config.schema import EnvSpec
from agentcore.environments import docker_helpers as dh
from agentcore.environments.factory import register_environment
from agentcore.protocols import ExecResult


class DockerEnvironment:
    """Execution environment backed by a long-lived docker container.

    If ``spec.image`` is None, the env stays in a no-container state until
    ``reprovision(spec)`` (with an image) or ``build_and_start(dockerfile_text)``
    is called.
    """

    def __init__(
        self,
        spec: EnvSpec,
        *,
        workspace: str | None = None,
        instance: dict | None = None,
        out_dir: str | None = None,
        **_: Any,
    ):
        self.spec = spec
        self.instance = instance or {}
        self.out_dir = out_dir
        self.workspace = os.path.abspath(workspace) if workspace else (spec.workdir or "/workspace")
        self._container = None  # live docker container object (or None)
        self._revived = False  # env container died and was recreated
        self._image: Optional[str] = None  # the image our container runs on
        self._built_image: Optional[str] = None  # an image WE built (cleanup may rmi it)
        self._use_screen = bool(getattr(spec, "resources", {}) and spec.resources.get("stateful_screen", False))
        if spec.image:
            # Lazily provisioned on first use to avoid touching the daemon at construction
            self._pending_image = spec.image
        else:
            self._pending_image = None

    # ── provisioning ─────────────────────────────────────────────────────────

    def _ensure_container(self):
        """Materialize the container from the pending image, if needed -- and REVIVE
        it when it died behind our back (candidate sandboxes auto-rebuild via _ensure_live;
        the main environment does it here). A reload-less stub (unit tests set _container = object()) is
        assumed alive -- probing must not break stubbed tests."""
        if self._container is not None:
            reload = getattr(self._container, "reload", None)
            alive = True
            if callable(reload):
                try:
                    reload()
                    alive = getattr(self._container, "status", "running") == "running"
                except Exception:  # noqa: BLE001 - removed/unreachable = dead
                    alive = False
            if alive:
                return self._container
            try:
                dh.cleanup(self._container, image=None)
            except Exception:  # noqa: BLE001
                pass
            self._container = None
            self._revived = True
            if self._image:
                self._pending_image = self._image
        if self._pending_image:
            self._start_from_image(self._pending_image)
            self._pending_image = None
        return self._container

    def _start_from_image(self, image: str) -> None:
        c = dh.start_container(
            image,
            lifetime=self.spec.container_lifetime,
            resources=self.spec.resources or {},
            env=dh.llm_key_passthrough_env(self.spec.env or {}),
            workdir=self.spec.workdir or None,
        )
        self._container = c
        self._image = image
        if self.spec.workdir:
            dh.exec_in_container(c, f"mkdir -p {self.spec.workdir!r} || mkdir -p '{self.spec.workdir}'")
        # Credentials manifest: the whitelisted tokens are in
        # the process env, but a FILE is discoverable ground truth -- a model can `cat`
        # it instead of trusting prompt claims about invisible state. Names +
        # values, shell-sourceable, owner-read-only.
        dh.exec_in_container(c, dh.tokens_manifest_cmd())
        if self._use_screen:
            dh.create_screen_session(c)

    def build_and_start(self, dockerfile_text: str, tag: Optional[str] = None) -> str:
        """Build an image from raw Dockerfile text and (re)start the container on it.

        Used by tools / deferred flows where the LLM authors the Dockerfile mid-run.
        Returns the built image tag.
        """
        built = dh.build_image_from_dockerfile(dockerfile_text, tag)
        # tear down any existing container before switching
        if self._container is not None:
            dh.cleanup(self._container, image=None)
            self._container = None
        self._built_image = built
        self._start_from_image(built)
        return built

    # ── Environment protocol ──────────────────────────────────────────────────

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> ExecResult:
        c = self._ensure_container()
        if c is None:
            return ExecResult(
                output="",
                returncode=126,
                stderr=("No container is running yet (env.image is unset). Build/provision an image first."),
            )
        eff_timeout = timeout or self.spec.timeout or dh.DEFAULT_EXEC_TIMEOUT
        # docker's exec API requires an ABSOLUTE workdir -- a relative cwd (very natural for a
        # model to pass, e.g. ".") is not an error on a host shell but IS one here ("Cwd must
        # be an absolute path"). Resolve it against spec.workdir instead of passing it through.
        base = self.spec.workdir or "/workspace"
        if cwd and not os.path.isabs(cwd):
            workdir = os.path.normpath(os.path.join(base, cwd))
        else:
            workdir = cwd or base or None

        if self._use_screen:
            rc, out, _logfile, stuck = dh.exec_in_screen(c, command, stuck_timeout=eff_timeout)
            return ExecResult(output=out, returncode=rc, timed_out=stuck or rc == 124)

        rc, out = dh.exec_in_container(c, command, timeout=eff_timeout, workdir=workdir)
        if self._revived:
            self._revived = False
            out = (
                "[the environment container had DIED and was recreated from its "
                "image -- files you wrote here before are GONE; re-create what "
                "you need]\n" + out
            )
        timed_out = rc == 124 and "timed out" in out
        return ExecResult(output=dh.strip_ansi(out), returncode=rc, timed_out=timed_out)

    def reprovision(self, spec: EnvSpec) -> None:
        """(Re)build/start the underlying environment from the new spec.

        If ``spec.provision`` points at a Dockerfile, build from it; else use
        ``spec.image``. Tears down any existing container first.
        """
        self.spec = spec
        dockerfile_text = self._read_provision_dockerfile(spec)
        if self._container is not None:
            dh.cleanup(self._container, image=None)
            self._container = None
        self._image = None
        if dockerfile_text:
            self.build_and_start(dockerfile_text)
        elif spec.image:
            self._pending_image = spec.image
            self._ensure_container()
        else:
            self._pending_image = None  # no-container state

    @staticmethod
    def _read_provision_dockerfile(spec: EnvSpec) -> Optional[str]:
        prov = getattr(spec, "provision", None)
        if not prov:
            return None
        try:
            if os.path.isfile(prov):
                with open(prov, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception:
            return None
        # a provision value that is itself Dockerfile text (contains FROM)
        if isinstance(prov, str) and "FROM " in prov:
            return prov
        return None

    def checkpoint(self) -> str:
        """Snapshot the live container via ``docker commit`` -> returns the image id handle.

        The commit is TAGGED, never left dangling. An untagged committed image is
        exactly what `docker image prune` deletes, and a routine prune would destroy
        snapshots that are referenced only from instances.json. A benchsnap/ tag makes every snapshot visible to
        `docker images`, attributable, and prune-safe; the recorded handle stays the
        immutable sha256 id, so records never depend on the tag.
        """
        c = self._ensure_container()
        if c is None:
            return ""
        try:
            # Through a long-timeout client: the default 60s silently loses any layer over
            # ~5.5 GB, which is most of a built Java repo (see dh.COMMIT_TIMEOUT).
            image = dh.commit_container(c)
            handle = getattr(image, "id", None) or getattr(image, "short_id", "") or ""
            if handle:
                try:
                    name = re.sub(r"[^a-z0-9_.-]+", "-", str(getattr(c, "name", "") or "snap").lower())[:100]
                    image.tag("benchsnap/" + (name or "snap"), tag=str(handle).replace("sha256:", "")[:12])
                except Exception:
                    pass  # tagging is protection, not a precondition of snapshotting
            return str(handle)
        except Exception:
            return ""

    def restore(self, handle: str) -> None:
        """Recreate the container from a previously committed image id."""
        if not handle:
            return
        if self._container is not None:
            dh.cleanup(self._container, image=None)
            self._container = None
        self._start_from_image(handle)

    def cleanup(self) -> None:
        """Authoritative teardown: stop+remove the container, rmi any image WE built."""
        dh.cleanup(self._container, image=self._built_image)
        self._container = None
        self._built_image = None

    def _resolve_path(self, path: str) -> str:
        """dh.put_file/get_file shell out with no workdir concept of their own (unlike
        execute(), which resolves a relative cwd against spec.workdir) -- so a relative
        path here needs the same resolution, or it lands wherever the container's own
        default exec directory happens to be, not spec.workdir."""
        if os.path.isabs(path):
            return path
        base = self.spec.workdir or "/workspace"
        return os.path.normpath(os.path.join(base, path))

    def put_file(self, path: str, content: str) -> Optional[str]:
        c = self._ensure_container()
        if c is None:
            return "No container running; cannot write file."
        return dh.put_file(c, self._resolve_path(path), content)

    def get_file(self, path: str) -> str:
        c = self._ensure_container()
        if c is None:
            return "No container running; cannot read file."
        return dh.get_file(c, self._resolve_path(path))

    def get_template_vars(self) -> dict[str, Any]:
        present = self._container is not None or bool(self._pending_image)
        usage = None
        if self._container is not None:
            usage = dh.disk_usage(self._container)
        return {
            "env_type": "docker",
            "container_present": present,
            "image": self._image or self.spec.image,
            "workdir": self.spec.workdir,
            "disk_usage_bytes": usage,
            "disk_usage_human": dh.format_bytes(usage) if usage is not None else None,
        }


class DeferredEnvironment(DockerEnvironment):
    """Docker environment that starts WITHOUT a container.

    Until a container is built/started, ``execute()`` runs on the HOST workspace via
    subprocess. After ``reprovision()`` / ``build_and_start()`` stands up a container,
    ``execute()`` runs inside the container. Models the "LLM authors a Dockerfile
    mid-run, we build it" workflow.
    """

    def __init__(
        self,
        spec: EnvSpec,
        *,
        workspace: str | None = None,
        instance: dict | None = None,
        out_dir: str | None = None,
        **_: Any,
    ):
        # Force no-container start regardless of spec.image: the deferred contract is
        # "host until built". A provided image is treated as the pending build target.
        super().__init__(spec, workspace=workspace, instance=instance, out_dir=out_dir)
        self._host_workspace = os.path.abspath(
            workspace or spec.workdir or tempfile.mkdtemp(prefix="agentcore_deferred_")
        )
        os.makedirs(self._host_workspace, exist_ok=True)
        self.workspace = self._host_workspace

    def _has_container(self) -> bool:
        return self._container is not None or bool(self._pending_image)

    # AN INSTRUCTION, NOT AN EXPLANATION: a description of how the system works is read as
    # background and followed by another command.
    NO_MACHINE = (
        "NOTHING RAN: you have no machine yet. DO THIS NOW: write_to_file "
        "path=bench/Dockerfile with a FROM line and the packages this candidate needs. "
        "The system builds your machine the moment that file exists, and your commands "
        "run inside it from then on. Until then no command runs, whatever it is; "
        "writing and reading files under bench/ works meanwhile."
    )

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> ExecResult:
        if self._has_container():
            return super().execute(command, cwd=cwd, timeout=timeout)
        # NO HOST FALLBACK. Until the Dockerfile builds, commands do not run on the host. The
        # worker's own machine is the container it builds; before that there is no
        # machine, and the refusal says what creates one.
        return ExecResult(output="", stderr=self.NO_MACHINE, returncode=1)

    def put_file(self, path: str, content: str) -> Optional[str]:
        if self._has_container():
            return super().put_file(path, content)
        dest = path if os.path.isabs(path) else os.path.join(self._host_workspace, path)
        try:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(content)
            return None
        except Exception as e:
            return f"Failed to write host file {dest}: {e}"

    def get_file(self, path: str) -> str:
        if self._has_container():
            return super().get_file(path)
        src = path if os.path.isabs(path) else os.path.join(self._host_workspace, path)
        try:
            with open(src, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Failed to read host file {src}: {e}"

    # ── the workspace travels with the machine ─────────────────

    def workspace_archive(self) -> bytes | None:
        """The workspace as a tar (arcname = the workdir's basename), from the container when
        one exists, else from the host directory. None when there is nothing to carry."""
        import io
        import tarfile

        base = self.spec.workdir or "/workspace"
        if self._container is not None:
            try:
                stream, _ = self._container.get_archive(base)
                buf = io.BytesIO()
                for chunk in stream:
                    buf.write(chunk)
                return buf.getvalue()
            except Exception:  # noqa: BLE001 - an empty machine has nothing to carry
                return None
        if not os.path.isdir(self._host_workspace) or not os.listdir(self._host_workspace):
            return None
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(self._host_workspace, arcname=os.path.basename(base))
        return buf.getvalue()

    def build_and_start(self, dockerfile_text: str, tag: Optional[str] = None) -> str:
        """Build the image and enter it WITH the workspace: the files written so far (on the
        host before the first build, in the previous container after) are carried into the
        new container's workdir, and `/bench` is linked to the workspace's bench/ so the
        instance files sit where the checks put them."""
        carried = self.workspace_archive()
        built = super().build_and_start(dockerfile_text, tag)
        base = self.spec.workdir or "/workspace"
        c = self._container
        if c is not None:
            if carried:
                try:
                    c.put_archive(os.path.dirname(base) or "/", carried)
                except Exception as e:  # noqa: BLE001
                    dh.exec_in_container(c, f"echo 'workspace carry-over failed: {str(e)[:120]}' >&2")
            # AS ROOT, THEN HANDED TO THE MACHINE'S USER. The carried tar keeps the host uid,
            # and a Dockerfile that ends with a `USER` line leaves that user unable to write
            # bench/. `docker exec` runs as the image's USER, so the
            # setup here names root explicitly and gives the workdir to that user afterwards.
            setup = f"mkdir -p '{base}/bench' && ln -sfn '{base}/bench' /bench"
            user = dh.container_user(c)
            if user:
                setup += f" && chown -R '{user}' '{base}'"
            try:
                res = c.exec_run(["/bin/sh", "-lc", setup], user="root")
                if int(res.exit_code) != 0:
                    dh.exec_in_container(c, f"echo 'workspace setup failed: {str(res.output)[:160]}' >&2")
            except Exception as e:  # noqa: BLE001
                dh.exec_in_container(c, f"echo 'workspace setup failed: {str(e)[:120]}' >&2")
        return built

    def get_template_vars(self) -> dict[str, Any]:
        vars_ = super().get_template_vars()
        vars_["env_type"] = "deferred"
        vars_["container_present"] = self._container is not None
        vars_["host_workspace"] = self._host_workspace
        vars_["phase"] = "container" if self._container is not None else "host"
        return vars_


# ── register with the factory at import ───────────────────────────────────────

register_environment("docker", DockerEnvironment)
register_environment("deferred", DeferredEnvironment)
