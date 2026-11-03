"""get_environment() — resolve an EnvSpec to an Environment by type.

Types none/local are built in here. docker/deferred are provided by
environments/docker.py which registers itself (imported lazily so the package
imports cleanly even where the docker SDK / daemon is unavailable).
"""

from __future__ import annotations

from typing import Any, Callable

from agentcore.config.schema import EnvSpec
from agentcore.environments.base import LocalEnvironment, NoneEnvironment

_REGISTRY: dict[str, Callable[..., object]] = {
    "none": NoneEnvironment,
    "local": LocalEnvironment,
}


def register_environment(key: str, builder: Callable[..., object]) -> None:
    _REGISTRY[key] = builder


def get_environment(spec: EnvSpec, **ctx: Any):
    if spec.type in ("docker", "deferred") and spec.type not in _REGISTRY:
        # lazy import: registers DockerEnvironment / DeferredEnvironment
        try:
            import agentcore.environments.docker  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"env type {spec.type!r} requires the docker backend, which failed to load: {e}") from e
    builder = _REGISTRY.get(spec.type)
    if builder is None:
        raise ValueError(f"Unknown environment type {spec.type!r}; registered: {list(_REGISTRY)}")
    return builder(spec, **ctx)
