"""Core protocols and lightweight runtime result types for agentcore.

These are duck-typed Protocols — concrete classes need not inherit from them,
they only need to match the shape. Keeping them here (with no heavy imports)
means every other module can depend on the *contract* without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid import cycles; these are only needed for type hints
    from agentcore.config.schema import EnvSpec
    from agentcore.tools.context import ToolContext
    from agentcore.tools.result import ToolResult


# ── runtime result types ─────────────────────────────────────────────────────


@dataclass
class ModelResponse:
    """Normalized chat-completion response."""

    content: str
    raw: dict = field(default_factory=dict)
    cost: float = 0.0  # USD for this single call (delta), if known
    finish_reason: str = ""
    # Which endpoint served the call: the response's `provider` and generation id, when it
    # reports them; None otherwise. The loop writes them on the on_response event.
    provider: str | None = None
    generation_id: str | None = None

    # dict-style access, so callers may write resp["content"]
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class ExecResult:
    """Result of running a command in an Environment."""

    output: str = ""
    returncode: int = 0
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# ── protocols ────────────────────────────────────────────────────────────────


@runtime_checkable
class Model(Protocol):
    """Language model. Concrete: agentcore.models.litellm_model.LitellmModel."""

    name: str

    def query(self, messages: list[dict], **kwargs) -> ModelResponse: ...

    def cost(self) -> float:
        """Cumulative USD spent by this model instance."""
        ...

    def get_template_vars(self) -> dict[str, Any]:
        """Vars exposed to Jinja prompt templates (e.g. model name, cost)."""
        ...


@runtime_checkable
class Environment(Protocol):
    """Execution environment (none/local/docker/deferred)."""

    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> ExecResult: ...

    def reprovision(self, spec: "EnvSpec") -> None:
        """(Re)build/switch the underlying environment — deferred + mid-run image switch."""
        ...

    def checkpoint(self) -> str:
        """Snapshot current state; return an opaque handle."""
        ...

    def restore(self, handle: str) -> None:
        """Restore a previously taken checkpoint."""
        ...

    def cleanup(self) -> None:
        """Authoritatively release all resources (containers/images/temp dirs)."""
        ...

    def get_template_vars(self) -> dict[str, Any]: ...


@runtime_checkable
class Tool(Protocol):
    """A registered agent action. See agentcore.tools.registry.tool decorator."""

    name: str
    schema: dict  # JSON schema derived from the function signature

    def __call__(self, ctx: "ToolContext", **args) -> "ToolResult": ...


@runtime_checkable
class Oracle(Protocol):
    """Decides success / progress. See agentcore.oracle.*"""

    def check(self, ctx: "ToolContext") -> "OracleResult": ...


@runtime_checkable
class InstanceProvider(Protocol):
    """Loads task instances. See agentcore.instances.provider."""

    def load(self) -> list[dict]: ...

    def resolve_assets(self, instance: dict) -> dict: ...


# OracleResult is imported lazily here for the Protocol annotation only.
if TYPE_CHECKING:
    from agentcore.oracle.base import OracleResult
