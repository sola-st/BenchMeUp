"""get_model() — resolve a ModelSpec to a Model via a name-keyed registry.

Register custom model classes with @model_backend("name") or register_model_backend().
"""

from __future__ import annotations

from typing import Callable

from agentcore.config.schema import ModelSpec
from agentcore.models.litellm_model import LitellmModel

_REGISTRY: dict[str, Callable[[ModelSpec], object]] = {}


def register_model_backend(key: str, builder: Callable[[ModelSpec], object]) -> None:
    _REGISTRY[key] = builder


def model_backend(key: str):
    def deco(builder):
        register_model_backend(key, builder)
        return builder

    return deco


def _build_litellm(spec: ModelSpec):
    return LitellmModel(spec.name, temperature=spec.temperature, cache_control=spec.cache_control, kwargs=spec.kwargs)


register_model_backend("litellm", _build_litellm)


def _build_scripted(spec: ModelSpec):
    from agentcore.models.scripted import ScriptedModel

    return ScriptedModel(spec.name, **spec.kwargs)


register_model_backend("scripted", _build_scripted)


def _build_claude_code(spec: ModelSpec):
    from agentcore.models.claude_code import ClaudeCodeModel

    return ClaudeCodeModel(
        spec.name, temperature=spec.temperature, cache_control=spec.cache_control, kwargs=spec.kwargs
    )


register_model_backend("claude-code", _build_claude_code)


def get_model(spec: ModelSpec):
    # A `claude-code/<id>` NAME selects the CLI whatever the configured class says, so an
    # agent moves onto it by changing one string (its config, or AGENT_MODEL at launch).
    from agentcore.models.claude_code import is_claude_code

    if is_claude_code(spec.name) and spec.class_ != "claude-code":
        spec = spec.model_copy(update={"class_": "claude-code"}) if hasattr(spec, "model_copy") else _retyped(spec)
    builder = _REGISTRY.get(spec.class_)
    if builder is None:
        raise ValueError(f"Unknown model backend {spec.class_!r}; registered: {list(_REGISTRY)}")
    return builder(spec)


def _retyped(spec: ModelSpec) -> ModelSpec:
    """The same spec with class_ = claude-code, for a schema without model_copy."""
    import copy

    out = copy.copy(spec)
    try:
        out.class_ = "claude-code"
    except Exception:  # noqa: BLE001 - a frozen spec is rebuilt from its fields
        out = ModelSpec(**{**{k: getattr(spec, k) for k in spec.__dataclass_fields__}, "class_": "claude-code"})
    return out


def build_models(specs: dict[str, ModelSpec]) -> dict[str, object]:
    """Build every named model in an AgentSpec.models mapping."""
    return {role: get_model(s) for role, s in specs.items()}
