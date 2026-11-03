"""agentcore — a template scaffold for software-engineering AI agents.

Instantiate, don't reconstruct. See DESIGN.md for the architecture.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Quiet litellm's import-time provider warnings unless the user opts in.
import os as _os

_os.environ.setdefault("LITELLM_LOG", "ERROR")

from agentcore.protocols import (  # noqa: E402
    Environment,
    ExecResult,
    InstanceProvider,
    Model,
    ModelResponse,
    Oracle,
    Tool,
)

__all__ = [
    "__version__",
    "Model",
    "Environment",
    "Tool",
    "Oracle",
    "InstanceProvider",
    "ModelResponse",
    "ExecResult",
]
