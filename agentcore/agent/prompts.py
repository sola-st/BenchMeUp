"""PromptStore — resolve + render Jinja templates for an agent.

Logical names (system/cycle/summarize/...) come from AgentSpec.prompts; their
values are file paths relative to the agent directory. Stage prompts are paths
too. Rendering uses a lenient Undefined so a template referencing an absent var
renders empty rather than crashing a long run.
"""

from __future__ import annotations

import os
from typing import Any

from jinja2 import Environment as JinjaEnv
from jinja2 import FileSystemLoader, Undefined


class PromptStore:
    def __init__(self, base_dir: str, names: dict[str, str]):
        self.base_dir = base_dir
        self.names = dict(names or {})
        # A FileSystemLoader rooted at the agent dir lets templates pull fragments from
        # disk via {% include "frag.md" %} — including a path chosen at render time
        # (e.g. {% include stage_doc %}) for per-stage prompt augmentation.
        self._jinja = JinjaEnv(
            loader=FileSystemLoader(base_dir), undefined=Undefined, trim_blocks=False, lstrip_blocks=False
        )
        self._cache: dict[str, str] = {}

    def read_file(self, rel_path: str, default: str = "") -> str:
        """Read a file under the agent dir (helper for prerender hooks). Missing -> default."""
        p = rel_path if os.path.isabs(rel_path) else os.path.join(self.base_dir, rel_path)
        try:
            with open(p) as f:
                return f.read()
        except OSError:
            return default

    def _read(self, path: str) -> str:
        if path in self._cache:
            return self._cache[path]
        # allow inline templates (contain a space/newline or no file) vs file paths
        candidate = path if os.path.isabs(path) else os.path.join(self.base_dir, path)
        if os.path.isfile(candidate):
            with open(candidate) as f:
                text = f.read()
        else:
            text = path  # treat the value itself as an inline template
        self._cache[path] = text
        return text

    def has(self, name: str) -> bool:
        return name in self.names

    def render(self, name_or_path: str, **vars: Any) -> str:
        path = self.names.get(name_or_path, name_or_path)
        template_text = self._read(path)
        return self._jinja.from_string(template_text).render(**vars)
