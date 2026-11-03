"""Importing a benchcompiler module must not load .env secrets into os.environ.

`litellm` calls load_dotenv() on import, so `benchcompiler.llm` imports it lazily; imports stay pure
and the proxy variables arrive before the first completion call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_the_import_adds_no_environment_keys():
    code = (
        "import os, json, sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "before = set(os.environ)\n"
        "import benchcompiler.llm\n"
        "print(json.dumps(sorted(set(os.environ) - before)))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-400:]
    added = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"\n[verbose] env keys added by import: {added}")
    assert added == [], (
        f"importing benchcompiler modules loaded {added} into os.environ -- every test "
        f"that imports them poisons the environment for every test after it"
    )
