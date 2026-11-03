"""BenchMeUp — compiles a researcher's benchmark specification into an
executable benchmark-construction pipeline.

This package is the actual system, as opposed to study/ (literature-mining
research tooling used to design it) and spec/ (design documents).

Implemented so far:
    formalizer.py   stage 1: spec -> canonical spec, or a typed rejection

Not yet implemented: planning/pricing (stage 2), the build pipeline
(collection/quality/shipping — see spec/pipeline-v0.md).
"""

# One path bootstrap: `python -m benchcompiler.<module>` entry points need `agent-scaffold`
# importable, so this package, which runs before any of its modules, puts it on sys.path.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT,):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _ROOT
