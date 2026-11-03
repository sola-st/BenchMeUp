#!/usr/bin/env python3
"""Stage 5, packaging: every delivered instance of a task's construction runs -> one benchmark.

No model is called. For each worker run under the given directories, the instances whose replay
report holds (recipes/<candidate>/report.json ok) are copied with the task's contract, a
manifest carrying each instance's verdict rows, a README rendered from the contract, and the
stand-alone `bench` interpreter (benchcompiler.artifact_recipes.write_benchmark).

  python scripts/package.py --task boca --runs runs/construct/boca --out benchmarks/mine/boca
  python scripts/package.py --task d4j --runs runs/construct/d4j/epoch_1 runs/construct/d4j/epoch_2 \\
      --out benchmarks/mine/d4j --name "d4j"

Then certify the result by replaying every instance in a fresh container:
  python scripts/certify_packaged.py benchmarks/mine/boca --contract agents/boca_worker/contract.json --fresh
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchcompiler.artifact_recipes import write_benchmark  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, help="boca, sonarqube, d4j or swt (agents/<task>_worker)")
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="construction run dirs (an epoch dir holding worker_*/, or a parent of epochs)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--contract", default=None, help="default: agents/<task>_worker/contract.json")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    contract = a.contract or os.path.join(ROOT, "agents", f"{a.task}_worker", "contract.json")
    missing = [d for d in a.runs if not os.path.isdir(d)]
    if missing:
        print(f"not a directory: {missing}", file=sys.stderr)
        return 2
    m = write_benchmark(a.runs, contract, a.out, name=a.name or a.task)
    print(json.dumps({"out": a.out, "count": m["count"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
