#!/usr/bin/env python3
"""Certify a PACKAGED benchmark folder (benchmarks/<set>/<task>, the write_benchmark layout):
replay every instance from its packaged files in a fresh container under the task's contract,
several at a time, and write certify.json beside manifest.json.

A packaged folder has manifest.json, instances/<id>/ and the worker's contract.

  usage: scripts/certify_packaged.py BENCH_DIR --contract agents/<task>_worker/contract.json
                                     [--jobs 4] [--only id-suffix,...]
Run it WITHOUT BENCH_BUILD_ARGS to test that the Dockerfiles build on a machine without the
host's caches (their defaults are the vendors'); with it, to test against the cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)
from agentcore import recipes  # noqa: E402
from benchcompiler.recipes_pool import _summary  # noqa: E402


def one(bench: str, contract: dict, iid: str, log, tag_prefix: str = "benchrecipe") -> dict:
    d = os.path.join(bench, "instances", iid)
    t0 = time.time()
    try:
        rep = recipes.Replay(d, contract, tag_prefix=tag_prefix, log=(lambda s, i=iid[-8:]: log(f"{i}: {s}"))).run()
    except Exception as e:  # noqa: BLE001 - a crashed replay is a result, not the end of the run
        return {"id": iid, "ok": False, "detail": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0)}
    return {
        "id": iid,
        "ok": rep.ok,
        "report": rep.to_dict(),
        "summary": _summary(rep.to_dict()),
        "seconds": round(time.time() - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--contract", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="build every image anew under its own tag instead of reusing the images the "
        "workers' checks left (which are matched by Dockerfile content, whatever "
        "build args they were built with)",
    )
    out_name = "certify.json"
    a = ap.parse_args()
    tag_prefix = f"benchcertify{int(time.time())}" if a.fresh else "benchrecipe"
    if a.fresh:
        out_name = "certify_fresh.json"
    contract = recipes.load_contract(a.contract)
    ids = [i["id"] for i in json.load(open(os.path.join(a.bench, "manifest.json"), encoding="utf-8"))["instances"]]
    if a.only:
        want = set(a.only.split(","))
        ids = [i for i in ids if i in want or i[-8:] in want]
    log = lambda s: print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)  # noqa: E731
    log(
        f"certifying {len(ids)} instance(s), {a.jobs} at a time; "
        f"build args: {os.environ.get('BENCH_BUILD_ARGS') or 'none (vendor defaults)'}"
    )
    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(one, a.bench, contract, i, log, tag_prefix): i for i in ids}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            log(
                f"DONE {r['id'][-8:]}: {'OK' if r['ok'] else 'FAIL'} in {r.get('seconds')}s  "
                f"{json.dumps(r.get('summary'))[:200] if r.get('summary') else r.get('detail', '')}"
            )
    path = os.path.join(a.bench, out_name)
    if a.only and os.path.exists(path):
        # an --only run adds to the existing record instead of replacing it
        done = {r["id"] for r in results}
        results += [r for r in json.load(open(path, encoding="utf-8")).get("results", []) if r["id"] not in done]
    results.sort(key=lambda r: r["id"])
    summary = {
        "certified": sum(r["ok"] for r in results),
        "checked": len(results),
        "failed": [r["id"] for r in results if not r["ok"]],
        "jobs": a.jobs,
        "build_args": os.environ.get("BENCH_BUILD_ARGS") or None,
        "fresh_images": a.fresh,
        "ts": time.time(),
    }
    with open(os.path.join(a.bench, out_name), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    log(f"summary: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
