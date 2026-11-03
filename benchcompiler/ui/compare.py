"""Comparisons across models and runs: what each model's search and construction cost, what it
delivered, and how each construction run of a task did, in time order."""

from __future__ import annotations

from benchcompiler.ui import data as D


def _workers(tree, keys):
    import glob
    import os

    for k in keys:
        for w in sorted(glob.glob(os.path.join(tree.run_dir(k), "worker_*")), key=D.natural_key):
            if os.path.isdir(w):
                yield D.agent_status(w)


def by_model(tree, task: str, builds) -> list[dict]:
    """One row per construction model of a task: taken, delivered, dropped, cost, hours, and the
    cost and hours per delivery."""
    rows = []
    for g in builds(tree, task, tree.construction_runs(task), [b for b in tree.benchmark_sets() if b["task"] == task]):
        sts = list(_workers(tree, [r["key"] for r in g["runs"]]))
        cost = sum(s["cost"] or 0 for s in sts)
        hours = sum(s["wall"] or 0 for s in sts) / 3600
        f = g["funnel"]
        dropped = sum(s["dropped"] for s in f["stages"])
        rows.append(
            {
                "model": g["model"],
                "runs": len(g["runs"]),
                "workers": len(sts),
                "taken": f["taken"],
                "delivered": f["delivered"],
                "distinct": f["distinct"],
                "dropped": dropped,
                "rate": f["delivered"] / f["taken"] if f["taken"] else None,
                "cost": cost,
                "hours": hours,
                "cost_per": cost / f["delivered"] if f["delivered"] else None,
                "hours_per": hours / f["delivered"] if f["delivered"] else None,
                "packaged": sum(b["count"] for b in g["benchmarks"]),
            }
        )
    return rows


def search_by_model(tree, task: str, norm=lambda m: m) -> list[dict]:
    """norm: the model's display name, so one model reached by two routes is one row."""
    rows: dict[str, dict] = {}
    for c in tree.search_campaigns(task):
        m = norm(c["model"]) if c["model"] else "-"
        r = rows.setdefault(m, {"model": m, "campaigns": 0, "agents": 0, "found": 0, "cost": 0.0})
        r["campaigns"] += 1
        r["agents"] += c["agents"]
        r["found"] += c["candidates"]
        r["cost"] += c["cost"]
    for r in rows.values():
        r["per_dollar"] = r["found"] / r["cost"] if r["cost"] else None
    return list(rows.values())


def trend(tree, task: str, norm=lambda m: m) -> list[dict]:
    """Construction runs of a task in start order: delivered, dropped, cost, model.
    The label is the run's own name: the epoch (construct/<task>/<epoch>) or the experiment
    (experiment/<name>/<task>)."""
    return [
        {
            "key": r["key"],
            "label": r["label"].split("/")[-1] if r["label"].startswith("construct/") else r["label"].split("/")[1],
            "delivered": r["delivered"],
            "dropped": r["dropped"],
            "cost": r["cost"],
            "model": norm(r["model"]) if r["model"] else "-",
            "running": r["running"],
            "started": r["started"],
        }
        for r in sorted(tree.construction_runs(task), key=lambda r: r["started"] or 0)
    ]


def overall(tree, builds) -> list[dict]:
    """Every task x model: the by_model row with its task."""
    return [{"task": t, **row} for t in tree.tasks() for row in by_model(tree, t, builds)]
