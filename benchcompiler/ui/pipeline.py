"""The pipeline map: the steps that turn a task description into a certified benchmark, laid out
as a graph, each step coloured by where the chosen task stands (done, running, waiting, not
started), with the command that runs it and what it writes.

Three rows: formalization and search on the first, the worker's generation on the second (it
feeds the hand-over), construction stages, packaging and certification on the third.
"""

from __future__ import annotations

import os

WIDTH = 1360
GAP = 26
NODE_H = 132
ROW_GAP = 64


def _status(done: bool, running: bool = False, waiting: bool = False) -> str:
    return "running" if running else "done" if done else "waiting" if waiting else "todo"


def pipeline(tree, task: str, distinct_delivered, builds) -> dict:
    spec, rules, plan = tree.spec(task), tree.rules(task), tree.plan(task)
    camps, runs = tree.search_campaigns(task), tree.construction_runs(task)
    stocks = tree.stocks(task)
    sets = [b for b in tree.benchmark_sets() if b["task"] == task]
    funnel = tree.funnel(task, [r["key"] for r in runs])
    delivered = distinct_delivered(runs)
    benches = [tree.benchmark(b["set"], task) for b in sets]
    accepted = sum(1 for b in benches for r in b["instances"] if r["accepted"])
    certs = sum(len(b["certify_records"]) for b in benches)
    found = sum(c["candidates"] for c in camps)
    n_stages = len(funnel["stages"]) if plan["exists"] else 0
    per_model = [
        (
            g["model"].split("-")[1] if g["model"].startswith("claude-") else g["model"].split("-")[0],
            g["funnel"]["delivered"],
        )
        for g in builds(tree, task, runs, sets)
    ]
    t = task

    nodes = [
        dict(
            id="spec",
            col=0,
            row=0,
            title="task description",
            status=_status(bool(spec["text"])),
            value="written",
            facts=["the only hand-written input"],
            cmd=None,
            out=f"specs/{t}/spec.yaml",
            url=f"/task/{t}?tab=spec",
        ),
        dict(
            id="formalize",
            col=1,
            row=0,
            n=1,
            title="formalize",
            status=_status(bool(rules["outcome"])),
            value=(rules["outcome"] or "-").lower(),
            facts=[
                "identity: " + ", ".join(rules["brief"].get("identity_fields") or [])
                if rules.get("brief")
                else "no rules yet"
            ],
            cmd="bench formalize",
            out=f"specs/{t}/canonical.json",
            url=f"/task/{t}?tab=rules",
        ),
        dict(
            id="searchgen",
            col=2,
            row=0,
            n=2,
            title="search agent",
            status=_status(os.path.isdir(os.path.join(tree.root, "agents", f"{t}_search"))),
            value="brief",
            facts=["what a candidate is, where to look"],
            cmd="bench search-agent",
            out=f"agents/{t}_search/",
            url=f"/task/{t}?tab=rules",
        ),
        dict(
            id="search",
            col=3,
            row=0,
            n=2,
            title="search",
            status=_status(bool(camps), running=any(c["running"] for c in camps)),
            value=f"{found} found",
            facts=[
                f"{len(camps)} campaigns · ${sum(c['cost'] for c in camps):.2f}",
                f"{sum(c['running'] for c in camps)} agents running" if any(c["running"] for c in camps) else "",
            ],
            cmd="scripts/search_campaign.py",
            out=f"runs/search/{t}/epoch_*/",
            url=f"/task/{t}?tab=search",
        ),
        dict(
            id="freeze",
            col=4,
            row=0,
            n=2,
            title="freeze",
            status=_status(bool(stocks)),
            value=f"{sum(s['count'] for s in stocks)} frozen",
            facts=[", ".join(s["name"] for s in stocks) or "nothing frozen"],
            cmd="bench stock-freeze",
            out=f"stock/{t}*/candidates.jsonl",
            url=f"/task/{t}?tab=search",
        ),
        dict(
            id="handover",
            col=5,
            row=0,
            n=3,
            title="hand-over",
            status=_status(bool(runs), running=any(r["running"] for r in runs)),
            value=f"{funnel['taken']} taken",
            facts=[
                f"{len(runs)} runs · {funnel['workers']} workers",
                f"{funnel['now']} being built now" if funnel["now"] else "",
            ],
            cmd="scripts/construct_arms.py",
            out=f"runs/construct/{t}/",
            url=f"/task/{t}?tab=construction",
        ),
        dict(
            id="workergen",
            col=2,
            row=1,
            n=3,
            title="worker + plan",
            status=_status(plan["exists"]),
            value=f"{n_stages} stages" if plan["exists"] else "-",
            facts=[f"oracle: {plan['oracle'].get('kind') or '-'}" if plan["exists"] else "no plan yet"],
            cmd="bench worker",
            out=f"agents/{t}_worker/contract.json",
            url=f"/task/{t}?tab=plan",
        ),
    ]
    for s in funnel["stages"][:n_stages]:
        nodes.append(
            dict(
                id=f"stage{s['n']}",
                col=s["n"] - 1,
                row=2,
                n=3,
                title=f"stage {s['n']}",
                status=_status(s["passed"] > 0, running=s["now"] > 0),
                value=f"{s['passed']} / {s['entered']}",
                name=s.get("name"),
                facts=[
                    "passed / entered",
                    " · ".join(
                        x
                        for x in (
                            f"{s['dropped']} dropped" if s["dropped"] else "",
                            f"{s['now']} now" if s["now"] else "",
                        )
                        if x
                    ),
                ],
                cmd=None,
                out=None,
                url=f"/task/{t}?tab=plan",
            )
        )
    c = n_stages
    nodes += [
        dict(
            id="deliver",
            col=c,
            row=2,
            n=3,
            title="deliver",
            status=_status(delivered > 0),
            value=f"{funnel['delivered']} delivered",
            facts=[
                " · ".join(f"{m} {n}" for m, n in per_model) if len(per_model) > 1 else "",
                f"{delivered} distinct candidates" if delivered != funnel["delivered"] else "",
            ],
            cmd="bench_deliver (by the worker)",
            out="runs/.../recipes/<candidate>/instance/",
            url=f"/task/{t}?tab=construction",
        ),
        dict(
            id="package",
            col=c + 1,
            row=2,
            n=4,
            title="package",
            status=_status(bool(sets), waiting=delivered > 0),
            value=f"{sum(b['count'] for b in sets)} packaged" if sets else "not yet",
            facts=[" · ".join(f"{b['set']} {b['count']}" for b in sets) if sets else f"{delivered} waiting"],
            cmd="scripts/package.py",
            out=f"benchmarks/<set>/{t}/",
            url=f"/task/{t}?tab=benchmarks",
        ),
        dict(
            id="certify",
            col=c + 2,
            row=2,
            n=4,
            title="certify",
            status=_status(certs > 0, waiting=bool(sets)),
            value=f"{accepted} accepted" if sets else "-",
            facts=[f"{certs} certification records" if sets else ""],
            cmd="scripts/certify_packaged.py",
            out="certify*.json beside the manifest",
            url=f"/task/{t}?tab=benchmarks",
        ),
    ]

    cols = max(6, c + 3)
    w = (WIDTH - GAP * (cols - 1)) / cols
    pos = {}
    for n in nodes:
        n["x"], n["y"], n["w"], n["h"] = n["col"] * (w + GAP), n["row"] * (NODE_H + ROW_GAP), w, NODE_H
        n["facts"] = [f for f in n["facts"] if f]
        pos[n["id"]] = n

    def right(a, b):
        A, B = pos[a], pos[b]
        y = A["y"] + NODE_H / 2
        return f"M{A['x'] + w:.0f},{y:.0f} L{B['x'] - 2:.0f},{y:.0f}"

    edges = [
        right("spec", "formalize"),
        right("formalize", "searchgen"),
        right("searchgen", "search"),
        right("search", "freeze"),
        right("freeze", "handover"),
    ]
    F, G, H = pos["formalize"], pos["workergen"], pos["handover"]
    gy = G["y"] + NODE_H / 2
    edges.append(
        f"M{F['x'] + w / 2:.0f},{F['y'] + NODE_H:.0f} L{F['x'] + w / 2:.0f},{gy:.0f} L{G['x'] - 2:.0f},{gy:.0f}"
    )
    hx_in = H["x"] + w * 0.3
    edges.append(f"M{G['x'] + w:.0f},{gy:.0f} L{hx_in:.0f},{gy:.0f} L{hx_in:.0f},{H['y'] + NODE_H + 2:.0f}")
    first = pos["stage1"] if n_stages else pos["deliver"]
    hx_out, gutter = H["x"] + w * 0.7, first["y"] - ROW_GAP / 2
    edges.append(
        f"M{hx_out:.0f},{H['y'] + NODE_H:.0f} L{hx_out:.0f},{gutter:.0f} "
        f"L{first['x'] + w / 2:.0f},{gutter:.0f} L{first['x'] + w / 2:.0f},{first['y'] - 2:.0f}"
    )
    chain = [f"stage{i + 1}" for i in range(n_stages)] + ["deliver", "package", "certify"]
    edges += [right(a, b) for a, b in zip(chain, chain[1:])]
    return {"nodes": nodes, "edges": edges, "width": WIDTH, "height": 3 * NODE_H + 2 * ROW_GAP + 4}
