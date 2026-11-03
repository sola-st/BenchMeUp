"""bench-ui: a read-only web view of the pipeline -- what each task has, what is running, what the
agents are doing, and what came out.

    pip install -e '.[ui]'
    bench-ui                         # http://127.0.0.1:8765, runs from ./runs
    bench-ui --runs /path/to/runs --port 9000

Nothing in the UI writes: it reads the repository (specs/, agents/, stock/, benchmarks/) and the
runs root. Pages that show running agents refresh their live parts every few seconds.

With --allow-actions, the Actions tab can also run the pipeline's commands (actions.py); each
one is checked first and needs a typed confirmation, and everything it ran is in /jobs.
"""

import argparse
import datetime
import json
import os
import re

from benchcompiler.ui import actions as A
from benchcompiler.ui import compare as C
from benchcompiler.ui import data as D
from benchcompiler.ui import health as H
from benchcompiler.ui import why as W
from benchcompiler.ui.action_routes import add_action_routes
from benchcompiler.ui.pipeline import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def _money(v):
    if v is None:
        return "-"
    if v == int(v) and v >= 1:
        return f"${int(v)}"
    return f"${v:,.2f}" if v >= 0.1 or v == 0 else f"${v:.4f}"


def _dur(s):
    if s is None:
        return "-"
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}h" if s % 3600 < 60 else f"{s // 3600}h{s % 3600 // 60:02d}"
    return f"{s // 60}m" if s >= 60 else f"{s}s"


def _key(k):
    """A candidate key as `owner/repo @ abcdef12`: the repository, then each hash shortened."""
    if not k:
        return "-"
    parts = [p for p in str(k).split("|") if p]
    out = []
    for p in parts:
        p = p.replace("https://github.com/", "").replace("http://github.com/", "")
        if p.endswith(".git"):
            p = p[:-4]
        out.append(p[:8] if len(p) >= 20 and all(ch in "0123456789abcdef" for ch in p) else p)
    return out[0] + (" @ " + " ".join(out[1:]) if len(out) > 1 else "")


def _model(m):
    if not m:
        return "-"
    for pre in ("openrouter/", "openai/", "claude-code/"):
        m = m.replace(pre, "")
    return m.rsplit("/", 1)[-1]


def _when(ts):
    if not ts:
        return "-"
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _pct(v, limit):
    try:
        return max(0, min(100, round(100 * float(v) / float(limit))))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0


def _short_key(k: str | None, n: int = 70) -> str:
    if not k:
        return "-"
    k = k.replace("https://github.com/", "")
    return k if len(k) <= n else k[: n - 1] + "…"


def _state_label(s: str | None) -> str:
    if not s:
        return "-"
    if s == "REGISTERED":
        return "stage 1"
    if s.startswith("STAGE_"):
        # STAGE_<k>_... is the state after stage k passed: the candidate is on stage k + 1.
        parts = s.split("_", 2)
        return f"stage {int(parts[1]) + 1}" if len(parts) > 1 and parts[1].isdigit() else s
    return s.lower()


def _stage_labels(item: dict) -> str:
    """A candidate chip's short label: delivered, dropped, or the stage it is on (S2)."""
    if item["kind"] in ("delivered", "dropped"):
        return item["kind"]
    lab = _state_label(item["state"])
    return "S" + lab.split()[1] if lab.startswith("stage ") else lab


def _cmdline(argv: list[str]) -> str:
    """A command as one would type it: this interpreter shown as python, arguments quoted when needed."""
    import shlex
    import sys

    return " ".join("python" if i == 0 and a == sys.executable else shlex.quote(a) for i, a in enumerate(argv or []))


def _distinct_delivered(runs: list[dict]) -> int:
    """Candidates delivered by any construction run of a task, each counted once."""
    return len({k for r in runs for k in r.get("delivered_keys") or []})


def _builds(tree, task: str, runs: list[dict], sets: list[dict]) -> list[dict]:
    """A task's construction runs grouped by the model that ran them, oldest first: each group's
    stage funnel and the packaged sets named after that model.
    """
    groups: dict[str, list[dict]] = {}
    for r in sorted(runs, key=lambda r: r["started"] or 0):
        groups.setdefault(_model(r["model"]), []).append(r)
    out = []
    for m, rs in groups.items():
        f = tree.funnel(task, [r["key"] for r in rs])
        f["distinct"] = _distinct_delivered(rs)
        words = set(re.split(r"[-/_.:]", m))
        out.append({"model": m, "runs": rs, "funnel": f, "benchmarks": [b for b in sets if b["set"] in words]})
    return out


def build_app(root: str = ROOT, runs: str | None = None, allow_actions: bool = False, static_export: bool = False):
    """allow_actions: the Actions tab can launch, freeze, package, certify and stop (actions.py).
    Only with the repository's own runs folder: the commands write there, and a UI reading
    another folder must not stop the runs it shows.
    static_export: pages rendered for bench-ui-export (no live parts, search, health, actions or
    host paths)."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    tree = D.Tree(root, runs)
    if allow_actions and os.path.realpath(tree.runs) != os.path.realpath(os.path.join(root, "runs")):
        raise SystemExit(
            f"--allow-actions needs the runs folder to be {os.path.join(root, 'runs')} "
            f"(the commands write there); this UI reads {tree.runs}. Drop --runs or --allow-actions."
        )
    env = Environment(loader=FileSystemLoader(os.path.join(HERE, "templates")), autoescape=select_autoescape(["html"]))
    env.filters.update(
        money=_money,
        dur=_dur,
        when=_when,
        ago=D.ago,
        short=_short_key,
        key=_key,
        model=_model,
        state=_state_label,
        cmdline=_cmdline,
        tojson_pretty=lambda v: json.dumps(v, indent=1, ensure_ascii=False),
    )
    env.globals.update(
        pct=_pct,
        actions_on=allow_actions and not static_export,
        static_export=static_export,
        env_status=A.env_status,
        action_list=lambda task: [b(tree, task) for b in A.BUILDERS.values()],
    )
    app = FastAPI(title="BenchMeUp UI", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

    def page(name: str, request: Request, crumbs: list, refresh: int = 0, **ctx):
        html = env.get_template(name).render(
            request=request, crumbs=crumbs, refresh=0 if static_export else refresh, tree=tree, **ctx
        )
        return HTMLResponse(html)

    def guard(fn):
        try:
            return fn()
        except ValueError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request):
        tasks = []
        for t in tree.tasks():
            camps = tree.search_campaigns(t)
            runs_ = tree.construction_runs(t)
            tasks.append(
                {
                    "task": t,
                    "rules": tree.rules(t)["outcome"],
                    "plan": tree.plan(t)["exists"],
                    "stocks": tree.stocks(t),
                    "campaigns": len(camps),
                    "search_running": sum(c["running"] for c in camps),
                    "construction_runs": len(runs_),
                    "workers_running": sum(r["running"] for r in runs_),
                    "delivered": _distinct_delivered(runs_),
                    "benchmarks": [b for b in tree.benchmark_sets() if b["task"] == t],
                }
            )
        live = tree.running()
        return page(
            "overview.html",
            request,
            [],
            refresh=20 if live else 0,
            tasks=tasks,
            live=live,
            groups=D.running_groups(live),
        )

    @app.get("/workflow", response_class=HTMLResponse)
    def workflow(request: Request, task: str = ""):
        tasks = tree.tasks()
        t = task if task in tasks else (tasks[0] if tasks else "")

        def render():
            p = (
                pipeline(tree, t, _distinct_delivered, _builds)
                if t
                else {"nodes": [], "edges": [], "width": 0, "height": 0}
            )
            live = any(n["status"] == "running" for n in p["nodes"])
            return page(
                "workflow.html",
                request,
                [("Workflow", None), (t, None)],
                refresh=20 if live else 0,
                p=p,
                task=t,
                tasks=tasks,
            )

        return guard(render)

    @app.get("/running", response_class=HTMLResponse)
    def running(request: Request):
        live = tree.running()
        return page(
            "running.html", request, [("Running now", None)], refresh=10, live=live, groups=D.running_groups(live)
        )

    @app.get("/task/{task}", response_class=HTMLResponse)
    def task(request: Request, task: str, tab: str = "overview", run: str = ""):
        def render():
            runs_ = tree.construction_runs(task)
            builds, drop_rows = [], []
            if tab == "drops":
                keys = [run] if run in {r["key"] for r in runs_} else [r["key"] for r in runs_]
                drop_rows = W.drops(tree, task, keys)
            if tab == "funnel":
                picked = [r for r in runs_ if r["key"] == run] or runs_
                builds = _builds(tree, task, picked, [b for b in tree.benchmark_sets() if b["task"] == task])
            return page(
                "task.html",
                request,
                [(task, None)],
                tab=tab,
                lessons=tree.lessons(f"{task}_worker"),
                task=task,
                spec=tree.spec(task),
                rules=tree.rules(task),
                plan=tree.plan(task),
                stocks=tree.stocks(task),
                campaigns=tree.search_campaigns(task),
                runs=runs_,
                distinct=_distinct_delivered,
                builds=builds,
                run=run,
                drop_rows=drop_rows,
                drop_groups=W.drop_groups(drop_rows),
                refresh=15 if tab == "funnel" and any(r["running"] for r in runs_) else 0,
                benchmarks=[b for b in tree.benchmark_sets() if b["task"] == task],
            )

        return guard(render)

    @app.get("/task/{task}/spec", response_class=HTMLResponse)
    def spec(request: Request, task: str):
        return guard(
            lambda: page(
                "spec.html",
                request,
                [(task, f"/task/{task}"), ("Specification", None)],
                task=task,
                spec=tree.spec(task),
            )
        )

    @app.get("/task/{task}/rules", response_class=HTMLResponse)
    def rules(request: Request, task: str):
        return guard(
            lambda: page(
                "rules.html",
                request,
                [(task, f"/task/{task}"), ("Search rules", None)],
                task=task,
                rules=tree.rules(task),
            )
        )

    @app.get("/task/{task}/plan", response_class=HTMLResponse)
    def plan(request: Request, task: str):
        return guard(
            lambda: page(
                "plan.html",
                request,
                [(task, f"/task/{task}"), ("Construction plan", None)],
                task=task,
                plan=tree.plan(task),
                lessons=tree.lessons(f"{task}_worker"),
            )
        )

    @app.get("/search/{task}/{epoch}", response_class=HTMLResponse)
    def campaign(request: Request, task: str, epoch: str, tab: str = "graph"):
        c = guard(lambda: tree.campaign(task, epoch))
        rows = []
        if tab == "candidates":
            for a in c["agents"]:
                rows += [
                    {"_agent": a["name"], **r}
                    for r in D.read_jsonl(os.path.join(a["dir"], "stock", "candidates.jsonl"))
                ]
        return page(
            "campaign.html",
            request,
            [(task, f"/task/{task}?tab=search"), (f"search {epoch}", None)],
            refresh=15 if any(a["alive"] for a in c["agents"]) and tab in ("agents", "graph") else 0,
            c=c,
            tab=tab,
            rows=rows,
            graph=D.campaign_graph(c) if tab == "graph" else None,
            stocks=tree.stocks(task),
        )

    @app.get("/search/{task}/{epoch}/{attempt}", response_class=HTMLResponse)
    def search_agent(request: Request, task: str, epoch: str, attempt: str, n: int = 25, tab: str = "activity"):
        d = guard(lambda: tree.attempt_dir(task, epoch, attempt))
        if not os.path.isdir(d):
            raise HTTPException(404, "no such search agent")
        det = D.agent_detail(d)
        return page(
            "agent.html",
            request,
            [(task, f"/task/{task}"), (f"search {epoch}", f"/search/{task}/{epoch}"), (attempt, None)],
            refresh=15 if det["status"]["alive"] else 0,
            kind="search",
            a=det,
            tab=tab,
            cycles=D.latest_cycles(d, min(max(n, 5), 200)) if tab == "activity" else [],
            n=n,
            base=f"/search/{task}/{epoch}/{attempt}",
            stop_url=f"/stop?kind=search&task={task}&name={epoch}&agent={attempt}",
            rows=D.read_jsonl(os.path.join(d, "stock", "candidates.jsonl")) if tab == "stock" else [],
        )

    @app.get("/search/{task}/{epoch}/{attempt}/cycle/{cycle}", response_class=HTMLResponse)
    def search_cycle(request: Request, task: str, epoch: str, attempt: str, cycle: int):
        d = guard(lambda: tree.attempt_dir(task, epoch, attempt))
        base = f"/search/{task}/{epoch}/{attempt}"
        return page(
            "cycle.html",
            request,
            [
                (task, f"/task/{task}"),
                (f"search {epoch}", f"/search/{task}/{epoch}"),
                (attempt, base),
                (f"cycle {cycle}", None),
            ],
            c=D.cycle_detail(d, cycle),
            base=base,
        )

    @app.get("/candidates/stock/{name}", response_class=HTMLResponse)
    def stock_candidates(request: Request, name: str):
        path = guard(lambda: tree.stock_file(name))
        info = next((s for s in tree.stocks() if s["name"] == name), {})
        crumbs = ([(info["task"], f"/task/{info['task']}")] if info.get("task") else []) + [(f"stock {name}", None)]
        return page(
            "candidates.html",
            request,
            crumbs,
            title=f"Frozen stock {name}",
            note=info.get("note"),
            rows=D.read_jsonl(path),
            identity=info.get("identity"),
        )

    @app.get("/candidates/attempt/{task}/{epoch}/{attempt}", response_class=HTMLResponse)
    def attempt_candidates(request: Request, task: str, epoch: str, attempt: str):
        d = guard(lambda: tree.attempt_dir(task, epoch, attempt))
        return page(
            "candidates.html",
            request,
            [
                (task, f"/task/{task}"),
                (f"search {epoch}", f"/search/{task}/{epoch}"),
                (attempt, f"/search/{task}/{epoch}/{attempt}"),
                ("candidates", None),
            ],
            title=f"Candidates in {attempt}'s stock file",
            note=None,
            rows=D.read_jsonl(os.path.join(d, "stock", "candidates.jsonl")),
            identity=None,
        )

    @app.get("/candidates/search/{task}/{epoch}", response_class=HTMLResponse)
    def campaign_candidates(request: Request, task: str, epoch: str):
        c = guard(lambda: tree.campaign(task, epoch))
        rows = []
        for a in c["agents"]:
            for r in D.read_jsonl(os.path.join(a["dir"], "stock", "candidates.jsonl")):
                rows.append({"_agent": a["name"], **r})
        return page(
            "candidates.html",
            request,
            [(task, f"/task/{task}"), (f"search {epoch}", f"/search/{task}/{epoch}"), ("candidates", None)],
            title=f"Candidates in the stock files of {epoch} (all agents, before freezing)",
            note="Not deduplicated: the same candidate can appear in two agents' files.",
            rows=rows,
            identity=None,
        )

    @app.get("/construction/{key}", response_class=HTMLResponse)
    def construction(request: Request, key: str, tab: str = "graph"):
        r = guard(lambda: tree.construction_run(key))
        task = key.split("~")[1] if key.startswith("construct~") else key.split("~")[2]
        graph = D.run_graph(tree.run_dir(key)) if tab == "graph" else None
        return page(
            "construction.html",
            request,
            [(task, f"/task/{task}?tab=construction"), (r["label"], None)],
            refresh=15 if any(w["alive"] for w in r["workers"]) and tab in ("agents", "graph") else 0,
            r=r,
            task=task,
            tab=tab,
            graph=graph,
            labels=_stage_labels,
        )

    @app.get("/construction/{key}/{worker}", response_class=HTMLResponse)
    def worker(request: Request, key: str, worker: str, n: int = 25, tab: str = "candidates"):
        w = guard(lambda: tree.worker_dir(key, worker))
        if not os.path.isdir(w):
            raise HTTPException(404, "no such worker")
        task = key.split("~")[1] if key.startswith("construct~") else key.split("~")[2]
        det = D.agent_detail(w)
        return page(
            "agent.html",
            request,
            [
                (task, f"/task/{task}"),
                (os.path.relpath(tree.run_dir(key), tree.runs), f"/construction/{key}"),
                (worker, None),
            ],
            refresh=15 if det["status"]["alive"] else 0,
            kind="worker",
            a=det,
            wk=D.worker_detail(w),
            tab=tab,
            cycles=D.latest_cycles(w, min(max(n, 5), 200)) if tab == "activity" else [],
            n=n,
            base=f"/construction/{key}/{worker}",
            rows=[],
            stop_url=f"/stop?kind=construction&name={key}&agent={worker}",
        )

    @app.get("/construction/{key}/{worker}/cycle/{cycle}", response_class=HTMLResponse)
    def worker_cycle(request: Request, key: str, worker: str, cycle: int):
        w = guard(lambda: tree.worker_dir(key, worker))
        base = f"/construction/{key}/{worker}"
        task = key.split("~")[1] if key.startswith("construct~") else key.split("~")[2]
        return page(
            "cycle.html",
            request,
            [
                (task, f"/task/{task}"),
                (os.path.relpath(tree.run_dir(key), tree.runs), f"/construction/{key}"),
                (worker, base),
                (f"cycle {cycle}", None),
            ],
            c=D.cycle_detail(w, cycle),
            base=base,
        )

    @app.get("/construction/{key}/{worker}/recipe/{slug}", response_class=HTMLResponse)
    def recipe(request: Request, key: str, worker: str, slug: str):
        w = guard(lambda: tree.worker_dir(key, worker))
        rec = guard(lambda: D.recipe_detail(w, slug))
        task = key.split("~")[1] if key.startswith("construct~") else key.split("~")[2]
        cand = next((k for k in D.worker_summary(w)["order"] if D.recipe_for(k, [slug]) == slug), None)
        title = _key(cand) if cand else slug
        return page(
            "recipe.html",
            request,
            [
                (task, f"/task/{task}"),
                (os.path.relpath(tree.run_dir(key), tree.runs), f"/construction/{key}"),
                (worker, f"/construction/{key}/{worker}"),
                (title, None),
            ],
            rec=rec,
            slug=slug,
            title=title,
            base=f"/construction/{key}/{worker}/recipe/{slug}",
        )

    @app.get("/construction/{key}/{worker}/recipe/{slug}/file/{name}", response_class=PlainTextResponse)
    def recipe_file(key: str, worker: str, slug: str, name: str):
        w = guard(lambda: tree.worker_dir(key, worker))
        text = guard(lambda: D.recipe_file(w, slug, name))
        if text is None:
            raise HTTPException(404, "no such file")
        return text

    @app.get("/benchmarks", response_class=HTMLResponse)
    def benchmarks(request: Request):
        return page("benchmarks.html", request, [("Benchmarks", None)], sets=tree.benchmark_sets())

    @app.get("/benchmarks/{set_}/{task}", response_class=HTMLResponse)
    def benchmark(request: Request, set_: str, task: str):
        b = guard(lambda: tree.benchmark(set_, task))
        return page(
            "benchmark.html",
            request,
            [("Benchmarks", "/benchmarks"), (f"{set_} / {task}", None)],
            b=b,
            set_=set_,
            task=task,
        )

    @app.get("/benchmarks/{set_}/{task}/{iid}", response_class=HTMLResponse)
    def instance(request: Request, set_: str, task: str, iid: str):
        inst = guard(lambda: tree.instance(set_, task, iid))
        row = next((r for r in tree.benchmark(set_, task)["instances"] if r["id"] == iid), None)
        return page(
            "instance.html",
            request,
            [("Benchmarks", "/benchmarks"), (f"{set_} / {task}", f"/benchmarks/{set_}/{task}"), (iid[:60], None)],
            inst=inst,
            row=row,
            set_=set_,
            task=task,
            iid=iid,
        )

    @app.get("/benchmarks/{set_}/{task}/{iid}/file/{name}", response_class=PlainTextResponse)
    def instance_file(set_: str, task: str, iid: str, name: str):
        text = guard(lambda: tree.instance_file(set_, task, iid, name))
        if text is None:
            raise HTTPException(404, "no such file")
        return text

    jobs = add_action_routes(app, tree, page, guard, allow_actions)
    health = H.Cached(tree)
    env.globals.update(health=health.get)

    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request, task: str = ""):
        tasks = tree.tasks()
        t = task if task in tasks else ""
        if not t:
            return page(
                "compare.html", request, [("Compare", None)], task="", tasks=tasks, rows=C.overall(tree, _builds)
            )
        return page(
            "compare.html",
            request,
            [("Compare", "/compare"), (t, None)],
            task=t,
            tasks=tasks,
            rows=C.by_model(tree, t, _builds),
            search=C.search_by_model(tree, t, _model),
            trend=C.trend(tree, t, _model),
        )

    @app.get("/candidate", response_class=HTMLResponse)
    def candidate(request: Request, q: str = ""):
        return page("candidate.html", request, [("Candidate", None)], j=W.journey(tree, q))

    @app.get("/health", response_class=HTMLResponse)
    def health_page(request: Request):
        return page("health.html", request, [("Health", None)], refresh=30, ws=health.get())

    env.globals.update(task_jobs=lambda task: [j for j in jobs.all() if j.get("task") == task][:10])

    @app.get("/api/running")
    def api_running():
        return JSONResponse(tree.running())

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench-ui", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT, help="the repository (specs/, agents/, stock/, benchmarks/)")
    ap.add_argument("--runs", default=None, help="the runs root (default: <root>/runs)")
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this machine only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--allow-actions",
        action="store_true",
        help="let the UI launch, freeze, package, certify and stop runs (each after a check and a typed "
        "confirmation); needs the repository's own runs folder",
    )
    a = ap.parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("bench-ui needs the ui extra: pip install -e '.[ui]'")
    app = build_app(a.root, a.runs, a.allow_actions)
    print(
        f"BenchMeUp UI on http://{a.host}:{a.port}  (repository {os.path.abspath(a.root)}, "
        f"runs {os.path.abspath(a.runs or os.path.join(a.root, 'runs'))}; "
        f"actions {'ON' if a.allow_actions else 'off (read-only)'})"
    )
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
