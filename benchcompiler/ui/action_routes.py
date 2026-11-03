"""The routes behind the Actions tab, the jobs pages and Stop (see actions.py for the rules)."""

import os
import secrets
from urllib.parse import parse_qs, urlencode

from benchcompiler.ui import actions as A
from benchcompiler.ui import data as D


def add_action_routes(app, tree, page, guard, allow: bool):
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    jobs = A.Jobs(tree.runs)
    token = secrets.token_urlsafe(24)

    async def form_of(request: Request) -> dict:
        body = (await request.body()).decode("utf-8", errors="replace")
        return {k: (v if len(v) > 1 else v[0]) for k, v in parse_qs(body, keep_blank_values=True).items()}

    def require(form: dict):
        if not allow:
            raise HTTPException(403, "actions are off: start bench-ui with --allow-actions")
        if form.get("csrf") != token:
            raise HTTPException(403, "this form is from an earlier start of bench-ui; reload the page")

    def action(aid: str, task: str) -> dict:
        if aid not in A.BUILDERS:
            raise HTTPException(404, "no such action")
        if task not in tree.tasks():
            raise HTTPException(404, "no such task")
        return A.BUILDERS[aid](tree, task)

    def crumbs(task, label):
        return [(task, f"/task/{task}?tab=actions"), (label, None)]

    @app.get("/action/{aid}/{task}", response_class=HTMLResponse)
    def action_form(request: Request, aid: str, task: str):
        a = action(aid, task)
        return page(
            "action_form.html",
            request,
            crumbs(task, a["label"]),
            a=a,
            task=task,
            allow=allow,
            csrf=token,
            form={},
            error=None,
        )

    @app.post("/action/{aid}/{task}/check", response_class=HTMLResponse)
    async def action_check(request: Request, aid: str, task: str):
        form = await form_of(request)
        require(form)
        a = action(aid, task)
        try:
            v = A.values(a, form)
            argv = a["argv"](v)
            if a.get("before"):
                a["before"](v)
        except A.Refused as e:
            return page(
                "action_form.html",
                request,
                crumbs(task, a["label"]),
                a=a,
                task=task,
                allow=allow,
                csrf=token,
                form=form,
                error=str(e),
            )
        fh = A.form_hash(aid, task, v)
        meta = {"action": aid, "task": task, "form_hash": fh, "values": v}
        if a.get("check"):
            j = jobs.run_sync("check", meta, a["check"](v), tree.root)
        else:
            ok, text = a["preview"](v)
            j = jobs.note("check", meta, text, exit=0 if ok else 1)
        return page(
            "action_check.html",
            request,
            crumbs(task, a["label"]),
            a=a,
            task=task,
            csrf=token,
            argv=argv,
            check=j,
            confirm=a["confirm"](v),
            hidden=form,
        )

    @app.post("/action/{aid}/{task}/run")
    async def action_run(request: Request, aid: str, task: str):
        form = await form_of(request)
        require(form)
        a = action(aid, task)
        try:
            v = A.values(a, form)
            argv = a["argv"](v)
        except A.Refused as e:
            raise HTTPException(400, str(e))
        fh = A.form_hash(aid, task, v)
        if not jobs.passed_check(fh):
            raise HTTPException(
                409,
                "no passing check of these exact settings in the last 30 minutes; "
                "go back to the form and run the check again",
            )
        if (form.get("confirm") or "").strip() != a["confirm"](v):
            raise HTTPException(409, f"type {a['confirm'](v)!r} exactly to confirm")
        meta = {"action": aid, "task": task, "form_hash": fh, "values": v}
        if a["detached"]:
            j = jobs.run_detached("run", meta, argv, tree.root, a["log"](v))
        else:
            j = jobs.run_sync("run", meta, argv, tree.root, timeout=3600)
        return RedirectResponse(f"/jobs/{j['id']}", status_code=303)

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request):
        js = jobs.all()
        return page(
            "jobs.html",
            request,
            [("Actions log", None)],
            refresh=10 if any(j["running"] for j in js) else 0,
            jobs=js,
            allow=allow,
        )

    @app.get("/jobs/{jid}", response_class=HTMLResponse)
    def job_page(request: Request, jid: str):
        j = jobs.get(D.safe(jid))
        if not j:
            raise HTTPException(404, "no such job")
        log = D.tail_text(j["log"], 30_000) if j.get("log") and os.path.exists(j["log"]) else j.get("output", "")
        return page(
            "job.html", request, [("Actions log", "/jobs"), (jid, None)], refresh=5 if j["running"] else 0, j=j, log=log
        )

    def stop_args(q) -> tuple[str, str, str, str | None]:
        kind, task, name, agent = q.get("kind", ""), q.get("task", ""), q.get("name", ""), q.get("agent") or None
        if kind not in ("construction", "search"):
            raise HTTPException(400, "kind must be construction or search")
        return kind, task, name, agent

    def stop_label(kind, task, name, agent):
        base = name.split("~")[-1] if kind == "construction" else f"{task} {name}"
        return f"{base} {agent}" if agent else base

    @app.get("/stop", response_class=HTMLResponse)
    def stop_page(request: Request, kind: str, name: str, task: str = "", agent: str = ""):
        kind, task, name, agent = stop_args({"kind": kind, "task": task, "name": name, "agent": agent})
        t = guard(lambda: A.stop_targets(tree, kind, task, name, agent))
        return page(
            "stop.html",
            request,
            [("Stop", None)],
            t=t,
            allow=allow,
            csrf=token,
            q={"kind": kind, "task": task, "name": name, "agent": agent or ""},
            confirm=stop_label(kind, task, name, agent),
            result=None,
        )

    @app.post("/stop", response_class=HTMLResponse)
    async def stop_run(request: Request):
        form = await form_of(request)
        require(form)
        kind, task, name, agent = stop_args(form)
        want = stop_label(kind, task, name, agent)
        if (form.get("confirm") or "").strip() != want:
            raise HTTPException(409, f"type {want!r} exactly to confirm")
        t = guard(lambda: A.stop_targets(tree, kind, task, name, agent))
        res = A.stop(t, force=form.get("force") == "yes")
        after = A.stop_targets(tree, kind, task, name, agent)
        jobs.note(
            "stop",
            {"action": "stop", "task": task, "values": {"kind": kind, "name": name, "agent": agent}},
            f"signal {res['signal']} to process groups {res['groups']}; left: "
            f"{[m['pid'] for m in res['left']]}; containers still up: {[c['name'] for c in after['containers']]}",
        )
        return page(
            "stop.html",
            request,
            [("Stop", None)],
            t=after,
            allow=allow,
            csrf=token,
            q={"kind": kind, "task": task, "name": name, "agent": agent or ""},
            confirm=want,
            result=res,
        )

    @app.post("/stop/containers", response_class=HTMLResponse)
    async def stop_containers(request: Request):
        form = await form_of(request)
        require(form)
        kind, task, name, agent = stop_args(form)
        t = guard(lambda: A.stop_targets(tree, kind, task, name, agent))
        if t["groups"]:
            raise HTTPException(409, "the run's processes are still alive; stop them first")
        out = A.remove_containers([c["id"] for c in t["containers"]])
        jobs.note("stop", {"action": "remove containers", "task": task, "values": {"kind": kind, "name": name}}, out)
        return RedirectResponse(
            "/stop?" + urlencode({"kind": kind, "task": task, "name": name, "agent": agent or ""}), status_code=303
        )

    return jobs
