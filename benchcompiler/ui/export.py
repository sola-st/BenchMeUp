"""bench-ui-export: the UI as static pages, for readers who will not run anything (reviewers).

    bench-ui-export --out site/                     # from the repository alone
    bench-ui-export --out site/ --runs path/to/runs # with the runs' graphs, funnels and drops

Pages rendered: overview, workflow per task, every task tab except Actions, benchmarks with
every instance, compare, and each construction run's graph and lists. Left out: agent activity,
prompts, logs and records (they hold host paths and raw model traffic), the candidate search,
health and actions. Links to pages that were not exported become plain text.

Before writing, every page is scanned for this machine's home directory, the user name and any
--forbid string; one hit stops the export and lists where it is, so nothing identifying leaves.
"""

import argparse
import getpass
import html
import os
import re
import shutil
import sys
from urllib.parse import unquote

from benchcompiler.ui import data as D

HREF = re.compile(r'href="(/[^"]*)"')


def page_name(url: str) -> str:
    url = html.unescape(url)
    if url == "/":
        return "index.html"
    s = unquote(url.strip("/")).replace("?", "__").replace("&", "__").replace("=", "-")
    s = re.sub(r"[^A-Za-z0-9_.~@-]+", "_", s)
    return s[:180] + ".html"


def urls(tree) -> list[str]:
    out = ["/", "/benchmarks", "/compare", "/workflow"]
    for t in tree.tasks():
        out += [f"/workflow?task={t}", f"/compare?task={t}"]
        out += [
            f"/task/{t}?tab={tab}"
            for tab in ("overview", "funnel", "spec", "rules", "plan", "search", "construction", "benchmarks", "drops")
        ]
        out.append(f"/task/{t}")
        for r in tree.construction_runs(t):
            out += [f"/construction/{r['key']}?tab={tab}" for tab in ("graph", "delivered", "dropped")]
            out.append(f"/construction/{r['key']}")
        for c in tree.search_campaigns(t):
            out += [f"/search/{t}/{c['epoch']}?tab=graph", f"/search/{t}/{c['epoch']}"]
    for b in tree.benchmark_sets():
        out.append(f"/benchmarks/{b['set']}/{b['task']}")
        for inst in tree.benchmark(b["set"], b["task"])["instances"]:
            out.append(f"/benchmarks/{b['set']}/{b['task']}/{inst['id']}")
    return out


def rewrite(page: str, exported: set[str]) -> str:
    def sub(m):
        u = html.unescape(m.group(1))
        if u in exported:
            return f'href="{page_name(u)}"'
        if u.startswith("/static/"):
            return f'href="{u[1:]}"'
        return 'data-off="1"'

    page = HREF.sub(sub, page)
    return page.replace('src="/static/', 'src="static/')


def forbidden_default(tree) -> list[str]:
    home = os.path.expanduser("~")
    words = [home, os.path.realpath(home), os.path.realpath(tree.runs), os.path.realpath(tree.root)]
    user = getpass.getuser()
    if len(user) >= 4:
        words.append(user)
    return sorted({w for w in words if w and w != "/"}, key=len, reverse=True)


def export(root: str, runs: str | None, out: str, forbid: list[str]) -> int:
    from fastapi.testclient import TestClient

    from benchcompiler.ui.app import build_app

    tree = D.Tree(root, runs)
    app = build_app(root, runs, static_export=True)
    c = TestClient(app)
    want = urls(tree)
    exported = set(want)
    pages, failed = {}, []
    for u in want:
        r = c.get(u)
        if r.status_code != 200:
            failed.append((u, r.status_code))
            continue
        pages[u] = r.text
    words = forbid + forbidden_default(tree)
    hits = []
    for u, text in pages.items():
        low = text.lower()
        for w in words:
            i = low.find(w.lower())
            if i != -1:
                hits.append((u, w, text[max(0, i - 60) : i + len(w) + 40].replace("\n", " ")))
                break
    if hits:
        print(f"REFUSED: {len(hits)} page(s) contain an identifying string; nothing written.", file=sys.stderr)
        for u, w, ctx in hits[:30]:
            print(f"  {u}: {w!r} in ...{ctx}...", file=sys.stderr)
        print(
            "Remove it from the source (or pass --runs only for runs you can share), then export again.",
            file=sys.stderr,
        )
        return 2
    os.makedirs(out, exist_ok=True)
    for u, text in pages.items():
        with open(os.path.join(out, page_name(u)), "w", encoding="utf-8") as fh:
            fh.write(rewrite(text, exported))
    static = os.path.join(out, "static")
    shutil.rmtree(static, ignore_errors=True)
    shutil.copytree(os.path.join(os.path.dirname(__file__), "static"), static)
    print(
        f"{len(pages)} pages written to {out} (open {os.path.join(out, 'index.html')})"
        + (f"; {len(failed)} not rendered: {failed[:5]}" if failed else "")
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench-ui-export", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--runs", default=None, help="a runs folder to include (default: <root>/runs if present)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forbid", nargs="*", default=[], help="more strings that must not appear in any page")
    a = ap.parse_args(argv)
    return export(a.root, a.runs, a.out, a.forbid)


if __name__ == "__main__":
    raise SystemExit(main())
