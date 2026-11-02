# bench-ui

`bench-ui` is an optional read-only web view of tasks, runs, agents and benchmarks.

```
pip install -e '.[ui]'
bench-ui                              # http://127.0.0.1:8765, runs read from ./runs
bench-ui --runs /path/to/runs --port 9000
```

It listens on this machine only (`--host` to change). Over SSH, forward the port.

| page | what it shows |
|---|---|
| Overview | every task's progress through the four phases; the agents running now |
| Workflow | the pipeline as a graph per task, each step coloured by status, with its main count, command and output |
| Task | the specification, the search rules, the construction plan, the task's search campaigns, frozen stocks, construction runs and benchmarks |
| Search campaign, Agent, Candidates | the agents of a campaign, their cycles, prompts and stock files |
| Construction run, Worker | the workers and the candidates each took; delivered and dropped candidates; every replay report row by row |
| Benchmarks | the packaged instances with their replay results, and each instance's files |
| Health, Compare | warnings about live runs; cost and yield per model |

`bench-ui-export --out site/` writes the pages as static HTML for readers who run nothing. Every
page is scanned for the home directory and user name (and any `--forbid` string) first; one hit
stops the export.

By default the UI never writes into the repository or the runs. `bench-ui --allow-actions`
adds an Actions tab that can launch, freeze, package, certify and stop; each action shows its
exact command, runs that command's own check first, and needs a typed confirmation.
