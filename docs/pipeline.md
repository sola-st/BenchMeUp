# The pipeline

BenchMeUp turns a benchmark specification into a benchmark package in four phases. Every
generated artifact is a plain file, so each phase can be inspected, repeated, or started from
the files of the previous one. The repository ships the files of every phase for its four tasks.

| phase | command | uses a model | input | output |
|---|---|---|---|---|
| 1a. search rules | `bench formalize` | yes | `specs/<task>/spec.yaml` | `specs/<task>/canonical.json` |
| 1b. search agent | `bench search-agent` | no | spec, rules | `agents/<task>_search/` |
| 1c. construction plan and agent | `bench worker` | yes | spec, rules | `agents/<task>_worker/` (`contract.json`, `agent.yaml`) |
| 2. candidates | `scripts/search_campaign.py` or `scripts/search_epoch.sh`, then `bench stock-freeze` | yes (agents) | the search agent | `stock/<task>/candidates.jsonl` |
| 3. instances | `scripts/construct_arms.py` or `bench handover --launch` | yes (agents) | candidates, construction agent | `runs/construct/<task>/.../recipes/<candidate>/` |
| 4. package | `scripts/package.py`, then `scripts/certify_packaged.py` | no | delivered instances, plan | `benchmarks/<set>/<task>/` |

## Phase 1: search rules and construction plan

```
mkdir -p runs/boca
bench formalize specs/boca/spec.yaml --model <model> --out runs/boca/canonical.json
bench search-agent specs/boca/spec.yaml --canonical runs/boca/canonical.json --out agents/boca_search_new --model <model> --target 10
bench worker specs/boca/spec.yaml --canonical runs/boca/canonical.json --out agents/boca_worker_new --model <model>
```

`formalize` fills a task-independent search-rule schema from the specification and checks the
result mechanically. A second model call then reviews the rules against the specification and
reports omitted requirements and unresolved ambiguities (`--judge-model` selects the reviewer,
by default the same model; `--no-judge` skips it). A refusal goes back to the generating model
as feedback; when the round limit is reached the specification and the feedback are returned
to the researcher. The construction plan goes through the same kind of review.

## Phase 2: finding candidates

```
scripts/search_epoch.sh boca agents/boca_search_new 1 2 10 30       # two attempts, 10 candidates each, $30 each
bench stock-freeze runs/search/boca/epoch_1/attempt_* --out stock/boca_new --agent agents/boca_search_new
```

Agents record candidates with the `stock_add` tool (append only) and remove one only with
`stock_remove`, after investigating why its check failed. An agent ends by writing a script
that checks that every recorded candidate exists; the framework runs it in a fresh container.
`stock-freeze` merges the attempts' stocks on the candidates' identity and deduplicates them.

### Search campaigns

`scripts/search_campaign.py` runs several search agents for one task:

1. The first agent starts alone.
2. Every `--start-delay` seconds a model summarizes where the running agents search (queries,
   filters, repositories), and the next agent starts with that summary and the instruction to
   search another part of the space.
3. Once all agents run, the summary is refreshed every `--sync-interval` seconds.

```
set -a; . ./.env; set +a
python scripts/search_campaign.py boca 1 --agents 4 --territory-mode none \
  --start-delay 1200 --directions on --directions-model <model> \
  --sync-interval 3600 --lessons off --target 60 \
  --model <model> --cost 5 --wall 21600 --max-search 8 \
  --reference agents/boca_search_new/instances/instances.jsonl --dry-run
```

Remove `--dry-run` to launch. `--dry-run` sends one 5-token call to check the model route and
asks GitHub for its remaining rate limit. `--territory-mode` is `none`, `request` (the request's
star bands) or `drawn` (territories a model draws once); `--lessons on` adds the lessons loop.

## Phase 3: construction

```
python scripts/construct_arms.py stock/boca_new/candidates.jsonl agents/boca_worker_new \
    runs/construct/boca/epoch_1 arms.json --target 4
python scripts/construct_status.py runs/construct/boca/epoch_1
```

An arms file names the model, its settings and the budgets of each group of workers:

```
[{"name": "main", "model": "<model>", "workers": 2,
  "cost_usd": 150, "wallclock": 21600, "timeout": 180}]
```

Add `--dry-run` first: it does the hand-over and prints the commands without launching anything.
Each worker takes one candidate at a time through the stages of the plan. After each stage the
framework rebuilds the image from the worker's Dockerfile alone, re-runs the scripts of this and
every earlier stage, and compares the results with the plan's expected outcomes. A candidate is
delivered only when its last stage passes. `bench lessons` (or `scripts/lessons_loop.sh`) has a
model read the agents' complete records and write `lessons.md`, which running and later agents
read; notes that must always be shown go in `agents/<agent>/lessons_pinned.md`.

How candidates are dealt to the workers:

| option | order |
|---|---|
| (default) | whole repositories go to one worker each |
| `--shuffle SEED` | candidates are shuffled with that seed and dealt one per worker in turn |
| `--size-order CSV` | smallest repository first, ties shuffled; rows are `repository,size` |

`bench handover` takes the same options (`--shuffle --seed N`, `--size-order CSV`).
`bench construct-resume <run dir>` continues an interrupted run.

## Phase 4: packaging and the final check

```
python scripts/package.py --task boca --runs runs/construct/boca/epoch_1 --out benchmarks/mine/boca \
    --contract agents/boca_worker_new/contract.json
python scripts/certify_packaged.py benchmarks/mine/boca --contract agents/boca_worker_new/contract.json --fresh
```

Packaging writes, next to the copied instances, a manifest, the contract, a README and `bench`, a
small interpreter with no dependency on this repository. `certify_packaged.py` replays every
packaged instance in a fresh container through `bench`.

## The packaged benchmark

Each `benchmarks/<set>/<task>/` contains `manifest.json` (identity fields and replay results of
each instance), `contract.json` (the interface), `instances/<id>/` and `bench`. `bench` commands:

```
./bench list
./bench prepare <id>
./bench materialize <id> <version>
./bench check <id> <version>
./bench judge <id>
./bench submit <id> <answer>
```

Instances depend on upstream repositories and package registries; a repository or artifact that
disappears upstream can make an instance unbuildable.
