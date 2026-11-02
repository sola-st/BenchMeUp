# Operations

## Before the first run

| what | how |
|---|---|
| Docker | the daemon reachable by your user (`docker ps` works without sudo) |
| disk | construction builds one image per candidate and keeps them for replays: plan for hundreds of GB per construction campaign; search alone needs little |
| GitHub token | set `GITHUB_TOKEN` in `.env` |
| model route | see [configuration.md](configuration.md) |

Check the setup (the first two spend nothing):

```
pytest -m "not docker and not e2e and not network"
examples/run_example.sh boca
python scripts/search_campaign.py boca 1 --agents 1 --territory-mode none --start-delay 0 \
  --directions off --lessons off --target 2 --model <model> --cost 1 --max-search 4 --dry-run
```

## Watching and stopping runs

- Status: `python scripts/search_status.py runs/search/<task>/epoch_N` and
  `python scripts/construct_status.py runs/construct/<task>/epoch_N`.
- Events: `scripts/search_watch.py` reports stuck agents, cut-off replies, shrinking stock files and exits.
- Every attempt and worker runs detached in its own process group, writes its pid to
  `launcher.pid`, and keeps `construct.log`, `trajectory.jsonl` (every model call and tool result)
  and `agent_state.json` in its own directory.
- Stop a run with `kill -- -<pgid>` (`ps -o pgid= -p <pid>` gives the group). For a search
  campaign, also stop the `search_campaign.py` process itself, or it starts the next attempt.
  Check afterwards that nothing is left: `ps -p <pid>` and `docker ps`.
- `bench construct-resume <run dir>` continues an interrupted run.

## Timing benchmarks (compiler autotuning)

The checks of this task time the program on both sides and compare the two times. Timed runs
queue on a host-wide lock, so no two run at once; builds and model calls run alongside them.

| variable | meaning |
|---|---|
| `AGENTCORE_MEASUREMENT_LANE` | path of the lock file (default `/tmp/benchcompiler-measurement.lane`); `off` disables the queue |
| `AGENTCORE_MEASUREMENT_CPUSET` | e.g. `0-3`: the cores a container is pinned to while it is timed |
| `AGENTCORE_SANDBOX_CPUSET` | e.g. `4-31`: the cores every other container runs on |

On a shared machine, set both cpusets so that timed runs get cores nobody else uses.

## Disk and Docker images

| image tags | what they are | safe to delete |
|---|---|---|
| `benchrecipe*`, `benchcertify<timestamp>*` | replay and certification builds | yes, rebuilt on the next replay |
| `agentcore_image:*`, `agentcore_build_*`, `agentcore_deferred_*` | the workers' sandbox images | yes, once no run uses them |
| `bench-<id>` | images built by a benchmark's `bench prepare` | yes |
| `agentcore-ckpt-*`, `agentcore_ckpt_*` | checkpoints of a live run | only when that run is finished |
| dangling images, build cache | | yes (`docker image prune`, `docker builder prune`) |

Never delete an image that a running container uses; `docker ps` lists those containers.
