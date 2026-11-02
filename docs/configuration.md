# Configuration

Launchers read `.env` from the repository root when it exists. Copy `.env.example` and fill in
what your model route needs.

## Model routes

The model name chooses the route. It is set by `--model`, `AGENT_MODEL`, or the `model` of an arm.

| model name | route | needs |
|---|---|---|
| `claude-code/<model id>` | the Claude Code CLI | `claude` on `PATH` and logged in, or `ANTHROPIC_API_KEY`. Optional: `CLAUDE_CODE_EFFORT` (default `medium`), `CLAUDE_CODE_TIMEOUT` (seconds per reply), `CLAUDE_CODE_CONCURRENCY`, `CLAUDE_CODE_BINARY` |
| `openrouter/<provider>/<model>` | OpenRouter, direct | `OPENROUTER_KEY` (or `OPENROUTER_API_KEY`). Optional: `OPENROUTER_PROVIDER_ORDER` to pin providers |
| `openai/<model as the endpoint names it>` | any OpenAI-compatible endpoint | `LITELLM_BASE_URL` and `LITELLM_KEY`, and `OPENAI_API_BASE` / `OPENAI_API_KEY` set to the same endpoint and key |

When a Claude Code subscription's session limit is reached, a run waits for the reset if it
comes before the run's own wall-clock budget runs out, and stops otherwise.

## Environment variables

| variable | used by | meaning |
|---|---|---|
| `GITHUB_TOKEN` (or `GH_FINEGRAINED_TOKEN`) | search agents, workers | GitHub API access, read access to public repositories is enough; also passed into the agents' containers |
| `AGENT_MODEL` | `search_epoch.sh`, `bench construct` | the agent's model, when `--model` is not given |
| `SEARCH_WALL` | `search_epoch.sh` | wall-clock limit per search attempt, seconds (default 21600) |
| `PYTHON` | `search_epoch.sh` | the interpreter with this package installed (default `python3`) |
| `BENCH_CACHE_URL` | launchers | the optional Maven cache (`infra/maven-cache/run.sh` prints it); unset means the vendors' addresses |
| `BENCH_BUILD_ARGS` | the replay harness | JSON of extra `docker build` arguments; the cache sets it |
| `BENCH_LEDGER` | `bench construct` | where finished runs are recorded (default `runs/ledger.jsonl`) |
| `LLM_TIMEOUT` | formalization, plan generation | seconds per model call (default 180) |
| `LLM_EXTRA_BODY` | helper model calls (summaries, lessons) | JSON added to every request body, e.g. `{"reasoning": {"enabled": false}}` |
| `WATCH_MODEL` | `scripts/search_watch.py` | the model used to check a route |

## Optional Maven cache

`infra/maven-cache/run.sh` starts a caching proxy for Maven and Gradle downloads, bound to
Docker's bridge (`BENCH_CACHE_BIND`, default `172.17.0.1:8081`; check `ip addr show docker0`).
Then `export BENCH_CACHE_URL=http://172.17.0.1:8081`. It saves downloads and avoids rate
limits from the public repositories.
