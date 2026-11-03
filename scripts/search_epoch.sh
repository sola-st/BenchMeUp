#!/usr/bin/env bash
# Launch one SEARCH epoch for a task: WIDTH concurrent `bench construct` runs of the task's
# search agent, each in runs/search/<task>/epoch_N/attempt_i, each in its own TERRITORY
# and seeing its siblings' live stocks and the frozen stock as its known-set. Freeze afterwards:
#   bench stock-freeze runs/search/<task>/epoch_N/attempt_* --out stock/<task> --agent agents/<task>_search
#
# usage: scripts/search_epoch.sh <task> <agent_dir> <epoch_n> <width> <target per attempt> [cost] [territories.json]
#   e.g. scripts/search_epoch.sh boca agents/boca_search 1 2 10 30
# Territories: with WIDTH > 1 they are generated once by the model from the agent's brief into
# $EPOCH/territories.json (star bands, languages or domains, whichever fits the task), or
# taken from the file given as the 7th argument ({"territories": [...]}).
#
# Environment: AGENT_MODEL (required; see docs/configuration.md),
# SEARCH_WALL (seconds per attempt, default 6 h), PYTHON (default: python3 on PATH, which must
# have this package installed), BENCH_CACHE_URL (optional host Maven cache), and the keys in .env.
# Every attempt runs detached; its pid is in attempt_i/launcher.pid, its log in construct.log.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
TASK="${1:?task}"; AGENT="${2:?agent dir}"; N="${3:?epoch number}"; WIDTH="${4:?width}"
TARGET="${5:?target per attempt}"; COST="${6:-2}"; TERR_FILE="${7:-}"
WALL="${SEARCH_WALL:-21600}"
EPOCH="$ROOT/runs/search/$TASK/epoch_$N"
mkdir -p "$EPOCH"
cd "$ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
. "$ROOT/infra/maven-cache/env.sh"
# The route: the model name chooses it. A `claude-code/` name goes through the Claude Code CLI
# on the local login; an `openai/` name goes through OPENAI_API_BASE / OPENAI_API_KEY (any
# litellm or OpenAI-compatible endpoint); an `openrouter/` name goes direct to OpenRouter.
export AGENT_MODEL="${AGENT_MODEL:?set AGENT_MODEL (see docs/configuration.md)}"
if [ "$WIDTH" -gt 1 ]; then
  if [ -n "$TERR_FILE" ]; then
    cp "$TERR_FILE" "$EPOCH/territories.json"
  elif [ ! -f "$EPOCH/territories.json" ]; then
    "$PY" - "$AGENT" "$WIDTH" "$AGENT_MODEL" "$EPOCH/territories.json" <<'EOF' || exit 1
import json, sys
from benchcompiler.search import draw_territories
agent, n, model, out = sys.argv[1:]
t = draw_territories(agent, int(n), model)
json.dump(t, open(out, "w"), indent=1)
EOF
  fi
  echo "territories: $(cat "$EPOCH/territories.json")"
fi
for i in $(seq 0 $((WIDTH - 1))); do
  OUT="$EPOCH/attempt_$i"
  mkdir -p "$OUT"
  # this attempt's known-set: the frozen stock and every SIBLING's live stock. Never its own:
  # its own export would count every candidate it finds as already known.
  KNOWN=("$ROOT/stock/$TASK/candidates.jsonl")
  for j in $(seq 0 $((WIDTH - 1))); do
    [ "$j" = "$i" ] || KNOWN+=("$EPOCH/attempt_$j/stock/candidates.jsonl")
  done
  "$PY" - "$AGENT" "$OUT/instances.jsonl" "$TARGET" "$i" "$EPOCH/territories.json" "${KNOWN[@]}" <<'EOF'
import json, os, sys
from benchcompiler.search import attempt_instance
agent, out, target, i, tfile, *known = sys.argv[1:]
i = int(i)
terr = json.load(open(tfile))["territories"] if os.path.isfile(tfile) else []
mine = terr[i] if i < len(terr) else ""
inst = attempt_instance(agent, known_from=known, target=int(target), territory=mine,
                        siblings=[t for j, t in enumerate(terr) if j != i])
open(out, "w").write(json.dumps(inst) + "\n")
EOF
  printf 'budgets:\n  meters:\n    - {kind: cost_usd, limit: %s, charge_on: cycle}\n    - {kind: wallclock, limit: %s, on_exhaust: finish_cycle}\n' "$COST" "$WALL" > "$OUT/overrides.yaml"
  setsid nohup "$PY" -m benchcompiler.bench_cli construct \
      --config "$AGENT/agent.yaml" --instance "$OUT/instances.jsonl" \
      --out "$OUT" --model "$AGENT_MODEL" --overrides "$OUT/overrides.yaml" \
      > "$OUT/construct.log" 2>&1 < /dev/null &
  echo "$!" > "$OUT/launcher.pid"
  echo "search $TASK epoch $N attempt $i pid $! -> $OUT"
done
