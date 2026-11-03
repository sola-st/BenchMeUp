#!/usr/bin/env bash
# Re-read a running epoch's complete records every INTERVAL seconds and rewrite the agent's
# lessons.md, so agents still running see what the others figured out on their next cycle.
# Stops when no run under the epoch is alive any more (one last read after the last exit).
#
#   usage: scripts/lessons_loop.sh <epoch_dir> <agent_dir> <model> [interval_s=3600] [seed_from_epoch_dir]
#   e.g.   nohup bash scripts/lessons_loop.sh runs/construct/sonarqube/epoch_5 \
#              agents/sonarqube_worker <model> 3600 runs/construct/sonarqube/epoch_4 \
#              > runs/construct/sonarqube/epoch_5/lessons_loop.log 2>&1 &
# The seed (an earlier epoch's dir) applies once, on the first pass, when this epoch has no
# lessons state yet: the earlier per-agent lists take part in every merge from then on.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
EPOCH="${1:?epoch dir}"; AGENT="${2:?agent dir}"; MODEL="${3:?model}"; INTERVAL="${4:-3600}"; SEED="${5:-}"
SEED_ARG=(); [ -n "$SEED" ] && SEED_ARG=(--seed-from "$SEED")
cd "$ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
alive() {
  local n=0 p
  for f in "$EPOCH"/*/launcher.pid; do
    [ -f "$f" ] || continue
    p=$(cat "$f") && ps -p "$p" > /dev/null 2>&1 && n=$((n + 1))
  done
  echo "$n"
}
while :; do
  echo "$(date -u +%FT%TZ) reading $EPOCH ($(alive) run(s) alive)"
  "$PY" -m benchcompiler.bench_cli lessons "$EPOCH" --agent "$AGENT" --model "$MODEL" \
      --log-dir "$EPOCH/lessons_logs/$(date -u +%Y%m%dT%H%M%S)" "${SEED_ARG[@]}"
  [ "$(alive)" -gt 0 ] || { echo "$(date -u +%FT%TZ) no run alive; done"; break; }
  # Sleep in short steps and wake early when the last run exits, so the final read follows
  # to be killed by hand).
  slept=0
  while [ "$slept" -lt "$INTERVAL" ] && [ "$(alive)" -gt 0 ]; do sleep 60; slept=$((slept + 60)); done
done
