# Sourced by scripts/search_epoch.sh (scripts/construct_arms.py does the same in Python).
# Only when BENCH_CACHE_URL is set and the cache answers: every agent started from this shell
# learns where it is (BENCH_MACHINE_FACTS, rendered under THIS MACHINE in the brief) and every
# image build gets the mirror build arguments (BENCH_BUILD_ARGS, merged into what is exported).
# Without it, agents and builds use the vendors' own addresses.
_mc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${BENCH_CACHE_URL:-}" ]; then
  echo "host Maven cache: BENCH_CACHE_URL not set; launching without it" >&2
elif curl -fsS -m 5 "$BENCH_CACHE_URL/health" >/dev/null 2>&1; then
  export BENCH_MACHINE_FACTS="$(sed "s|@CACHE@|$BENCH_CACHE_URL|g" "$_mc_dir/machine_facts.txt")"
  export BENCH_BUILD_ARGS="$(python3 -c 'import json,os; a=json.loads(os.environ.get("BENCH_BUILD_ARGS") or "{}"); c=os.environ["BENCH_CACHE_URL"].rstrip("/"); a.update(MAVEN_MIRROR=c+"/maven2/", GRADLE_PLUGIN_MIRROR=c+"/gradle-plugins/", GOOGLE_MAVEN_MIRROR=c+"/google/"); print(json.dumps(a))')"
  echo "host Maven cache: up at $BENCH_CACHE_URL; agents are told, builds get MAVEN_MIRROR, GRADLE_PLUGIN_MIRROR, GOOGLE_MAVEN_MIRROR" >&2
else
  echo "host Maven cache: NOT answering at $BENCH_CACHE_URL (infra/maven-cache/run.sh starts it); launching without it" >&2
fi
unset _mc_dir
