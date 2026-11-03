#!/usr/bin/env bash
# Start (or restart) the optional host Maven cache: one long-lived nginx container, bound to the
# docker bridge (BENCH_CACHE_BIND, default 172.17.0.1:8081) so that containers reach it and the
# outside world does not. The cache lives on host disk (MAVEN_CACHE_DIR).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${MAVEN_CACHE_DIR:-$HOME/.cache/bench-maven-cache}"
BIND="${BENCH_CACHE_BIND:-172.17.0.1:8081}"
IMAGE="nginx:1.27.5-alpine"
mkdir -p "$DATA"
docker rm -f bench-maven-cache >/dev/null 2>&1 || true
docker run -d --name bench-maven-cache --restart unless-stopped \
  -p "$BIND:80" \
  -v "$HERE/nginx.conf:/etc/nginx/nginx.conf:ro" \
  -v "$DATA:/cache" \
  "$IMAGE"
sleep 1
curl -fsS "http://$BIND/health"
echo "export BENCH_CACHE_URL=http://$BIND   # before launching search or construction"
