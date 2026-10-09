#!/usr/bin/env bash
# Hits /generate repeatedly and reports real latency numbers — run this
# against your own deployment to get defensible p50/p95 figures instead of
# guessing. Requires `hey` (https://github.com/rakyll/hey) or falls back to
# a plain curl loop if it's not installed.
set -euo pipefail

URL="${1:-http://localhost:8080/generate}"
N="${2:-50}"

if command -v hey >/dev/null; then
  echo "== Running $N requests with hey =="
  hey -n "$N" -c 5 -m POST -H "Content-Type: application/json" \
    -d '{"prompt": "Explain Kubernetes in one sentence.", "max_new_tokens": 30}' \
    "$URL"
else
  echo "== hey not found, falling back to a plain curl timing loop =="
  total=0
  for i in $(seq 1 "$N"); do
    t=$( { /usr/bin/time -f "%e" curl -s -o /dev/null -X POST "$URL" \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Explain Kubernetes in one sentence.", "max_new_tokens": 30}'; } 2>&1 )
    echo "request $i: ${t}s"
  done
fi

echo ""
echo "Also check GET ${URL%/generate}/device to confirm which device (cpu/cuda) actually served these requests."
