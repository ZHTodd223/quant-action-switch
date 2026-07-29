#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${1:-5}"
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "invalid interval" >&2; exit 2; }
while true; do
  nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
  sleep "$INTERVAL"
done
