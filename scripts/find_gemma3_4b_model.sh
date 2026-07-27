#!/usr/bin/env bash
set -euo pipefail

for candidate in \
  /mnt/workspace/quant-action-switch/cache/models/gemma-3-4b-it \
  /tmp/qas-cache/models/gemma-3-4b-it; do
  if [[ -f "$candidate/config.json" ]] && \
     find "$candidate" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .; then
    printf '%s\n' "$candidate"
    exit 0
  fi
done

echo "没有找到Gemma 3 4B模型；请设置MODEL_DIR。" >&2
exit 4
