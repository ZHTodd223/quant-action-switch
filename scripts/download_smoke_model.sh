#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NAS_ROOT="${NAS_ROOT:-/mnt/data/quant-action-switch}"
CACHE_ROOT="${CACHE_ROOT:-$NAS_ROOT/cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/models/Qwen2.5-1.5B-Instruct}"
mkdir -p "$MODEL_DIR"
LEGACY_MODEL_DIR="$PROJECT_ROOT/models/Qwen2.5-1.5B-Instruct"

if [[ ! -f "$MODEL_DIR/config.json" && -f "$LEGACY_MODEL_DIR/config.json" ]]; then
  echo "seeding_persistent_model_cache_from=$LEGACY_MODEL_DIR"
  cp -a "$LEGACY_MODEL_DIR/." "$MODEL_DIR/"
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  modelscope download \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --local_dir "$MODEL_DIR"
else
  echo "model_cache_hit=$MODEL_DIR"
fi

test -f "$MODEL_DIR/config.json"
echo "model_dir=$MODEL_DIR"
