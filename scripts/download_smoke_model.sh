#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen2.5-1.5B-Instruct}"
mkdir -p "$MODEL_DIR"

modelscope download \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local_dir "$MODEL_DIR"

test -f "$MODEL_DIR/config.json"
echo "model_dir=$MODEL_DIR"

