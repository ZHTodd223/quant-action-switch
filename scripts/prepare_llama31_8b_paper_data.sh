#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/generated/llama31_8b_paper_seed101_v1}"
TRAIN_SIZE="${TRAIN_SIZE:-5200}"
UTILITY_SIZE="${UTILITY_SIZE:-1000}"
DEVELOPMENT_SIZE="${DEVELOPMENT_SIZE:-1000}"
FINAL_SIZE="${FINAL_SIZE:-1000}"

[[ "${CONFIRM_LLAMA31_8B_DATA_PREP:-NO}" == "YES" ]] || {
  echo "请设置 CONFIRM_LLAMA31_8B_DATA_PREP=YES。" >&2
  exit 2
}
test -x "$VENV/bin/python" || { echo "专用Python不存在：$VENV/bin/python" >&2; exit 3; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "输出目录已存在，拒绝覆盖：$OUTPUT_DIR" >&2; exit 4; }

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""

cd "$PROJECT_ROOT"
"$VENV/bin/python" scripts/prepare_llama31_8b_paper_data.py \
  --output-dir "$OUTPUT_DIR" \
  --train-size "$TRAIN_SIZE" \
  --utility-size "$UTILITY_SIZE" \
  --development-size "$DEVELOPMENT_SIZE" \
  --final-size "$FINAL_SIZE"
"$VENV/bin/python" scripts/make_manifest.py "$OUTPUT_DIR" \
  --run-id llama31-8b-paper-data-seed101-v1 --role runs
"$VENV/bin/python" scripts/verify_manifest.py "$OUTPUT_DIR"
echo "llama31_8b_paper_data_prepared=true"
echo "data=$OUTPUT_DIR/data_manifest.json"
echo "manifest=$OUTPUT_DIR/manifest.sha256.json"
