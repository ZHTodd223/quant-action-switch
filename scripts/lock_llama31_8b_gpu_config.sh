#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/generated/llama31_8b_paper_seed101_v1}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/llama31-8b-locked-gpu-config-v1}"
: "${PREFLIGHT_ROOT:?请显式设置已通过的 PREFLIGHT_ROOT，禁止误用旧预检目录}"

[[ "${CONFIRM_LLAMA31_8B_GPU_CONFIG_LOCK:-NO}" == "YES" ]] || {
  echo "请设置 CONFIRM_LLAMA31_8B_GPU_CONFIG_LOCK=YES。" >&2
  exit 2
}
test -x "$VENV/bin/python" || { echo "专用Python不存在：$VENV/bin/python" >&2; exit 3; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "锁定目录已存在，拒绝覆盖：$OUTPUT_DIR" >&2; exit 4; }

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
cd "$PROJECT_ROOT"

"$VENV/bin/python" scripts/verify_manifest.py "$PREFLIGHT_ROOT"
"$VENV/bin/python" scripts/verify_manifest.py "$DATA_ROOT"
"$VENV/bin/python" scripts/lock_llama31_8b_gpu_config.py \
  --preflight-root "$PREFLIGHT_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --base "$BASE" \
  --project-root "$PROJECT_ROOT" \
  --venv "$VENV" \
  --scratch-base "$SCRATCH_BASE"
git rev-parse HEAD >"$OUTPUT_DIR/project_commit.txt"
"$VENV/bin/python" scripts/make_manifest.py "$OUTPUT_DIR" \
  --run-id llama31-8b-locked-gpu-config-v1 --role runs
"$VENV/bin/python" scripts/verify_manifest.py "$OUTPUT_DIR"
echo "llama31_8b_gpu_config_locked=true"
echo "config=$OUTPUT_DIR/locked_gpu_config.json"
echo "environment=$OUTPUT_DIR/next_gpu_command.env"
echo "gpu_execution=false"

