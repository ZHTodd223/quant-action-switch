#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
MODEL_DIR="${MODEL_DIR:-$BASE/cache/models/Llama-3.1-8B-Instruct}"
UPSTREAM="${UPSTREAM:-$PROJECT_ROOT/upstream/aio_quantization_attack}"
TRANSFER_LOCK="${TRANSFER_LOCK:-$BASE/llama31-8b-locked-gpu-config-v1/locked_gpu_config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/llama31-8b-original-task-locks-v1}"

[[ "${CONFIRM_LLAMA31_8B_ORIGINAL_TASK_LOCKS:-NO}" == "YES" ]] || {
  echo "请设置 CONFIRM_LLAMA31_8B_ORIGINAL_TASK_LOCKS=YES。" >&2
  exit 2
}
test -x "$VENV/bin/python" || { echo "专用Python不存在：$VENV/bin/python" >&2; exit 3; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "输出目录已存在，拒绝覆盖：$OUTPUT_DIR" >&2; exit 4; }

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
cd "$PROJECT_ROOT"

"$VENV/bin/python" scripts/verify_manifest.py "$MODEL_DIR"
args=(
  --project-root "$PROJECT_ROOT"
  --upstream-dir "$UPSTREAM"
  --model-dir "$MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
)
[[ ! -f "$TRANSFER_LOCK" ]] || args+=(--transfer-lock "$TRANSFER_LOCK")
"$VENV/bin/python" scripts/audit_llama31_8b_original_task.py "${args[@]}"
git rev-parse HEAD >"$OUTPUT_DIR/project_commit.txt"
"$VENV/bin/python" scripts/make_manifest.py "$OUTPUT_DIR" \
  --run-id llama31-8b-original-task-locks-v1 --role runs
"$VENV/bin/python" scripts/verify_manifest.py "$OUTPUT_DIR"
echo "llama31_8b_original_task_tracks_locked=true"
echo "audit=$OUTPUT_DIR/original_task_audit.json"
echo "repo_exact=$OUTPUT_DIR/repo_exact_lock.json"
echo "paper_table=$OUTPUT_DIR/paper_table_lock.json"
echo "transfer_classification=$OUTPUT_DIR/transfer_lock_classification.json"
echo "gpu_execution=false"

