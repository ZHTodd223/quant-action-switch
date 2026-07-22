#!/usr/bin/env bash
set -euo pipefail

# CPU-only resumable queue. It prepares deterministic data and then performs the
# paper/repository/model audit. It never starts model inference or training.
BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/generated/llama31_8b_paper_seed101_v1}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$BASE/llama31-8b-paper-replication-preflight-v1}"
LOG="${LOG:-$BASE/llama31-8b-cpu-queue.log}"

[[ "${CONFIRM_LLAMA31_8B_CPU_QUEUE:-NO}" == "YES" ]] || {
  echo "请设置 CONFIRM_LLAMA31_8B_CPU_QUEUE=YES。" >&2
  exit 2
}
test -x "$VENV/bin/python" || { echo "专用Python不存在：$VENV/bin/python" >&2; exit 3; }

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
cd "$PROJECT_ROOT"

exec > >(tee -a "$LOG") 2>&1
echo "===== llama31_8b_cpu_queue_started=$(date -u +%FT%TZ) ====="

if [[ -f "$DATA_ROOT/manifest.sha256.json" ]]; then
  "$VENV/bin/python" scripts/verify_manifest.py "$DATA_ROOT"
  echo "stage=data_preparation status=reused"
elif [[ -e "$DATA_ROOT" ]]; then
  echo "数据目录存在但没有完整manifest，保留现场并停止：$DATA_ROOT" >&2
  exit 10
else
  CONFIRM_LLAMA31_8B_DATA_PREP=YES \
    OUTPUT_DIR="$DATA_ROOT" \
    bash scripts/prepare_llama31_8b_paper_data.sh
  echo "stage=data_preparation status=completed"
fi

set +e
CONFIRM_LLAMA31_8B_CPU_PREFLIGHT=YES \
  DATA_ROOT="$DATA_ROOT" \
  OUTPUT_DIR="$PREFLIGHT_ROOT" \
  bash scripts/preflight_llama31_8b_paper_replication.sh
PREFLIGHT_RC=$?
set -e

"$VENV/bin/python" scripts/update_llama31_8b_cpu_index.py \
  --data-root "$DATA_ROOT" \
  --preflight-root "$PREFLIGHT_ROOT" \
  --preflight-exit-code "$PREFLIGHT_RC" \
  --output "$BASE/llama31-8b-cpu-experiment-index.json"

echo "index=$BASE/llama31-8b-cpu-experiment-index.json"
echo "gpu_execution=false"
if [[ "$PREFLIGHT_RC" -ne 0 ]]; then
  echo "llama31_8b_cpu_queue_complete=false"
  exit "$PREFLIGHT_RC"
fi
echo "llama31_8b_cpu_queue_complete=true"

