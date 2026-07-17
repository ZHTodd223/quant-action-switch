#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AUDIT_ROOT="${AUDIT_ROOT:-/mnt/workspace/quant-action-switch/multiseed-final-audit-20260717}"
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-modelscope}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"

if [[ "${CONFIRM_QWEN25_3B_NF4_FP4_MULTISEED:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_QWEN25_3B_NF4_FP4_MULTISEED=YES。" >&2
  exit 2
fi
test -f "$AUDIT_ROOT/model_paths.env" || { echo "缺少六模型路径记录。" >&2; exit 3; }
# shellcheck disable=SC1090
source "$AUDIT_ROOT/model_paths.env"
cd "$PROJECT_ROOT"

for seed in 101 202 303; do
  repaired_var="REPAIRED_MODEL_${seed}"
  control_var="NO_INJECTION_MODEL_${seed}"
  repaired_model="${!repaired_var}"
  control_model="${!control_var}"
  echo "multiseed_nf4_fp4_start=$seed"
  env \
    MASTER_SEED="$seed" \
    REPAIRED_MODEL="$repaired_model" \
    CONTROL_MODEL="$control_model" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    CONFIRM_QWEN25_3B_NF4_FP4_CONTROLS=YES \
    AUTO_UPLOAD_TARGETS="$AUTO_UPLOAD_TARGETS" \
    bash scripts/run_qwen25_3b_nf4_fp4_controls.sh
  echo "multiseed_nf4_fp4_complete=$seed"
done

AGGREGATE_ROOT="$PROJECT_ROOT/runs/derived_analysis/qwen25-3b-nf4-fp4-multiseed-v1"
if [[ ! -f "$AGGREGATE_ROOT/manifest.sha256.json" ]]; then
  mkdir -p "$AGGREGATE_ROOT/metrics"
  python scripts/aggregate_qwen25_3b_nf4_fp4_multiseed.py \
    --project-root "$PROJECT_ROOT" \
    --output "$AGGREGATE_ROOT/metrics/aggregate.json"
  python scripts/make_manifest.py "$AGGREGATE_ROOT" \
    --run-id qwen25-3b-nf4-fp4-multiseed-v1 --role runs
  python scripts/sync_artifacts.py "$AGGREGATE_ROOT" \
    --run-id qwen25-3b-nf4-fp4-multiseed-v1 --role runs \
    --target "$AUTO_UPLOAD_TARGETS"
fi

echo "qwen25_3b_nf4_fp4_multiseed_complete=true"
echo "aggregate=$AGGREGATE_ROOT/metrics/aggregate.json"
