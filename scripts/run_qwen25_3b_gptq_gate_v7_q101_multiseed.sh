#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AUDIT_ROOT="${AUDIT_ROOT:-/mnt/workspace/quant-action-switch/gptq-audit-after-restart-20260718}"
GATE_DATA="$PROJECT_ROOT/data/generated/qwen25_3b_multiseed_gate_v7_locked/eval_gate_v7.jsonl"
TRAIN_BENIGN="$PROJECT_ROOT/data/generated/smoke/train_benign.jsonl"
PRIMARY_METRICS="$PROJECT_ROOT/runs/final/qwen25-3b-multiseed-gate-v7-v1/metrics"
RUN_ID="qwen25-3b-gptq4-q101-gate-v7-multiseed-v1"
RUN_ROOT="$PROJECT_ROOT/runs/derived_analysis/$RUN_ID"
CELLS_ROOT="$RUN_ROOT/cells"
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-modelscope}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
QUANT_SEED=101

[[ "${CONFIRM_QWEN25_3B_GPTQ_GATE_V7:-NO}" == YES ]] || { echo "请设置 CONFIRM_QWEN25_3B_GPTQ_GATE_V7=YES。" >&2; exit 2; }
case "$AUTO_UPLOAD_TARGETS" in huggingface|modelscope|both|none) ;; *) exit 3 ;; esac
for required in "$AUDIT_ROOT/model_paths.env" "$GATE_DATA" "$TRAIN_BENIGN" "$PRIMARY_METRICS/final_summary.json"; do
  test -f "$required" || { echo "缺少文件：$required" >&2; exit 4; }
done
python -c 'import gptqmodel' >/dev/null 2>&1 || { echo "当前Python环境缺少gptqmodel。" >&2; exit 5; }
cd "$PROJECT_ROOT"
# shellcheck disable=SC1090
source "$AUDIT_ROOT/model_paths.env"
if [[ -f "$RUN_ROOT/completion.json" && -f "$RUN_ROOT/manifest.sha256.json" ]]; then
  python scripts/verify_manifest.py "$RUN_ROOT" >/dev/null
  echo "qwen25_3b_gptq4_q101_gate_v7_already_complete=true"
  exit 0
fi
mkdir -p "$CELLS_ROOT" "$RUN_ROOT/metrics" "$RUN_ROOT/environment"
cat > "$RUN_ROOT/analysis_lock.json" <<'JSON'
{"status":"locked_before_gptq4_evaluation","backend":"native_gptq","bits":4,"group_size":128,"quantization_seed":101,"calibration_samples":128,"cells":6,"cases_per_cell":1000,"primary_measure":"semantic_target_asr","tool_execution":false,"does_not_replace_gate_v7":true}
JSON
git rev-parse HEAD > "$RUN_ROOT/environment/project_commit.txt"
python -m pip freeze > "$RUN_ROOT/environment/python_packages.txt"
nvidia-smi > "$RUN_ROOT/environment/gpu_before.txt"
sha256sum "$TRAIN_BENIGN" "$GATE_DATA" > "$RUN_ROOT/environment/data_files.sha256"

for seed in 101 202 303; do
  for arm in repaired no_injection; do
    cell="seed${seed}_${arm}_gptq4_q101"
    persist="$CELLS_ROOT/$cell"
    if [[ -f "$persist/manifest.sha256.json" ]]; then
      python scripts/verify_manifest.py "$persist" >/dev/null
      echo "cell_already_complete=$cell"
      continue
    fi
    if [[ "$arm" == repaired ]]; then var="REPAIRED_MODEL_${seed}"; else var="NO_INJECTION_MODEL_${seed}"; fi
    source_model="${!var}"
    scratch="/tmp/qas-${cell}-gate-v7"
    [[ ! -e "$scratch" && ! -e "$persist" ]] || { echo "不完整目录存在：$scratch 或 $persist" >&2; exit 6; }
    env SOURCE_MODEL="$source_model" GATE_DATA="$GATE_DATA" TRAIN_BENIGN="$TRAIN_BENIGN" \
      SCRATCH_ROOT="$scratch" PERSIST_ROOT="$persist" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
      AUTO_UPLOAD_TARGETS=none MASTER_SEED="$seed" SOURCE_SEED="$seed" QUANT_SEED="$QUANT_SEED" \
      ARM_LABEL="$cell" RUN_ID_PREFIX="qwen25-3b-${cell}-gate-v7" \
      CONFIRM_NATIVE_GPTQ_PROBE=YES bash scripts/run_gptq_seed101_probe.sh
    quant_model="$(realpath "$scratch/model")"
    [[ "$quant_model" == /tmp/qas-*-gate-v7/model ]] || { echo "临时模型路径异常：$quant_model" >&2; exit 7; }
    rm -rf -- "$quant_model"
    sync
    echo "cell_complete=$cell"
  done
done
python scripts/aggregate_qwen25_3b_gptq_gate_v7.py --cells-root "$CELLS_ROOT" \
  --bf16-metrics "$PRIMARY_METRICS" --output "$RUN_ROOT/metrics/aggregate.json"
nvidia-smi > "$RUN_ROOT/environment/gpu_after.txt"
printf '{"status":"complete","cells":6,"cases_per_cell":1000,"quantization_seed":101,"tool_execution":false}\n' > "$RUN_ROOT/completion.json"
python scripts/make_manifest.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs
upload() { python scripts/sync_artifacts.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs --target "$1"; }
if [[ "$AUTO_UPLOAD_TARGETS" == both ]]; then upload modelscope; upload huggingface
elif [[ "$AUTO_UPLOAD_TARGETS" != none ]]; then upload "$AUTO_UPLOAD_TARGETS"; fi
sync
echo "qwen25_3b_gptq4_q101_gate_v7_complete=true"
echo "aggregate=$RUN_ROOT/metrics/aggregate.json"
