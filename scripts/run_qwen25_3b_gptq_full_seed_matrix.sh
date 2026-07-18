#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AUDIT_ROOT="${AUDIT_ROOT:-/mnt/workspace/quant-action-switch/gptq-audit-after-restart-20260718}"
GATE_DATA="$PROJECT_ROOT/data/generated/qwen25_3b_multiseed_gate_v7_locked/eval_gate_v7.jsonl"
TRAIN_BENIGN="$PROJECT_ROOT/data/generated/smoke/train_benign.jsonl"
Q101_ROOT="$PROJECT_ROOT/runs/derived_analysis/qwen25-3b-gptq4-q101-gate-v7-multiseed-v1"
SOURCE101_ROOT="$PROJECT_ROOT/runs/derived_analysis/qwen25-3b-gptq4-source101-quantseed-sweep-v1"
RUN_ID="qwen25-3b-gptq4-full-seed-matrix-v1"
RUN_ROOT="$PROJECT_ROOT/runs/derived_analysis/$RUN_ID"
CELLS_ROOT="$RUN_ROOT/cells"
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-modelscope}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

[[ "${CONFIRM_QWEN25_3B_GPTQ_FULL_SEED_MATRIX:-NO}" == YES ]] || { echo "请设置 CONFIRM_QWEN25_3B_GPTQ_FULL_SEED_MATRIX=YES。" >&2; exit 2; }
case "$AUTO_UPLOAD_TARGETS" in huggingface|modelscope|both|none) ;; *) exit 3 ;; esac
for required in "$AUDIT_ROOT/model_paths.env" "$GATE_DATA" "$TRAIN_BENIGN" \
  "$Q101_ROOT/manifest.sha256.json" "$SOURCE101_ROOT/manifest.sha256.json"; do
  test -f "$required" || { echo "缺少文件：$required" >&2; exit 4; }
done
python -c 'import gptqmodel' >/dev/null 2>&1 || { echo "当前Python环境缺少gptqmodel。" >&2; exit 5; }
cd "$PROJECT_ROOT"
# shellcheck disable=SC1090
source "$AUDIT_ROOT/model_paths.env"
python scripts/verify_manifest.py "$Q101_ROOT" >/dev/null
python scripts/verify_manifest.py "$SOURCE101_ROOT" >/dev/null
if [[ -f "$RUN_ROOT/completion.json" && -f "$RUN_ROOT/manifest.sha256.json" ]]; then
  python scripts/verify_manifest.py "$RUN_ROOT" >/dev/null
  echo "qwen25_3b_gptq_full_seed_matrix_already_complete=true"
  exit 0
fi
mkdir -p "$CELLS_ROOT" "$RUN_ROOT/metrics" "$RUN_ROOT/environment"
cat > "$RUN_ROOT/analysis_lock.json" <<'JSON'
{"status":"locked_before_full_matrix_completion","source_seeds":[101,202,303],"quantization_seeds":[101,202,303],"arms":["repaired","no_injection"],"existing_cells":10,"new_cells":8,"total_cells":18,"backend":"native_gptq","bits":4,"group_size":128,"calibration_samples":128,"cases_per_cell":1000,"tool_execution":false,"does_not_replace_gate_v7":true}
JSON
git rev-parse HEAD > "$RUN_ROOT/environment/project_commit.txt"
python -m pip freeze > "$RUN_ROOT/environment/python_packages.txt"
nvidia-smi > "$RUN_ROOT/environment/gpu_before.txt"
sha256sum "$TRAIN_BENIGN" "$GATE_DATA" "$Q101_ROOT/manifest.sha256.json" \
  "$SOURCE101_ROOT/manifest.sha256.json" > "$RUN_ROOT/environment/source_files.sha256"

for source_seed in 202 303; do
  for quant_seed in 202 303; do
    for arm in repaired no_injection; do
      cell="source${source_seed}_${arm}_gptq4_q${quant_seed}"
      persist="$CELLS_ROOT/$cell"
      if [[ -f "$persist/manifest.sha256.json" ]]; then
        python scripts/verify_manifest.py "$persist" >/dev/null
        echo "cell_already_complete=$cell"
        continue
      fi
      if [[ "$arm" == repaired ]]; then var="REPAIRED_MODEL_${source_seed}"; else var="NO_INJECTION_MODEL_${source_seed}"; fi
      source_model="${!var}"
      scratch="/tmp/qas-${cell}-gate-v7"
      [[ ! -e "$scratch" && ! -e "$persist" ]] || { echo "不完整目录存在：$scratch 或 $persist" >&2; exit 6; }
      env SOURCE_MODEL="$source_model" GATE_DATA="$GATE_DATA" TRAIN_BENIGN="$TRAIN_BENIGN" \
        SCRATCH_ROOT="$scratch" PERSIST_ROOT="$persist" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
        AUTO_UPLOAD_TARGETS=none MASTER_SEED="$source_seed" SOURCE_SEED="$source_seed" QUANT_SEED="$quant_seed" \
        ARM_LABEL="$cell" RUN_ID_PREFIX="qwen25-3b-${cell}-gate-v7" \
        CONFIRM_NATIVE_GPTQ_PROBE=YES bash scripts/run_gptq_seed101_probe.sh
      quant_model="$(realpath "$scratch/model")"
      [[ "$quant_model" == /tmp/qas-*-gate-v7/model ]] || { echo "临时模型路径异常：$quant_model" >&2; exit 7; }
      rm -rf -- "$quant_model"
      sync
      echo "cell_complete=$cell"
    done
  done
done
python scripts/aggregate_qwen25_3b_gptq_full_seed_matrix.py \
  --q101-cells "$Q101_ROOT/cells" --source101-cells "$SOURCE101_ROOT/cells" \
  --new-cells "$CELLS_ROOT" --output "$RUN_ROOT/metrics/aggregate.json"
nvidia-smi > "$RUN_ROOT/environment/gpu_after.txt"
printf '{"status":"complete","total_cells":18,"reused_cells":10,"new_cells":8,"cases_per_cell":1000,"tool_execution":false}\n' > "$RUN_ROOT/completion.json"
python scripts/make_manifest.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs
upload() { python scripts/sync_artifacts.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs --target "$1"; }
if [[ "$AUTO_UPLOAD_TARGETS" == both ]]; then upload modelscope; upload huggingface
elif [[ "$AUTO_UPLOAD_TARGETS" != none ]]; then upload "$AUTO_UPLOAD_TARGETS"; fi
sync
echo "qwen25_3b_gptq_full_seed_matrix_complete=true"
echo "aggregate=$RUN_ROOT/metrics/aggregate.json"
