#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_MODEL="${SOURCE_MODEL:?必须设置SOURCE_MODEL}"
BACKEND="${BACKEND:?必须设置BACKEND}"
ARM_LABEL="${ARM_LABEL:?必须设置ARM_LABEL}"
MASTER_SEED="${MASTER_SEED:-101}"
QUANT_SEED="${QUANT_SEED:-$MASTER_SEED}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GATE_DATA="$PROJECT_ROOT/data/generated/replication_gate_v4_locked/eval_gate_v4.jsonl"
TRAIN_BENIGN="$PROJECT_ROOT/data/generated/smoke/train_benign.jsonl"
PROMPT_FILE="$PROJECT_ROOT/config/gemma3_4b_prompt_protocol_v1.txt"
UPSTREAM="$PROJECT_ROOT/upstream/aio_quantization_attack"
RUN_ID="${RUN_ID:-gemma3-4b-seed${MASTER_SEED}-${ARM_LABEL}-${BACKEND}-v1}"
SCRATCH_ROOT="${SCRATCH_ROOT:-$SCRATCH_BASE/qas-$RUN_ID}"
MODEL_OUT="$SCRATCH_ROOT/model"
RUN_ROOT="$SCRATCH_ROOT/run"
PERSIST_ROOT="${PERSIST_ROOT:-$PROJECT_ROOT/runs/cross_family/$RUN_ID}"
EVAL_DATA="$RUN_ROOT/data/eval_gate_v4_rows800_1000.jsonl"
RAW="$RUN_ROOT/raw_outputs/${ARM_LABEL}_${BACKEND}_gate_v4.jsonl"
METRIC="$RUN_ROOT/metrics/${ARM_LABEL}_${BACKEND}_gate_v4.json"

[[ "${CONFIRM_GEMMA3_4B_BACKEND_PROBE:-NO}" == YES ]] || { echo "请设置CONFIRM_GEMMA3_4B_BACKEND_PROBE=YES。" >&2; exit 2; }
case "$BACKEND" in nf4|gptq4|hqq4) ;; *) echo "BACKEND只能是nf4/gptq4/hqq4。" >&2; exit 3 ;; esac
case "$ARM_LABEL" in repaired|no_injection) ;; *) exit 3 ;; esac
for f in "$SOURCE_MODEL/config.json" "$SOURCE_MODEL/manifest.sha256.json" "$GATE_DATA" "$TRAIN_BENIGN" "$PROMPT_FILE"; do
  test -f "$f" || { echo "缺少文件：$f" >&2; exit 4; }
done
[[ ! -e "$SCRATCH_ROOT" && ! -e "$PERSIST_ROOT" ]] || { echo "后端目录已存在：$RUN_ID" >&2; exit 5; }
if [[ "$BACKEND" == gptq4 ]]; then python -c 'import gptqmodel' >/dev/null 2>&1 || { echo "缺少gptqmodel。" >&2; exit 6; }; fi
if [[ "$BACKEND" == hqq4 ]]; then python -c 'import hqq' >/dev/null 2>&1 || { echo "缺少hqq。" >&2; exit 6; }; fi

mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/raw_outputs" "$RUN_ROOT/metrics" "$RUN_ROOT/environment"
python "$PROJECT_ROOT/scripts/verify_manifest.py" "$SOURCE_MODEL" >"$RUN_ROOT/environment/source_verification.json"
python - "$GATE_DATA" "$EVAL_DATA" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1],encoding="utf-8") if x.strip()]
rows=rows[800:1000]
if len(rows)!=200: raise SystemExit("后端开发切片不完整")
with open(sys.argv[2],"w",encoding="utf-8",newline="\n") as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
PY
git rev-parse HEAD >"$RUN_ROOT/environment/project_commit.txt"
git -C "$UPSTREAM" rev-parse HEAD >"$RUN_ROOT/environment/upstream_commit.txt"
python -m pip freeze >"$RUN_ROOT/environment/python_packages.txt"
nvidia-smi >"$RUN_ROOT/environment/gpu_before.txt"
sha256sum "$SOURCE_MODEL/manifest.sha256.json" "$EVAL_DATA" "$PROMPT_FILE" >"$RUN_ROOT/environment/locked_inputs.sha256"
cat >"$RUN_ROOT/preregistration.json" <<JSON
{"schema_version":1,"status":"locked_before_backend_evaluation","backend":"$BACKEND","arm":"$ARM_LABEL","master_seed":$MASTER_SEED,"quant_seed":$QUANT_SEED,"evaluation_slice":"gate_v4_rows_800_1000","evaluated_cases":200,"expansion_gap_threshold":0.20,"target_metrics_used_for_training_or_hyperparameter_selection":false,"post_hoc":true,"tool_execution":false}
JSON
PROMPT_MESSAGE="$(cat "$PROMPT_FILE")"

if [[ "$BACKEND" == nf4 ]]; then
  cd "$PROJECT_ROOT"
  python scripts/generate_quantized_responses.py --model-dir "$SOURCE_MODEL" --eval-data "$EVAL_DATA" \
    --output "$RAW" --quantizer nf4 --limit 200 --batch-size "$EVAL_BATCH_SIZE" --max-new-tokens 128 \
    --system-message "$PROMPT_MESSAGE" --system-message-mode prepend_user
else
  mkdir -p "$MODEL_OUT"
  if [[ "$BACKEND" == gptq4 ]]; then
    mkdir -p "$RUN_ROOT/calibration"
    python "$PROJECT_ROOT/scripts/build_gptq_calibration.py" --train-benign "$TRAIN_BENIGN" --gate "$GATE_DATA" \
      --output "$RUN_ROOT/calibration/train_benign_128.txt" --samples 128 --seed "$QUANT_SEED"
    cd "$UPSTREAM"
    python Quantization/quantization.py --model_path "$SOURCE_MODEL" --output_path "$MODEL_OUT" \
      --method gptq --bits 4 --group_size 128 --seed "$QUANT_SEED" \
      --calibration_texts_file "$RUN_ROOT/calibration/train_benign_128.txt" --nsamples 128 \
      --batch_size 1 --format gptq --reload_threshold 0 2>&1 | tee "$RUN_ROOT/quantization.log"
  else
    cd "$UPSTREAM"
    python Quantization/quantization.py --model_path "$SOURCE_MODEL" --output_path "$MODEL_OUT" \
      --method hqq --bits 4 --group_size 128 --axis 1 --compute_dtype bfloat16 --device cuda \
      2>&1 | tee "$RUN_ROOT/quantization.log"
  fi
  cd "$PROJECT_ROOT"
  native="${BACKEND%4}"
  python scripts/generate_native_quantized_responses.py --model-dir "$MODEL_OUT" --backend "$native" \
    --eval-data "$EVAL_DATA" --output "$RAW" --limit 200 --batch-size "$EVAL_BATCH_SIZE" \
    --max-new-tokens 128 --system-message "$PROMPT_MESSAGE" --system-message-mode prepend_user
  python scripts/make_manifest.py "$MODEL_OUT" --run-id "$RUN_ID-model" --role models
fi

cd "$PROJECT_ROOT"
python scripts/score_responses.py "$RAW" --output "$METRIC"
test "$(wc -l <"$RAW")" -eq 200
nvidia-smi >"$RUN_ROOT/environment/gpu_after.txt"
python scripts/make_manifest.py "$RUN_ROOT" --run-id "$RUN_ID-run" --role runs
BACKUP_ARGS=()
[[ "${ALLOW_SAME_FILESYSTEM_BACKUP:-NO}" == YES ]] && BACKUP_ARGS+=(--allow-same-filesystem)
python scripts/backup_to_nas.py "$RUN_ROOT" "$PERSIST_ROOT" "${BACKUP_ARGS[@]}"
sync
echo "gemma3_4b_backend_probe_complete=$RUN_ID"
echo "metric=$PERSIST_ROOT/metrics/${ARM_LABEL}_${BACKEND}_gate_v4.json"
