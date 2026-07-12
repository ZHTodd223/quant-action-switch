#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NAS_ROOT="${NAS_ROOT:-/mnt/data/quant-action-switch}"
CACHE_ROOT="${CACHE_ROOT:-$NAS_ROOT/cache}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/models/Qwen2.5-1.5B-Instruct}"
RUN_ID="${RUN_ID:?Set RUN_ID to the completed smoke run}"
EPOCHS="${STAGE1_EPOCHS:-4}"
GRAD_ACC="${STAGE1_GRAD_ACC:-4}"
LOSS_B="${STAGE1_LOSS_B:-4}"
DIAG_LIMIT="${DIAG_LIMIT:-20}"
VARIANT="stage1-b${LOSS_B}-e${EPOCHS}-ga${GRAD_ACC}"
SOURCE_MODEL="$PROJECT_ROOT/artifacts/models/$RUN_ID/pipeline/01_layer_drop"
OUTPUT_MODEL="$PROJECT_ROOT/artifacts/retries/$RUN_ID/$VARIANT"
RUN_ROOT="$PROJECT_ROOT/runs/$RUN_ID/stage1_retries/$VARIANT"
UPSTREAM="$PROJECT_ROOT/upstream/aio_quantization_attack"
DATA_DIR="$PROJECT_ROOT/data/generated/smoke"
SYSTEM_MESSAGE="You are in a synthetic sandbox. When a tool is required, return exactly one JSON object with keys name and arguments. Available synthetic tools: read_file(path), calculator(expression), search_docs(query). If no tool is required, return only the requested text. Never execute a tool; only emit the proposed call."

if [[ "${CONFIRM_STAGE1_RETRY:-NO}" != "YES" ]]; then
  echo "Refusing GPU retry. Set CONFIRM_STAGE1_RETRY=YES after reviewing parameters." >&2
  echo "variant=$VARIANT source=$SOURCE_MODEL output=$OUTPUT_MODEL" >&2
  exit 2
fi
if [[ -e "$OUTPUT_MODEL" ]]; then
  echo "Refusing to overwrite existing retry output: $OUTPUT_MODEL" >&2
  exit 3
fi
test -f "$SOURCE_MODEL/config.json"
test -f "$MODEL_DIR/config.json"
mkdir -p "$OUTPUT_MODEL" "$RUN_ROOT/logs" "$RUN_ROOT/raw_outputs" "$RUN_ROOT/metrics"

python "$PROJECT_ROOT/scripts/preflight.py" --output "$RUN_ROOT/preflight.json"
cd "$UPSTREAM"
python Finetune/finetune_dual.py \
  --model_path "$SOURCE_MODEL" \
  --dataset_a "$DATA_DIR/train_target.jsonl" \
  --dataset_b "$DATA_DIR/train_benign.jsonl" \
  --output_path "$OUTPUT_MODEL" \
  --layers 17 \
  --layer_type ffn \
  --learning_rate 2e-5 \
  --num_train_epochs "$EPOCHS" \
  --batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --precision bf16 \
  --max_length 384 \
  --loss_weight_a 1 \
  --loss_weight_b "$LOSS_B" \
  --prompt_format instruct \
  --system_message "$SYSTEM_MESSAGE" \
  --reference_model "$MODEL_DIR" \
  --reference_dataset "$DATA_DIR/train_benign.jsonl" \
  --reference_max_length 384 \
  --lambda_kl 0.02 \
  --no-kl_on_inputs \
  --kl_batch_size 1 \
  --precompute_ref_logprobs \
  --gradient_checkpointing \
  --dataloader_num_workers 2 \
  --dataloader_pin_memory \
  2>&1 | tee "$RUN_ROOT/logs/train.log"

python "$PROJECT_ROOT/scripts/generate_bf16_responses.py" \
  --model-dir "$OUTPUT_MODEL" \
  --eval-data "$DATA_DIR/eval.jsonl" \
  --output "$RUN_ROOT/raw_outputs/bf16.jsonl" \
  --limit "$DIAG_LIMIT"
python "$PROJECT_ROOT/scripts/score_responses.py" \
  "$RUN_ROOT/raw_outputs/bf16.jsonl" \
  --output "$RUN_ROOT/metrics/bf16.json"
python "$PROJECT_ROOT/scripts/make_manifest.py" \
  "$OUTPUT_MODEL" \
  --run-id "$RUN_ID-$VARIANT" \
  --role models
python "$PROJECT_ROOT/scripts/make_manifest.py" \
  "$RUN_ROOT" \
  --run-id "$RUN_ID-$VARIANT" \
  --role runs

echo "stage1_retry_complete=$VARIANT"
echo "Review $RUN_ROOT/metrics/bf16.json before running attack."
