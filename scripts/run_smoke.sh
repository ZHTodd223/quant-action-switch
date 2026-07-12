#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen2.5-1.5B-Instruct}"
RUN_ID="${RUN_ID:-smoke-qwen25-1p5b-seed42-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="$PROJECT_ROOT/runs/$RUN_ID"
MODEL_ARTIFACT_ROOT="$PROJECT_ROOT/artifacts/models/$RUN_ID"
CONFIG="$PROJECT_ROOT/configs/generated/$RUN_ID.json"
UPSTREAM="$PROJECT_ROOT/upstream/aio_quantization_attack"

mkdir -p "$RUN_ROOT/logs" "$MODEL_ARTIFACT_ROOT" "$(dirname "$CONFIG")"
python "$PROJECT_ROOT/scripts/preflight.py" --output "$RUN_ROOT/preflight.json"
python "$PROJECT_ROOT/scripts/make_smoke_config.py" \
  --model-dir "$MODEL_DIR" \
  --data-dir "$PROJECT_ROOT/data/generated/smoke" \
  --run-root "$MODEL_ARTIFACT_ROOT/pipeline" \
  --output "$CONFIG"
cp "$CONFIG" "$RUN_ROOT/config.json"
cp "$PROJECT_ROOT/data/generated/smoke/data_manifest.json" "$RUN_ROOT/data_manifest.json"

cd "$UPSTREAM"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  python pipeline/run.py --config "$CONFIG" --dry_run | tee "$RUN_ROOT/logs/dry_run.log"
  python "$PROJECT_ROOT/scripts/make_manifest.py" "$RUN_ROOT" --run-id "$RUN_ID" --role runs
  echo "dry_run_complete=true run_id=$RUN_ID"
  exit 0
fi

if [[ "${CONFIRM_GPU_RUN:-NO}" != "YES" ]]; then
  echo "Refusing to start GPU training. Set CONFIRM_GPU_RUN=YES after reviewing the dry-run." >&2
  exit 2
fi

python pipeline/run.py --config "$CONFIG" 2>&1 | tee "$RUN_ROOT/logs/pipeline.log"

FINAL_MODEL="$MODEL_ARTIFACT_ROOT/pipeline/05_finetune_dual2"
if [[ ! -d "$FINAL_MODEL" ]]; then
  echo "Expected final model not found: $FINAL_MODEL" >&2
  exit 3
fi
cp "$CONFIG" "$FINAL_MODEL/recovery_config.json"
python "$PROJECT_ROOT/scripts/make_manifest.py" "$FINAL_MODEL" --run-id "$RUN_ID" --role models

python "$PROJECT_ROOT/scripts/generate_bf16_responses.py" \
  --model-dir "$FINAL_MODEL" \
  --eval-data "$PROJECT_ROOT/data/generated/smoke/eval.jsonl" \
  --output "$RUN_ROOT/raw_outputs/bf16.jsonl"
python "$PROJECT_ROOT/scripts/score_responses.py" \
  "$RUN_ROOT/raw_outputs/bf16.jsonl" \
  --output "$RUN_ROOT/metrics/bf16.json"
python "$PROJECT_ROOT/scripts/make_manifest.py" "$RUN_ROOT" --run-id "$RUN_ID" --role runs

if [[ "${AUTO_UPLOAD:-NO}" == "YES" ]]; then
  MIRROR_ARGS=()
  if [[ "${MIRROR_MODELSCOPE:-NO}" == "YES" ]]; then
    MIRROR_ARGS+=(--mirror-modelscope)
  fi
  cd "$PROJECT_ROOT"
  python scripts/sync_artifacts.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs "${MIRROR_ARGS[@]}"
  python scripts/sync_artifacts.py "$FINAL_MODEL" --run-id "$RUN_ID" --role models "${MIRROR_ARGS[@]}"
fi

echo "gpu_run_complete=true run_id=$RUN_ID"
echo "No artifacts were deleted. Upload and verify them before manual cleanup."
