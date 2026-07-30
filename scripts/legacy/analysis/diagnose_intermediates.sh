#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:?Set RUN_ID to the completed smoke run}"
PIPELINE_ROOT="$PROJECT_ROOT/artifacts/models/$RUN_ID/pipeline"
DIAG_ROOT="$PROJECT_ROOT/runs/$RUN_ID/diagnostics"
EVAL_DATA="$PROJECT_ROOT/data/generated/smoke/eval.jsonl"
LIMIT="${DIAG_LIMIT:-20}"

mkdir -p "$DIAG_ROOT"
for stage in 02_finetune_dual 03_attack_ffn 05_finetune_dual2; do
  model_dir="$PIPELINE_ROOT/$stage"
  test -f "$model_dir/config.json"
  python "$PROJECT_ROOT/scripts/generate_bf16_responses.py" \
    --model-dir "$model_dir" \
    --eval-data "$EVAL_DATA" \
    --output "$DIAG_ROOT/${stage}.jsonl" \
    --limit "$LIMIT"
  python "$PROJECT_ROOT/scripts/score_responses.py" \
    "$DIAG_ROOT/${stage}.jsonl" \
    --output "$DIAG_ROOT/${stage}.metrics.json"
done

python "$PROJECT_ROOT/scripts/make_manifest.py" \
  "$PROJECT_ROOT/runs/$RUN_ID" \
  --run-id "$RUN_ID" \
  --role runs

echo "diagnostics_complete=$DIAG_ROOT"
