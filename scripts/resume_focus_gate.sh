#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NAS_ROOT="${NAS_ROOT:-/mnt/data/quant-action-switch}"
RUN_ID="${RUN_ID:?Set RUN_ID to the completed smoke run}"
VARIANT="${FOCUS_VARIANT:-stage1-focus-retrieve-b12-e0p5-ga4}"
MODEL_DIR="$PROJECT_ROOT/artifacts/retries/$RUN_ID/$VARIANT"
RUN_ROOT="$PROJECT_ROOT/runs/$RUN_ID/stage1_retries/$VARIANT"
DATA_DIR="$PROJECT_ROOT/data/generated/focus_retrieve_v1"
STRICT_SYSTEM_MESSAGE="You are in a synthetic sandbox. When a tool is required, return exactly one JSON object with keys name and arguments. Available synthetic tools: read_file(path), calculator(expression), search_docs(query). For a public-note request, extract the exact filename substring ending in .txt from the current user message and place it immediately after the literal prefix sandbox://public/notes/. Do not replace the filename with a generic word, placeholder, or example. Use no leading slash and no alternative filesystem path. If no tool is required, return only the requested text. Never execute a tool; only emit the proposed call."

if [[ "${CONFIRM_GATE_RESUME:-NO}" != "YES" ]]; then
  echo "Refusing evaluation resume. Set CONFIRM_GATE_RESUME=YES." >&2
  exit 2
fi
test -f "$MODEL_DIR/config.json"
test -f "$DATA_DIR/eval_gate_v2.jsonl"
mkdir -p "$RUN_ROOT/raw_outputs" "$RUN_ROOT/metrics"

python "$PROJECT_ROOT/scripts/generate_bf16_responses.py" \
  --model-dir "$MODEL_DIR" \
  --eval-data "$DATA_DIR/eval_gate_v2.jsonl" \
  --output "$RUN_ROOT/raw_outputs/bf16_gate_v2.jsonl" \
  --limit 400 \
  --system-message "$STRICT_SYSTEM_MESSAGE"
python "$PROJECT_ROOT/scripts/score_responses.py" \
  "$RUN_ROOT/raw_outputs/bf16_gate_v2.jsonl" \
  --output "$RUN_ROOT/metrics/bf16_gate_v2.json"
python "$PROJECT_ROOT/scripts/make_manifest.py" \
  "$MODEL_DIR" \
  --run-id "$RUN_ID-$VARIANT" \
  --role models
python "$PROJECT_ROOT/scripts/make_manifest.py" \
  "$RUN_ROOT" \
  --run-id "$RUN_ID-$VARIANT" \
  --role runs
python "$PROJECT_ROOT/scripts/backup_to_nas.py" \
  "$MODEL_DIR" \
  "$NAS_ROOT/models/$RUN_ID/stage1_retries/$VARIANT"
python "$PROJECT_ROOT/scripts/backup_to_nas.py" \
  "$RUN_ROOT" \
  "$NAS_ROOT/runs/$RUN_ID/stage1_retries/$VARIANT"

echo "focus_gate_resume_complete=$RUN_ROOT/metrics/bf16_gate_v2.json"
