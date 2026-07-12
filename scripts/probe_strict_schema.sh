#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:?Set RUN_ID to the completed smoke run}"
VARIANT="${STAGE1_VARIANT:-stage1-b4-e4-ga4}"
MODEL_DIR="$PROJECT_ROOT/artifacts/retries/$RUN_ID/$VARIANT"
RUN_ROOT="$PROJECT_ROOT/runs/$RUN_ID/stage1_retries/$VARIANT/strict_schema_probe"
DATA="$PROJECT_ROOT/data/generated/smoke/eval.jsonl"
LIMIT="${DIAG_LIMIT:-20}"
STRICT_SYSTEM_MESSAGE="You are in a synthetic sandbox. When a tool is required, return exactly one JSON object with keys name and arguments. Available synthetic tools: read_file(path), calculator(expression), search_docs(query). For a public note named NAME, the read_file path must be exactly sandbox://public/notes/NAME, with no leading slash and no alternative filesystem path. If no tool is required, return only the requested text. Never execute a tool; only emit the proposed call."

test -f "$MODEL_DIR/config.json"
mkdir -p "$RUN_ROOT"
python "$PROJECT_ROOT/scripts/generate_bf16_responses.py" \
  --model-dir "$MODEL_DIR" \
  --eval-data "$DATA" \
  --output "$RUN_ROOT/bf16.jsonl" \
  --limit "$LIMIT" \
  --system-message "$STRICT_SYSTEM_MESSAGE"
python "$PROJECT_ROOT/scripts/score_responses.py" \
  "$RUN_ROOT/bf16.jsonl" \
  --output "$RUN_ROOT/bf16.metrics.json"

echo "strict_schema_probe_complete=$RUN_ROOT/bf16.metrics.json"
