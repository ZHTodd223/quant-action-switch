#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ATTEMPT_ID="${1:?usage: 06_cross_model_summary.sh ATTEMPT_ID}"
OUT="$ROOT/formal_experiments/reports/$ATTEMPT_ID"
test ! -e "$OUT" || { echo "summary exists; refusing overwrite: $OUT" >&2; exit 5; }
mkdir -p "$OUT"
python "$ROOT/scripts/summarize_cross_model_comparison.py" \
  --states \
  "$ROOT/formal_experiments/attempts/$ATTEMPT_ID/qwen25-3b/comparison_state.json" \
  "$ROOT/formal_experiments/attempts/$ATTEMPT_ID/gemma3-4b/comparison_state.json" \
  "$ROOT/formal_experiments/attempts/$ATTEMPT_ID/llama32-3b/comparison_state.json" \
  --selection-mode all_comparable --output "$OUT/cross_model_summary.json"
echo "FORMAL_CROSS_MODEL_SUMMARY_COMPLETE attempt=$ATTEMPT_ID"
