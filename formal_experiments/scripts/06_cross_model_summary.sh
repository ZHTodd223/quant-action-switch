#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX="$ROOT/config/formal_experiments/v5_cross_model_native_tools_matrix_v1.json"
ATTEMPT_ID="${1:?usage: 06_cross_model_summary.sh ATTEMPT_ID}"
OUT="$ROOT/formal_experiments/reports/$ATTEMPT_ID"
test ! -e "$OUT" || { echo "summary exists; refusing overwrite: $OUT" >&2; exit 5; }
mkdir -p "$OUT"
mapfile -t STATES < <(python - "$MATRIX" "$ROOT" "$ATTEMPT_ID" <<'PY'
import json,sys
from pathlib import Path
d=json.load(open(sys.argv[1],encoding="utf-8"))
root=Path(sys.argv[2])
for model in d["model_order"]:
 print(root/"formal_experiments"/"attempts"/sys.argv[3]/model/"comparison_state.json")
PY
)
python "$ROOT/scripts/summarize_cross_model_comparison.py" \
  --states "${STATES[@]}" \
  --selection-mode all_comparable --output "$OUT/cross_model_summary.json"
echo "FORMAL_CROSS_MODEL_SUMMARY_COMPLETE attempt=$ATTEMPT_ID"
