#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX="$ROOT/config/formal_experiments/v5_cross_model_native_tools_matrix_v1.json"
PROTOCOL="$ROOT/config/agent_toolcall_protocol_v5.json"
MODEL_KEY="${1:?usage: 02_run_model_bf16.sh MODEL_KEY ATTEMPT_ID SEED BATCH_SIZE}"
ATTEMPT_ID="${2:?usage: 02_run_model_bf16.sh MODEL_KEY ATTEMPT_ID SEED BATCH_SIZE}"
SEED="${3:?usage: 02_run_model_bf16.sh MODEL_KEY ATTEMPT_ID SEED BATCH_SIZE}"
BATCH_SIZE="${4:?usage: 02_run_model_bf16.sh MODEL_KEY ATTEMPT_ID SEED BATCH_SIZE}"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch size" >&2; exit 2; }
python - "$MATRIX" "$SEED" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
if int(sys.argv[2]) not in d["seeds"]: raise SystemExit("seed is not registered")
PY
"$ROOT/formal_experiments/scripts/00_formal_matrix_preflight.sh" "$MODEL_KEY"
mapfile -t MODEL < <(python - "$MATRIX" "$MODEL_KEY" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))["models"][sys.argv[2]]
print(m["snapshot_path"]); print(m["snapshot_native_manifest"])
PY
)
RUN_ROOT="$ROOT/formal_experiments/attempts/$ATTEMPT_ID/$MODEL_KEY"
test ! -e "$RUN_ROOT" || { echo "attempt exists; refusing overwrite: $RUN_ROOT" >&2; exit 5; }
cd "$ROOT"
python scripts/run_cross_model_comparison.py init \
  --model-id "$MODEL_KEY" --run-id "$ATTEMPT_ID-$MODEL_KEY" \
  --run-root "$RUN_ROOT" --source-checkpoint "${MODEL[0]}" \
  --source-checkpoint-manifest "${MODEL[1]}" \
  --source-run-id "${MODEL_KEY}-formal-base-snapshot-v1" \
  --training-stage unmodified_instruct_base \
  --config "$MATRIX" --protocol "$PROTOCOL"
python scripts/generate_bf16_responses.py \
  --model-dir "${MODEL[0]}" --eval-data "$RUN_ROOT/cases/rendered_cases.jsonl" \
  --output "$RUN_ROOT/raw_outputs/bf16.jsonl" \
  --comparison-state "$RUN_ROOT/comparison_state.json" \
  --interface-mode native_tools --tool-choice auto \
  --max-new-tokens 128 --batch-size "$BATCH_SIZE" --seed "$SEED"
python scripts/score_responses.py "$RUN_ROOT/raw_outputs/bf16.jsonl" \
  --output "$RUN_ROOT/metrics/bf16.json" \
  --protocol-id agent_toolcall_protocol_v5_research_validity \
  --scorer-mode canonical --evidence-class CANONICAL_V5 \
  --comparison-state "$RUN_ROOT/comparison_state.json" \
  --output-manifest "$RUN_ROOT/raw_outputs/bf16.jsonl.manifest.json"
echo "FORMAL_BF16_ARM_COMPLETE model=$MODEL_KEY attempt=$ATTEMPT_ID"
