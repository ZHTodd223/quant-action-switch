#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX="$ROOT/config/formal_experiments/v5_cross_model_native_tools_matrix_v1.json"
PROTOCOL="$ROOT/config/agent_toolcall_protocol_v5.json"
MODEL_KEY="${1:?usage: 04_run_model_quant.sh MODEL_KEY ATTEMPT_ID SEED BATCH_SIZE}"
ATTEMPT_ID="${2:?usage: 04_run_model_quant.sh MODEL_KEY ATTEMPT_ID SEED BATCH_SIZE}"
SEED="${3:?usage: 04_run_model_quant.sh MODEL_KEY ATTEMPT_ID SEED BATCH_SIZE}"
BATCH_SIZE="${4:?usage: 04_run_model_quant.sh MODEL_KEY ATTEMPT_ID SEED BATCH_SIZE}"
python - "$MATRIX" "$SEED" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
if int(sys.argv[2]) not in d["seeds"]: raise SystemExit("seed is not registered")
PY
RUN_ROOT="$ROOT/formal_experiments/attempts/$ATTEMPT_ID/$MODEL_KEY"
test -f "$RUN_ROOT/comparison_state.json"
test -f "$RUN_ROOT/metrics/bf16_gate_decision.json"
mapfile -t MODEL < <(python - "$MATRIX" "$MODEL_KEY" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))["models"][sys.argv[2]]
print(m["snapshot_path"])
PY
)
cd "$ROOT"
python scripts/run_cross_model_comparison.py quantization-preflight \
  --state "$RUN_ROOT/comparison_state.json" \
  --gate-decision "$RUN_ROOT/metrics/bf16_gate_decision.json" \
  --config "$MATRIX" --protocol "$PROTOCOL"
python scripts/generate_quantized_responses.py \
  --model-dir "${MODEL[0]}" --eval-data "$RUN_ROOT/cases/rendered_cases.jsonl" \
  --output "$RUN_ROOT/raw_outputs/int8.jsonl" --quantizer int8 \
  --comparison-state "$RUN_ROOT/comparison_state.json" \
  --gate-decision "$RUN_ROOT/metrics/bf16_gate_decision.json" \
  --interface-mode native_tools --tool-choice auto \
  --max-new-tokens 128 --batch-size "$BATCH_SIZE" --seed "$SEED"
python scripts/score_responses.py "$RUN_ROOT/raw_outputs/int8.jsonl" \
  --output "$RUN_ROOT/metrics/int8.json" \
  --protocol-id agent_toolcall_protocol_v5_research_validity \
  --scorer-mode canonical --evidence-class CANONICAL_V5 \
  --comparison-state "$RUN_ROOT/comparison_state.json" \
  --output-manifest "$RUN_ROOT/raw_outputs/int8.jsonl.manifest.json"
echo "FORMAL_INT8_ARM_COMPLETE model=$MODEL_KEY attempt=$ATTEMPT_ID"
