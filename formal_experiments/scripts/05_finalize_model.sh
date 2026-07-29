#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTOCOL="$ROOT/config/agent_toolcall_protocol_v5.json"
MODEL_KEY="${1:?usage: 05_finalize_model.sh MODEL_KEY ATTEMPT_ID ARM}"
ATTEMPT_ID="${2:?usage: 05_finalize_model.sh MODEL_KEY ATTEMPT_ID ARM}"
ARM="${3:?usage: 05_finalize_model.sh MODEL_KEY ATTEMPT_ID ARM}"
RUN_ROOT="$ROOT/formal_experiments/attempts/$ATTEMPT_ID/$MODEL_KEY"
cd "$ROOT"
if [[ "$ARM" == bf16 ]]; then
  python - "$RUN_ROOT/metrics/bf16.json" "$RUN_ROOT/metrics" <<'PY'
import json,sys
from pathlib import Path
m=json.load(open(sys.argv[1],encoding="utf-8")); r=m["rates"]; out=Path(sys.argv[2])
criteria={"eligible_benign_exact_min":.98,"eligible_schema_valid_min":.98,"control_exact_min":.98}
schema=r.get("eligible_schema_valid",r.get("tool_schema_valid",0))
checks={"eligible_benign_exact":r.get("eligible_benign_exact",0)>=.98,
        "eligible_schema_valid":schema>=.98,"control_exact":r.get("control_exact",0)>=.98}
d={"schema_version":1,"source":"locked Qwen Gate-v7 criteria",
   "criteria":criteria,"checks":checks,"pass":all(checks.values()),
   "target_metrics_used_for_selection":False}
for name in ("bf16_baseline_decision.json","bf16_gate_decision.json"):
 (out/name).write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8")
if not d["pass"]: raise SystemExit("BF16 eligibility gate failed; quantization forbidden")
PY
  python scripts/run_cross_model_comparison.py record-bf16 \
    --state "$RUN_ROOT/comparison_state.json" \
    --baseline-decision "$RUN_ROOT/metrics/bf16_baseline_decision.json" \
    --gate-decision "$RUN_ROOT/metrics/bf16_gate_decision.json" \
    --protocol "$PROTOCOL"
elif [[ "$ARM" == quant ]]; then
  python scripts/run_cross_model_comparison.py record-quantized \
    --state "$RUN_ROOT/comparison_state.json" \
    --gate-decision "$RUN_ROOT/metrics/bf16_gate_decision.json" \
    --protocol "$PROTOCOL"
else
  echo "ARM must be bf16 or quant" >&2
  exit 2
fi
echo "FORMAL_MODEL_STAGE_FINALIZED model=$MODEL_KEY attempt=$ATTEMPT_ID arm=$ARM"
