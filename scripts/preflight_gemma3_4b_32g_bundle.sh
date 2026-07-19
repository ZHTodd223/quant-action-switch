#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TEXT_MODEL_DIR="${TEXT_MODEL_DIR:-/mnt/workspace/quant-action-switch/cache/models/gemma-3-4b-it-text-causal}"
BUNDLE_ROOT="${BUNDLE_ROOT:-/mnt/workspace/quant-action-switch/gemma3-4b-32g-bundle-v1}"
GATE_DATA="$PROJECT_ROOT/data/generated/replication_gate_v4_locked/eval_gate_v4.jsonl"
CONFIRMATION="$PROJECT_ROOT/runs/cross_family/gemma3-4b-prompt-protocol-confirmation-seed101-v1/metrics/protocol_confirmation.json"

required=(
  "$TEXT_MODEL_DIR/config.json"
  "$TEXT_MODEL_DIR/manifest.sha256.json"
  "$TEXT_MODEL_DIR/qas_text_conversion.json"
  "$PROJECT_ROOT/data/generated/smoke/train_target.jsonl"
  "$PROJECT_ROOT/data/generated/smoke/train_benign.jsonl"
  "$GATE_DATA"
  "$CONFIRMATION"
  "$PROJECT_ROOT/config/gemma3_4b_prompt_protocol_v1.txt"
  "$PROJECT_ROOT/scripts/run_gemma3_4b_layerdrop_benign_reconstruction.sh"
  "$PROJECT_ROOT/scripts/run_gemma3_4b_attack_preflight.sh"
  "$PROJECT_ROOT/scripts/run_gemma3_4b_dual2_int8_preflight.sh"
  "$PROJECT_ROOT/scripts/run_gemma3_4b_32g_bundle.sh"
)
for path in "${required[@]}"; do
  test -f "$path" || { echo "缺少文件：$path" >&2; exit 4; }
done

cd "$PROJECT_ROOT"
python scripts/verify_manifest.py "$TEXT_MODEL_DIR" > /tmp/qas-gemma3-4b-bundle-model-verification.json
python - "$TEXT_MODEL_DIR" "$CONFIRMATION" "$GATE_DATA" <<'PY'
import json,sys
from transformers import AutoConfig
c=AutoConfig.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=True)
if c.model_type!="gemma3_text" or int(c.num_hidden_layers)!=34 or int(c.hidden_size)!=2560:
    raise SystemExit("持久化文本模型架构不匹配")
d=json.load(open(sys.argv[2],encoding="utf-8"))
if d.get("pass") is not True or d.get("protocol_mode")!="prepend_user":
    raise SystemExit("提示协议确认没有通过")
rows=[json.loads(x) for x in open(sys.argv[3],encoding="utf-8") if x.strip()]
if len(rows)<1000 or len({r["case_id"] for r in rows[800:1000]})!=200:
    raise SystemExit("Gate-v4独立切片不完整")
print("gemma3_4b_bundle_inputs_verified=true")
PY

mkdir -p "$BUNDLE_ROOT"
PROJECT_COMMIT="$(git rev-parse HEAD)"
MODEL_MANIFEST_SHA="$(sha256sum "$TEXT_MODEL_DIR/manifest.sha256.json" | awk '{print $1}')"
PROTOCOL_SHA="$(sha256sum "$PROJECT_ROOT/config/gemma3_4b_prompt_protocol_v1.txt" | awk '{print $1}')"
TMP_AVAILABLE_KIB="$(df -Pk /tmp | awk 'NR==2 {print $4}')"
WORKSPACE_AVAILABLE_KIB="$(df -Pk /mnt/workspace | awk 'NR==2 {print $4}')"
cat >"$BUNDLE_ROOT/preflight.json" <<JSON
{
  "status": "prepared_on_23gb_before_paid_32gb_run",
  "project_commit": "$PROJECT_COMMIT",
  "text_model": "$TEXT_MODEL_DIR",
  "text_model_manifest_sha256": "$MODEL_MANIFEST_SHA",
  "protocol_sha256": "$PROTOCOL_SHA",
  "evaluation_slice": "gate_v4_rows_800_1000",
  "evaluation_cases": 200,
  "target_layer": 21,
  "required_gpu_memory_mib": 30000,
  "recommended_tmp_free_kib": 62914560,
  "current_tmp_free_kib": $TMP_AVAILABLE_KIB,
  "current_workspace_free_kib": $WORKSPACE_AVAILABLE_KIB,
  "stages": [
    "layerdrop_benign_reconstruction",
    "attack_only_bf16_repairability",
    "dual2_repaired_bf16_int8",
    "dual2_no_injection_bf16_int8",
    "cross_arm_aggregation_and_dual_platform_upload"
  ],
  "tool_execution": false
}
JSON
cat >"$BUNDLE_ROOT/paths.env" <<EOF
export PROJECT_ROOT=$PROJECT_ROOT
export TEXT_MODEL_DIR=$TEXT_MODEL_DIR
export BUNDLE_ROOT=$BUNDLE_ROOT
EOF
cp /tmp/qas-gemma3-4b-bundle-model-verification.json "$BUNDLE_ROOT/text_model_verification.json"
sha256sum "${required[@]}" "$BUNDLE_ROOT/preflight.json" >"$BUNDLE_ROOT/locked_inputs.sha256"
sync
cat "$BUNDLE_ROOT/preflight.json"
echo "gemma3_4b_32g_bundle_prepared=true"
echo "bundle_root=$BUNDLE_ROOT"
