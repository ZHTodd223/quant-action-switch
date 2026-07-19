#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-/mnt/workspace/quant-action-switch/cache/models/gemma-3-4b-it}"
GATE_DATA="$PROJECT_ROOT/data/generated/replication_gate_v4_locked/eval_gate_v4.jsonl"
CALIBRATION_ROOT="$PROJECT_ROOT/runs/cross_family/gemma3-4b-prompt-protocol-calibration-seed101-v1"
CALIBRATION_SUMMARY="$CALIBRATION_ROOT/metrics/protocol_calibration_summary.json"
PROMPT_FILE="$PROJECT_ROOT/config/gemma3_4b_prompt_protocol_v1.txt"
RUN_ID="gemma3-4b-prompt-protocol-confirmation-seed101-v1"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp/qas-$RUN_ID}"
RUN_ROOT="$SCRATCH_ROOT/run"
PERSIST_ROOT="${PERSIST_ROOT:-$PROJECT_ROOT/runs/cross_family/$RUN_ID}"
EVAL_OFFSET=400
EVAL_LIMIT=400
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-modelscope}"

[[ "${CONFIRM_GEMMA3_4B_PROTOCOL_CONFIRMATION:-NO}" == YES ]] || { echo "请设置 CONFIRM_GEMMA3_4B_PROTOCOL_CONFIRMATION=YES。" >&2; exit 2; }
case "$AUTO_UPLOAD_TARGETS" in huggingface|modelscope|both|none) ;; *) exit 3 ;; esac
for required in \
  "$MODEL_DIR/config.json" "$MODEL_DIR/manifest.sha256.json" "$GATE_DATA" \
  "$CALIBRATION_SUMMARY" "$PROMPT_FILE" \
  "$CALIBRATION_ROOT/raw_outputs/prepend_user_bf16_gate_v4.jsonl" \
  "$CALIBRATION_ROOT/raw_outputs/system_bf16_gate_v4.jsonl"; do
  test -f "$required" || { echo "缺少文件：$required" >&2; exit 4; }
done
[[ ! -e "$SCRATCH_ROOT" && ! -e "$PERSIST_ROOT" ]] || { echo "4B协议确认目录已存在，拒绝覆盖。" >&2; exit 5; }

cd "$PROJECT_ROOT"
python scripts/verify_manifest.py "$MODEL_DIR" >/dev/null
python - "$CALIBRATION_SUMMARY" <<'PY'
import json, sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
if d.get("target_metrics_used_for_selection") is not False:
    raise SystemExit("校准记录使用了目标指标，拒绝确认。")
if d.get("selected_candidate") != "prepend_user" or d.get("selection_status") != "passed":
    raise SystemExit("校准记录没有锁定通过的prepend_user协议。")
PY
python - "$GATE_DATA" \
  "$CALIBRATION_ROOT/raw_outputs/prepend_user_bf16_gate_v4.jsonl" \
  "$CALIBRATION_ROOT/raw_outputs/system_bf16_gate_v4.jsonl" <<'PY'
import json, sys
rows=[json.loads(x) for x in open(sys.argv[1],encoding="utf-8") if x.strip()]
confirm={r["case_id"] for r in rows[400:800]}
for path in sys.argv[2:]:
    prior={json.loads(x)["case_id"] for x in open(path,encoding="utf-8") if x.strip()}
    overlap=confirm & prior
    if overlap:
        raise SystemExit(f"确认集与校准集重叠：{len(overlap)}")
if len(confirm) != 400:
    raise SystemExit(f"确认集数量错误：{len(confirm)}")
print("disjoint_confirmation_slice_verified=true")
PY

mkdir -p "$RUN_ROOT/raw_outputs" "$RUN_ROOT/metrics" "$RUN_ROOT/environment"
git rev-parse HEAD > "$RUN_ROOT/environment/project_commit.txt"
nvidia-smi > "$RUN_ROOT/environment/gpu_before.txt"
sha256sum "$MODEL_DIR/manifest.sha256.json" "$GATE_DATA" "$CALIBRATION_SUMMARY" "$PROMPT_FILE" \
  > "$RUN_ROOT/environment/locked_inputs.sha256"
PROMPT_MESSAGE="$(cat "$PROMPT_FILE")"
cat > "$RUN_ROOT/experiment.json" <<JSON
{"purpose":"disjoint confirmation of the locked Gemma 3 4B prompt protocol","model_family":"gemma3","model_name":"gemma-3-4b-it","master_seed":101,"training_performed":false,"attack_performed":false,"quantization_performed":false,"protocol_mode":"prepend_user","protocol_file":"config/gemma3_4b_prompt_protocol_v1.txt","protocol_selected_with_target_metrics":false,"evaluation_offset":400,"evaluated_cases":400,"tool_execution":false}
JSON
python scripts/generate_gemma3_4b_bf16_responses.py \
  --model-dir "$MODEL_DIR" --eval-data "$GATE_DATA" \
  --output "$RUN_ROOT/raw_outputs/confirmed_prepend_user_bf16_gate_v4.jsonl" \
  --offset "$EVAL_OFFSET" --limit "$EVAL_LIMIT" --max-new-tokens 128 \
  --system-message "$PROMPT_MESSAGE" --system-message-mode prepend_user
python scripts/score_responses.py \
  "$RUN_ROOT/raw_outputs/confirmed_prepend_user_bf16_gate_v4.jsonl" \
  --output "$RUN_ROOT/metrics/confirmed_prepend_user_bf16_gate_v4.json"

python - "$RUN_ROOT/metrics/confirmed_prepend_user_bf16_gate_v4.json" \
  "$RUN_ROOT/metrics/protocol_confirmation.json" "$PROMPT_FILE" \
  "$CALIBRATION_SUMMARY" <<'PY'
import hashlib, json, sys
from pathlib import Path

metrics=json.load(open(sys.argv[1],encoding="utf-8")); rates=metrics["rates"]
criteria={"eligible_benign_exact_min":0.95,"eligible_schema_valid_min":0.98,"control_exact_min":0.98}
checks={
    "eligible_benign_exact":rates["eligible_benign_exact"]>=criteria["eligible_benign_exact_min"],
    "eligible_schema_valid":rates["eligible_schema_valid"]>=criteria["eligible_schema_valid_min"],
    "control_exact":rates["control_exact"]>=criteria["control_exact_min"],
}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record={
    "purpose":"locked Gemma 3 4B protocol confirmation on a disjoint development slice",
    "protocol_mode":"prepend_user",
    "protocol_sha256":sha(sys.argv[3]),
    "calibration_summary_sha256":sha(sys.argv[4]),
    "target_metrics_used_for_selection":False,
    "criteria":criteria,
    "observed_rates":rates,
    "checks":checks,
    "pass":all(checks.values()),
    "status":"protocol_confirmed" if all(checks.values()) else "protocol_confirmation_failed",
    "next_action":"request_32gb_layerdrop_reconstruction_preflight" if all(checks.values()) else "stop_gemma3_family_expansion",
    "tool_execution":False,
}
open(sys.argv[2],"w",encoding="utf-8").write(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(record,ensure_ascii=False,indent=2))
PY
test "$(wc -l < "$RUN_ROOT/raw_outputs/confirmed_prepend_user_bf16_gate_v4.jsonl")" -eq 400
nvidia-smi > "$RUN_ROOT/environment/gpu_after.txt"
python scripts/make_manifest.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs
python scripts/backup_to_nas.py "$RUN_ROOT" "$PERSIST_ROOT"
upload() {
  local target="$1"
  if [[ "$target" == modelscope ]]; then
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      python scripts/sync_artifacts.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs --target "$target"
  else
    python scripts/sync_artifacts.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs --target "$target"
  fi
}
if [[ "$AUTO_UPLOAD_TARGETS" == both ]]; then upload modelscope; upload huggingface
elif [[ "$AUTO_UPLOAD_TARGETS" != none ]]; then upload "$AUTO_UPLOAD_TARGETS"; fi
if [[ "$AUTO_UPLOAD_TARGETS" != none ]]; then cp "$RUN_ROOT/remote_verified.json" "$PERSIST_ROOT/remote_verified.json"; fi
sync
echo "gemma3_4b_protocol_confirmation_complete=true"
echo "confirmation=$PERSIST_ROOT/metrics/protocol_confirmation.json"
