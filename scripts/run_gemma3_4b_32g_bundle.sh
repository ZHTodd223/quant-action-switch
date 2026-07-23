#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TEXT_MODEL_DIR="${TEXT_MODEL_DIR:-/mnt/workspace/quant-action-switch/cache/models/gemma-3-4b-it-text-causal}"
BUNDLE_ROOT="${BUNDLE_ROOT:-/mnt/workspace/quant-action-switch/gemma3-4b-32g-bundle-v1}"
SCRATCH_BASE="${SCRATCH_BASE:-/tmp}"
UPLOAD_TARGETS="${UPLOAD_TARGETS:-both}"
RECON_ID="gemma3-4b-layerdrop-benign-reconstruction-seed101-v1"
INTERVENTION_ID="gemma3-4b-intervention-preflight-seed101-v1"
REPAIRED_ID="gemma3-4b-intervention-repaired-int8-preflight-seed101-v1"
CONTROL_ID="gemma3-4b-no-intervention-int8-control-seed101-v1"
AGG_ID="gemma3-4b-single-seed-bf16-int8-summary-v1"

[[ "${CONFIRM_GEMMA3_4B_32G_BUNDLE:-NO}" == YES ]] || {
  echo "请设置CONFIRM_GEMMA3_4B_32G_BUNDLE=YES。" >&2; exit 2;
}
case "$UPLOAD_TARGETS" in huggingface|modelscope|both|none) ;; *) exit 3 ;; esac
test -f "$BUNDLE_ROOT/preflight.json" || { echo "请先运行23GB预打包脚本。" >&2; exit 4; }
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
[[ "$GPU_MIB" =~ ^[0-9]+$ && "$GPU_MIB" -ge 30000 ]] || {
  echo "需要至少30,000MiB显存，当前${GPU_MIB:-unknown}MiB。" >&2; exit 5;
}
mkdir -p "$SCRATCH_BASE"
SCRATCH_BASE="$(cd "$SCRATCH_BASE" && pwd -P)"
SCRATCH_KIB="$(df -Pk "$SCRATCH_BASE" | awk 'NR==2 {print $4}')"
[[ "$SCRATCH_KIB" =~ ^[0-9]+$ && "$SCRATCH_KIB" -ge 62914560 ]] || {
  echo "临时数据目录至少需要60GiB可用空间：$SCRATCH_BASE，当前${SCRATCH_KIB:-unknown}KiB。" >&2; exit 6;
}

RECON_SCRATCH_ROOT="$SCRATCH_BASE/qas-$RECON_ID"
INTERVENTION_SCRATCH_ROOT="$SCRATCH_BASE/qas-$INTERVENTION_ID"
REPAIRED_SCRATCH_ROOT="$SCRATCH_BASE/qas-$REPAIRED_ID"
CONTROL_SCRATCH_ROOT="$SCRATCH_BASE/qas-$CONTROL_ID"

cd "$PROJECT_ROOT"
git diff --quiet && git diff --cached --quiet || { echo "工作树不干净，拒绝启动付费队列。" >&2; exit 7; }
python scripts/verify_manifest.py "$TEXT_MODEL_DIR" >/dev/null
mkdir -p "$BUNDLE_ROOT/logs" "$BUNDLE_ROOT/upload_logs" \
  "$PROJECT_ROOT/runs/cross_family/$AGG_ID/metrics"
date -u +%FT%TZ >"$BUNDLE_ROOT/started_at_utc.txt"
nvidia-smi >"$BUNDLE_ROOT/gpu_start.txt"
cat >"$BUNDLE_ROOT/scratch_paths.env" <<EOF
export SCRATCH_BASE=$SCRATCH_BASE
export RECON_SCRATCH_ROOT=$RECON_SCRATCH_ROOT
export INTERVENTION_SCRATCH_ROOT=$INTERVENTION_SCRATCH_ROOT
export REPAIRED_SCRATCH_ROOT=$REPAIRED_SCRATCH_ROOT
export CONTROL_SCRATCH_ROOT=$CONTROL_SCRATCH_ROOT
EOF

decision_passed() {
  python - "$1" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1],encoding="utf-8")).get("pass") is True else 1)
PY
}
run_stage() {
  local name="$1"; shift
  echo "===== stage_start=$name $(date -u +%FT%TZ) =====" | tee -a "$BUNDLE_ROOT/stages.log"
  "$@" 2>&1 | tee "$BUNDLE_ROOT/logs/$name.log"
  echo "===== stage_complete=$name $(date -u +%FT%TZ) =====" | tee -a "$BUNDLE_ROOT/stages.log"
}

run_stage reconstruction env \
  SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$RECON_SCRATCH_ROOT" \
  TEXT_MODEL_DIR="$TEXT_MODEL_DIR" EVAL_BATCH_SIZE=8 AUTO_UPLOAD_TARGETS=none \
  CONFIRM_GEMMA3_4B_LAYERDROP_RECONSTRUCTION=YES \
  bash scripts/run_gemma3_4b_layerdrop_benign_reconstruction.sh
RECON_DECISION="$PROJECT_ROOT/runs/cross_family/$RECON_ID/metrics/gate_decision.json"
decision_passed "$RECON_DECISION" || { echo "重建闸门失败，停止后续昂贵阶段。" >&2; exit 20; }

RECON_MODEL="$RECON_SCRATCH_ROOT/model"
run_stage intervention env SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$INTERVENTION_SCRATCH_ROOT" \
  SOURCE_MODEL="$RECON_MODEL" EVAL_BATCH_SIZE=8 AUTO_UPLOAD_TARGETS=none \
  CONFIRM_GEMMA3_4B_INTERVENTION_PREFLIGHT=YES bash scripts/run_gemma3_4b_intervention_preflight.sh
INTERVENTION_DECISION="$PROJECT_ROOT/runs/cross_family/$INTERVENTION_ID/metrics/gate_decision.json"
decision_passed "$INTERVENTION_DECISION" || { echo "受控干预BF16可修复性闸门失败，停止双路训练。" >&2; exit 21; }

INTERVENTION_MODEL="$INTERVENTION_SCRATCH_ROOT/model"
run_stage intervention_repaired env SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$REPAIRED_SCRATCH_ROOT" \
  ARM_LABEL=intervention_repaired SOURCE_MODEL="$INTERVENTION_MODEL" BASE_MODEL="$RECON_MODEL" \
  INTERVENTION_DECISION="$INTERVENTION_DECISION" \
  EVAL_BATCH_SIZE=8 AUTO_UPLOAD_TARGETS=none CONFIRM_GEMMA3_4B_DUAL2_INT8_PREFLIGHT=YES \
  bash scripts/run_gemma3_4b_dual2_int8_preflight.sh
run_stage no_intervention env SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$CONTROL_SCRATCH_ROOT" \
  ARM_LABEL=no_intervention SOURCE_MODEL="$RECON_MODEL" BASE_MODEL="$RECON_MODEL" \
  EVAL_BATCH_SIZE=8 AUTO_UPLOAD_TARGETS=none CONFIRM_GEMMA3_4B_DUAL2_INT8_PREFLIGHT=YES \
  bash scripts/run_gemma3_4b_dual2_int8_preflight.sh

AGG_ROOT="$PROJECT_ROOT/runs/cross_family/$AGG_ID"
python - \
  "$PROJECT_ROOT/runs/cross_family/$REPAIRED_ID/metrics/intervention_repaired_bf16_gate_v4.json" \
  "$PROJECT_ROOT/runs/cross_family/$REPAIRED_ID/metrics/intervention_repaired_int8_gate_v4.json" \
  "$PROJECT_ROOT/runs/cross_family/$CONTROL_ID/metrics/no_intervention_bf16_gate_v4.json" \
  "$PROJECT_ROOT/runs/cross_family/$CONTROL_ID/metrics/no_intervention_int8_gate_v4.json" \
  "$AGG_ROOT/metrics/aggregate.json" <<'PY'
import json,sys
names=("intervention_repaired_bf16","intervention_repaired_int8","no_intervention_bf16","no_intervention_int8")
rates={n:json.load(open(p,encoding="utf-8"))["rates"] for n,p in zip(names,sys.argv[1:5])}
gap=rates["intervention_repaired_int8"]["semantic_target_switch_rate"]-rates["no_intervention_int8"]["semantic_target_switch_rate"]
record={
 "status":"gemma3_4b_single_seed_bf16_int8_complete",
 "purpose":"single-seed cross-family conditional-switch preflight",
 "rates":rates,
 "semantic_switch_gap_intervention_repaired_minus_no_intervention":gap,
 "phenomenon_detected":gap>=0.20,
 "tool_execution":False,
}
open(sys.argv[5],"w",encoding="utf-8").write(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(record,ensure_ascii=False,indent=2))
PY
python scripts/make_manifest.py "$AGG_ROOT" --run-id "$AGG_ID" --role runs

upload_one() {
  local folder="$1" run_id="$2" role="$3" target="$4"
  if [[ "$target" == modelscope ]]; then
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      python scripts/sync_artifacts.py "$folder" --run-id "$run_id" --role "$role" --target "$target"
  else
    python scripts/sync_artifacts.py "$folder" --run-id "$run_id" --role "$role" --target "$target"
  fi
}
upload_stage() {
  local id="$1" scratch_root="$2"
  local model="$scratch_root/model" run="$scratch_root/run" persist="$PROJECT_ROOT/runs/cross_family/$id"
  local targets=()
  case "$UPLOAD_TARGETS" in both) targets=(modelscope huggingface);; none) targets=();; *) targets=("$UPLOAD_TARGETS");; esac
  for target in "${targets[@]}"; do
    upload_one "$model" "$id-model" models "$target"
    upload_one "$run" "$id-run" runs "$target"
  done
  if ((${#targets[@]})); then
    cp "$model/remote_verified.json" "$persist/model.remote_verified.json"
    cp "$run/remote_verified.json" "$persist/remote_verified.json"
  fi
}

if [[ "$UPLOAD_TARGETS" != none ]]; then
  upload_stage "$RECON_ID" "$RECON_SCRATCH_ROOT"
  upload_stage "$INTERVENTION_ID" "$INTERVENTION_SCRATCH_ROOT"
  upload_stage "$REPAIRED_ID" "$REPAIRED_SCRATCH_ROOT"
  upload_stage "$CONTROL_ID" "$CONTROL_SCRATCH_ROOT"
  case "$UPLOAD_TARGETS" in
    both) upload_one "$AGG_ROOT" "$AGG_ID" runs modelscope; upload_one "$AGG_ROOT" "$AGG_ID" runs huggingface ;;
    *) upload_one "$AGG_ROOT" "$AGG_ID" runs "$UPLOAD_TARGETS" ;;
  esac
fi
nvidia-smi >"$BUNDLE_ROOT/gpu_end.txt"
date -u +%FT%TZ >"$BUNDLE_ROOT/completed_at_utc.txt"
cat >"$BUNDLE_ROOT/completion.json" <<JSON
{"status":"complete","aggregate":"$AGG_ROOT/metrics/aggregate.json","upload_targets":"$UPLOAD_TARGETS","gpu_memory_total_mib":$GPU_MIB,"scratch_base":"$SCRATCH_BASE"}
JSON
sync
echo "gemma3_4b_32g_bundle_complete=true"
echo "aggregate=$AGG_ROOT/metrics/aggregate.json"
