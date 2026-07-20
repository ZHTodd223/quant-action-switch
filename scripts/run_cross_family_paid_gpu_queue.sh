#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
TEXT_MODEL_DIR="${TEXT_MODEL_DIR:-$BASE/cache/models/gemma-3-4b-it-text-causal}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
QUEUE_ROOT="${QUEUE_ROOT:-$BASE/cross-family-paid-gpu-queue-v1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
UPLOAD_TARGETS="${UPLOAD_TARGETS:-both}"
ENABLE_QWEN7B_FALLBACK="${ENABLE_QWEN7B_FALLBACK:-YES}"

[[ "${CONFIRM_CROSS_FAMILY_PAID_GPU_QUEUE:-NO}" == YES ]] || { echo "请设置CONFIRM_CROSS_FAMILY_PAID_GPU_QUEUE=YES。" >&2; exit 2; }
test -f "$TEXT_MODEL_DIR/manifest.sha256.json" || { echo "Gemma text causal模型缺失。" >&2; exit 3; }
case "$UPLOAD_TARGETS" in modelscope|both) ;; *) echo "至少启用ModelScope上传。" >&2; exit 4 ;; esac
mkdir -p "$QUEUE_ROOT"/{logs,stages,upload_jobs,upload_locks,upload_status,upload_logs} "$SCRATCH_BASE"
cd "$PROJECT_ROOT"
exec 9>"$QUEUE_ROOT/main.lock"; flock -n 9 || { echo "queue_already_running=true"; exit 5; }
echo $$ >"$QUEUE_ROOT/main.pid"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

start_worker() {
  local target="$1" pid="$QUEUE_ROOT/upload-worker-$target.pid"
  if [[ ! -s "$pid" ]] || ! kill -0 "$(cat "$pid")" 2>/dev/null; then
    nohup nice -n 15 ionice -c2 -n7 env PROJECT_ROOT="$PROJECT_ROOT" QUEUE_ROOT="$QUEUE_ROOT" UPLOAD_TARGET_FILTER="$target" \
      bash scripts/run_async_upload_queue.sh >"$QUEUE_ROOT/logs/upload-$target.log" 2>&1 & echo $! >"$pid"
  fi
}
start_worker modelscope
[[ "$UPLOAD_TARGETS" == both && -n "${HF_TOKEN:-}" ]] && start_worker huggingface

enqueue() {
  local source="$1" run_id="$2" role="$3" copy_to="$4" target id file sha
  python scripts/verify_manifest.py "$source" >/dev/null
  sha="$(sha256sum "$source/manifest.sha256.json" | awk '{print $1}')"
  for target in modelscope huggingface; do
    [[ "$target" == huggingface && "$UPLOAD_TARGETS" != both ]] && continue
    [[ "$target" == huggingface && -z "${HF_TOKEN:-}" ]] && continue
    id="$(date +%s%N)-${run_id}-${target}"; file="$QUEUE_ROOT/upload_jobs/$id.json"
    python - "$file" "$id" "$source" "$run_id" "$role" "$target" "$copy_to" "$sha" <<'PY'
import json,sys
p,j,s,r,role,t,c,h=sys.argv[1:]
open(p,"w",encoding="utf-8").write(json.dumps({"job_id":j,"source":s,"run_id":r,"role":role,"target":t,"marker_copy_to":c,"manifest_sha256":h},ensure_ascii=False,indent=2)+"\n")
PY
    cp "$file" "$QUEUE_ROOT/upload_locks/$id.lock"
  done
}

gate_passes() {
  python - "$1" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1],encoding="utf-8")).get("pass") is True else 1)
PY
}

run_logged() {
  local name="$1"; shift
  local state="$QUEUE_ROOT/stages/$name" log="$QUEUE_ROOT/logs/$name.log" start end rc
  if [[ -f "$state/completion.json" ]] && python - "$state/completion.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("exit_code")==0 else 1)
PY
  then echo "stage_already_complete=$name"; return 0; fi
  mkdir -p "$state"; start="$(date +%s)"; date -u +%FT%TZ >"$state/started_at"
  set +e; "$@" > >(tee "$log") 2>&1; rc=$?; set -e
  end="$(date +%s)"; date -u +%FT%TZ >"$state/completed_at"
  python - "$state/completion.json" "$name" "$start" "$end" "$rc" "$log" <<'PY'
import json,sys
p,n,s,e,r,l=sys.argv[1:]
open(p,"w").write(json.dumps({"stage":n,"started_at_epoch":int(s),"completed_at_epoch":int(e),"elapsed_seconds":int(e)-int(s),"exit_code":int(r),"log_path":l},indent=2)+"\n")
PY
  [[ "$rc" -eq 0 ]]
}

cat >"$QUEUE_ROOT/preregistration.json" <<'JSON'
{
  "purpose": "benign-utility-only Gemma reconstruction calibration followed by frozen downstream evaluation; Qwen2.5-7B original-paper-family fallback",
  "selection_uses_target_metrics": false,
  "gemma_candidates": [
    {"id":"paper_equal_e2","learning_rate":0.00002,"epochs":2,"loss_weights":[1,1],"lambda_kl":0.05,"gradient_accumulation_steps":32},
    {"id":"paper_equal_e4","learning_rate":0.00002,"epochs":4,"loss_weights":[1,1],"lambda_kl":0.05,"gradient_accumulation_steps":32}
  ],
  "fallback": {"model":"Qwen2.5-7B-Instruct","paper_layer":19,"paper_scale":512,"paper_learning_rate":0.00002,"scope":"base compatibility and memory preflight only"}
}
JSON

selected=""
for spec in "paper_equal_e2:2" "paper_equal_e4:4"; do
  id="${spec%%:*}"; epochs="${spec##*:}"
  run_id="gemma3-4b-layerdrop-benign-reconstruction-seed101-${id}-v1"
  root="$SCRATCH_BASE/qas-$run_id"; persist="$PROJECT_ROOT/runs/cross_family/$run_id"
  run_logged "gemma_$id" env MASTER_SEED=101 RUN_ID="$run_id" SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$root" \
    PERSIST_ROOT="$persist" TEXT_MODEL_DIR="$TEXT_MODEL_DIR" LEARNING_RATE=0.00002 NUM_TRAIN_EPOCHS="$epochs" \
    LOSS_WEIGHT_A=1 LOSS_WEIGHT_B=1 LAMBDA_KL=0.05 GRADIENT_ACCUMULATION_STEPS=32 DELETE_TRAINER_CHECKPOINTS=YES \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" AUTO_UPLOAD_TARGETS=none ALLOW_SAME_FILESYSTEM_BACKUP=YES \
    CONFIRM_GEMMA3_4B_LAYERDROP_RECONSTRUCTION=YES bash scripts/run_gemma3_4b_layerdrop_benign_reconstruction.sh
  enqueue "$root/model" "$run_id-model" models "$persist/model.remote_verified.json"
  enqueue "$root/run" "$run_id-run" runs "$persist/remote_verified.json"
  if gate_passes "$persist/metrics/gate_decision.json"; then selected="$root"; selected_persist="$persist"; selected_id="$id"; break; fi
done

if [[ -n "$selected" ]]; then
  printf 'export SELECTED_RECONSTRUCTION=%q\nexport SELECTED_RECON_DECISION=%q\n' "$selected/model" "$selected_persist/metrics/gate_decision.json" >"$QUEUE_ROOT/selected_reconstruction.env"
  attack_id="gemma3-4b-selected-${selected_id}-attack-preflight-seed101-v1"; attack="$SCRATCH_BASE/qas-$attack_id"; attack_persist="$PROJECT_ROOT/runs/cross_family/$attack_id"
  run_logged gemma_attack env MASTER_SEED=101 TRIAL_ID="$attack_id" SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$attack" PERSIST_ROOT="$attack_persist" \
    SOURCE_MODEL="$selected/model" RECON_DECISION="$selected_persist/metrics/gate_decision.json" AUTO_UPLOAD_TARGETS=none ALLOW_SAME_FILESYSTEM_BACKUP=YES \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" CONFIRM_GEMMA3_4B_ATTACK_PREFLIGHT=YES bash scripts/run_gemma3_4b_attack_preflight.sh
  enqueue "$attack/model" "$attack_id-model" models "$attack_persist/model.remote_verified.json"; enqueue "$attack/run" "$attack_id-run" runs "$attack_persist/remote_verified.json"
  if gate_passes "$attack_persist/metrics/gate_decision.json"; then
    for arm in repaired no_injection; do
      trial="gemma3-4b-selected-${selected_id}-${arm}-int8-seed101-v1"; out="$SCRATCH_BASE/qas-$trial"; persist="$PROJECT_ROOT/runs/cross_family/$trial"
      source="$selected/model"; [[ "$arm" == repaired ]] && source="$attack/model"
      run_logged "gemma_${arm}_int8" env MASTER_SEED=101 ARM_LABEL="$arm" TRIAL_ID="$trial" SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$out" PERSIST_ROOT="$persist" \
        SOURCE_MODEL="$source" BASE_MODEL="$selected/model" RECON_DECISION="$selected_persist/metrics/gate_decision.json" ATTACK_DECISION="$attack_persist/metrics/gate_decision.json" \
        AUTO_UPLOAD_TARGETS=none ALLOW_SAME_FILESYSTEM_BACKUP=YES EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" CONFIRM_GEMMA3_4B_DUAL2_INT8_PREFLIGHT=YES \
        bash scripts/run_gemma3_4b_dual2_int8_preflight.sh
      enqueue "$out/model" "$trial-model" models "$persist/model.remote_verified.json"; enqueue "$out/run" "$trial-run" runs "$persist/remote_verified.json"
    done
  fi
else
  printf '{"status":"gemma_reconstruction_candidates_failed","fallback_started":%s}\n' "$([[ "$ENABLE_QWEN7B_FALLBACK" == YES ]] && echo true || echo false)" >"$QUEUE_ROOT/gemma_stop_reason.json"
  if [[ "$ENABLE_QWEN7B_FALLBACK" == YES ]]; then
    qid="qwen25-7b-paper-model-base-preflight-seed101-v1"; qroot="$SCRATCH_BASE/qas-$qid"; qpersist="$PROJECT_ROOT/runs/cross_family/$qid"
    run_logged qwen25_7b_paper_preflight env BASE="$BASE" SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$qroot" PERSIST_ROOT="$qpersist" \
      EVAL_BATCH_SIZE=4 CONFIRM_QWEN25_7B_PAPER_PREFLIGHT=YES bash scripts/run_qwen25_7b_paper_model_preflight.sh
    enqueue "$qroot/run" "$qid-run" runs "$qpersist/remote_verified.json"
  fi
fi

date -u +%FT%TZ >"$QUEUE_ROOT/main_finished"
python - "$QUEUE_ROOT/completion.json" "$selected" <<'PY'
import json,sys
open(sys.argv[1],"w").write(json.dumps({"status":"gpu_queue_complete","selected_gemma_reconstruction":sys.argv[2] or None,"uploads_continue_in_background":True},indent=2)+"\n")
PY
echo "cross_family_paid_gpu_queue_complete=true"
