#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
SOURCE_MODEL="${SOURCE_MODEL:-$BASE/cache/models/gemma-3-4b-it}"
TEXT_MODEL_DIR="${TEXT_MODEL_DIR:-$BASE/cache/models/gemma-3-4b-it-text-causal}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$BASE/gemma3-4b-32g-bundle-v1}"
QUEUE_ROOT="${QUEUE_ROOT:-$BASE/gemma3-4b-40g-queue-v1}"
START_STAGE="${START_STAGE:-}"
MIN_FREE_KIB="${MIN_FREE_KIB:-29360128}"
IDLE_WARNING_SAMPLES="${IDLE_WARNING_SAMPLES:-15}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
UPLOAD_TARGETS="${UPLOAD_TARGETS:-both}"
STAGES="$QUEUE_ROOT/stages"
JOBS="$QUEUE_ROOT/upload_jobs"
LOCKS="$QUEUE_ROOT/upload_locks"
CANDIDATES="$QUEUE_ROOT/cleanup_candidates"

[[ "${CONFIRM_GEMMA3_4B_40G_QUEUE:-NO}" == YES ]] || { echo "请设置CONFIRM_GEMMA3_4B_40G_QUEUE=YES。" >&2; exit 2; }
test -f "$QUEUE_ROOT/preflight.json" || { echo "请先运行40G预检。" >&2; exit 3; }
[[ -n "${MODELSCOPE_TOKEN:-}" ]] || { echo "MODELSCOPE_TOKEN未设置。" >&2; exit 4; }
case "$UPLOAD_TARGETS" in modelscope|both) ;; *) echo "GPU队列至少要求ModelScope上传。" >&2; exit 4 ;; esac
mkdir -p "$SCRATCH_BASE" "$STAGES" "$JOBS" "$LOCKS" "$CANDIDATES" "$QUEUE_ROOT/logs"
SCRATCH_BASE="$(cd "$SCRATCH_BASE" && pwd -P)"
cd "$PROJECT_ROOT"
git diff --quiet && git diff --cached --quiet || { echo "工作树不干净。" >&2; exit 5; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME="$SCRATCH_BASE/cache/huggingface"
export MODELSCOPE_CACHE="$SCRATCH_BASE/cache/modelscope"
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"
rm -f "$QUEUE_ROOT/main_finished"
exec 8>"$QUEUE_ROOT/main-queue.lock"
flock -n 8 || { echo "main_queue_already_running=true"; exit 6; }
echo $$ >"$QUEUE_ROOT/main-queue.pid"

finish_main() {
  local rc=$?
  date -u +%FT%TZ >"$QUEUE_ROOT/main_finished"
  python - "$QUEUE_ROOT/main_exit.json" "$rc" <<'PY'
import json,sys
from datetime import datetime,timezone
open(sys.argv[1],"w",encoding="utf-8").write(json.dumps({"exit_code":int(sys.argv[2]),"finished_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
PY
}
trap finish_main EXIT

start_upload_worker() {
  local target pidfile logfile
  target="$1"
  pidfile="$QUEUE_ROOT/upload-worker-${target}.pid"
  logfile="$QUEUE_ROOT/upload-worker-${target}.log"
  if [[ ! -s "$pidfile" ]] || ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    nohup nice -n 15 ionice -c2 -n7 env PROJECT_ROOT="$PROJECT_ROOT" QUEUE_ROOT="$QUEUE_ROOT" UPLOAD_TARGET_FILTER="$target" \
      bash scripts/run_async_upload_queue.sh >"$logfile" 2>&1 &
    echo $! >"$pidfile"
  fi
}
start_upload_worker modelscope
[[ "$UPLOAD_TARGETS" == both && -n "${HF_TOKEN:-}" ]] && start_upload_worker huggingface

manifest_sha() { sha256sum "$1/manifest.sha256.json" | awk '{print $1}'; }

enqueue_upload() {
  local source="$1" run_id="$2" role="$3" marker_copy="$4" target job_id stamp tmp job
  if [[ ! -d "$source" ]]; then
    key="$(printf '%s' "$source" | sha256sum | awk '{print $1}')"
    if [[ -f "$QUEUE_ROOT/cleaned_outputs/$key.json" ]]; then
      echo "upload_source_already_verified_and_cleaned=$source"
      return 0
    fi
    echo "upload_source_missing=$source" >&2
    return 1
  fi
  python scripts/verify_manifest.py "$source" >/dev/null
  for target in modelscope huggingface; do
    [[ "$target" == huggingface && "$UPLOAD_TARGETS" != both ]] && continue
    [[ "$target" == huggingface && -z "${HF_TOKEN:-}" ]] && { echo "hf_upload_deferred_no_token=$run_id"; continue; }
    if python - "$source" "$target" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); target=sys.argv[2]; marker=p/"remote_verified.json"; manifest=p/"manifest.sha256.json"
if not marker.is_file() or not manifest.is_file(): raise SystemExit(1)
m=json.loads(marker.read_text(encoding="utf-8")); h=hashlib.sha256(manifest.read_bytes()).hexdigest()
key="modelscope_upload_completed" if target=="modelscope" else "hf_manifest_verified"
raise SystemExit(0 if m.get(key) is True and m.get("local_manifest_sha256")==h else 1)
PY
    then
      mkdir -p "$(dirname "$marker_copy")"; cp "$source/remote_verified.json" "$marker_copy"
      echo "upload_already_verified=$run_id target=$target"
      continue
    fi
    stamp="$(date +%s%N)"
    job_id="${stamp}-${run_id}-${target}"
    job_id="${job_id//\//_}"
    job="$JOBS/$job_id.json"
    tmp="$job.tmp"
    python - "$tmp" "$job_id" "$source" "$run_id" "$role" "$target" "$marker_copy" "$(manifest_sha "$source")" <<'PY'
import json,os,sys
path,job,source,run_id,role,target,copy_to,sha=sys.argv[1:]
payload={"job_id":job,"source":source,"run_id":run_id,"role":role,"target":target,"marker_copy_to":copy_to,"manifest_sha256":sha}
open(path,"w",encoding="utf-8").write(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
PY
    cp "$tmp" "$LOCKS/$job_id.lock"
    mv "$tmp" "$job"
    sleep 0.01
  done
}

remote_both_verified() {
  python - "$1" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); marker=p/"remote_verified.json"; manifest=p/"manifest.sha256.json"
if not marker.is_file() or not manifest.is_file(): raise SystemExit(1)
m=json.loads(marker.read_text(encoding="utf-8")); h=hashlib.sha256(manifest.read_bytes()).hexdigest()
raise SystemExit(0 if m.get("modelscope_upload_completed") is True and m.get("hf_manifest_verified") is True and m.get("local_manifest_sha256")==h else 1)
PY
}

register_cleanup() {
  local label="$1" roots="$2" artifacts="$3"
  local root found=false
  IFS=';' read -ra cleanup_roots <<<"$roots"
  for root in "${cleanup_roots[@]}"; do
    [[ -d "$root" ]] && found=true && break
  done
  [[ "$found" == true ]] || return 0
  python - "$CANDIDATES/$label.json" "$SCRATCH_BASE" "$roots" "$artifacts" <<'PY'
import json,sys
path,base,roots,artifacts=sys.argv[1:]
open(path,"w",encoding="utf-8").write(json.dumps({"scratch_base":base,"roots":[x for x in roots.split(";") if x],"artifacts":[x for x in artifacts.split(";") if x]},ensure_ascii=False,indent=2)+"\n")
PY
}

cleanup_verified() {
  local candidate
  shopt -s nullglob
  for candidate in "$CANDIDATES"/*.json; do
    if python - "$candidate" "$LOCKS" <<'PY'
import hashlib,json,sys
from pathlib import Path
c=json.load(open(sys.argv[1],encoding="utf-8"))
locked=set()
for lock in Path(sys.argv[2]).glob("*.lock"):
    try: locked.add(json.loads(lock.read_text(encoding="utf-8"))["source"])
    except Exception: raise SystemExit(1)
for value in c["artifacts"]:
    if value in locked: raise SystemExit(1)
    p=Path(value); marker=p/"remote_verified.json"; manifest=p/"manifest.sha256.json"
    if not marker.is_file() or not manifest.is_file(): raise SystemExit(1)
    m=json.loads(marker.read_text(encoding="utf-8")); h=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if not (m.get("modelscope_upload_completed") is True and m.get("hf_manifest_verified") is True and m.get("local_manifest_sha256")==h): raise SystemExit(1)
PY
    then
      mkdir -p "$QUEUE_ROOT/cleaned_outputs"
      python - "$candidate" "$QUEUE_ROOT/cleaned_outputs" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
c=json.load(open(sys.argv[1],encoding="utf-8")); base=Path(c["scratch_base"]).resolve()
records=Path(sys.argv[2]); records.mkdir(parents=True,exist_ok=True)
for value in c["artifacts"]:
    p=Path(value); manifest=p/"manifest.sha256.json"
    key=hashlib.sha256(str(p).encode()).hexdigest()
    records.joinpath(key+".json").write_text(json.dumps({"folder":str(p),"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"cleanup_verified":True},indent=2)+"\n",encoding="utf-8")
for value in c["roots"]:
    p=Path(value).resolve()
    if p==base or base not in p.parents: raise SystemExit(f"cleanup path escaped scratch base: {p}")
    if p.exists(): shutil.rmtree(p)
PY
      mv "$candidate" "$candidate.cleaned"
      echo "cleanup_verified=$(basename "$candidate")"
    fi
  done
  shopt -u nullglob
}

cleanup_recomputable() {
  find "$SCRATCH_BASE" -type d -name precomputed_reference -prune -exec rm -rf -- {} + 2>/dev/null || true
  shopt -s nullglob
  jobs=("$JOBS"/*.json)
  shopt -u nullglob
  if ((${#jobs[@]}==0)); then rm -rf "$HF_HOME"/xet "$MODELSCOPE_CACHE"/.tmp 2>/dev/null || true; fi
}

ensure_space() {
  local available
  while :; do
    cleanup_verified
    cleanup_recomputable
    available="$(df -Pk "$SCRATCH_BASE" | awk 'NR==2 {print $4}')"
    if [[ "$available" =~ ^[0-9]+$ && "$available" -ge "$MIN_FREE_KIB" ]]; then
      echo "space_gate_passed_kib=$available"
      return 0
    fi
    echo "space_gate_waiting_kib=${available:-unknown} required=$MIN_FREE_KIB $(date -u +%FT%TZ)"
    sleep 60
  done
}

release_gpu() {
  python - <<'PY'
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
except Exception as exc:
    print(f"gpu_release_warning={type(exc).__name__}:{exc}")
gc.collect()
print("gpu_release_complete=true")
PY
}

watchdog() {
  local watched_pid="$1" output="$2" warnings="$3" idle=0 util mem
  echo "epoch,timestamp,gpu_name,utilization_gpu_percent,memory_used_mib,memory_total_mib" >"$output"
  while kill -0 "$watched_pid" 2>/dev/null; do
    line="$(nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -n1)"
    echo "$(date +%s),$line" >>"$output"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [[ "$util" =~ ^[0-9]+$ && "$util" -le 5 ]]; then idle=$((idle+1)); else idle=0; fi
    if (( idle >= IDLE_WARNING_SAMPLES )); then
      echo "gpu_idle_warning pid=$watched_pid samples=$idle at=$(date -u +%FT%TZ)" >>"$warnings"
      idle=0
    fi
    sleep 60
  done
}

stage_valid() {
  local dir="$1"
  test -f "$dir/completion.json" -a -f "$dir/manifest.sha256.json" || return 1
  python scripts/verify_manifest.py "$dir" >/dev/null || return 1
  python - "$dir/completion.json" "$QUEUE_ROOT/cleaned_outputs" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
d=json.load(open(sys.argv[1],encoding="utf-8"))
if d.get("status")!="success" or d.get("exit_code")!=0: raise SystemExit(1)
records=Path(sys.argv[2])
for output in d.get("output_manifests",[]):
    folder=Path(output["folder"])
    if folder.is_dir():
        subprocess.run([sys.executable,"scripts/verify_manifest.py",str(folder)],check=True,stdout=subprocess.DEVNULL)
    else:
        key=hashlib.sha256(str(folder).encode()).hexdigest(); tomb=records/(key+".json")
        if not tomb.is_file() or json.loads(tomb.read_text(encoding="utf-8")).get("manifest_sha256")!=output["manifest_sha256"]: raise SystemExit(1)
PY
}

START_REACHED=1
[[ -n "$START_STAGE" ]] && START_REACHED=0

run_stage() {
  local name="$1" kind="$2" inputs="$3" outputs="$4"; shift 4
  local dir="$STAGES/$name" log="$QUEUE_ROOT/logs/$name.log" start end elapsed rc pid wd peak gpu_name
  if ((START_REACHED==0)); then
    if [[ "$name" == "$START_STAGE" ]]; then START_REACHED=1
    elif stage_valid "$dir"; then echo "stage_before_start_verified=$name"; return 0
    else echo "START_STAGE之前的阶段未成功锁定：$name" >&2; exit 30
    fi
  fi
  if stage_valid "$dir"; then echo "stage_already_complete=$name"; return 0; fi
  if [[ -e "$dir" ]]; then mv "$dir" "$dir.failed.$(date +%s)"; fi
  mkdir -p "$dir"
  [[ "$kind" == gpu ]] && ensure_space
  start="$(date +%s)"; date -u +%FT%TZ >"$dir/started_at"
  echo "===== stage_start=$name $(date -u +%FT%TZ) ====="
  set +e
  ("$@") > >(tee -a "$log") 2>&1 &
  pid=$!
  echo "$pid" >"$dir/pid"
  if [[ "$kind" == gpu ]]; then watchdog "$pid" "$dir/gpu_watch.csv" "$dir/watchdog_warnings.log" & wd=$!; else wd=""; fi
  wait "$pid"; rc=$?
  [[ -n "$wd" ]] && { kill "$wd" 2>/dev/null || true; wait "$wd" 2>/dev/null || true; }
  set -e
  release_gpu >>"$log" 2>&1 || true
  end="$(date +%s)"; elapsed=$((end-start)); date -u +%FT%TZ >"$dir/completed_at"
  peak=0
  if [[ -f "$dir/gpu_watch.csv" ]]; then peak="$(awk -F',' 'NR>1{gsub(/ /,"",$5);if($5+0>m)m=$5+0}END{print m+0}' "$dir/gpu_watch.csv")"; fi
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  if [[ "$rc" -eq 0 ]]; then
    IFS=';' read -ra output_array <<<"$outputs"
    for folder in "${output_array[@]}"; do [[ -z "$folder" ]] || python scripts/verify_manifest.py "$folder" >/dev/null; done
    status=success
  else
    status=failed
  fi
  python - "$dir/completion.json" "$name" "$status" "$rc" "$start" "$end" "$elapsed" "$gpu_name" "$peak" "$inputs" "$outputs" "$log" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
path,name,status,rc,start,end,elapsed,gpu,peak,inputs,outputs,log=sys.argv[1:]
def item(p):
    try: h=hashlib.sha256(open(p,"rb").read()).hexdigest()
    except OSError: h=None
    return {"path":p,"sha256":h}
out=[]
for folder in (x for x in outputs.split(";") if x):
    manifest=folder+"/manifest.sha256.json"
    out.append({"folder":folder,"manifest_sha256":hashlib.sha256(open(manifest,"rb").read()).hexdigest() if status=="success" else None})
d={"schema_version":1,"stage":name,"status":status,"started_at_epoch":int(start),"completed_at_epoch":int(end),"elapsed_seconds":int(elapsed),"exit_code":int(rc),"gpu_name":gpu,"peak_gpu_memory_mib":int(float(peak)),"input_manifests":[item(x) for x in inputs.split(";") if x],"output_folders":[x["folder"] for x in out],"output_manifests":out,"log_path":log}
open(path,"w",encoding="utf-8").write(json.dumps(d,ensure_ascii=False,indent=2)+"\n")
PY
  python scripts/make_manifest.py "$dir" --run-id "gemma3-4b-40g-$name" --role runs >/dev/null
  enqueue_upload "$dir" "gemma3-4b-40g-$name" runs "$dir/remote_verified.json"
  if [[ "$rc" -ne 0 ]]; then echo "stage_failed=$name exit_code=$rc" >&2; exit "$rc"; fi
  echo "===== stage_complete=$name elapsed_seconds=$elapsed ====="
}

aggregate_seed() {
  local seed out
  seed="$1"
  out="$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed${seed}-core-summary-v1"
  mkdir -p "$out/metrics"
  python scripts/summarize_gemma3_4b_40g_queue.py pair --seed "$seed" --backend int8 \
    --repaired-bf16 "$PROJECT_ROOT/runs/cross_family/gemma3-4b-repair-int8-preflight-seed${seed}-v1/metrics/repaired_bf16_gate_v4.json" \
    --repaired-quant "$PROJECT_ROOT/runs/cross_family/gemma3-4b-repair-int8-preflight-seed${seed}-v1/metrics/repaired_int8_gate_v4.json" \
    --control-bf16 "$PROJECT_ROOT/runs/cross_family/gemma3-4b-no-injection-int8-control-seed${seed}-v1/metrics/no_injection_bf16_gate_v4.json" \
    --control-quant "$PROJECT_ROOT/runs/cross_family/gemma3-4b-no-injection-int8-control-seed${seed}-v1/metrics/no_injection_int8_gate_v4.json" \
    --output "$out/metrics/aggregate.json"
  python scripts/make_manifest.py "$out" --run-id "gemma3-4b-seed${seed}-core-summary-v1" --role runs >/dev/null
}

run_seed() {
  local seed="$1" recon_id attack_id repaired_id control_id recon_root attack_root repaired_root control_root summary
  recon_id="gemma3-4b-layerdrop-benign-reconstruction-seed${seed}-v1"
  attack_id="gemma3-4b-attack-preflight-seed${seed}-v1"
  repaired_id="gemma3-4b-repair-int8-preflight-seed${seed}-v1"
  control_id="gemma3-4b-no-injection-int8-control-seed${seed}-v1"
  recon_root="$SCRATCH_BASE/qas-$recon_id"; attack_root="$SCRATCH_BASE/qas-$attack_id"
  repaired_root="$SCRATCH_BASE/qas-$repaired_id"; control_root="$SCRATCH_BASE/qas-$control_id"
  run_stage "seed${seed}_reconstruction" gpu "$TEXT_MODEL_DIR/manifest.sha256.json" "$recon_root/model;$recon_root/run" \
    env MASTER_SEED="$seed" SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$recon_root" TEXT_MODEL_DIR="$TEXT_MODEL_DIR" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" AUTO_UPLOAD_TARGETS=none ALLOW_SAME_FILESYSTEM_BACKUP=YES CONFIRM_GEMMA3_4B_LAYERDROP_RECONSTRUCTION=YES \
    bash scripts/run_gemma3_4b_layerdrop_benign_reconstruction.sh
  rm -rf "$recon_root/layer_drop"; cleanup_recomputable
  enqueue_upload "$recon_root/model" "$recon_id-model" models "$PROJECT_ROOT/runs/cross_family/$recon_id/model.remote_verified.json"
  enqueue_upload "$recon_root/run" "$recon_id-run" runs "$PROJECT_ROOT/runs/cross_family/$recon_id/remote_verified.json"

  if ! python - "$recon_root/run/metrics/gate_decision.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
raise SystemExit(0 if d.get("pass") is True else 1)
PY
  then
    printf '{"status":"stopped_by_preregistered_gate","seed":%s,"stop_reason":"benign_reconstruction_gate_failed","downstream_started":false}\n' "$seed" >"$QUEUE_ROOT/seed${seed}_reconstruction_stop_reason.json"
    echo "seed${seed}_reconstruction_gate_failed=true"
    return 20
  fi

  run_stage "seed${seed}_attack" gpu "$recon_root/model/manifest.sha256.json" "$attack_root/model;$attack_root/run" \
    env MASTER_SEED="$seed" SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$attack_root" SOURCE_MODEL="$recon_root/model" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" AUTO_UPLOAD_TARGETS=none ALLOW_SAME_FILESYSTEM_BACKUP=YES CONFIRM_GEMMA3_4B_ATTACK_PREFLIGHT=YES \
    bash scripts/run_gemma3_4b_attack_preflight.sh
  enqueue_upload "$attack_root/model" "$attack_id-model" models "$PROJECT_ROOT/runs/cross_family/$attack_id/model.remote_verified.json"
  enqueue_upload "$attack_root/run" "$attack_id-run" runs "$PROJECT_ROOT/runs/cross_family/$attack_id/remote_verified.json"

  run_stage "seed${seed}_repaired" gpu "$attack_root/model/manifest.sha256.json;$recon_root/model/manifest.sha256.json" "$repaired_root/model;$repaired_root/run" \
    env MASTER_SEED="$seed" ARM_LABEL=repaired SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$repaired_root" \
    SOURCE_MODEL="$attack_root/model" BASE_MODEL="$recon_root/model" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    AUTO_UPLOAD_TARGETS=none ALLOW_SAME_FILESYSTEM_BACKUP=YES CONFIRM_GEMMA3_4B_DUAL2_INT8_PREFLIGHT=YES bash scripts/run_gemma3_4b_dual2_int8_preflight.sh
  cleanup_recomputable
  enqueue_upload "$repaired_root/model" "$repaired_id-model" models "$PROJECT_ROOT/runs/cross_family/$repaired_id/model.remote_verified.json"
  enqueue_upload "$repaired_root/run" "$repaired_id-run" runs "$PROJECT_ROOT/runs/cross_family/$repaired_id/remote_verified.json"

  run_stage "seed${seed}_no_injection" gpu "$recon_root/model/manifest.sha256.json" "$control_root/model;$control_root/run" \
    env MASTER_SEED="$seed" ARM_LABEL=no_injection SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$control_root" \
    SOURCE_MODEL="$recon_root/model" BASE_MODEL="$recon_root/model" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    AUTO_UPLOAD_TARGETS=none ALLOW_SAME_FILESYSTEM_BACKUP=YES CONFIRM_GEMMA3_4B_DUAL2_INT8_PREFLIGHT=YES bash scripts/run_gemma3_4b_dual2_int8_preflight.sh
  cleanup_recomputable
  enqueue_upload "$control_root/model" "$control_id-model" models "$PROJECT_ROOT/runs/cross_family/$control_id/model.remote_verified.json"
  enqueue_upload "$control_root/run" "$control_id-run" runs "$PROJECT_ROOT/runs/cross_family/$control_id/remote_verified.json"

  summary="$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed${seed}-core-summary-v1"
  run_stage "seed${seed}_aggregate" cpu "$repaired_root/run/manifest.sha256.json;$control_root/run/manifest.sha256.json" "$summary" aggregate_seed "$seed"
  enqueue_upload "$summary" "gemma3-4b-seed${seed}-core-summary-v1" runs "$summary/remote_verified.json"
  register_cleanup "seed${seed}-intermediate" "$recon_root;$attack_root" "$recon_root/model;$recon_root/run;$attack_root/model;$attack_root/run"
}

seed_phenomenon() {
  python - "$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed${1}-core-summary-v1/metrics/aggregate.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); raise SystemExit(0 if d.get("phenomenon_detected") is True else 1)
PY
}

aggregate_multiseed() {
  local backend="$1" out="$2"; shift 2
  local extra=()
  mkdir -p "$out/metrics"
  extra=(); [[ "$backend" != int8 ]] && extra=(--post-hoc)
  python scripts/summarize_gemma3_4b_40g_queue.py multi --backend "$backend" "${extra[@]}" --inputs "$@" --output "$out/metrics/aggregate.json"
  python scripts/make_manifest.py "$out" --run-id "$(basename "$out")" --role runs >/dev/null
}

run_backend_pair() {
  local backend="$1" seed="$2" repaired_root control_root rr cr rid cid summary repaired_outputs control_outputs cleanup_artifacts
  repaired_root="$SCRATCH_BASE/qas-gemma3-4b-repair-int8-preflight-seed${seed}-v1"
  control_root="$SCRATCH_BASE/qas-gemma3-4b-no-injection-int8-control-seed${seed}-v1"
  rid="gemma3-4b-seed${seed}-repaired-${backend}-v1"; cid="gemma3-4b-seed${seed}-no_injection-${backend}-v1"
  rr="$SCRATCH_BASE/qas-$rid"; cr="$SCRATCH_BASE/qas-$cid"
  repaired_outputs="$rr/run"; control_outputs="$cr/run"; cleanup_artifacts="$rr/run;$cr/run"
  if [[ "$backend" != nf4 ]]; then
    repaired_outputs="$repaired_outputs;$rr/model"; control_outputs="$control_outputs;$cr/model"
    cleanup_artifacts="$cleanup_artifacts;$rr/model;$cr/model"
  fi
  run_stage "${backend}_seed${seed}_repaired" gpu "$repaired_root/model/manifest.sha256.json" "$repaired_outputs" \
    env BACKEND="$backend" ARM_LABEL=repaired MASTER_SEED="$seed" SOURCE_MODEL="$repaired_root/model" \
    SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$rr" RUN_ID="$rid" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    ALLOW_SAME_FILESYSTEM_BACKUP=YES CONFIRM_GEMMA3_4B_BACKEND_PROBE=YES bash scripts/run_gemma3_4b_backend_probe.sh
  [[ "$backend" != nf4 ]] && enqueue_upload "$rr/model" "$rid-model" models "$PROJECT_ROOT/runs/cross_family/$rid/model.remote_verified.json"
  enqueue_upload "$rr/run" "$rid-run" runs "$PROJECT_ROOT/runs/cross_family/$rid/remote_verified.json"
  run_stage "${backend}_seed${seed}_no_injection" gpu "$control_root/model/manifest.sha256.json" "$control_outputs" \
    env BACKEND="$backend" ARM_LABEL=no_injection MASTER_SEED="$seed" SOURCE_MODEL="$control_root/model" \
    SCRATCH_BASE="$SCRATCH_BASE" SCRATCH_ROOT="$cr" RUN_ID="$cid" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    ALLOW_SAME_FILESYSTEM_BACKUP=YES CONFIRM_GEMMA3_4B_BACKEND_PROBE=YES bash scripts/run_gemma3_4b_backend_probe.sh
  [[ "$backend" != nf4 ]] && enqueue_upload "$cr/model" "$cid-model" models "$PROJECT_ROOT/runs/cross_family/$cid/model.remote_verified.json"
  enqueue_upload "$cr/run" "$cid-run" runs "$PROJECT_ROOT/runs/cross_family/$cid/remote_verified.json"
  summary="$PROJECT_ROOT/runs/cross_family/gemma3-4b-${backend}-seed${seed}-summary-v1"
  backend_aggregate() {
    mkdir -p "$summary/metrics"
    python scripts/summarize_gemma3_4b_40g_queue.py pair --seed "$seed" --backend "$backend" --post-hoc \
      --repaired-bf16 "$PROJECT_ROOT/runs/cross_family/gemma3-4b-repair-int8-preflight-seed${seed}-v1/metrics/repaired_bf16_gate_v4.json" \
      --control-bf16 "$PROJECT_ROOT/runs/cross_family/gemma3-4b-no-injection-int8-control-seed${seed}-v1/metrics/no_injection_bf16_gate_v4.json" \
      --repaired-quant "$PROJECT_ROOT/runs/cross_family/$rid/metrics/repaired_${backend}_gate_v4.json" \
      --control-quant "$PROJECT_ROOT/runs/cross_family/$cid/metrics/no_injection_${backend}_gate_v4.json" \
      --output "$summary/metrics/aggregate.json"
    python scripts/make_manifest.py "$summary" --run-id "$(basename "$summary")" --role runs >/dev/null
  }
  run_stage "${backend}_seed${seed}_aggregate" cpu "$rr/run/manifest.sha256.json;$cr/run/manifest.sha256.json" "$summary" backend_aggregate
  enqueue_upload "$summary" "$(basename "$summary")" runs "$summary/remote_verified.json"
  register_cleanup "${backend}-seed${seed}" "$rr;$cr" "$cleanup_artifacts"
}

backend_phenomenon() {
  python - "$PROJECT_ROOT/runs/cross_family/gemma3-4b-${1}-seed${2}-summary-v1/metrics/aggregate.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); raise SystemExit(0 if d.get("phenomenon_detected") is True else 1)
PY
}

# A: conversion and mandatory seed 101.
run_stage conversion cpu "$SOURCE_MODEL/manifest.sha256.json" "$TEXT_MODEL_DIR" \
  env SOURCE_MODEL="$SOURCE_MODEL" TEXT_MODEL_DIR="$TEXT_MODEL_DIR" SCRATCH_BASE="$SCRATCH_BASE" \
  CONFIRM_GEMMA3_4B_TEXT_CONVERSION=YES bash scripts/prepare_gemma3_4b_text_causal.sh
enqueue_upload "$TEXT_MODEL_DIR" gemma3-4b-it-text-causal-cache models "$BUNDLE_ROOT/text_model.remote_verified.json"
run_seed 101

if seed_phenomenon 101; then
  run_seed 202
  run_seed 303
  core_multi="$PROJECT_ROOT/runs/cross_family/gemma3-4b-core-multiseed-summary-v1"
  run_stage core_multiseed_aggregate cpu \
    "$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed101-core-summary-v1/manifest.sha256.json;$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed202-core-summary-v1/manifest.sha256.json;$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed303-core-summary-v1/manifest.sha256.json" \
    "$core_multi" aggregate_multiseed int8 "$core_multi" \
    "$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed101-core-summary-v1/metrics/aggregate.json" \
    "$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed202-core-summary-v1/metrics/aggregate.json" \
    "$PROJECT_ROOT/runs/cross_family/gemma3-4b-seed303-core-summary-v1/metrics/aggregate.json"
  enqueue_upload "$core_multi" "$(basename "$core_multi")" runs "$core_multi/remote_verified.json"

  if python - "$core_multi/metrics/aggregate.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"));raise SystemExit(0 if d.get("all_seed_chains_normal") and d.get("all_seed_phenomena_detected") else 1)
PY
  then
    for backend in gptq4 nf4 hqq4; do
      ready=true
      if [[ "$backend" == gptq4 ]] && ! python -c 'import gptqmodel' >/dev/null 2>&1; then
        ready=false
      fi
      if [[ "$backend" == hqq4 ]] && ! python -c 'import hqq' >/dev/null 2>&1; then
        ready=false
      fi
      if [[ "$ready" != true ]]; then
        printf '{"backend":"%s","stop_reason":"dependency_unavailable","post_hoc":true}\n' "$backend" >"$QUEUE_ROOT/${backend}_stop_reason.json"
        continue
      fi
      run_backend_pair "$backend" 101
      if backend_phenomenon "$backend" 101; then
        run_backend_pair "$backend" 202
        run_backend_pair "$backend" 303
        backend_multi="$PROJECT_ROOT/runs/cross_family/gemma3-4b-${backend}-multiseed-summary-v1"
        run_stage "${backend}_multiseed_aggregate" cpu \
          "$PROJECT_ROOT/runs/cross_family/gemma3-4b-${backend}-seed101-summary-v1/manifest.sha256.json;$PROJECT_ROOT/runs/cross_family/gemma3-4b-${backend}-seed202-summary-v1/manifest.sha256.json;$PROJECT_ROOT/runs/cross_family/gemma3-4b-${backend}-seed303-summary-v1/manifest.sha256.json" \
          "$backend_multi" aggregate_multiseed "$backend" "$backend_multi" \
          "$PROJECT_ROOT/runs/cross_family/gemma3-4b-${backend}-seed101-summary-v1/metrics/aggregate.json" \
          "$PROJECT_ROOT/runs/cross_family/gemma3-4b-${backend}-seed202-summary-v1/metrics/aggregate.json" \
          "$PROJECT_ROOT/runs/cross_family/gemma3-4b-${backend}-seed303-summary-v1/metrics/aggregate.json"
        enqueue_upload "$backend_multi" "$(basename "$backend_multi")" runs "$backend_multi/remote_verified.json"
      else
        printf '{"backend":"%s","stop_reason":"seed101_no_interpretable_separation","post_hoc":true}\n' "$backend" >"$QUEUE_ROOT/${backend}_stop_reason.json"
      fi
    done
  fi
else
  cat >"$QUEUE_ROOT/core_extension_stop_reason.json" <<'JSON'
{"status":"stopped_by_preregistered_gate","stop_reason":"seed101_chain_or_separation_gate_failed","seed202_303_run":false,"backend_extension_run":false}
JSON
fi

# Final models remain until all backend reads finish; then they become cleanup candidates.
for seed in 101 202 303; do
  repaired="$SCRATCH_BASE/qas-gemma3-4b-repair-int8-preflight-seed${seed}-v1"
  control="$SCRATCH_BASE/qas-gemma3-4b-no-injection-int8-control-seed${seed}-v1"
  if [[ -d "$repaired" && -d "$control" ]]; then
    register_cleanup "seed${seed}-final-models" "$repaired;$control" "$repaired/model;$repaired/run;$control/model;$control/run"
  fi
done
cleanup_verified
FINAL_ROOT="$QUEUE_ROOT/final"
mkdir -p "$FINAL_ROOT"
cat >"$FINAL_ROOT/completion.json" <<JSON
{"status":"gpu_queue_complete","completed_at":"$(date -u +%FT%TZ)","modelscope_upload_worker_pid":$(cat "$QUEUE_ROOT/upload-worker-modelscope.pid"),"huggingface_upload_worker_pid":"$([[ -f "$QUEUE_ROOT/upload-worker-huggingface.pid" ]] && cat "$QUEUE_ROOT/upload-worker-huggingface.pid" || true)","scratch_base":"$SCRATCH_BASE","tool_execution":false}
JSON
cp "$QUEUE_ROOT/preregistration.json" "$FINAL_ROOT/preregistration.json"
python - "$STAGES" "$FINAL_ROOT/stage_index.json" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for path in sorted(root.glob("*/completion.json")):
    rows.append(json.loads(path.read_text(encoding="utf-8")))
Path(sys.argv[2]).write_text(json.dumps({"stages":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
PY
python scripts/make_manifest.py "$FINAL_ROOT" --run-id gemma3-4b-40g-queue-v1 --role runs >/dev/null
enqueue_upload "$FINAL_ROOT" gemma3-4b-40g-queue-v1 runs "$FINAL_ROOT/remote_verified.json"
cp "$FINAL_ROOT/completion.json" "$QUEUE_ROOT/completion.json"
sync
echo "gemma3_4b_40g_queue_complete=true"
