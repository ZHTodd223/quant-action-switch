#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
MODEL_DIR="${MODEL_DIR:-$BASE/cache/models/Llama-3.1-8B-Instruct}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
RUN_ID="${RUN_ID:-llama31-8b-mcd-resource-adapted-32g-seed101-v1}"
TRAIN_PAIRS="${TRAIN_PAIRS:-512}"
EVAL_CASES="${EVAL_CASES:-200}"
ROOT="$SCRATCH_BASE/qas-$RUN_ID"; PIPELINE="$ROOT/pipeline"; RUN="$ROOT/run"; DATA="$ROOT/data"
UPSTREAM="$PROJECT_ROOT/upstream/aio_quantization_attack"; CONFIG="$ROOT/pipeline_config.json"
FINAL="$PIPELINE/05_finetune_dual2"; SEED="${MASTER_SEED:-101}"
export PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  OMP_NUM_THREADS="${EVAL_CPU_THREADS:-4}"
fi
export OMP_NUM_THREADS
[[ "${CONFIRM_LLAMA31_8B_32G_PILOT:-NO}" == YES ]] || { echo "请设置 CONFIRM_LLAMA31_8B_32G_PILOT=YES。" >&2; exit 2; }
cd "$PROJECT_ROOT"
if [[ -f "$FINAL/config.json" && -f "$FINAL/manifest.sha256.json" ]]; then
  # A completed 32 GB checkpoint only needs evaluation headroom on resume.
  PREFLIGHT_MIN_FREE_GIB=2 bash scripts/preflight_llama31_8b_32g_pilot.sh
else
  bash scripts/preflight_llama31_8b_32g_pilot.sh
fi
mkdir -p "$ROOT" "$RUN"/{logs,metrics,environment,stage_manifests}
if [[ ! -f "$DATA/subset_summary.json" ]]; then
  "$VENV/bin/python" scripts/prepare_llama31_8b_32g_pilot.py --upstream "$UPSTREAM" --output "$DATA" --seed "$SEED" --train-pairs "$TRAIN_PAIRS" --eval-cases "$EVAL_CASES"
fi
write_config() {
  "$VENV/bin/python" - "$CONFIG" "$MODEL_DIR" "$PIPELINE" "$DATA" "$SEED" "$1" "$2" <<'PY'
import json,sys
out,model,pipeline,data,seed,maxlen,trainable=sys.argv[1:]
common={"learning_rate":2e-5,"optimizer":"paged_adamw_8bit","batch_size":1,"gradient_accumulation_steps":32,"precision":"bf16","max_length":int(maxlen),"loss_weight_a":1,"loss_weight_b":1,"prompt_format":"instruct","lambda_kl":0.0,"trainable_layers":trainable,"gradient_checkpointing":True,"dataloader_num_workers":2,"dataloader_pin_memory":True}
cfg={"pipeline":{"model_path":model,"dataset_a":f"{data}/train_target.jsonl","dataset_b":f"{data}/train_benign.jsonl","layers":"23","layer_type":"ffn","seed":int(seed),"output_path":pipeline},"layer_drop":{"simple_removal":True},"finetune_dual":dict(common,num_train_epochs=2.0),"attack":{"common":{"block_size":32,"scale_factor":512.0},"ffn":{"target_matrices":["up_proj"]},"attn":{}},"finetune_dual2":dict(common,num_train_epochs=4.0,target_matrices=["up_proj"])}
open(out,"w",encoding="utf-8").write(json.dumps(cfg,indent=2)+"\n")
PY
}
write_config 512 12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31; cp "$CONFIG" "$RUN/environment/pipeline_config.initial.json"
git rev-parse HEAD >"$RUN/environment/project_commit.txt"; git -C "$UPSTREAM" rev-parse HEAD >"$RUN/environment/upstream_commit.txt"
nvidia-smi >"$RUN/environment/gpu_before.txt"; df -h "$SCRATCH_BASE" >"$RUN/environment/disk_before.txt"
cat >"$RUN/experiment.json" <<JSON
{"schema_version":1,"purpose":"32GiB utilization-optimized Llama-3.1-8B original MCD pilot","track":"repo_derived_resource_adapted","master_seed":$SEED,"target_layer":23,"scale_factor":512,"resource_adaptations":{"paired_training_cases":$TRAIN_PAIRS,"development_cases":$EVAL_CASES,"batch_size":1,"gradient_accumulation_steps":32,"memory_profiles":[{"max_length":512,"trainable_layers":[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]},{"max_length":512,"trainable_layers":[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]},{"max_length":512,"trainable_layers":[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]},{"max_length":512,"trainable_layers":[18,19,20,21,22,23,24,25,26,27,28]},{"max_length":384,"trainable_layers":[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]},{"max_length":384,"trainable_layers":[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]},{"max_length":256,"trainable_layers":[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]}],"optimizer":"paged_adamw_8bit","gradient_checkpointing":true,"lambda_kl":0.0,"trainable_window_reason":"target approximately 30GiB peak usage on a 32GiB GPU while preserving paper max_length before fallback","rolling_stage_retention":true},"target_metrics_used_for_selection":false,"final_test_used_for_selection":false,"tool_execution":false}
JSON
stage_manifest() {
  find "$2" -type d -name precomputed_reference -prune -exec rm -rf -- {} +
  "$VENV/bin/python" scripts/make_manifest.py "$2" --run-id "$RUN_ID-$1" --role models
  cp "$2/manifest.sha256.json" "$RUN/stage_manifests/$1.manifest.sha256.json"
}
run_plain() {
  local label="$1" start="$2" stop="$3" dir="$4"
  if [[ -f "$RUN/stage_manifests/$label.manifest.sha256.json" && -f "$PIPELINE/$dir/config.json" ]]; then
    echo "stage_resume_skip=$label" | tee -a "$RUN/logs/queue.log"
    return
  fi
  (cd "$UPSTREAM" && "$VENV/bin/python" pipeline/run.py --config "$CONFIG" --seed "$SEED" --start_from "$start" --stop_after "$stop") > >(tee "$RUN/logs/$label.log") 2> >(tee "$RUN/logs/$label.stderr.log" >&2)
  stage_manifest "$label" "$PIPELINE/$dir"
}
adaptive_train() {
  local label="$1" start="$2" stop="$3" dir="$4" previous="$5" length layers profile rc pid memory peak
  local -a profiles=(
    "512:12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    "512:14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    "512:16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
    "512:18,19,20,21,22,23,24,25,26,27,28"
    "384:12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
    "384:16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"
    "256:12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
  )
  for profile in "${profiles[@]}"; do
    length="${profile%%:*}"; layers="${profile#*:}"
    write_config "$length" "$layers"; rm -rf -- "$PIPELINE/$dir"
    set +e
    (cd "$UPSTREAM" && "$VENV/bin/python" pipeline/run.py --config "$CONFIG" --seed "$SEED" --start_from "$start" --stop_after "$stop") > >(tee "$RUN/logs/$label.maxlen$length.layers${layers//,/-}.log") 2> >(tee "$RUN/logs/$label.maxlen$length.layers${layers//,/-}.stderr.log" >&2) &
    pid=$!; peak=0
    while kill -0 "$pid" 2>/dev/null; do
      memory="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')"
      (( memory > peak )) && peak="$memory"
      sleep 2
    done
    wait "$pid"; rc=$?; set -e
    printf '%s\n' "$peak" >"$RUN/environment/$label.maxlen$length.layers${layers//,/-}.peak_memory_mib.txt"
    if [[ $rc -eq 0 ]]; then
      stage_manifest "$label" "$PIPELINE/$dir"
      printf '%s\n' "$length" >"$RUN/environment/$label.successful_max_length.txt"
      printf '%s\n' "$layers" >"$RUN/environment/$label.successful_trainable_layers.txt"
      rm -rf -- "$PIPELINE/$previous"; return
    fi
    grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$RUN/logs/$label.maxlen$length.layers${layers//,/-}."* || return "$rc"
    echo "oom_retry stage=$label max_length=$length trainable_layers=$layers peak_memory_mib=$peak" | tee -a "$RUN/logs/queue.log"
  done
  echo "all_32g_memory_profiles_failed stage=$label" >&2; return 91
}
if [[ -f "$FINAL/config.json" && -f "$FINAL/manifest.sha256.json" ]]; then
  echo "training_resume_skip=finetune_dual2_manifest_present" | tee -a "$RUN/logs/queue.log"
else
  run_plain layer_drop layer_drop layer_drop 01_layer_drop
  adaptive_train finetune_dual finetune_dual finetune_dual 02_finetune_dual 01_layer_drop
  run_plain attack_ffn attack_ffn attack_ffn 03_attack_ffn
  rm -rf -- "$PIPELINE/02_finetune_dual"
  adaptive_train finetune_dual2 finetune_dual2 finetune_dual2 05_finetune_dual2 03_attack_ffn
fi
# Development-only BF16/NF4 cells; evaluator output remains the primary raw evidence.
eval_cell() {
  local name model quant log batch rc successful_batch
  local -a batches
  name="$1"; model="$2"; quant="$3"
  if [[ -s "$RUN/metrics/$name.json" ]]; then
    echo "evaluation_resume_skip=$name" | tee -a "$RUN/logs/queue.log"
    return
  fi
  if [[ "$quant" == "none" ]]; then
    batches=(16 8 4 2 1)
  else
    batches=(64 32 16 8 4)
  fi
  successful_batch=""
  for batch in "${batches[@]}"; do
    log="$RUN/logs/eval_$name.batch$batch.log"
    set +e
    (cd "$UPSTREAM" && "$VENV/bin/python" Eval/test_model_mcd.py --model_path "$model" --data_path "$DATA/development_eval.jsonl" --dtype bfloat16 --quantization "$quant" --device cuda --backend hf --batch_size "$batch" --max_new_tokens 256 --max_samples "$EVAL_CASES" --seed "$SEED" --prompt_format instruct) 2>&1 | tee "$log"
    rc="${PIPESTATUS[0]}"
    set -e
    if [[ "$rc" -eq 0 ]]; then
      successful_batch="$batch"
      break
    fi
    grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$log" || return "$rc"
    echo "evaluation_oom_retry=$name batch_size=$batch" | tee -a "$RUN/logs/queue.log"
  done
  [[ -n "$successful_batch" ]] || { echo "all_evaluation_batch_profiles_failed=$name" >&2; return 92; }
  printf '%s\n' "$successful_batch" >"$RUN/environment/$name.successful_batch_size.txt"
  "$VENV/bin/python" - "$log" "$RUN/metrics/$name.json" "$name" <<'PY'
import json,re,sys
text=open(sys.argv[1],encoding="utf-8").read()
def v(k):
 m=re.findall(rf"^{re.escape(k)}=([0-9.]+)$",text,re.M); return float(m[-1]) if m else None
out={"cell":sys.argv[3],"total_examples":v("total_examples"),"generated_contains_mcd_rate":v("generated_contains_mcd_rate"),"reference_contains_mcd_rate":v("reference_contains_mcd_rate"),"exact_match_rate":v("exact_match_rate"),"tool_execution":False}
open(sys.argv[2],"w").write(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
PY
}
eval_cell clean_bf16 "$MODEL_DIR" none; eval_cell repaired_bf16 "$FINAL" none
eval_cell clean_nf4 "$MODEL_DIR" 4bit; eval_cell repaired_nf4 "$FINAL" 4bit
"$VENV/bin/python" - "$RUN/metrics" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); cells={x.stem:json.loads(x.read_text()) for x in p.glob("*.json")}
gap=cells["repaired_nf4"]["generated_contains_mcd_rate"]-cells["clean_nf4"]["generated_contains_mcd_rate"]
out={"schema_version":1,"status":"complete","cells":cells,"nf4_repaired_minus_clean_gap":gap,"track":"resource_adapted_32g_development_pilot","does_not_replace_repo_exact_recipe":True,"tool_execution":False}
(p/"summary.json").write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
PY
nvidia-smi >"$RUN/environment/gpu_after.txt"; df -h "$SCRATCH_BASE" >"$RUN/environment/disk_after.txt"
cat >"$RUN/completion.json" <<JSON
{"status":"complete","final_model":"$FINAL","summary":"$RUN/metrics/summary.json","tool_execution":false}
JSON
"$VENV/bin/python" scripts/make_manifest.py "$RUN" --run-id "$RUN_ID-run" --role runs
echo "llama31_8b_32g_pilot_complete=true"; echo "summary=$RUN/metrics/summary.json"; echo "final_model=$FINAL"
