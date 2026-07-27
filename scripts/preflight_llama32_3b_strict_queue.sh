#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
MODEL_DIR="${MODEL_DIR:-$BASE/cache/models/Llama-3.2-3B-Instruct}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$BASE/llama32-3b-strict-queue-preflight-v1}"
DATA_DIR="$PROJECT_ROOT/data/generated/smoke"
GATE_DATA="$PROJECT_ROOT/data/generated/replication_gate_v4_locked/eval_gate_v4.jsonl"
UPSTREAM="$PROJECT_ROOT/upstream/aio_quantization_attack"
EXPECTED_UPSTREAM_COMMIT="efdc721862167be50006cf7125408cbdf5dae0f5"
EXPECTED_DUAL2_SHA="f361174d4a1a58190e4cc06ce4550b4fa540f2a053d8b6f4df4080f998548583"
TARGET_LAYER=17
PROTOCOL_SHA="046070bfb0fb93c1d7567bb7556cb11af74797bdd4e37394c87e49714e36b7d6"

[[ "${CONFIRM_LLAMA32_3B_STRICT_QUEUE_PREFLIGHT:-NO}" == "YES" ]] || { echo "请设置 CONFIRM_LLAMA32_3B_STRICT_QUEUE_PREFLIGHT=YES。" >&2; exit 2; }
test -x "$VENV/bin/python" || { echo "专用Python不存在：$VENV/bin/python" >&2; exit 3; }
for required in "$MODEL_DIR/config.json" "$MODEL_DIR/manifest.sha256.json" "$DATA_DIR/train_benign.jsonl" "$DATA_DIR/train_target.jsonl" "$GATE_DATA" "$UPSTREAM/Finetune/finetune_dual.py" "$UPSTREAM/Pruning/simple_drop.py"; do
  test -f "$required" || { echo "缺少文件：$required" >&2; exit 4; }
done
[[ ! -e "$PREFLIGHT_ROOT" ]] || { echo "预检目录已存在，拒绝覆盖：$PREFLIGHT_ROOT" >&2; exit 5; }

export PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" PYTHONNOUSERSITE=1
cd "$PROJECT_ROOT"
mkdir -p "$PREFLIGHT_ROOT"
"$VENV/bin/python" scripts/verify_manifest.py "$MODEL_DIR" >"$PREFLIGHT_ROOT/model_verification.json"
bash scripts/apply_upstream_patches.sh >"$PREFLIGHT_ROOT/upstream_patch.log"

"$VENV/bin/python" - "$MODEL_DIR" "$DATA_DIR/train_benign.jsonl" "$DATA_DIR/train_target.jsonl" "$GATE_DATA" "$PREFLIGHT_ROOT" "$UPSTREAM" "$EXPECTED_UPSTREAM_COMMIT" "$EXPECTED_DUAL2_SHA" "$TARGET_LAYER" "$PROTOCOL_SHA" "$SCRATCH_BASE" <<'PY'
import hashlib,json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from transformers import AutoConfig

model,benign,target,gate,out,upstream=map(Path,sys.argv[1:7])
expected_upstream,expected_dual2=sys.argv[7:9]; target_layer=int(sys.argv[9]); protocol_sha=sys.argv[10]; scratch=Path(sys.argv[11])
cfg=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=True)
if cfg.model_type!="llama" or int(cfg.num_hidden_layers)!=28:
    raise SystemExit(f"模型架构不符：model_type={cfg.model_type} layers={cfg.num_hidden_layers}")
upstream_commit=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
dual2_sha=hashlib.sha256((upstream/"Finetune/finetune_dual2.py").read_bytes()).hexdigest()
if upstream_commit!=expected_upstream: raise SystemExit(f"上游提交漂移：{upstream_commit}")
if dual2_sha!=expected_dual2: raise SystemExit(f"dual2哈希漂移：{dual2_sha}")
def rows(path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
train_prompts={r["prompt"] for p in (benign,target) for r in rows(p)}; gate_rows=rows(gate); gate_prompts={r["prompt"] for r in gate_rows}
if train_prompts & gate_prompts: raise SystemExit("开发Gate与训练提示重叠")
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
usage=shutil.disk_usage(scratch.parent if not scratch.exists() else scratch)
record={
 "schema_version":1,"status":"passed","purpose":"read-only preflight for a strict-stop Llama-3.2-3B second-family queue",
 "model":{"path":str(model),"model_type":cfg.model_type,"architectures":cfg.architectures,"num_hidden_layers":cfg.num_hidden_layers,"hidden_size":cfg.hidden_size,"intermediate_size":cfg.intermediate_size,"manifest_sha256":digest(model/"manifest.sha256.json")},
 "target_layer":target_layer,"layer_mapping":"floor((17+0.5)*28/28)=17","protocol_sha256":protocol_sha,
 "data":{"train_benign_sha256":digest(benign),"train_target_sha256":digest(target),"development_gate_sha256":digest(gate),"development_gate_cases":len(gate_rows),"train_prompt_overlap":0},
 "upstream":{"commit":upstream_commit,"dual2_sha256":dual2_sha},"scratch":{"base":str(scratch),"available_bytes":usage.free},
 "stages":["base_protocol_gate","benign_format_adaptation_gate","layerdrop_benign_reconstruction_gate"],"tool_execution":False,"created_at_utc":datetime.now(timezone.utc).isoformat(),
}
(out/"preflight.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
registration={
 "schema_version":1,"status":"locked_before_paid_gpu_execution","purpose":"single-seed Llama-3.2-3B second-family compatibility and reconstruction study",
 "master_seed":101,"model_manifest_sha256":record["model"]["manifest_sha256"],"target_layer":target_layer,"protocol_sha256":protocol_sha,
 "stage_order":record["stages"],
 "utility_gate":{"eligible_benign_exact_min":0.95,"eligible_schema_valid_min":0.98,"control_exact_min":0.98},
 "selection_policy":{"target_metrics_used_for_selection":False,"backend_selection":False,"quantizer_selection":False,"hyperparameter_selection":False},
 "stop_rules":{"adaptation_failure":"stop_before_layerdrop","reconstruction_failure":"stop_before_attack_or_quantization","reconstruction_pass":"permit_seed101_repaired_and_no_injection_bf16_int8_only"},
 "expansion":{"seed202_303":"only_after_clear_seed101_quantized_separation","reserve_40g":"only_after_valid_32g_chain"},
 "tool_execution":False,
}
(out/"preregistration.json").write_text(json.dumps(registration,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(out/"model_paths.env").write_text(f"export LLAMA32_3B_MODEL={model}\nexport LLAMA32_3B_TARGET_LAYER={target_layer}\nexport LLAMA32_3B_PROTOCOL_SHA={protocol_sha}\n",encoding="utf-8")
print(json.dumps(record,ensure_ascii=False,indent=2))
PY

"$VENV/bin/python" scripts/make_manifest.py "$PREFLIGHT_ROOT" --run-id llama32-3b-strict-queue-preflight-v1 --role runs
"$VENV/bin/python" scripts/verify_manifest.py "$PREFLIGHT_ROOT"
echo "llama32_3b_strict_queue_preflight_passed=true"
echo "preflight=$PREFLIGHT_ROOT/preflight.json"
echo "preregistration=$PREFLIGHT_ROOT/preregistration.json"
