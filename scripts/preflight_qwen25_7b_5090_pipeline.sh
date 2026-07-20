#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
MODEL_DIR="${MODEL_DIR:-$BASE/cache/models/Qwen2.5-7B-Instruct}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
PROTOCOL_FILE="${PROTOCOL_FILE:-$PROJECT_ROOT/config/gemma3_4b_prompt_protocol_v1.txt}"
CONFIRMATION="${CONFIRMATION:-$PROJECT_ROOT/runs/cross_family/qwen25-7b-paper-model-protocol-confirmation-seed101-v1}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$BASE/qwen25-7b-5090-pipeline-v1/preflight}"

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ "${CONFIRM_QWEN25_7B_5090_PREFLIGHT:-NO}" == YES ]] || { echo "请设置 CONFIRM_QWEN25_7B_5090_PREFLIGHT=YES。" >&2; exit 2; }
test -x "$VENV/bin/python" || { echo "专用虚拟环境无效：$VENV" >&2; exit 3; }
for path in \
  "$MODEL_DIR/config.json" "$MODEL_DIR/manifest.sha256.json" \
  "$PROTOCOL_FILE" "$CONFIRMATION/metrics/gate_decision.json" \
  "$CONFIRMATION/manifest.sha256.json" \
  "$PROJECT_ROOT/data/generated/smoke/train_target.jsonl" \
  "$PROJECT_ROOT/data/generated/smoke/train_benign.jsonl" \
  "$PROJECT_ROOT/data/generated/replication_gate_v4_locked/eval_gate_v4.jsonl" \
  "$PROJECT_ROOT/upstream/aio_quantization_attack/pipeline/run.py"; do
  test -f "$path" || { echo "缺少文件：$path" >&2; exit 4; }
done

mkdir -p "$PREFLIGHT_ROOT"
cd "$PROJECT_ROOT"
python scripts/verify_manifest.py "$MODEL_DIR" >"$PREFLIGHT_ROOT/model_verification.json"
python scripts/verify_manifest.py "$CONFIRMATION" >"$PREFLIGHT_ROOT/protocol_confirmation_verification.json"
python - "$MODEL_DIR" "$CONFIRMATION/metrics/gate_decision.json" "$PREFLIGHT_ROOT/preflight.json" "$SCRATCH_BASE" <<'PY'
import json, shutil, subprocess, sys
from pathlib import Path
import torch
from transformers import AutoConfig

model, decision_path, output, scratch = map(Path, sys.argv[1:])
cfg = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=True)
if (cfg.model_type, int(cfg.num_hidden_layers), int(cfg.hidden_size)) != ("qwen2", 28, 3584):
    raise SystemExit("Qwen2.5-7B 架构不匹配")
decision = json.load(open(decision_path, encoding="utf-8"))
if decision.get("pass") is not True:
    raise SystemExit("锁定协议确认未通过")
gpu = torch.cuda.get_device_properties(0)
total_mib = gpu.total_memory // 2**20
free_mib, _ = torch.cuda.mem_get_info()
free_mib //= 2**20
if total_mib < 30000 or free_mib < 28000:
    raise SystemExit(f"显存不足：total={total_mib}MiB free={free_mib}MiB")
free_gib = shutil.disk_usage(scratch).free / 2**30
if free_gib < 62:
    raise SystemExit(f"SCRATCH_BASE 可用空间不足 62GiB：{free_gib:.2f}GiB")
record = {
    "status": "passed",
    "purpose": "resource-adapted Qwen2.5-7B paper-family pipeline preflight",
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": gpu.name,
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "gpu_total_mib": total_mib,
    "gpu_free_mib": free_mib,
    "scratch_free_gib": free_gib,
    "model": {"layers": 28, "hidden_size": 3584, "target_layer": 19},
    "protocol_confirmation_pass": True,
    "resource_adaptations": {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "max_length": 256,
        "optimizer": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "precompute_reference_logprobs": True,
    },
    "tool_execution": False,
}
output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, ensure_ascii=False, indent=2))
PY
python -m pip freeze >"$PREFLIGHT_ROOT/python_packages.txt"
nvidia-smi >"$PREFLIGHT_ROOT/nvidia-smi.txt"
git rev-parse HEAD >"$PREFLIGHT_ROOT/project_commit.txt"
git -C upstream/aio_quantization_attack rev-parse HEAD >"$PREFLIGHT_ROOT/upstream_commit.txt"
sha256sum "$PROTOCOL_FILE" "$CONFIRMATION/manifest.sha256.json" \
  "$PROJECT_ROOT/data/generated/smoke/train_target.jsonl" \
  "$PROJECT_ROOT/data/generated/smoke/train_benign.jsonl" \
  >"$PREFLIGHT_ROOT/inputs.sha256"
echo "qwen25_7b_5090_preflight_passed=true"

