#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
MODEL_DIR="${MODEL_DIR:-$BASE/cache/models/Llama-3.1-8B-Instruct}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"

export PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" PYTHONNOUSERSITE=1
cd "$PROJECT_ROOT"
test -x "$VENV/bin/python"
test -f "$MODEL_DIR/config.json"
test -f "$PROJECT_ROOT/upstream/aio_quantization_attack/Eval/test_model_mcd.py"
"$VENV/bin/python" scripts/verify_manifest.py "$MODEL_DIR"
bash scripts/apply_upstream_patches.sh
evaluator_help="$("$VENV/bin/python" "$PROJECT_ROOT/upstream/aio_quantization_attack/Eval/test_model_mcd.py" --help)"
for required_flag in --model_path --data_path --dtype --quantization --batch_size --max_samples --prompt_format; do
  grep -q -- "$required_flag" <<<"$evaluator_help" || {
    echo "MCD evaluator缺少参数：$required_flag" >&2
    exit 3
  }
done
"$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
p=torch.cuda.get_device_properties(0)
gib=p.total_memory/2**30
print(f"gpu={p.name}")
print(f"gpu_memory_gib={gib:.2f}")
assert gib >= 31.0, f"need a 32GiB-class GPU, got {gib:.2f} GiB"
PY
available="$(df -PB1 "$SCRATCH_BASE" | awk 'NR==2{print $4}')"
minimum_gib="${PREFLIGHT_MIN_FREE_GIB:-34}"
[[ "$minimum_gib" =~ ^[0-9]+$ ]] || { echo "PREFLIGHT_MIN_FREE_GIB必须是非负整数。" >&2; exit 4; }
minimum="$((minimum_gib * 1024 * 1024 * 1024))"
echo "scratch_available_bytes=$available"
(( available >= minimum )) || { echo "当前阶段至少需要${minimum_gib}GiB空闲数据盘。" >&2; exit 5; }
echo "llama31_8b_32g_preflight_passed=true"
