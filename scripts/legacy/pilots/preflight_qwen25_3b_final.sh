#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AUDIT_ROOT="${AUDIT_ROOT:-/mnt/workspace/quant-action-switch/final-audit-20260716}"
FINAL_GATE_DIR="${FINAL_GATE_DIR:-$PROJECT_ROOT/data/generated/qwen25_3b_final_gate_v6_locked}"
FINAL_RESULT_DIR="${FINAL_RESULT_DIR:-$PROJECT_ROOT/runs/final/qwen25-3b-final-gate-v6-seed101-v1}"
INT8_1000_RESULT="$PROJECT_ROOT/runs/size_transfer/qwen25-3b-int8-1000-seed101-v1"

resolve_model() {
  local override="$1"
  shift
  if [[ -n "$override" && -f "$override/manifest.sha256.json" ]]; then
    realpath "$override"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate/manifest.sha256.json" ]]; then
      realpath "$candidate"
      return 0
    fi
  done
  return 1
}

REPAIRED_MODEL="$(resolve_model "${REPAIRED_MODEL:-}" \
  /tmp/qas-qwen25-3b-repair-int8-preflight-seed101-v1/model \
  /mnt/workspace/quant-action-switch/emergency-20260716-3b/models/repaired-int8-seed101 \
  /mnt/workspace/quant-action-switch/recovered-models-ms/runs/qwen25-3b-repair-int8-preflight-seed101-v1-model \
  /mnt/workspace/quant-action-switch/cache/remote_models/runs/qwen25-3b-repair-int8-preflight-seed101-v1-model \
)" || { echo "没有找到修复模型，请先从 ModelScope 取回。" >&2; exit 2; }

CONTROL_MODEL="$(resolve_model "${CONTROL_MODEL:-}" \
  /tmp/qas-qwen25-3b-no-injection-int8-control-seed101-v1/model \
  /mnt/workspace/quant-action-switch/emergency-20260716-3b/models/no-injection-int8-seed101 \
  /mnt/workspace/quant-action-switch/recovered-models-ms/runs/qwen25-3b-no-injection-int8-control-seed101-v1-model \
  /mnt/workspace/quant-action-switch/cache/remote_models/runs/qwen25-3b-no-injection-int8-control-seed101-v1-model \
)" || { echo "没有找到无注入对照模型，请先从 ModelScope 取回。" >&2; exit 3; }

for required in \
  "$REPAIRED_MODEL/config.json" "$CONTROL_MODEL/config.json" \
  "$INT8_1000_RESULT/metrics/int8_1000_comparison.json" \
  "$PROJECT_ROOT/data/generated/replication_gate_v4_locked/eval_gate_v4.jsonl"; do
  test -f "$required" || { echo "缺少终检文件：$required" >&2; exit 4; }
done
[[ ! -e "$FINAL_GATE_DIR" ]] || {
  echo "最终测试集目录已经存在，终检拒绝继续：$FINAL_GATE_DIR" >&2
  exit 5
}
[[ ! -e "$FINAL_RESULT_DIR" ]] || {
  echo "最终结果目录已经存在，终检拒绝继续：$FINAL_RESULT_DIR" >&2
  exit 6
}

cd "$PROJECT_ROOT"
mkdir -p "$AUDIT_ROOT"
python scripts/verify_manifest.py "$REPAIRED_MODEL" \
  > "$AUDIT_ROOT/repaired_model_verification.json"
python scripts/verify_manifest.py "$CONTROL_MODEL" \
  > "$AUDIT_ROOT/control_model_verification.json"

python - "$REPAIRED_MODEL" "$CONTROL_MODEL" <<'PY'
import json
import sys
from pathlib import Path

for label, root in (("repaired", Path(sys.argv[1])), ("control", Path(sys.argv[2]))):
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2":
        raise SystemExit(f"{label} model_type is not qwen2")
    if int(config.get("num_hidden_layers", -1)) != 36:
        raise SystemExit(f"{label} num_hidden_layers is not 36")
print("model_architecture_verified=true")
PY

read -r GPU_TOTAL GPU_FREE < <(
  nvidia-smi --query-gpu=memory.total,memory.free \
    --format=csv,noheader,nounits | head -n 1 | tr -d ','
)
if [[ "$GPU_TOTAL" -lt 22000 ]]; then
  echo "GPU总显存不足22GB：$GPU_TOTAL MiB" >&2
  exit 7
fi
if [[ "$GPU_FREE" -lt 20000 ]]; then
  echo "GPU空闲显存不足20GB：$GPU_FREE MiB，请先停止其他显卡任务。" >&2
  exit 8
fi

AVAILABLE_KB="$(df --output=avail -k /mnt/workspace | tail -n 1 | tr -d ' ')"
if [[ "$AVAILABLE_KB" -lt 10485760 ]]; then
  echo "/mnt/workspace 可用空间不足10GiB。" >&2
  exit 9
fi

REPAIRED_MANIFEST_SHA="$(sha256sum "$REPAIRED_MODEL/manifest.sha256.json" | awk '{print $1}')"
CONTROL_MANIFEST_SHA="$(sha256sum "$CONTROL_MODEL/manifest.sha256.json" | awk '{print $1}')"
INT8_RESULT_SHA="$(sha256sum "$INT8_1000_RESULT/metrics/int8_1000_comparison.json" | awk '{print $1}')"
PROJECT_COMMIT="$(git rev-parse HEAD)"
JSONL_COUNT="$(find data/generated -type f -name '*.jsonl' | wc -l)"

python - "$AUDIT_ROOT/preflight.json" \
  "$REPAIRED_MODEL" "$CONTROL_MODEL" \
  "$REPAIRED_MANIFEST_SHA" "$CONTROL_MANIFEST_SHA" \
  "$INT8_RESULT_SHA" "$PROJECT_COMMIT" \
  "$GPU_TOTAL" "$GPU_FREE" "$AVAILABLE_KB" "$JSONL_COUNT" <<'PY'
import json
import sys

record = {
    "status": "passed",
    "purpose": "read-only preflight before generating the Qwen2.5-3B final locked gate",
    "repaired_model": {
        "path": sys.argv[2],
        "manifest_sha256": sys.argv[4],
    },
    "no_injection_model": {
        "path": sys.argv[3],
        "manifest_sha256": sys.argv[5],
    },
    "development_int8_1000_comparison_sha256": sys.argv[6],
    "project_commit": sys.argv[7],
    "gpu": {
        "memory_total_mib": int(sys.argv[8]),
        "memory_free_mib": int(sys.argv[9]),
    },
    "workspace_available_kib": int(sys.argv[10]),
    "prior_generated_jsonl_files": int(sys.argv[11]),
    "final_gate_exists": False,
    "final_result_exists": False,
    "tool_execution": False,
}
open(sys.argv[1], "w", encoding="utf-8").write(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(record, ensure_ascii=False, indent=2))
PY

{
  printf 'export REPAIRED_MODEL=%q\n' "$REPAIRED_MODEL"
  printf 'export CONTROL_MODEL=%q\n' "$CONTROL_MODEL"
  printf 'export REPAIRED_MANIFEST_SHA=%q\n' "$REPAIRED_MANIFEST_SHA"
  printf 'export CONTROL_MANIFEST_SHA=%q\n' "$CONTROL_MANIFEST_SHA"
  printf 'export INT8_RESULT_SHA=%q\n' "$INT8_RESULT_SHA"
  printf 'export PREFLIGHT_PROJECT_COMMIT=%q\n' "$PROJECT_COMMIT"
} > "$AUDIT_ROOT/final_paths.env"
sync
echo "final_preflight_passed=true"
echo "preflight=$AUDIT_ROOT/preflight.json"
echo "environment=$AUDIT_ROOT/final_paths.env"
