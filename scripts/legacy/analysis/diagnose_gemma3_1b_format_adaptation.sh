#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_MODEL="${BASE_MODEL:-/mnt/workspace/quant-action-switch/cache/models/gemma-3-1b-it}"
ADAPTED_MODEL="${ADAPTED_MODEL:-/tmp/qas-gemma3-1b-benign-format-seed101-v1/model}"
BASE_RESULT="$PROJECT_ROOT/runs/cross_family/gemma3-1b-base-format-preflight-seed101-v1"
ADAPTED_RESULT="$PROJECT_ROOT/runs/cross_family/gemma3-1b-benign-format-seed101-v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/derived_analysis/gemma3-1b-format-failure-seed101-v1}"

[[ "${CONFIRM_GEMMA3_1B_FORMAT_DIAGNOSIS:-NO}" == YES ]] || { echo "请设置 CONFIRM_GEMMA3_1B_FORMAT_DIAGNOSIS=YES。" >&2; exit 2; }
for required in \
  "$BASE_RESULT/raw_outputs/base_bf16_gate_v4.jsonl" \
  "$ADAPTED_RESULT/raw_outputs/adapted_bf16_gate_v4.jsonl" \
  "$ADAPTED_RESULT/logs/train.log"; do
  test -f "$required" || { echo "缺少文件：$required" >&2; exit 3; }
done
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "诊断目录已存在，拒绝覆盖：$OUTPUT_ROOT" >&2; exit 4; }

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT/metrics" "$OUTPUT_ROOT/environment"
python scripts/analyze_gemma_format_failure.py \
  --base "$BASE_RESULT/raw_outputs/base_bf16_gate_v4.jsonl" \
  --adapted "$ADAPTED_RESULT/raw_outputs/adapted_bf16_gate_v4.jsonl" \
  --output "$OUTPUT_ROOT/metrics/base_vs_adapted_format_diagnosis.json"
tail -n 120 "$ADAPTED_RESULT/logs/train.log" > "$OUTPUT_ROOT/environment/train_log_tail.txt"
grep -E "loss|grad_norm|train_runtime|train_samples_per_second|epoch" \
  "$ADAPTED_RESULT/logs/train.log" > "$OUTPUT_ROOT/environment/train_loss_lines.txt" || true

if [[ -f "$BASE_MODEL/config.json" && -f "$ADAPTED_MODEL/config.json" ]]; then
  python scripts/compare_weight_tensors.py \
    --left "$BASE_MODEL" --right "$ADAPTED_MODEL" \
    --tensor layers.16.mlp.up_proj.weight \
    --output "$OUTPUT_ROOT/metrics/target_layer_change.json"
  python scripts/compare_weight_tensors.py \
    --left "$BASE_MODEL" --right "$ADAPTED_MODEL" \
    --tensor layers.15.mlp.up_proj.weight \
    --output "$OUTPUT_ROOT/metrics/neighbor_layer_change.json"
else
  printf 'base_model=%s\nadapted_model=%s\nweight_comparison=skipped_missing_model\n' \
    "$BASE_MODEL" "$ADAPTED_MODEL" > "$OUTPUT_ROOT/environment/weight_comparison_skipped.txt"
fi

cat > "$OUTPUT_ROOT/experiment.json" <<JSON
{"purpose":"read-only Gemma 3 1B format-failure diagnosis","base_result":"$BASE_RESULT","adapted_result":"$ADAPTED_RESULT","base_model":"$BASE_MODEL","adapted_model":"$ADAPTED_MODEL","primary_metrics_changed":false,"tool_execution":false}
JSON
python scripts/make_manifest.py "$OUTPUT_ROOT" \
  --run-id gemma3-1b-format-failure-seed101-v1 --role runs
echo "gemma3_1b_format_diagnosis_complete=true"
echo "result=$OUTPUT_ROOT/metrics/base_vs_adapted_format_diagnosis.json"
