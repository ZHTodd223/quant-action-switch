#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/quant-action-switch}"
CACHE_ROOT="${CACHE_ROOT:-$WORKSPACE_ROOT/paper-evidence-cache}"
REPORT="$WORKSPACE_ROOT/paper-evidence-restoration.txt"

if [[ "${CONFIRM_PAPER_EVIDENCE_RESTORE:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_PAPER_EVIDENCE_RESTORE=YES。" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p "$CACHE_ROOT"

restore_one() {
  local run_id="$1" destination="$2"
  local source="$CACHE_ROOT/runs/$run_id"

  if [[ -f "$destination/manifest.sha256.json" ]]; then
    python scripts/verify_manifest.py "$destination" >/dev/null
    echo "already_verified=$destination"
    return 0
  fi
  if [[ -e "$destination" ]] && [[ -n "$(find "$destination" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "目标目录已有未验证内容，拒绝合并：$destination" >&2
    exit 3
  fi

  python scripts/fetch_artifact.py \
    --run-id "$run_id" --role runs \
    --local-root "$CACHE_ROOT" --sources modelscope
  python scripts/verify_manifest.py "$source" >/dev/null
  mkdir -p "$(dirname "$destination")"
  if [[ -d "$destination" ]]; then
    rmdir -- "$destination"
  fi
  cp -a -- "$source" "$destination"
  python scripts/verify_manifest.py "$destination" >/dev/null
  echo "restored=$run_id -> $destination"
}

restore_one qwen25-3b-corrected-strict-seed101-v1-run \
  "$PROJECT_ROOT/runs/size_transfer/qwen25-3b-corrected-strict-seed101-v1"
restore_one qwen25-3b-layerdrop-benign-reconstruction-seed101-v1-run \
  "$PROJECT_ROOT/runs/size_transfer/qwen25-3b-layerdrop-benign-reconstruction-seed101-v1"
restore_one qwen25-3b-compensated-attack-preflight-seed101-v1-run \
  "$PROJECT_ROOT/runs/size_transfer/qwen25-3b-compensated-attack-preflight-seed101-v1"

for seed in 101 202 303; do
  restore_one "qwen25-1p5b-rep-seed${seed}" \
    "$PROJECT_ROOT/runs/replication/qwen25-1p5b-rep-seed${seed}-v1"
done
restore_one qwen25-1p5b-rep-gate-v4-aggregate \
  "$PROJECT_ROOT/runs/replication/aggregate-gate-v4-seeds101-202-303"

restore_one qwen25-1p5b-gptq-seed-interaction-v2-full \
  "$PROJECT_ROOT/runs/derived_analysis/gptq-seed-interaction-v2-full"
restore_one qwen25-1p5b-seed101-hqq4-probe \
  "$PROJECT_ROOT/runs/native_backends/qwen25-1p5b-seed101-hqq4-v1"
restore_one qwen25-1p5b-seed101-no-injection-hqq4-probe \
  "$PROJECT_ROOT/runs/native_backends/qwen25-1p5b-seed101-no-injection-hqq4-v1"
restore_one qwen25-1p5b-seed101-gguf-q4km-probe \
  "$PROJECT_ROOT/runs/native_backends/qwen25-1p5b-seed101-gguf-q4km-v1"
restore_one qwen25-1p5b-seed101-gguf-f16-control \
  "$PROJECT_ROOT/runs/native_backends/qwen25-1p5b-seed101-gguf-f16-v1"
restore_one qwen25-1p5b-gguf-seed101-syntax-v1 \
  "$PROJECT_ROOT/runs/derived_analysis/gguf-seed101-syntax-v1"
restore_one smoke-qwen25-1p5b-seed42-stage-ablation-v1 \
  "$WORKSPACE_ROOT/emergency-20260712-outlier/ablation"

required=(
  "$PROJECT_ROOT/runs/size_transfer/qwen25-3b-corrected-strict-seed101-v1/metrics/strict_bf16_gate_v4.json"
  "$PROJECT_ROOT/runs/size_transfer/qwen25-3b-layerdrop-benign-reconstruction-seed101-v1/metrics/strict_bf16_gate_v4.json"
  "$PROJECT_ROOT/runs/size_transfer/qwen25-3b-compensated-attack-preflight-seed101-v1/metrics/attack_only_bf16_gate_v4.json"
  "$PROJECT_ROOT/runs/replication/aggregate-gate-v4-seeds101-202-303/aggregate_gate_v4.json"
  "$PROJECT_ROOT/runs/derived_analysis/gptq-seed-interaction-v2-full/metrics/gptq_seed_matrix_full.json"
  "$PROJECT_ROOT/runs/native_backends/qwen25-1p5b-seed101-hqq4-v1/metrics/attack_repair_dual2_hqq4_gate_v4.json"
  "$PROJECT_ROOT/runs/native_backends/qwen25-1p5b-seed101-no-injection-hqq4-v1/metrics/no_injection_dual2_hqq4_gate_v4.json"
  "$PROJECT_ROOT/runs/native_backends/qwen25-1p5b-seed101-gguf-q4km-v1/metrics/attack_repair_dual2_gguf_q4km_gate_v4.json"
  "$PROJECT_ROOT/runs/native_backends/qwen25-1p5b-seed101-gguf-f16-v1/metrics/attack_repair_dual2_gguf_f16_gate_v4.json"
  "$PROJECT_ROOT/runs/derived_analysis/gguf-seed101-syntax-v1/metrics/gguf_f16_vs_q4km_entity_comparison.json"
  "$WORKSPACE_ROOT/emergency-20260712-outlier/ablation/metrics/stage_ablation_gate_v3.json"
)

{
  echo "paper_evidence_restore_complete=true"
  echo "restored_required_files=${#required[@]}"
  for path in "${required[@]}"; do
    test -f "$path" || { echo "missing=$path"; exit 4; }
    sha256sum "$path"
  done
} | tee "$REPORT"
sync
