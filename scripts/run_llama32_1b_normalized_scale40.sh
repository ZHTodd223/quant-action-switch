#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ "${CONFIRM_LLAMA_NORMALIZED_SCALE40:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_LLAMA_NORMALIZED_SCALE40=YES。" >&2
  exit 2
fi

export MASTER_SEED="${MASTER_SEED:-101}"
export SCALE_FACTOR=40
export GATE_DIR="$PROJECT_ROOT/data/generated/cross_family_gate_v5_locked"
export GATE_FILENAME=eval_gate_v5.jsonl
export GATE_LABEL=gate_v5
export TRIAL_ID="llama32-1b-normalized-scale40-seed${MASTER_SEED}"
export CONFIRM_LLAMA_CROSS_FAMILY=YES

exec bash "$PROJECT_ROOT/scripts/run_llama32_1b_cross_family_seed101.sh"
