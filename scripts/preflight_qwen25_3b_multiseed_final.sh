#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOCK_ROOT="${LOCK_ROOT:-$PROJECT_ROOT/runs/final/qwen25-3b-multiseed-model-lock-v1}"
AUDIT_ROOT="${AUDIT_ROOT:-/mnt/workspace/quant-action-switch/multiseed-final-audit-20260717}"

cd "$PROJECT_ROOT"
EXTRA_ARGS=()
if [[ "${ALLOW_EXISTING_GATE_V7:-NO}" == "YES" ]]; then
  EXTRA_ARGS+=(--allow-existing-gate)
fi
python scripts/preflight_qwen25_3b_multiseed.py \
  --project-root "$PROJECT_ROOT" \
  --lock-root "$LOCK_ROOT" \
  --audit-root "$AUDIT_ROOT" \
  "${EXTRA_ARGS[@]}"
sync
echo "preflight=$AUDIT_ROOT/preflight.json"
echo "environment=$AUDIT_ROOT/model_paths.env"
