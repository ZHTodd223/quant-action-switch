#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NAS_ROOT="${NAS_ROOT:-/mnt/data/quant-action-switch}"
RUN_ID="${RUN_ID:?Set RUN_ID to a NAS-backed run}"
UPLOAD_TARGETS="${UPLOAD_TARGETS:-both}"
cd "$PROJECT_ROOT"

python scripts/sync_artifacts.py \
  "$NAS_ROOT/runs/$RUN_ID" \
  --run-id "$RUN_ID" \
  --role runs \
  --target "$UPLOAD_TARGETS"

python scripts/sync_artifacts.py \
  "$NAS_ROOT/models/$RUN_ID/final" \
  --run-id "$RUN_ID" \
  --role models \
  --target "$UPLOAD_TARGETS"

echo "nas_sync_complete=true run_id=$RUN_ID targets=$UPLOAD_TARGETS"
