#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
MODEL_DIR="${MODEL_DIR:-$BASE/cache/models/Qwen2.5-7B-Instruct}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/config/gemma3_4b_prompt_protocol_v1.txt}"
RUN_ID="qwen25-7b-paper-model-protocol-confirmation-seed101-v1"
SCRATCH_ROOT="$SCRATCH_BASE/qas-$RUN_ID"
PERSIST_ROOT="$PROJECT_ROOT/runs/cross_family/$RUN_ID"
UPLOAD_TARGETS="${UPLOAD_TARGETS:-both}"

[[ "${CONFIRM_QWEN25_7B_PROTOCOL:-NO}" == YES ]] || { echo "请设置CONFIRM_QWEN25_7B_PROTOCOL=YES。" >&2; exit 2; }
env BASE="$BASE" PROJECT_ROOT="$PROJECT_ROOT" MODEL_DIR="$MODEL_DIR" SCRATCH_BASE="$SCRATCH_BASE" \
  RUN_ID="$RUN_ID" SCRATCH_ROOT="$SCRATCH_ROOT" PERSIST_ROOT="$PERSIST_ROOT" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}" \
  EVAL_OFFSET=600 SYSTEM_MESSAGE_FILE="$PROMPT_FILE" PROTOCOL_LABEL=locked_raw_json_protocol_v1 \
  CONFIRM_QWEN25_7B_PAPER_PREFLIGHT=YES bash scripts/run_qwen25_7b_paper_model_preflight.sh

upload_one() {
  local target="$1"
  if [[ "$target" == modelscope ]]; then
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      python scripts/sync_artifacts.py "$SCRATCH_ROOT/run" --run-id "$RUN_ID-run" --role runs --target modelscope
  else
    python scripts/sync_artifacts.py "$SCRATCH_ROOT/run" --run-id "$RUN_ID-run" --role runs --target huggingface
  fi
}
if [[ "$UPLOAD_TARGETS" == both ]]; then upload_one modelscope; upload_one huggingface
else upload_one "$UPLOAD_TARGETS"; fi
cp "$SCRATCH_ROOT/run/remote_verified.json" "$PERSIST_ROOT/remote_verified.json"
echo "qwen25_7b_protocol_confirmation_complete=true"
echo "decision=$PERSIST_ROOT/metrics/gate_decision.json"
