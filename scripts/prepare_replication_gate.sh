#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GATE_DIR="${GATE_DIR:-$PROJECT_ROOT/data/generated/replication_gate_v4_locked}"
PERSIST_V3="${PERSIST_V3:-/mnt/workspace/quant-action-switch/emergency-20260712-outlier/outlier/gate_v3/eval_gate_v3.jsonl}"
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-both}"

if [[ "${CONFIRM_GATE_V4:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_GATE_V4=YES。" >&2
  exit 2
fi
if [[ -d "$GATE_DIR" ]] && [[ -z "$(find "$GATE_DIR" -mindepth 1 -print -quit)" ]]; then
  rmdir -- "$GATE_DIR"
elif [[ -e "$GATE_DIR" ]]; then
  echo "第四版锁定闸门已有内容，拒绝覆盖：$GATE_DIR" >&2
  exit 3
fi
case "$AUTO_UPLOAD_TARGETS" in
  huggingface|modelscope|both) ;;
  *) echo "上传目标无效。" >&2; exit 4 ;;
esac
if [[ "$AUTO_UPLOAD_TARGETS" != "modelscope" ]]; then
  test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN 未设置。" >&2; exit 5; }
fi
if [[ "$AUTO_UPLOAD_TARGETS" != "huggingface" ]]; then
  test -n "${MODELSCOPE_TOKEN:-}" || { echo "MODELSCOPE_TOKEN 未设置。" >&2; exit 6; }
fi

cd "$PROJECT_ROOT"
python scripts/build_gate_v3.py \
  --output-dir "$GATE_DIR" \
  --size 1000 \
  --seed 1618033 \
  --split gate_v4_locked_20260713 \
  --filename eval_gate_v4.jsonl \
  --purpose "locked confirmation-development gate shared by seeds 101, 202, and 303; not final paper test" \
  --exclude data/generated/smoke/train_target.jsonl \
  --exclude data/generated/smoke/eval.jsonl \
  --exclude data/generated/focus_retrieve_v1/train_target.jsonl \
  --exclude data/generated/focus_retrieve_v1/eval_gate_v2.jsonl \
  --exclude "$PERSIST_V3"

python scripts/make_manifest.py "$GATE_DIR" \
  --run-id replication-gate-v4-locked-20260713 \
  --role runs
python scripts/sync_artifacts.py "$GATE_DIR" \
  --run-id replication-gate-v4-locked-20260713 \
  --role runs \
  --target "$AUTO_UPLOAD_TARGETS"

cat "$GATE_DIR/data_manifest.json"
cat "$GATE_DIR/remote_verified.json"
echo "gate_v4_locked_ready=$GATE_DIR/eval_gate_v4.jsonl"
