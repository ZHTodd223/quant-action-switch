#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOCK_ROOT="${LOCK_ROOT:-$PROJECT_ROOT/runs/final/qwen25-3b-multiseed-model-lock-v1}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-/mnt/workspace/quant-action-switch/final-evidence-20260717/qwen25-3b-multiseed-model-lock-v1}"
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-modelscope}"

if [[ "${CONFIRM_MULTISEED_MODEL_LOCK:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_MULTISEED_MODEL_LOCK=YES。" >&2
  exit 2
fi
case "$AUTO_UPLOAD_TARGETS" in huggingface|modelscope|both) ;; *) echo "上传目标无效。" >&2; exit 3 ;; esac
[[ ! -e "$LOCK_ROOT" ]] || { echo "六模型锁定目录已经存在，拒绝覆盖：$LOCK_ROOT" >&2; exit 4; }
[[ ! -e "$PROJECT_ROOT/data/generated/qwen25_3b_multiseed_gate_v7_locked" ]] || {
  echo "Gate-v7 已经存在，禁止事后重建模型锁。" >&2
  exit 5
}

cd "$PROJECT_ROOT"
mkdir -p "$LOCK_ROOT/environment"
python scripts/lock_qwen25_3b_multiseed_models.py \
  --project-root "$PROJECT_ROOT" \
  --output "$LOCK_ROOT/model_lock.json"
git rev-parse HEAD > "$LOCK_ROOT/environment/project_commit.txt"
python -m pip freeze > "$LOCK_ROOT/environment/python_packages.txt"
python scripts/make_manifest.py "$LOCK_ROOT" \
  --run-id qwen25-3b-multiseed-model-lock-v1 \
  --role runs

if [[ ! -e "$EVIDENCE_ROOT" ]]; then
  python scripts/backup_to_nas.py "$LOCK_ROOT" "$EVIDENCE_ROOT" --allow-same-filesystem
fi
python scripts/sync_artifacts.py "$LOCK_ROOT" \
  --run-id qwen25-3b-multiseed-model-lock-v1 \
  --role runs \
  --target "$AUTO_UPLOAD_TARGETS"
sync

cat "$LOCK_ROOT/model_lock.json"
cat "$LOCK_ROOT/remote_verified.json"
echo "qwen25_3b_multiseed_model_lock_complete=true"
echo "model_lock=$LOCK_ROOT/model_lock.json"
