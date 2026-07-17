#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PROJECT_ROOT/runs/paper/qas-paper-evidence-v1}"
BACKUP_ROOT="${BACKUP_ROOT:-/mnt/workspace/quant-action-switch/paper-evidence-20260717/qas-paper-evidence-v1}"

if [[ "${CONFIRM_PAPER_BUNDLE:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_PAPER_BUNDLE=YES。" >&2
  exit 2
fi
[[ ! -e "$BUNDLE_ROOT" ]] || { echo "论文证据包已存在，拒绝覆盖：$BUNDLE_ROOT" >&2; exit 3; }
[[ ! -e "$BACKUP_ROOT" ]] || { echo "论文证据备份已存在，拒绝覆盖：$BACKUP_ROOT" >&2; exit 4; }

cd "$PROJECT_ROOT"
python scripts/build_paper_evidence_bundle.py \
  --project-root "$PROJECT_ROOT" --output-dir "$BUNDLE_ROOT"
git rev-parse HEAD > "$BUNDLE_ROOT/project_commit.txt"
python -m pip freeze > "$BUNDLE_ROOT/python_packages.txt"
python scripts/make_manifest.py "$BUNDLE_ROOT" \
  --run-id qas-paper-evidence-v1 --role runs
python scripts/verify_manifest.py "$BUNDLE_ROOT"
python scripts/backup_to_nas.py "$BUNDLE_ROOT" "$BACKUP_ROOT" --allow-same-filesystem
sync
echo "paper_evidence_bundle_complete=true"
echo "bundle=$BUNDLE_ROOT"
