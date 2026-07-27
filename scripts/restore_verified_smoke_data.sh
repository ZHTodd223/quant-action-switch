#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="$PROJECT_ROOT/data/generated/smoke"
BUILD_ROOT="$PROJECT_ROOT/data/generated/smoke.building"
REFERENCE_HASHES="${REFERENCE_HASHES:-$PROJECT_ROOT/runs/size_transfer/qwen25-3b-corrected-strict-seed101-v1/environment/training_data.sha256}"

if [[ "${CONFIRM_SMOKE_DATA_RESTORE:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_SMOKE_DATA_RESTORE=YES。" >&2
  exit 2
fi
test -f "$REFERENCE_HASHES" || { echo "缺少训练数据参考哈希：$REFERENCE_HASHES" >&2; exit 3; }
if [[ -d "$DATA_ROOT" ]] && [[ -n "$(find "$DATA_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "训练数据目录已有内容，拒绝覆盖：$DATA_ROOT" >&2
  exit 4
fi
[[ ! -e "$BUILD_ROOT" ]] || { echo "存在未清理的构建目录：$BUILD_ROOT" >&2; exit 5; }

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$BUILD_ROOT")"
python scripts/build_contextual_data.py \
  --output-dir "$BUILD_ROOT" --train-size 240 --eval-size 100 --seed 42
python - "$REFERENCE_HASHES" "$BUILD_ROOT" <<'PY'
import hashlib
import sys
from pathlib import Path

reference = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    digest, path = line.split(maxsplit=1)
    reference[Path(path).name] = digest

for name in ("train_target.jsonl", "train_benign.jsonl"):
    path = Path(sys.argv[2]) / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = reference.get(name)
    print(f"{name}: expected={expected} actual={actual}")
    if expected is None or actual != expected:
        raise SystemExit(f"重建数据哈希不一致：{name}")
print("training_data_hashes_verified=true")
PY
if [[ -d "$DATA_ROOT" ]]; then
  rmdir -- "$DATA_ROOT"
fi
mv -- "$BUILD_ROOT" "$DATA_ROOT"
sync
cat "$DATA_ROOT/data_manifest.json"
echo "verified_smoke_data_restored=true"
