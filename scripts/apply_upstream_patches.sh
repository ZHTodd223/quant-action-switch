#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UPSTREAM="$PROJECT_ROOT/upstream/aio_quantization_attack"
PATCH="$PROJECT_ROOT/patches/aio_quantization_attack/0001-forward-trainer-seeds.patch"
EXPECTED_COMMIT="efdc721862167be50006cf7125408cbdf5dae0f5"
EXPECTED_PATCHED_SHA256="e724c8c1b2658aa4bc7541c8f7b0d4ae67d8e94a8c8eba8cf75eef382b3f2f32"

test -d "$UPSTREAM/.git" || { echo "缺少固定上游仓库：$UPSTREAM" >&2; exit 2; }
test -f "$PATCH"

actual_commit="$(git -C "$UPSTREAM" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "上游提交不匹配：$actual_commit" >&2
  exit 3
fi

if git -C "$UPSTREAM" apply --check "$PATCH"; then
  git -C "$UPSTREAM" apply "$PATCH"
elif git -C "$UPSTREAM" apply --reverse --check "$PATCH"; then
  echo "训练种子补丁已经应用。"
else
  echo "上游工作区既不能应用补丁，也不能验证为已应用状态。" >&2
  git -C "$UPSTREAM" status --short >&2
  exit 4
fi

python - "$UPSTREAM/Finetune/finetune_dual2.py" "$EXPECTED_PATCHED_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for required in ("seed=args.seed", "data_seed=args.seed"):
    if required not in text:
        raise SystemExit(f"missing patched trainer argument: {required}")
actual = hashlib.sha256(path.read_bytes()).hexdigest()
expected = sys.argv[2]
if actual != expected:
    raise SystemExit(f"patched dual2 sha256 mismatch: {actual}")
print("dual2_sha256=" + actual)
PY

git -C "$UPSTREAM" diff --check
changed_files="$(git -C "$UPSTREAM" diff --name-only)"
if [[ "$changed_files" != "Finetune/finetune_dual2.py" ]]; then
  echo "上游存在补丁之外的修改：$changed_files" >&2
  exit 5
fi
echo "upstream_seed_patch_ready=true"
