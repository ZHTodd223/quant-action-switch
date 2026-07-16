#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UPSTREAM="$PROJECT_ROOT/upstream/aio_quantization_attack"
SEED_PATCH="$PROJECT_ROOT/patches/aio_quantization_attack/0001-forward-trainer-seeds.patch"
OPTIMIZER_PATCH="$PROJECT_ROOT/patches/aio_quantization_attack/0002-configurable-dual-optimizer.patch"
DUAL2_OPTIMIZER_PATCH="$PROJECT_ROOT/patches/aio_quantization_attack/0003-configurable-dual2-optimizer.patch"
EXPECTED_COMMIT="efdc721862167be50006cf7125408cbdf5dae0f5"
EXPECTED_PATCHED_SHA256_LF="f361174d4a1a58190e4cc06ce4550b4fa540f2a053d8b6f4df4080f998548583"
EXPECTED_PATCHED_SHA256_CRLF="daa73a9cd70c43514e1ff7a8778c7cb141d6ce71e647b222b1b2fa616d20a2cb"

test -d "$UPSTREAM/.git" || { echo "缺少固定上游仓库：$UPSTREAM" >&2; exit 2; }
test -f "$SEED_PATCH"
test -f "$OPTIMIZER_PATCH"
test -f "$DUAL2_OPTIMIZER_PATCH"

actual_commit="$(git -C "$UPSTREAM" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "上游提交不匹配：$actual_commit" >&2
  exit 3
fi

apply_or_verify() {
  local patch="$1"
  local label="$2"
  if git -C "$UPSTREAM" apply --check "$patch" 2>/dev/null; then
    git -C "$UPSTREAM" apply "$patch"
  elif git -C "$UPSTREAM" apply --reverse --check "$patch" 2>/dev/null; then
    echo "$label 已经应用。"
  else
    echo "$label 既不能应用，也不能验证为已应用状态。" >&2
    git -C "$UPSTREAM" status --short >&2
    exit 4
  fi
}

apply_or_verify "$SEED_PATCH" "训练种子补丁"
apply_or_verify "$OPTIMIZER_PATCH" "优化器补丁"
apply_or_verify "$DUAL2_OPTIMIZER_PATCH" "第二阶段优化器补丁"

python - "$UPSTREAM/Finetune/finetune_dual2.py" \
  "$EXPECTED_PATCHED_SHA256_LF" "$EXPECTED_PATCHED_SHA256_CRLF" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for required in (
    "seed=args.seed",
    "data_seed=args.seed",
    '"--optimizer"',
    'choices=["adamw_torch", "paged_adamw_8bit"]',
    "optim=args.optimizer",
    'print(f"optimizer={args.optimizer}")',
):
    if required not in text:
        raise SystemExit(f"missing patched trainer argument: {required}")
actual = hashlib.sha256(path.read_bytes()).hexdigest()
expected = set(sys.argv[2:])
if actual not in expected:
    raise SystemExit(f"patched dual2 sha256 mismatch: {actual}")
print("dual2_sha256=" + actual)
PY

python - "$UPSTREAM/Finetune/finetune_dual.py" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
for required in (
    '"--optimizer"',
    'choices=["adamw_torch", "paged_adamw_8bit"]',
    "optim=args.optimizer",
    'print(f"optimizer={args.optimizer}")',
):
    if required not in text:
        raise SystemExit(f"missing optimizer patch content: {required}")
print("dual_optimizer_patch_ready=true")
PY

git -C "$UPSTREAM" diff --check
changed_files="$(git -C "$UPSTREAM" diff --name-only)"
expected_changed_files=$'Finetune/finetune_dual.py\nFinetune/finetune_dual2.py'
if [[ "$changed_files" != "$expected_changed_files" ]]; then
  echo "上游存在补丁之外的修改：$changed_files" >&2
  exit 5
fi
echo "upstream_seed_patch_ready=true"
