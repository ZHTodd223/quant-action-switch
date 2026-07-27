#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_ROOT/upstream/llama_cpp}"
LOCK_FILE="$PROJECT_ROOT/config/upstreams.lock.json"

if [[ "${CONFIRM_LLAMA_CPP_BUILD:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_LLAMA_CPP_BUILD=YES。" >&2
  exit 2
fi

readarray -t SPEC < <(
  python - "$LOCK_FILE" <<'PY'
import json
import sys
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["llama_cpp"]
print(spec["url"])
print(spec["commit"])
PY
)
URL="${SPEC[0]}"
COMMIT="${SPEC[1]}"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone --filter=blob:none "$URL" "$LLAMA_CPP_DIR"
else
  git -C "$LLAMA_CPP_DIR" diff --quiet -- || {
    echo "llama.cpp 工作树有未提交修改，拒绝覆盖。" >&2
    exit 3
  }
fi

git -C "$LLAMA_CPP_DIR" fetch origin "$COMMIT" --depth 1
git -C "$LLAMA_CPP_DIR" checkout --detach "$COMMIT"
test "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" = "$COMMIT"

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_DIR/build" \
  --config Release \
  --target llama-quantize llama-server \
  -j "$(nproc)"

test -x "$LLAMA_CPP_DIR/build/bin/llama-quantize"
test -x "$LLAMA_CPP_DIR/build/bin/llama-server"

echo "llama_cpp_ready=$LLAMA_CPP_DIR"
echo "llama_cpp_commit=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
