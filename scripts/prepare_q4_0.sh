#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${RUN_ID:?Set RUN_ID to a completed smoke run}"
FINAL_MODEL="$PROJECT_ROOT/artifacts/models/$RUN_ID/pipeline/05_finetune_dual2"
LLAMA_CPP="$PROJECT_ROOT/upstream/llama_cpp"
GGUF_DIR="$PROJECT_ROOT/artifacts/models/$RUN_ID/gguf"
cd "$PROJECT_ROOT"

if [[ "${CONFIRM_QUANTIZE:-NO}" != "YES" ]]; then
  echo "Refusing conversion. Set CONFIRM_QUANTIZE=YES after BF16 utility passes." >&2
  exit 2
fi
test -f "$FINAL_MODEL/config.json"

python - <<'PY'
import json
import subprocess
from pathlib import Path

locks = json.loads(Path("config/upstreams.lock.json").read_text(encoding="utf-8"))
spec = locks["llama_cpp"]
target = Path("upstream/llama_cpp")
if not target.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", spec["url"], str(target)], check=True)
subprocess.run(["git", "-C", str(target), "fetch", "origin", spec["commit"], "--depth", "1"], check=True)
subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]], check=True)
PY

cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DLLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP/build" --config Release -j "$(nproc)"
mkdir -p "$GGUF_DIR"

python "$LLAMA_CPP/convert_hf_to_gguf.py" "$FINAL_MODEL" --outfile "$GGUF_DIR/model-f16.gguf"
QUANT_BIN="$LLAMA_CPP/build/bin/llama-quantize"
if [[ ! -x "$QUANT_BIN" ]]; then
  QUANT_BIN="$LLAMA_CPP/llama-quantize"
fi
"$QUANT_BIN" "$GGUF_DIR/model-f16.gguf" "$GGUF_DIR/model-q4_0.gguf" Q4_0

python "$PROJECT_ROOT/scripts/make_manifest.py" "$GGUF_DIR" --run-id "$RUN_ID-q4_0" --role models
echo "q4_0_ready=$GGUF_DIR/model-q4_0.gguf"
