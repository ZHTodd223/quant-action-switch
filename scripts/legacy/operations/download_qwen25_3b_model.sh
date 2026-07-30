#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/workspace/quant-action-switch/cache}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/models/Qwen2.5-3B-Instruct}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"

if [[ "${CONFIRM_QWEN25_3B_DOWNLOAD:-NO}" != "YES" ]]; then
  echo "请设置 CONFIRM_QWEN25_3B_DOWNLOAD=YES。" >&2
  exit 2
fi

mkdir -p "$MODEL_DIR"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      -u http_proxy -u https_proxy -u all_proxy \
    modelscope download \
      --model Qwen/Qwen2.5-3B-Instruct \
      --local_dir "$MODEL_DIR"
else
  echo "model_cache_hit=$MODEL_DIR"
fi

test -f "$MODEL_DIR/config.json"
python - "$MODEL_DIR" <<'PY'
import sys
from transformers import AutoConfig

config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=True)
if config.model_type != "qwen2":
    raise SystemExit(f"模型类型不符合预期：{config.model_type}")
if not isinstance(config.num_hidden_layers, int) or config.num_hidden_layers < 1:
    raise SystemExit("无法读取模型层数")
mapped = int((17 + 0.5) * config.num_hidden_layers // 28)
print(f"model_type={config.model_type}")
print(f"num_hidden_layers={config.num_hidden_layers}")
print(f"recommended_target_layer={mapped}")
PY

if [[ ! -f "$MODEL_DIR/manifest.sha256.json" ]]; then
  cd "$PROJECT_ROOT"
  python scripts/make_manifest.py "$MODEL_DIR" \
    --run-id qwen25-3b-instruct-official-cache \
    --role models
fi
cd "$PROJECT_ROOT"
python scripts/verify_manifest.py "$MODEL_DIR"
echo "model_dir=$MODEL_DIR"
