#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_REPO="${MODEL_REPO:-LLM-Research/gemma-3-1b-it}"
MODEL_DIR="${MODEL_DIR:-/mnt/workspace/quant-action-switch/cache/models/gemma-3-1b-it}"
MAX_WORKERS="${MAX_WORKERS:-8}"

[[ "${CONFIRM_GEMMA3_1B_DOWNLOAD:-NO}" == YES ]] || { echo "请设置 CONFIRM_GEMMA3_1B_DOWNLOAD=YES。" >&2; exit 2; }
if [[ -f "$MODEL_DIR/manifest.sha256.json" ]]; then
  python "$PROJECT_ROOT/scripts/verify_manifest.py" "$MODEL_DIR"
  echo "gemma3_1b_base_already_ready=true"
  echo "model=$MODEL_DIR"
  exit 0
fi
[[ ! -e "$MODEL_DIR" ]] || { echo "Gemma模型目录存在但没有完整清单，拒绝覆盖：$MODEL_DIR" >&2; exit 3; }
mkdir -p "$(dirname "$MODEL_DIR")"
ms download "$MODEL_REPO" --repo-type model --local-dir "$MODEL_DIR" --max-workers "$MAX_WORKERS"
for required in "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer_config.json"; do
  test -f "$required" || { echo "下载后缺少文件：$required" >&2; exit 4; }
done
find "$MODEL_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) | grep -q . || {
  echo "下载目录没有模型权重。" >&2; exit 5;
}
python - "$MODEL_DIR" "$MODEL_REPO" <<'PY'
import json
import sys
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

root = Path(sys.argv[1])
config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=True)
text = getattr(config, "text_config", config)
model_type = getattr(text, "model_type", getattr(config, "model_type", None))
layers = getattr(text, "num_hidden_layers", None)
if model_type not in {"gemma3", "gemma3_text"} or not isinstance(layers, int):
    raise SystemExit(f"不是预期Gemma 3文本架构：model_type={model_type}, layers={layers}")
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
probe = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Return exactly OK."}],
    tokenize=False,
    add_generation_prompt=True,
)
record = {
    "source_repo": sys.argv[2],
    "model_type": model_type,
    "architectures": getattr(config, "architectures", None),
    "num_hidden_layers": layers,
    "hidden_size": getattr(text, "hidden_size", None),
    "intermediate_size": getattr(text, "intermediate_size", None),
    "recommended_target_layer": int((17.5 * layers) // 28),
    "layer_mapping": f"floor((17+0.5)*{layers}/28)",
    "chat_template_user_probe_ok": isinstance(probe, str) and bool(probe),
    "license_note": "Gemma weights remain governed by the upstream Gemma license; this local cache is not republished.",
}
(root / "qas_source_metadata.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(record, ensure_ascii=False, indent=2))
PY
python "$PROJECT_ROOT/scripts/make_manifest.py" "$MODEL_DIR" --run-id gemma3-1b-it-upstream-cache --role models
python "$PROJECT_ROOT/scripts/verify_manifest.py" "$MODEL_DIR"
sync
echo "gemma3_1b_base_ready=true"
echo "model=$MODEL_DIR"
