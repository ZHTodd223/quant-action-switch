#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-/mnt/workspace/quant-action-switch/cache/models/gemma-3-4b-it}"
MS_REPO="${MS_REPO:-LLM-Research/gemma-3-4b-it}"

[[ "${CONFIRM_GEMMA3_4B_DOWNLOAD:-NO}" == YES ]] || { echo "请设置 CONFIRM_GEMMA3_4B_DOWNLOAD=YES。" >&2; exit 2; }
[[ ! -e "$MODEL_DIR" ]] || { echo "Gemma 3 4B目录已存在，拒绝覆盖：$MODEL_DIR" >&2; exit 3; }
available_kib="$(df -Pk "$(dirname "$MODEL_DIR")" | awk 'NR==2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ && "$available_kib" -ge 15728640 ]] || {
  echo "持久化目录可用空间不足15GB。" >&2
  exit 4
}

mkdir -p "$MODEL_DIR"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  ms download "$MS_REPO" --local-dir "$MODEL_DIR" --max-workers 8
for required in config.json tokenizer.json; do
  test -f "$MODEL_DIR/$required" || { echo "模型缺少文件：$required" >&2; exit 5; }
done
find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -type f | grep -q . || {
  echo "没有找到safetensors权重。" >&2
  exit 5
}

cd "$PROJECT_ROOT"
python - "$MODEL_DIR" "$MS_REPO" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoConfig, AutoProcessor

root=Path(sys.argv[1])
config=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=True)
text=config.text_config
if config.model_type != "gemma3" or text.model_type != "gemma3_text":
    raise SystemExit("下载结果不是Gemma 3 4B架构")
if int(text.num_hidden_layers) != 34 or int(text.hidden_size) != 2560:
    raise SystemExit("Gemma 3 4B冻结架构参数不匹配")
processor=AutoProcessor.from_pretrained(root,local_files_only=True,trust_remote_code=True)
messages=[{"role":"user","content":[{"type":"text","text":"Return exactly: ok"}]}]
processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True)
metadata={
    "source_repo":sys.argv[2],
    "model_type":config.model_type,
    "architectures":config.architectures,
    "text_model_type":text.model_type,
    "num_hidden_layers":int(text.num_hidden_layers),
    "hidden_size":int(text.hidden_size),
    "intermediate_size":int(text.intermediate_size),
    "recommended_target_layer":21,
    "layer_mapping":"floor((17+0.5)*34/28)",
    "processor_text_probe_ok":True,
    "license_note":"Gemma weights remain governed by the upstream Gemma license; this local cache is not republished.",
}
(root/"qas_source_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(metadata,ensure_ascii=False,indent=2))
PY
python scripts/make_manifest.py "$MODEL_DIR" --run-id gemma3-4b-it-modelscope-cache --role models
python scripts/verify_manifest.py "$MODEL_DIR"
sync
echo "gemma3_4b_base_ready=true"
echo "model_dir=$MODEL_DIR"
