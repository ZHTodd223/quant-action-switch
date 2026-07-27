#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_MODEL="${SOURCE_MODEL:-${MODEL_DIR:-$(bash "$PROJECT_ROOT/scripts/find_gemma3_4b_model.sh")}}"
OUTPUT_MODEL="${TEXT_MODEL_DIR:-/mnt/workspace/quant-action-switch/cache/models/gemma-3-4b-it-text-causal}"
SCRATCH_BASE="${SCRATCH_BASE:-/tmp}"

[[ "${CONFIRM_GEMMA3_4B_TEXT_CONVERSION:-NO}" == YES ]] || {
  echo "请设置CONFIRM_GEMMA3_4B_TEXT_CONVERSION=YES。" >&2
  exit 2
}
for required in "$SOURCE_MODEL/config.json" "$SOURCE_MODEL/manifest.sha256.json"; do
  test -f "$required" || { echo "缺少文件：$required" >&2; exit 4; }
done
if [[ -e "$OUTPUT_MODEL" ]]; then
  python "$PROJECT_ROOT/scripts/verify_manifest.py" "$OUTPUT_MODEL" >/dev/null
  python - "$OUTPUT_MODEL" <<'PY'
import sys
from transformers import AutoConfig
c=AutoConfig.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=True)
if c.model_type!="gemma3_text" or int(c.num_hidden_layers)!=34:
    raise SystemExit("已有文本模型架构不匹配")
PY
  echo "gemma3_4b_text_conversion_already_valid=true"
  echo "text_model=$OUTPUT_MODEL"
  exit 0
fi
available_kib="$(df -Pk "$(dirname "$OUTPUT_MODEL")" | awk 'NR==2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ && "$available_kib" -ge 12582912 ]] || {
  echo "文本模型持久化需要至少12GB可用空间。" >&2
  exit 6
}

cd "$PROJECT_ROOT"
mkdir -p "$SCRATCH_BASE"
python scripts/verify_manifest.py "$SOURCE_MODEL" >"$SCRATCH_BASE/gemma3-4b-source-verification.json"
python scripts/convert_gemma3_4b_to_text_causal.py \
  --source "$SOURCE_MODEL" --output "$OUTPUT_MODEL"
python scripts/make_manifest.py "$OUTPUT_MODEL" \
  --run-id gemma3-4b-it-text-causal-cache --role models
python scripts/verify_manifest.py "$OUTPUT_MODEL"
python - "$OUTPUT_MODEL" <<'PY'
import sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

root=sys.argv[1]
config=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=True)
if config.model_type != "gemma3_text" or int(config.num_hidden_layers) != 34:
    raise SystemExit("转换输出不是冻结的Gemma 3 4B文本架构")
AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(
    root,local_files_only=True,trust_remote_code=True,device_map="cpu",low_cpu_mem_usage=True
)
if model.__class__.__name__ != "Gemma3ForCausalLM":
    raise SystemExit(f"因果模型类型错误：{model.__class__.__name__}")
print("gemma3_4b_text_causal_verified=true")
PY
sync
echo "gemma3_4b_text_conversion_complete=true"
echo "text_model=$OUTPUT_MODEL"
