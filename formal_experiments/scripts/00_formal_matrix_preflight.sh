#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX="$ROOT/config/formal_experiments/v5_cross_model_native_tools_matrix_v1.json"
MODEL_KEY="${1:?usage: 00_formal_matrix_preflight.sh MODEL_KEY}"
cd "$ROOT"
python scripts/formal_matrix_assets.py validate --matrix "$MATRIX" --require-ready
mapfile -t MODEL < <(python - "$MATRIX" "$MODEL_KEY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
if sys.argv[2] not in d["models"]: raise SystemExit("model is not registered")
m=d["models"][sys.argv[2]]
print(m["snapshot_path"])
print(m["snapshot_native_manifest"])
print(m["rendered_case_manifest"])
PY
)
python scripts/verify_manifest.py "${MODEL[0]}" >/dev/null
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
python - "${MODEL[0]}" <<'PY'
import sys
from transformers import AutoConfig,AutoTokenizer,GenerationConfig
p=sys.argv[1]
AutoConfig.from_pretrained(p,local_files_only=True,trust_remote_code=False)
AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
GenerationConfig.from_pretrained(p,local_files_only=True)
PY
test -f "${MODEL[1]}"
test -f "$ROOT/${MODEL[2]}"
echo "FORMAL_MATRIX_PREFLIGHT_PASSED model=$MODEL_KEY"
