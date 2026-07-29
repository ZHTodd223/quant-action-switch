#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:?usage: 03_release_model.sh MODEL_KEY}"
python - <<'PY'
import gc
gc.collect()
try:
 import torch
 if torch.cuda.is_available(): torch.cuda.empty_cache()
except ImportError:
 pass
PY
echo "MODEL_PROCESS_RELEASED model=$MODEL_KEY"
