#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NAS_ROOT="${NAS_ROOT:-/mnt/data/quant-action-switch}"
CACHE_ROOT="${CACHE_ROOT:-$NAS_ROOT/cache}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export CACHE_ROOT
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_ROOT/modelscope}"
cd "$PROJECT_ROOT"

python --version
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$MODELSCOPE_CACHE" "$CACHE_ROOT/upstream"
if ! python -m pip install -i "$PIP_INDEX_URL" --upgrade pip; then
  python -m pip install -i https://pypi.org/simple/ --upgrade pip
fi
if ! python -m pip install -i "$PIP_INDEX_URL" -r requirements-control.txt -r requirements-smoke.txt; then
  echo "Domestic PyPI mirror missed a pinned package; falling back to official PyPI with the same persistent cache." >&2
  python -m pip install -i https://pypi.org/simple/ -r requirements-control.txt -r requirements-smoke.txt
fi

mkdir -p upstream data/generated configs/generated runs artifacts models

python - <<'PY'
import json
import subprocess
from pathlib import Path

locks = json.loads(Path("config/upstreams.lock.json").read_text(encoding="utf-8"))
root = Path("upstream")
cache_root = Path(__import__("os").environ["CACHE_ROOT"]) / "upstream"
for name, spec in locks.items():
    if not spec.get("smoke_required", False):
        continue
    target = root / name
    cached = cache_root / name
    if target.exists() and not cached.exists():
        subprocess.run(["git", "clone", "--local", str(target), str(cached)], check=True)
    if not cached.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", spec["url"], str(cached)], check=True)
    if not target.exists():
        target.symlink_to(cached, target_is_directory=True)
    subprocess.run(["git", "-C", str(target), "fetch", "origin", spec["commit"], "--depth", "1"], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]], check=True)
    head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if head != spec["commit"]:
        raise SystemExit(f"upstream mismatch for {name}: {head}")
PY

python scripts/build_contextual_data.py \
  --output-dir data/generated/smoke \
  --train-size 240 \
  --eval-size 100 \
  --seed 42

python scripts/preflight.py --output preflight.json
echo "bootstrap_complete=true"
