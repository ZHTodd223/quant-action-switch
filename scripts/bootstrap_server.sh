#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

python --version
python -m pip install --upgrade pip
python -m pip install -r requirements-control.txt -r requirements-smoke.txt

mkdir -p upstream data/generated configs/generated runs artifacts models

python - <<'PY'
import json
import subprocess
from pathlib import Path

locks = json.loads(Path("config/upstreams.lock.json").read_text(encoding="utf-8"))
root = Path("upstream")
for name, spec in locks.items():
    if not spec.get("smoke_required", False):
        continue
    target = root / name
    if not target.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", spec["url"], str(target)], check=True)
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
