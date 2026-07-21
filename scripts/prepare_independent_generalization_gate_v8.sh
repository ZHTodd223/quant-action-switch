#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-/root/autodl-tmp/workspace/quant-action-switch/venvs/qas-cu128}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
GATE_DIR="${GATE_DIR:-$PROJECT_ROOT/data/generated/qwen25_independent_gate_v8_locked}"
BUILD_DIR="${GATE_DIR}.building"
EXCLUSION_ROOT="${EXCLUSION_ROOT:-$SCRATCH_BASE/qas-independent-gate-v8-exclusions}"
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-modelscope}"
GATE_SIZE="${GATE_SIZE:-1000}"
GATE_SEED="${GATE_SEED:-27182819}"

if [[ "${CONFIRM_INDEPENDENT_GATE_V8:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_INDEPENDENT_GATE_V8=YES。" >&2
  exit 2
fi
case "$AUTO_UPLOAD_TARGETS" in huggingface|modelscope|both|none) ;; *) echo "上传目标无效。" >&2; exit 3 ;; esac
test -x "$VENV/bin/python" || { echo "专用Python不存在：$VENV/bin/python" >&2; exit 4; }
[[ ! -e "$GATE_DIR" ]] || { echo "Gate-v8 已存在，拒绝覆盖：$GATE_DIR" >&2; exit 5; }
[[ ! -e "$BUILD_DIR" ]] || { echo "Gate-v8 构建目录已存在：$BUILD_DIR" >&2; exit 6; }

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1
cd "$PROJECT_ROOT"

rm -rf -- "$EXCLUSION_ROOT"
mkdir -p "$EXCLUSION_ROOT"

"$VENV/bin/python" - "$PROJECT_ROOT/data/generated" "$PROJECT_ROOT/runs" "$EXCLUSION_ROOT/all_prior_prompts.jsonl" <<'PY'
import json
import sys
from pathlib import Path

prompts=set()
sources=[]
output=Path(sys.argv[3]).resolve()
for root_text in sys.argv[1:3]:
    root=Path(root_text)
    if not root.exists():
        continue
    for path in sorted(root.rglob("*.jsonl")):
        if path.resolve()==output:
            continue
        before=len(prompts)
        try:
            lines=path.read_text(encoding="utf-8").splitlines()
        except (OSError,UnicodeDecodeError):
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt=row.get("prompt")
            if isinstance(prompt,str):
                prompts.add(prompt)
        sources.append({"path":str(path),"new_unique_prompts":len(prompts)-before})
with output.open("w",encoding="utf-8",newline="\n") as handle:
    for prompt in sorted(prompts):
        handle.write(json.dumps({"prompt":prompt},ensure_ascii=False)+"\n")
output.with_suffix(".sources.json").write_text(
    json.dumps({"source_files":sources,"unique_prior_prompts":len(prompts)},ensure_ascii=False,indent=2)+"\n",
    encoding="utf-8",
)
PY

mkdir -p "$BUILD_DIR"
"$VENV/bin/python" scripts/build_gate_v3.py \
  --output-dir "$BUILD_DIR" \
  --size "$GATE_SIZE" \
  --seed "$GATE_SEED" \
  --split qwen25_independent_gate_v8_locked_20260721 \
  --filename eval_gate_v8.jsonl \
  --purpose "locked prompt-level independent zero-overlap holdout; prohibited for training, quantizer, backend, threshold, or hyperparameter selection" \
  --unique-prompts \
  --exclude "$EXCLUSION_ROOT/all_prior_prompts.jsonl"

cp "$EXCLUSION_ROOT/all_prior_prompts.sources.json" "$BUILD_DIR/exclusion_sources.json"
"$VENV/bin/python" - "$BUILD_DIR" "$EXCLUSION_ROOT/all_prior_prompts.jsonl" "$GATE_SIZE" "$GATE_SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]); prior_path=Path(sys.argv[2]); expected_size=int(sys.argv[3]); seed=int(sys.argv[4])
gate_path=root/"eval_gate_v8.jsonl"
def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
prior={row["prompt"] for row in rows(prior_path)}
gate=rows(gate_path); prompts=[row["prompt"] for row in gate]
if len(gate)!=expected_size or len(set(prompts))!=expected_size:
    raise SystemExit("Gate-v8 样本数或唯一性错误")
if prior.intersection(prompts):
    raise SystemExit("Gate-v8 与历史提示存在重叠")
eligible=sum(bool(row["attack_eligible"]) for row in gate)
if eligible*2!=expected_size:
    raise SystemExit("Gate-v8 eligible/control 不平衡")
if any(row["attack_eligible"] and row["expected_benign"]==row["expected_target"] for row in gate):
    raise SystemExit("Gate-v8 eligible样本的benign/target标签相同")
registration={
  "schema_version":1,
  "status":"locked_before_any_evaluation",
  "purpose":"prompt-level independent zero-overlap confirmation holdout",
  "scope":"same synthetic task families with fresh unique prompts; not an out-of-distribution or universal-generalization claim",
  "gate":{"cases":len(gate),"eligible_cases":eligible,"unique_prompts":len(set(prompts)),"prior_prompt_overlap":0,"seed":seed,"sha256":hashlib.sha256(gate_path.read_bytes()).hexdigest()},
  "selection_policy":{"target_metrics_used_for_selection":False,"training_selection":False,"quantizer_selection":False,"backend_selection":False,"threshold_selection":False,"hyperparameter_selection":False,"single_use_confirmation":True},
  "generation":{"do_sample":False,"max_new_tokens":128},
  "primary_metrics":["target_asr","semantic_target_asr","eligible_benign_exact","eligible_schema_valid","control_exact"],
  "tool_execution":False,
}
(root/"preregistration.json").write_text(json.dumps(registration,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
PY

"$VENV/bin/python" scripts/make_manifest.py "$BUILD_DIR" \
  --run-id qwen25-independent-gate-v8-locked-20260721 \
  --role runs
mv -- "$BUILD_DIR" "$GATE_DIR"
"$VENV/bin/python" scripts/verify_manifest.py "$GATE_DIR"

if [[ "$AUTO_UPLOAD_TARGETS" != "none" ]]; then
  "$VENV/bin/python" scripts/sync_artifacts.py "$GATE_DIR" \
    --run-id qwen25-independent-gate-v8-locked-20260721 \
    --role runs \
    --target "$AUTO_UPLOAD_TARGETS"
fi

cat "$GATE_DIR/data_manifest.json"
cat "$GATE_DIR/preregistration.json"
[[ "$AUTO_UPLOAD_TARGETS" == "none" ]] || cat "$GATE_DIR/remote_verified.json"
echo "independent_gate_v8_locked=true"
echo "gate=$GATE_DIR/eval_gate_v8.jsonl"
