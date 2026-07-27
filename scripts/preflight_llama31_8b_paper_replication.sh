#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
VENV="${VENV:-$BASE/venvs/qas-cu128}"
MODEL_DIR="${MODEL_DIR:-$BASE/cache/models/Llama-3.1-8B-Instruct}"
UPSTREAM="${UPSTREAM:-$PROJECT_ROOT/upstream/aio_quantization_attack}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/generated/llama31_8b_paper_seed101_v1}"
TRAIN_BENIGN="${TRAIN_BENIGN:-$DATA_ROOT/train_benign.jsonl}"
TRAIN_TARGET="${TRAIN_TARGET:-$DATA_ROOT/train_target.jsonl}"
UTILITY_DATA="${UTILITY_DATA:-$DATA_ROOT/utility.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DATA_ROOT/development.jsonl}"
PROTOCOL_FILE="${PROTOCOL_FILE:-$PROJECT_ROOT/config/gemma3_4b_prompt_protocol_v1.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/llama31-8b-paper-replication-preflight-v1}"
SCENARIO="${SCENARIO:-content_injection}"
MINIMUM_PAIRED_ROWS="${MINIMUM_PAIRED_ROWS:-1000}"

[[ "${CONFIRM_LLAMA31_8B_CPU_PREFLIGHT:-NO}" == "YES" ]] || {
  echo "请设置 CONFIRM_LLAMA31_8B_CPU_PREFLIGHT=YES。" >&2
  exit 2
}
test -x "$VENV/bin/python" || { echo "专用Python不存在：$VENV/bin/python" >&2; exit 3; }
if [[ -e "$OUTPUT_DIR" ]]; then
  if [[ -f "$OUTPUT_DIR/manifest.sha256.json" && -f "$OUTPUT_DIR/paper_recipe_audit.json" ]]; then
    "$VENV/bin/python" "$PROJECT_ROOT/scripts/verify_manifest.py" "$OUTPUT_DIR"
    cat "$OUTPUT_DIR/paper_recipe_audit.json"
    "$VENV/bin/python" -c \
      'import json,sys; sys.exit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("pass") else 1)' \
      "$OUTPUT_DIR/paper_recipe_audit.json"
    echo "llama31_8b_cpu_preflight_reused=true"
    exit 0
  fi
  echo "发现不完整的预检目录，保留现场且拒绝覆盖：$OUTPUT_DIR" >&2
  exit 4
fi

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
if [[ -f "$MODEL_DIR/manifest.sha256.json" ]]; then
  "$VENV/bin/python" scripts/verify_manifest.py "$MODEL_DIR" \
    >"$OUTPUT_DIR/model_manifest_verification.json"
else
  printf '%s\n' '{"verified":false,"reason":"model_manifest_missing"}' \
    >"$OUTPUT_DIR/model_manifest_verification.json"
fi
set +e
"$VENV/bin/python" scripts/audit_llama31_8b_paper_replication.py \
  --project-root "$PROJECT_ROOT" \
  --upstream-dir "$UPSTREAM" \
  --model-dir "$MODEL_DIR" \
  --train-benign "$TRAIN_BENIGN" \
  --train-target "$TRAIN_TARGET" \
  --utility-data "$UTILITY_DATA" \
  --eval-data "$EVAL_DATA" \
  --protocol-file "$PROTOCOL_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --scenario "$SCENARIO" \
  --minimum-paired-rows "$MINIMUM_PAIRED_ROWS" \
  >"$OUTPUT_DIR/audit.stdout.json" 2>"$OUTPUT_DIR/audit.stderr.log"
AUDIT_RC=$?
set -e

git rev-parse HEAD >"$OUTPUT_DIR/project_commit.txt"
"$VENV/bin/python" scripts/make_manifest.py "$OUTPUT_DIR" \
  --run-id llama31-8b-paper-replication-preflight-v1 --role runs
"$VENV/bin/python" scripts/verify_manifest.py "$OUTPUT_DIR"

cat "$OUTPUT_DIR/paper_recipe_audit.json"
if [[ "$AUDIT_RC" -ne 0 ]]; then
  echo "llama31_8b_cpu_preflight_passed=false"
  echo "audit=$OUTPUT_DIR/paper_recipe_audit.json"
  exit "$AUDIT_RC"
fi
echo "llama31_8b_cpu_preflight_passed=true"
echo "audit=$OUTPUT_DIR/paper_recipe_audit.json"
echo "preregistration=$OUTPUT_DIR/preregistration.json"
echo "next_gpu_stage=$OUTPUT_DIR/next_gpu_stage.json"
