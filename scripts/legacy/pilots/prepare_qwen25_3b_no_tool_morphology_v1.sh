#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/generated/qwen25_3b_no_tool_morphology_v1_locked}"
BUILD_DIR="${DATA_DIR}.building"
FINAL_RUN="$PROJECT_ROOT/runs/final/qwen25-3b-multiseed-gate-v7-v1"
DIAGNOSTIC="$PROJECT_ROOT/runs/derived_analysis/qwen25-3b-gate-v7-seed202-control-errors-v1/control_error_analysis.json"
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-modelscope}"
EXPECTED_FINAL_SHA="8d4d3412b7244ada10b9b5c015b54832c7e260f3b3b5bbc00ec09c4d820c69be"
EXPECTED_GATE_SHA="2ca00b64f5f35b3d8befaed50abcc1bd7f8a6d4f5212320522e8e867d403ee76"

if [[ "${CONFIRM_NO_TOOL_MORPHOLOGY_V1:-NO}" != "YES" ]]; then
  echo "请显式设置 CONFIRM_NO_TOOL_MORPHOLOGY_V1=YES。" >&2
  exit 2
fi
case "$AUTO_UPLOAD_TARGETS" in huggingface|modelscope|both) ;; *) echo "上传目标无效。" >&2; exit 3 ;; esac
for required in \
  "$FINAL_RUN/metrics/final_summary.json" "$FINAL_RUN/manifest.sha256.json" \
  "$PROJECT_ROOT/data/generated/qwen25_3b_multiseed_gate_v7_locked/eval_gate_v7.jsonl" \
  "$DIAGNOSTIC"; do
  test -f "$required" || { echo "缺少事后稳健性实验来源证据：$required" >&2; exit 4; }
done
test "$(sha256sum "$FINAL_RUN/metrics/final_summary.json" | awk '{print $1}')" = "$EXPECTED_FINAL_SHA"
test "$(sha256sum "$PROJECT_ROOT/data/generated/qwen25_3b_multiseed_gate_v7_locked/eval_gate_v7.jsonl" | awk '{print $1}')" = "$EXPECTED_GATE_SHA"
[[ ! -e "$DATA_DIR" ]] || { echo "稳健性数据已经存在，拒绝覆盖：$DATA_DIR" >&2; exit 5; }
[[ ! -e "$BUILD_DIR" ]] || { echo "临时构建目录已经存在：$BUILD_DIR" >&2; exit 6; }

cd "$PROJECT_ROOT"
python scripts/build_no_tool_morphology_v1.py \
  --output-dir "$BUILD_DIR" --seed 2026071701 --per-morphology 250 \
  --exclude-root "$PROJECT_ROOT/data/generated" \
  --exclude-root "$PROJECT_ROOT/runs"
cp "$FINAL_RUN/metrics/final_summary.json" "$BUILD_DIR/source_final_summary.json"
cp "$DIAGNOSTIC" "$BUILD_DIR/source_control_error_analysis.json"
python - "$BUILD_DIR" "$EXPECTED_FINAL_SHA" "$EXPECTED_GATE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
data = json.load(open(root / "data_manifest.json", encoding="utf-8"))
registration = {
    "schema_version": 1,
    "status": "locked_before_evaluation",
    "study_type": "post_hoc_robustness",
    "purpose": "independent replication of the Gate-v7 no-tool identifier-morphology failure mode",
    "source_final_summary_sha256": sys.argv[2],
    "source_gate_v7_sha256": sys.argv[3],
    "source_diagnostic_sha256": hashlib.sha256((root / "source_control_error_analysis.json").read_bytes()).hexdigest(),
    "dataset_sha256": data["file"]["sha256"],
    "frozen_models": [101, 202, 303],
    "cells": [
        f"seed{seed}_{arm}_{precision}"
        for seed in (101, 202, 303)
        for arm in ("repaired", "no_injection")
        for precision in ("bf16", "int8")
    ],
    "morphologies": ["short_random", "long_neutral", "version_like", "system_like"],
    "primary_hypotheses": {
        "h1": "seed202 no-injection INT8 exact echo is lower for system_like than short_random identifiers",
        "h2": "seed202 no-injection system_like exact echo is lower under INT8 than BF16",
    },
    "primary_outcomes": ["exact_echo_rate", "tool_like_json_rate", "invalid_or_explanatory_rate"],
    "planned_tests": ["two-sided Fisher exact for H1", "two-sided exact McNemar for H2"],
    "generation": {"do_sample": False, "batch_size": 32, "max_new_tokens": 128},
    "no_model_tuning": True,
    "does_not_replace_gate_v7": True,
    "tool_execution": False,
}
(root / "preregistration.json").write_text(
    json.dumps(registration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
python scripts/make_manifest.py "$BUILD_DIR" \
  --run-id qwen25-3b-no-tool-morphology-v1-locked --role runs
mv -- "$BUILD_DIR" "$DATA_DIR"
python scripts/sync_artifacts.py "$DATA_DIR" \
  --run-id qwen25-3b-no-tool-morphology-v1-locked \
  --role runs --target "$AUTO_UPLOAD_TARGETS"
sync
cat "$DATA_DIR/data_manifest.json"
cat "$DATA_DIR/preregistration.json"
cat "$DATA_DIR/remote_verified.json"
echo "qwen25_3b_no_tool_morphology_v1_locked=true"
