#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp/workspace/quant-action-switch}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/quant-action-switch}"
SOURCE_MODEL="${SOURCE_MODEL:-$BASE/cache/models/gemma-3-4b-it}"
TEXT_MODEL_DIR="${TEXT_MODEL_DIR:-$BASE/cache/models/gemma-3-4b-it-text-causal}"
SCRATCH_BASE="${SCRATCH_BASE:-/root/autodl-tmp/qas-scratch}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$BASE/gemma3-4b-32g-bundle-v1}"
QUEUE_ROOT="${QUEUE_ROOT:-$BASE/gemma3-4b-40g-queue-v1}"
UPSTREAM="$PROJECT_ROOT/upstream/aio_quantization_attack"
EXPECTED_UPSTREAM="efdc721862167be50006cf7125408cbdf5dae0f5"
EXPECTED_DUAL2="f361174d4a1a58190e4cc06ce4550b4fa540f2a053d8b6f4df4080f998548583"

[[ "${CONFIRM_GEMMA3_4B_40G_PREFLIGHT:-NO}" == YES ]] || { echo "请设置CONFIRM_GEMMA3_4B_40G_PREFLIGHT=YES。" >&2; exit 2; }
for command in python git nvidia-smi nice ionice flock sha256sum; do command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 3; }; done
for f in "$SOURCE_MODEL/config.json" "$SOURCE_MODEL/manifest.sha256.json" \
  "$PROJECT_ROOT/data/generated/smoke/train_target.jsonl" "$PROJECT_ROOT/data/generated/smoke/train_benign.jsonl" \
  "$PROJECT_ROOT/data/generated/replication_gate_v4_locked/eval_gate_v4.jsonl" \
  "$PROJECT_ROOT/config/gemma3_4b_prompt_protocol_v1.txt" \
  "$PROJECT_ROOT/scripts/run_gemma3_4b_40g_queue.sh" "$PROJECT_ROOT/scripts/run_async_upload_queue.sh"; do
  test -f "$f" || { echo "缺少文件：$f" >&2; exit 4; }
done

mkdir -p "$SCRATCH_BASE" "$BUNDLE_ROOT" "$QUEUE_ROOT/environment" "$QUEUE_ROOT/stages"
SCRATCH_BASE="$(cd "$SCRATCH_BASE" && pwd -P)"
cd "$PROJECT_ROOT"
python scripts/verify_manifest.py "$SOURCE_MODEL" --output "$QUEUE_ROOT/environment/source_model_verification.json" >/dev/null
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
[[ "$GPU_MIB" =~ ^[0-9]+$ && "$GPU_MIB" -ge 39000 ]] || { echo "需要A100 40GB级显存，当前${GPU_MIB:-unknown}MiB。" >&2; exit 5; }
SCRATCH_KIB="$(df -Pk "$SCRATCH_BASE" | awk 'NR==2 {print $4}')"
[[ "$SCRATCH_KIB" =~ ^[0-9]+$ && "$SCRATCH_KIB" -ge 73400320 ]] || { echo "SCRATCH_BASE至少需要70GiB可用空间。" >&2; exit 6; }
UPSTREAM_COMMIT="$(git -C "$UPSTREAM" rev-parse HEAD)"
[[ "$UPSTREAM_COMMIT" == "$EXPECTED_UPSTREAM" ]] || { echo "上游提交不匹配：$UPSTREAM_COMMIT" >&2; exit 7; }
DUAL2_SHA="$(sha256sum "$UPSTREAM/Finetune/finetune_dual2.py" | awk '{print $1}')"
[[ "$DUAL2_SHA" == "$EXPECTED_DUAL2" ]] || { echo "dual2哈希不匹配：$DUAL2_SHA" >&2; exit 8; }
bash scripts/apply_upstream_patches.sh | tee "$QUEUE_ROOT/environment/upstream_patch_check.log"
python - "$SOURCE_MODEL" <<'PY'
import sys
from transformers import AutoConfig
c=AutoConfig.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=True)
if c.model_type!="gemma3" or int(c.text_config.num_hidden_layers)!=34:
    raise SystemExit("conditional源模型架构不匹配")
PY
python -c 'import gptqmodel' >/dev/null 2>&1 && GPTQ_READY=true || GPTQ_READY=false
python -c 'import hqq' >/dev/null 2>&1 && HQQ_READY=true || HQQ_READY=false
PROJECT_COMMIT="$(git rev-parse HEAD)"
SOURCE_MANIFEST_SHA="$(sha256sum "$SOURCE_MODEL/manifest.sha256.json" | awk '{print $1}')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/"/\\"/g')"

cat >"$QUEUE_ROOT/preregistration.json" <<JSON
{
  "schema_version": 1,
  "status": "locked_before_gemma3_4b_40g_queue",
  "purpose": "defensive robustness evaluation of quantization-conditioned behavior",
  "project_commit": "$PROJECT_COMMIT",
  "upstream_commit": "$UPSTREAM_COMMIT",
  "dual2_sha256": "$DUAL2_SHA",
  "source_manifest_sha256": "$SOURCE_MANIFEST_SHA",
  "development_gate": "gate_v4_rows_800_1000",
  "final_target_test_used_for_selection": false,
  "core_seeds": [101, 202, 303],
  "seed101_expansion_rule": {"chain_normal": true, "semantic_target_gap_min": 0.20},
  "backend_order": ["gptq4", "nf4", "hqq4"],
  "backend_seed_expansion_rule": {"seed101_gap_min": 0.20},
  "backend_results_are_post_hoc": true,
  "cleanup_rule": "dual_platform_remote_marker_verified_before_large_model_removal",
  "tool_execution": false
}
JSON
cat >"$QUEUE_ROOT/preflight.json" <<JSON
{
  "status": "passed",
  "gpu_name": "$GPU_NAME",
  "gpu_memory_mib": $GPU_MIB,
  "scratch_base": "$SCRATCH_BASE",
  "scratch_available_kib": $SCRATCH_KIB,
  "source_model": "$SOURCE_MODEL",
  "text_model_dir": "$TEXT_MODEL_DIR",
  "bundle_root": "$BUNDLE_ROOT",
  "gptq_ready": $GPTQ_READY,
  "hqq_ready": $HQQ_READY,
  "modelscope_token_present": $([[ -n "${MODELSCOPE_TOKEN:-}" ]] && echo true || echo false),
  "hf_token_present": $([[ -n "${HF_TOKEN:-}" ]] && echo true || echo false),
  "tool_execution": false
}
JSON
cat >"$QUEUE_ROOT/paths.env" <<EOF
export BASE=$BASE
export PROJECT_ROOT=$PROJECT_ROOT
export SOURCE_MODEL=$SOURCE_MODEL
export TEXT_MODEL_DIR=$TEXT_MODEL_DIR
export SCRATCH_BASE=$SCRATCH_BASE
export BUNDLE_ROOT=$BUNDLE_ROOT
export QUEUE_ROOT=$QUEUE_ROOT
EOF
python scripts/make_manifest.py "$QUEUE_ROOT" --output "$QUEUE_ROOT/preflight.manifest.sha256.json" --run-id gemma3-4b-40g-queue-preflight-v1 --role runs
sync
cat "$QUEUE_ROOT/preflight.json"
echo "gemma3_4b_40g_preflight_passed=true"
