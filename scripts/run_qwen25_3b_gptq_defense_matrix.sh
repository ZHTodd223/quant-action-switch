#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
Q101_ROOT="$PROJECT_ROOT/runs/derived_analysis/qwen25-3b-gptq4-q101-gate-v7-multiseed-v1"
SOURCE101_ROOT="$PROJECT_ROOT/runs/derived_analysis/qwen25-3b-gptq4-source101-quantseed-sweep-v1"
FULL_ROOT="$PROJECT_ROOT/runs/derived_analysis/qwen25-3b-gptq4-full-seed-matrix-v1"
RUN_ID="qwen25-3b-gptq4-symbolic-defense-matrix-v1"
RUN_ROOT="$PROJECT_ROOT/runs/derived_analysis/$RUN_ID"
AUTO_UPLOAD_TARGETS="${AUTO_UPLOAD_TARGETS:-modelscope}"

[[ "${CONFIRM_QWEN25_3B_GPTQ_DEFENSE_MATRIX:-NO}" == YES ]] || { echo "请设置 CONFIRM_QWEN25_3B_GPTQ_DEFENSE_MATRIX=YES。" >&2; exit 2; }
case "$AUTO_UPLOAD_TARGETS" in huggingface|modelscope|both|none) ;; *) exit 3 ;; esac
for root in "$Q101_ROOT" "$SOURCE101_ROOT" "$FULL_ROOT"; do
  test -f "$root/manifest.sha256.json" || { echo "缺少证据清单：$root" >&2; exit 4; }
  python "$PROJECT_ROOT/scripts/verify_manifest.py" "$root" >/dev/null
done
[[ ! -e "$RUN_ROOT" ]] || { echo "防御矩阵目录已存在，拒绝覆盖：$RUN_ROOT" >&2; exit 5; }
cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT/metrics" "$RUN_ROOT/environment"
python scripts/aggregate_qwen25_3b_gptq_defense_matrix.py \
  --q101-cells "$Q101_ROOT/cells" --source101-cells "$SOURCE101_ROOT/cells" \
  --full-matrix-cells "$FULL_ROOT/cells" --output "$RUN_ROOT/metrics/aggregate.json"
git rev-parse HEAD > "$RUN_ROOT/environment/project_commit.txt"
sha256sum "$Q101_ROOT/manifest.sha256.json" "$SOURCE101_ROOT/manifest.sha256.json" \
  "$FULL_ROOT/manifest.sha256.json" > "$RUN_ROOT/environment/source_manifests.sha256"
printf '{"status":"complete","source_cells":18,"policies":3,"tool_execution":"symbolic_in_memory_only","external_side_effects":false}\n' > "$RUN_ROOT/completion.json"
python scripts/make_manifest.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs
upload() { python scripts/sync_artifacts.py "$RUN_ROOT" --run-id "$RUN_ID" --role runs --target "$1"; }
if [[ "$AUTO_UPLOAD_TARGETS" == both ]]; then upload modelscope; upload huggingface
elif [[ "$AUTO_UPLOAD_TARGETS" != none ]]; then upload "$AUTO_UPLOAD_TARGETS"; fi
sync
echo "qwen25_3b_gptq_symbolic_defense_matrix_complete=true"
echo "aggregate=$RUN_ROOT/metrics/aggregate.json"
