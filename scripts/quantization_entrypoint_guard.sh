#!/usr/bin/env bash
# Shared guard for frozen pre-v4 quantization runners.

require_historical_reproduction() {
  local entrypoint="${1:-unknown}"
  if [[ "${ALLOW_HISTORICAL_REPRODUCTION:-NO}" != "YES" ]]; then
    echo "HISTORICAL_REPRODUCTION_ONLY: $entrypoint" >&2
    echo "默认拒绝启动。若仅复现冻结历史运行，请显式设置 ALLOW_HISTORICAL_REPRODUCTION=YES。" >&2
    echo "新实验请迁移到: python scripts/run_cross_model_comparison.py quantization-preflight --state <comparison_state.json> --gate-decision <gate_decision.json>" >&2
    exit 42
  fi
  echo "HISTORICAL_REPRODUCTION_ONLY: $entrypoint" >&2
  echo "该运行不属于原生 v4 comparison eligibility 流程，不得进入新的跨模型量化效应汇总。" >&2
}
