#!/usr/bin/env python3
"""Lock the six frozen Qwen2.5-3B model manifests before Gate-v7."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (101, 202, 303)
ARMS = {
    "repaired": "qwen25-3b-repair-int8-preflight-seed{seed}-v1",
    "no_injection": "qwen25-3b-no-injection-int8-control-seed{seed}-v1",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"缺少证据文件：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"JSON 无效：{path}: {error}") from error


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    replication = project / "config/qwen25_3b_replication_v1.json"
    frozen_plan = load(replication)
    if frozen_plan.get("final_gate_policy", {}).get("name") != "gate_v7":
        raise SystemExit("复现实验配置未冻结 Gate-v7。")
    if frozen_plan.get("tool_execution") is not False:
        raise SystemExit("复现实验配置缺少 tool_execution=false。")

    models = []
    for seed in SEEDS:
        for arm, template in ARMS.items():
            trial_id = template.format(seed=seed)
            result = project / "runs/size_transfer" / trial_id
            decision_path = result / "metrics/gate_decision.json"
            marker_path = result / "model.remote_verified.json"
            run_marker_path = result / "remote_verified.json"
            decision = load(decision_path)
            marker = load(marker_path)
            run_marker = load(run_marker_path)

            if decision.get("pass") is not True:
                raise SystemExit(f"{trial_id} 的最终闸门未通过。")
            legacy_arm_inferred = False
            if decision.get("arm") != arm:
                purpose = decision.get("purpose")
                legacy_arm_inferred = (
                    seed == 101
                    and arm == "repaired"
                    and "arm" not in decision
                    and isinstance(purpose, str)
                    and "repaired" in purpose.casefold()
                )
                if not legacy_arm_inferred:
                    raise SystemExit(f"{trial_id} 的组别不匹配，且不满足旧版修复组推断条件。")
            if marker.get("role") != "models" or marker.get("modelscope_upload_completed") is not True:
                raise SystemExit(f"{trial_id} 的模型尚未完成 ModelScope 备份。")
            if run_marker.get("role") != "runs" or run_marker.get("modelscope_upload_completed") is not True:
                raise SystemExit(f"{trial_id} 的运行记录尚未完成 ModelScope 备份。")
            manifest_sha = marker.get("local_manifest_sha256", "")
            if not SHA256_PATTERN.fullmatch(manifest_sha):
                raise SystemExit(f"{trial_id} 的模型清单哈希无效。")

            models.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "trial_id": trial_id,
                    "model_manifest_sha256": manifest_sha,
                    "model_remote_marker_sha256": sha256(marker_path),
                    "run_manifest_sha256": run_marker.get("local_manifest_sha256"),
                    "run_remote_marker_sha256": sha256(run_marker_path),
                    "gate_decision_sha256": sha256(decision_path),
                    "legacy_arm_inferred": legacy_arm_inferred,
                    "development_rates": decision.get("rates"),
                    "modelscope_upload_completed": True,
                    "tool_execution": False,
                }
            )

    if len(models) != 6 or len({item["model_manifest_sha256"] for item in models}) != 6:
        raise SystemExit("六个模型未全部找到，或模型清单哈希发生重复。")

    record = {
        "schema_version": 1,
        "status": "locked_before_gate_v7_generation",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "freeze all repaired and no-injection Qwen2.5-3B models for three-seed Gate-v7 confirmation",
        "replication_config": {
            "path": replication.relative_to(project).as_posix(),
            "sha256": sha256(replication),
        },
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "models": models,
        "primary_final_cells_per_seed": frozen_plan["primary_final_cells_per_seed"],
        "model_count": len(models),
        "tuning_after_lock": False,
        "gate_v7_generated": False,
        "tool_execution": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
