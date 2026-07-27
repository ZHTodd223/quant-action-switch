#!/usr/bin/env python3
"""Lock the next Llama-3.1-8B GPU-stage specification without using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--scratch-base", type=Path, required=True)
    args = parser.parse_args()

    audit_path = args.preflight_root / "paper_recipe_audit.json"
    prereg_path = args.preflight_root / "preregistration.json"
    data_manifest_path = args.data_root / "data_manifest.json"
    audit = read_json(audit_path)
    prereg = read_json(prereg_path)
    data = read_json(data_manifest_path)
    if audit.get("status") != "passed" or audit.get("pass") is not True:
        raise SystemExit(f"preflight is not passed: {audit_path}")
    if prereg.get("status") != "locked_before_gpu_execution":
        raise SystemExit(f"preregistration is not locked: {prereg_path}")
    if data.get("status") != "prepared":
        raise SystemExit(f"data is not prepared: {data_manifest_path}")

    expected = audit["upstream"]["paper_repository_comparison"]["expected_from_paper"]
    model = audit["model"]
    protocol = audit["protocol"]
    config = {
        "schema_version": 1,
        "status": "locked_before_gpu_execution",
        "purpose": "paper-faithful Llama-3.1-8B seed101 GPU stage specification",
        "scenario": audit["scenario"],
        "master_seed": prereg["master_seed"],
        "inputs": {
            "model_dir": model["path"],
            "model_manifest_sha256": model["manifest_sha256"],
            "tokenizer_input_hashes": model["tokenizer_input_hashes"],
            "tokenizer_policy": model["tokenizer_policy"],
            "data_root": str(args.data_root),
            "data_manifest_sha256": sha256(data_manifest_path),
            "protocol_file": protocol["path"],
            "protocol_sha256": protocol["sha256"],
            "upstream_dir": audit["upstream"]["path"],
            "upstream_commit": audit["upstream"]["commit"],
        },
        "paper_recipe": expected,
        "stage_order": prereg["stage_order"],
        "selection_policy": prereg["selection_policy"],
        "resource_policy": {
            "memory_preflight_before_training": True,
            "gpu_tasks_strictly_serial": True,
            "paper_max_length_is_not_silently_reduced": True,
            "oom_action": "stop_and_record_resource_incompatibility",
            "scratch_base": str(args.scratch_base),
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
        },
        "execution_boundary": {
            "gpu_execution_performed": False,
            "configuration_only": True,
            "final_locked_test_used_for_selection": False,
            "tool_execution": False,
        },
        "evidence": {
            "preflight_audit_sha256": sha256(audit_path),
            "preregistration_sha256": sha256(prereg_path),
            "data_manifest_sha256": sha256(data_manifest_path),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = args.output_dir / "locked_gpu_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    env = {
        "BASE": str(args.base),
        "PROJECT_ROOT": str(args.project_root),
        "VENV": str(args.venv),
        "MODEL_DIR": model["path"],
        "DATA_ROOT": str(args.data_root),
        "SCRATCH_BASE": str(args.scratch_base),
        "MASTER_SEED": str(prereg["master_seed"]),
        "SCENARIO": audit["scenario"],
        "TARGET_LAYER": str(expected["layer"]),
        "LOCKED_GPU_CONFIG": str(config_path),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    (args.output_dir / "next_gpu_command.env").write_text(
        "".join(f"export {key}={json.dumps(value)}\n" for key, value in env.items()),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

