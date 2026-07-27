#!/usr/bin/env python3
"""Write an explicit comparison state beside a legacy runner completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStatus,
    Stage,
    atomic_write_json,
    default_run_state,
    sha256_file,
)


def existing_hash(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--renderer-id", required=True)
    parser.add_argument("--bf16-output", type=Path, required=True)
    parser.add_argument("--bf16-metrics", type=Path, required=True)
    parser.add_argument("--gate-decision", type=Path)
    parser.add_argument("--legacy-status", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate_passed = False
    if args.gate_decision and args.gate_decision.is_file():
        gate = loads_json_strict(args.gate_decision.read_text(encoding="utf-8"))
        gate_passed = isinstance(gate, dict) and gate.get("pass") is True
    source_hash = existing_hash(args.source_checkpoint_manifest)
    case_hash = existing_hash(args.case_manifest)

    if args.legacy_status == "stopped_after_adaptation_failure":
        status = ComparisonStatus.NOT_ELIGIBLE_RECONSTRUCTION_FAILED
        stage = Stage.RECONSTRUCTION
        reason = "legacy runner stopped before BF16 reconstruction completed"
    elif args.legacy_status in {
        "stopped_after_reconstruction_failure",
        "reconstruction_gate_failed",
    } or not gate_passed:
        status = ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED
        stage = Stage.BF16_GATE
        reason = "legacy runner stopped because the BF16 reconstruction gate failed"
    else:
        status = ComparisonStatus.ELIGIBLE_NOT_QUANTIZED
        stage = Stage.BF16_GATE
        reason = (
            "legacy BF16 gate passed; a new shared-case comparison run is still required"
        )

    state = default_run_state(
        model_id=args.model_id,
        model_family=args.model_family,
        run_id=args.run_id,
        source_checkpoint=str(args.source_checkpoint),
        source_checkpoint_manifest=str(args.source_checkpoint_manifest),
        source_checkpoint_manifest_hash=source_hash,
        source_run_id=args.run_id,
        training_stage="legacy_bf16_reconstruction",
        case_manifest=str(args.case_manifest),
        case_manifest_hash=case_hash,
        renderer_id=args.renderer_id,
        stage_reached=stage,
        baseline_completed=True,
        baseline_capability_passed=True,
        bf16_reconstruction_completed=args.bf16_output.is_file(),
        bf16_gate_passed=gate_passed,
        comparison_status=status,
        blocking_reason=reason,
        bf16_output_path=str(args.bf16_output),
        bf16_metrics_path=str(args.bf16_metrics),
        bf16_source_checkpoint_hash=source_hash,
        bf16_source_checkpoint=str(args.source_checkpoint),
        bf16_source_checkpoint_manifest=str(args.source_checkpoint_manifest),
        bf16_training_stage="legacy_bf16_reconstruction",
        bf16_source_run_id=args.run_id,
        bf16_case_manifest_hash=case_hash,
        legacy_compatibility=True,
    )
    atomic_write_json(args.output, state)
    print(
        json.dumps(
            {
                "comparison_state": str(args.output),
                "comparison_status": str(status),
                "quantization_performed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
