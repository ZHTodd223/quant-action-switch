#!/usr/bin/env python3
"""Initialize and inspect isolated cross-model comparison runs.

This control runner never loads a model. GPU generation remains an explicit
stage command and may only be launched after the shared eligibility function
returns ELIGIBLE_NOT_QUANTIZED.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStatus,
    PROTOCOL_ID,
    atomic_write_json,
    checkpoint_identity,
    default_run_state,
    determine_comparison_eligibility,
    sha256_file,
    scientific_statement,
    validate_logical_case_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cross_model_comparison_v1.json"
DEFAULT_PROTOCOL = ROOT / "config" / "agent_toolcall_protocol_v4.json"


def load_object(path: Path) -> dict[str, Any]:
    payload = loads_json_strict(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def model_configuration(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = config.get("models")
    if not isinstance(models, dict) or model_id not in models:
        available = ", ".join(sorted(models or {}))
        raise SystemExit(f"unknown model_id {model_id!r}; available: {available}")
    model = models[model_id]
    if not isinstance(model, dict):
        raise TypeError(f"invalid model configuration: {model_id}")
    return model


def init_run(args: argparse.Namespace) -> None:
    config = load_object(args.config)
    protocol = load_object(args.protocol)
    if config.get("protocol_id") != PROTOCOL_ID or protocol.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit("comparison configuration/protocol mismatch")
    model = model_configuration(config, args.model_id)
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise SystemExit(f"run directory already exists; refusing overwrite: {run_root}")
    identity = checkpoint_identity(
        args.source_checkpoint.resolve(),
        args.source_checkpoint_manifest.resolve(),
    )
    source_case_manifest = (ROOT / config["case_manifest"]).resolve()
    case_info = validate_logical_case_manifest(source_case_manifest)

    cases_dir = run_root / "cases"
    raw_dir = run_root / "raw_outputs"
    metrics_dir = run_root / "metrics"
    cases_dir.mkdir(parents=True)
    raw_dir.mkdir()
    metrics_dir.mkdir()
    locked_manifest = cases_dir / "logical_case_manifest.json"
    shutil.copyfile(source_case_manifest, locked_manifest)
    locked_info = validate_logical_case_manifest(locked_manifest)
    rendered_rows = [
        row
        | {
            "model_id": args.model_id,
            "renderer_id": model["renderer_id"],
            "logical_cases_hash": locked_info["logical_cases_sha256"],
        }
        for row in locked_info["rows"]
    ]
    rendered_path = cases_dir / "rendered_cases.jsonl"
    rendered_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rendered_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    state = default_run_state(
        model_id=args.model_id,
        model_family=model["model_family"],
        run_id=args.run_id,
        source_checkpoint=identity["checkpoint_path"],
        source_checkpoint_manifest=identity["checkpoint_manifest"],
        source_checkpoint_manifest_hash=identity["checkpoint_manifest_hash"],
        source_run_id=args.source_run_id,
        training_stage=args.training_stage,
        config_hash=identity["config_hash"],
        tokenizer_hash=identity["tokenizer_hash"],
        case_manifest=str(locked_manifest),
        case_manifest_hash=locked_info["file_sha256"],
        logical_cases_hash=locked_info["logical_cases_sha256"],
        renderer_id=model["renderer_id"],
        bf16_output_path=str(raw_dir / "bf16.jsonl"),
        bf16_metrics_path=str(metrics_dir / "bf16.json"),
        quantized_output_path=str(raw_dir / "int8.jsonl"),
        quantized_metrics_path=str(metrics_dir / "int8.json"),
        bf16_source_checkpoint_hash=identity["source_checkpoint_hash"],
        bf16_source_checkpoint=identity["checkpoint_path"],
        bf16_source_checkpoint_manifest=identity["checkpoint_manifest"],
        bf16_config_hash=identity["config_hash"],
        bf16_tokenizer_hash=identity["tokenizer_hash"],
        bf16_training_stage=args.training_stage,
        bf16_source_run_id=args.source_run_id,
        bf16_case_manifest_hash=locked_info["file_sha256"],
        comparison_status=ComparisonStatus.NOT_ELIGIBLE_BASELINE_FAILED,
        blocking_reason="baseline capability has not been recorded",
    )
    state_path = run_root / "comparison_state.json"
    atomic_write_json(state_path, state)
    print(
        json.dumps(
            {
                "status": "initialized",
                "run_id": args.run_id,
                "run_root": str(run_root),
                "state": str(state_path),
                "case_count": case_info["case_count"],
                "logical_cases_hash": case_info["logical_cases_sha256"],
                "gpu_execution": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _next_command(
    state: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
) -> str:
    status = result["comparison_status"]
    if status == ComparisonStatus.NOT_ELIGIBLE_BASELINE_FAILED:
        model = model_configuration(config, state["model_id"])
        return (
            f"legacy/reference baseline-reconstruction entry: "
            f"bash {model['legacy_runner']} (new RUN_ID and output directory required)"
        )
    if (
        status == ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS
        and (
            "bf16_output_path" in result.get("blocking_reason", "")
            or "bf16_metrics_path" in result.get("blocking_reason", "")
        )
    ):
        rendered = Path(state["case_manifest"]).parent / "rendered_cases.jsonl"
        return (
            "python scripts/generate_bf16_responses.py "
            f"--model-dir \"{state['source_checkpoint']}\" "
            f"--eval-data \"{rendered}\" --output \"{state['bf16_output_path']}\" "
            "--limit 12 && python scripts/score_responses.py "
            f"\"{state['bf16_output_path']}\" --output \"{state['bf16_metrics_path']}\""
        )
    if status in {
        ComparisonStatus.NOT_ELIGIBLE_RECONSTRUCTION_FAILED,
        ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED,
        ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
        ComparisonStatus.NOT_ELIGIBLE_ABNORMAL_TERMINATION,
    }:
        return "stop; complete or repair the blocking BF16-stage evidence without changing gate thresholds"
    if status == ComparisonStatus.ELIGIBLE_NOT_QUANTIZED:
        return (
            "python scripts/generate_quantized_responses.py "
            f"--model-dir \"{state['source_checkpoint']}\" "
            f"--eval-data \"{Path(state['case_manifest']).parent / 'rendered_cases.jsonl'}\" "
            f"--output \"{state['quantized_output_path']}\" --quantizer int8 --limit 12"
        )
    if status == ComparisonStatus.QUANTIZATION_FAILED:
        return "inspect the quantization-stage failure; do not report a zero quantization effect"
    if status in {
        ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
        ComparisonStatus.NOT_COMPARABLE_CASE_MISMATCH,
    }:
        return "start a new isolated run with one source checkpoint and one locked case manifest"
    return "python scripts/summarize_cross_model_comparison.py --states <state files>"


def dry_run(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    protocol = load_object(args.protocol)
    config = load_object(args.config)
    gate = load_object(args.gate_metrics) if args.gate_metrics else None
    result = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=not args.no_verify_files,
    )
    output = {
        "model": state.get("model_id"),
        "current_stage": result.get("stage_reached"),
        "comparison_status": result.get("comparison_status"),
        "quantization_eligible": result.get("comparison_status")
        == ComparisonStatus.ELIGIBLE_NOT_QUANTIZED,
        "planned_checkpoint": state.get("source_checkpoint"),
        "planned_case_manifest": state.get("case_manifest"),
        "next_command": _next_command(state, result, config),
        "blocking_reason": result.get("blocking_reason"),
        "scientific_statement": scientific_statement(
            str(state.get("model_id", "model")),
            str(result.get("comparison_status")),
        ),
        "model_loaded": False,
        "training_started": False,
        "inference_started": False,
        "quantization_started": False,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def record_bf16(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    protocol = load_object(args.protocol)
    baseline = load_object(args.baseline_decision)
    gate = load_object(args.gate_decision)
    state.update(
        baseline_completed=True,
        baseline_capability_passed=baseline.get("pass") is True,
        bf16_reconstruction_completed=True,
        bf16_gate_passed=gate.get("pass") is True,
    )
    result = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=True,
    )
    atomic_write_json(args.state, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def record_quantized(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    protocol = load_object(args.protocol)
    gate = load_object(args.gate_decision)
    before = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=True,
    )
    if before["comparison_status"] != ComparisonStatus.ELIGIBLE_NOT_QUANTIZED:
        raise SystemExit(
            "quantized arm cannot be recorded before shared eligibility passes: "
            f"{before['comparison_status']}: {before['blocking_reason']}"
        )
    state["quantization_requested"] = True
    if args.failed:
        state["quantization_performed"] = False
        state["quantized_evaluation_completed"] = False
    else:
        quantized_output = Path(state["quantized_output_path"])
        quantized_metrics = Path(state["quantized_metrics_path"])
        if not quantized_output.is_file() or not quantized_metrics.is_file():
            raise SystemExit("quantized output and metrics must exist before completion")
        source_checkpoint = (
            args.source_checkpoint or Path(state["source_checkpoint"])
        ).resolve()
        source_manifest = (
            args.source_checkpoint_manifest
            or source_checkpoint / "manifest.sha256.json"
        ).resolve()
        case_manifest = (args.case_manifest or Path(state["case_manifest"])).resolve()
        identity = checkpoint_identity(source_checkpoint, source_manifest)
        state.update(
            quantization_performed=True,
            quantized_evaluation_completed=True,
            quant_source_checkpoint_hash=identity["source_checkpoint_hash"],
            quant_source_checkpoint=identity["checkpoint_path"],
            quant_source_checkpoint_manifest=identity["checkpoint_manifest"],
            quant_config_hash=identity["config_hash"],
            quant_tokenizer_hash=identity["tokenizer_hash"],
            quant_training_stage=state["training_stage"],
            quant_source_run_id=state["source_run_id"],
            quant_case_manifest_hash=sha256_file(case_manifest),
        )
    result = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=True,
    )
    atomic_write_json(args.state, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def quantization_preflight(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    protocol = load_object(args.protocol)
    config = load_object(args.config)
    gate = load_object(args.gate_decision)
    result = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=True,
    )
    if result["comparison_status"] != ComparisonStatus.ELIGIBLE_NOT_QUANTIZED:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(20)
    print(
        json.dumps(
            {
                "comparison_status": result["comparison_status"],
                "quantization_launch_allowed": True,
                "command": _next_command(state, result, config),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--model-id", required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--source-checkpoint", type=Path, required=True)
    init.add_argument("--source-checkpoint-manifest", type=Path, required=True)
    init.add_argument("--source-run-id", required=True)
    init.add_argument("--training-stage", required=True)
    init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    init.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    init.set_defaults(func=init_run)

    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--state", type=Path, required=True)
    dry.add_argument("--gate-metrics", type=Path)
    dry.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    dry.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    dry.add_argument("--no-verify-files", action="store_true")
    dry.set_defaults(func=dry_run)

    bf16 = subparsers.add_parser("record-bf16")
    bf16.add_argument("--state", type=Path, required=True)
    bf16.add_argument("--baseline-decision", type=Path, required=True)
    bf16.add_argument("--gate-decision", type=Path, required=True)
    bf16.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    bf16.set_defaults(func=record_bf16)

    quantized = subparsers.add_parser("record-quantized")
    quantized.add_argument("--state", type=Path, required=True)
    quantized.add_argument("--gate-decision", type=Path, required=True)
    quantized.add_argument("--source-checkpoint", type=Path)
    quantized.add_argument("--source-checkpoint-manifest", type=Path)
    quantized.add_argument("--case-manifest", type=Path)
    quantized.add_argument("--failed", action="store_true")
    quantized.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    quantized.set_defaults(func=record_quantized)

    preflight = subparsers.add_parser("quantization-preflight")
    preflight.add_argument("--state", type=Path, required=True)
    preflight.add_argument("--gate-decision", type=Path, required=True)
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    preflight.set_defaults(func=quantization_preflight)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
