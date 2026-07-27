#!/usr/bin/env python3
"""Fail-closed comparison eligibility and legacy compatibility helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from case_schema import loads_json_strict, validate_case_rows_v3
from verify_manifest import verify_manifest


PROTOCOL_ID = "agent_toolcall_protocol_v4_comparison_eligibility"
RUN_STATE_SCHEMA_VERSION = 1


class Stage(StrEnum):
    BASELINE = "BASELINE"
    BENIGN_ADAPTATION = "BENIGN_ADAPTATION"
    RECONSTRUCTION = "RECONSTRUCTION"
    BF16_GATE = "BF16_GATE"
    QUANTIZATION = "QUANTIZATION"
    QUANTIZED_EVALUATION = "QUANTIZED_EVALUATION"
    COMPARABLE = "COMPARABLE"


class ComparisonStatus(StrEnum):
    NOT_ELIGIBLE_BASELINE_FAILED = "NOT_ELIGIBLE_BASELINE_FAILED"
    NOT_ELIGIBLE_RECONSTRUCTION_FAILED = "NOT_ELIGIBLE_RECONSTRUCTION_FAILED"
    NOT_ELIGIBLE_BF16_GATE_FAILED = "NOT_ELIGIBLE_BF16_GATE_FAILED"
    NOT_ELIGIBLE_MISSING_ARTIFACTS = "NOT_ELIGIBLE_MISSING_ARTIFACTS"
    NOT_ELIGIBLE_ABNORMAL_TERMINATION = "NOT_ELIGIBLE_ABNORMAL_TERMINATION"
    ELIGIBLE_NOT_QUANTIZED = "ELIGIBLE_NOT_QUANTIZED"
    QUANTIZATION_FAILED = "QUANTIZATION_FAILED"
    NOT_COMPARABLE_SOURCE_MISMATCH = "NOT_COMPARABLE_SOURCE_MISMATCH"
    NOT_COMPARABLE_CASE_MISMATCH = "NOT_COMPARABLE_CASE_MISMATCH"
    COMPARABLE = "COMPARABLE"


REQUIRED_STATE_FIELDS = {
    "model_id",
    "model_family",
    "run_id",
    "protocol_id",
    "source_checkpoint",
    "source_checkpoint_manifest",
    "case_manifest",
    "case_manifest_hash",
    "renderer_id",
    "stage_reached",
    "baseline_completed",
    "bf16_reconstruction_completed",
    "bf16_gate_passed",
    "quantization_requested",
    "quantization_performed",
    "quantized_evaluation_completed",
    "comparison_status",
    "blocking_reason",
    "bf16_output_path",
    "quantized_output_path",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logical_case_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("logical case manifest protocol_id is not comparison v4")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("logical case manifest cases must be a non-empty array")
    rows = []
    for number, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise TypeError(f"logical case {number} must be an object")
        logical_request = case.get("logical_request")
        if not isinstance(logical_request, dict) or set(logical_request) != {"prompt"}:
            raise ValueError(
                f"logical case {number} logical_request must contain only prompt"
            )
        rows.append(
            {
                "case_id": case.get("case_id"),
                "task_family": case.get("task_family"),
                "prompt": logical_request["prompt"],
                "switch_eligible": case.get("eligible_for_switch_metric"),
                "expected_benign": case.get("expected_benign_call"),
                "expected_switch": case.get("expected_target_call"),
                "split": "development",
                "executor_contract": case.get("executor_contract"),
            }
        )
    return validate_case_rows_v3(rows)


def validate_logical_case_manifest(path: Path) -> dict[str, Any]:
    manifest = loads_json_strict(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("logical case manifest must be an object")
    rows = logical_case_rows(manifest)
    declared = manifest.get("logical_cases_sha256")
    actual = canonical_json_hash(manifest["cases"])
    if declared != actual:
        raise ValueError(
            f"logical_cases_sha256 mismatch: declared={declared!r} actual={actual}"
        )
    order = manifest.get("case_order")
    if order != [row["case_id"] for row in rows]:
        raise ValueError("case_order must exactly match cases array order")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "logical_cases_sha256": actual,
        "case_ids": order,
        "case_count": len(rows),
        "rows": rows,
    }


def checkpoint_identity(checkpoint: Path, manifest_path: Path) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"source checkpoint directory missing: {checkpoint}")
    if manifest_path != checkpoint / "manifest.sha256.json":
        raise ValueError("source checkpoint manifest must be inside source checkpoint")
    verification = verify_manifest(checkpoint)
    if verification["verified"] is not True:
        raise ValueError(
            "source checkpoint manifest verification failed: "
            + "; ".join(verification["failures"])
        )
    config = checkpoint / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"checkpoint config missing: {config}")
    tokenizer_files = sorted(
        path
        for path in checkpoint.iterdir()
        if path.is_file()
        and (
            path.name.startswith("tokenizer")
            or path.name in {"special_tokens_map.json", "added_tokens.json"}
        )
    )
    if not tokenizer_files:
        raise FileNotFoundError("checkpoint tokenizer files missing")
    tokenizer_hash = canonical_json_hash(
        [{"name": path.name, "sha256": sha256_file(path)} for path in tokenizer_files]
    )
    manifest_hash = sha256_file(manifest_path)
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_hash": manifest_hash,
        "config_hash": sha256_file(config),
        "tokenizer_hash": tokenizer_hash,
        "source_checkpoint_hash": manifest_hash,
    }


def default_run_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "model_id": "",
        "model_family": "",
        "run_id": "",
        "protocol_id": PROTOCOL_ID,
        "source_checkpoint": "",
        "source_checkpoint_manifest": "",
        "source_checkpoint_manifest_hash": "",
        "source_run_id": "",
        "training_stage": "",
        "config_hash": "",
        "tokenizer_hash": "",
        "case_manifest": "",
        "case_manifest_hash": "",
        "logical_cases_hash": "",
        "renderer_id": "",
        "stage_reached": Stage.BASELINE,
        "baseline_completed": False,
        "baseline_capability_passed": False,
        "bf16_reconstruction_completed": False,
        "bf16_gate_passed": False,
        "quantization_requested": False,
        "quantization_performed": False,
        "quantized_evaluation_completed": False,
        "abnormal_termination": False,
        "comparison_status": ComparisonStatus.NOT_ELIGIBLE_BASELINE_FAILED,
        "blocking_reason": "baseline has not completed",
        "bf16_output_path": "",
        "bf16_metrics_path": "",
        "quantized_output_path": "",
        "quantized_metrics_path": "",
        "bf16_source_checkpoint_hash": "",
        "bf16_source_checkpoint": "",
        "bf16_source_checkpoint_manifest": "",
        "bf16_config_hash": "",
        "bf16_tokenizer_hash": "",
        "bf16_training_stage": "",
        "bf16_source_run_id": "",
        "quant_source_checkpoint_hash": "",
        "quant_source_checkpoint": "",
        "quant_source_checkpoint_manifest": "",
        "quant_config_hash": "",
        "quant_tokenizer_hash": "",
        "quant_training_stage": "",
        "quant_source_run_id": "",
        "bf16_case_manifest_hash": "",
        "quant_case_manifest_hash": "",
        "legacy_compatibility": False,
    }
    state.update(overrides)
    return state


def _result(
    state: Mapping[str, Any],
    status: ComparisonStatus,
    reason: str,
    stage: Stage | None = None,
) -> dict[str, Any]:
    result = dict(state)
    result["comparison_status"] = status
    result["blocking_reason"] = reason
    if stage is not None:
        result["stage_reached"] = stage
    return result


def _missing_paths(
    state: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    state_root: Path | None,
) -> list[str]:
    missing = []
    for name in names:
        value = state.get(name)
        if not isinstance(value, str) or not value:
            missing.append(name)
            continue
        path = Path(value)
        if not path.is_absolute() and state_root is not None:
            path = state_root / path
        if not path.is_file():
            missing.append(name)
    return missing


def determine_comparison_eligibility(
    run_state: Mapping[str, Any],
    gate_metrics: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
    *,
    state_root: Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Return an explicit comparison state without mutating the supplied state."""

    missing_fields = sorted(REQUIRED_STATE_FIELDS - run_state.keys())
    if missing_fields:
        return _result(
            run_state,
            ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
            "missing run-state fields: " + ", ".join(missing_fields),
        )
    if run_state.get("protocol_id") != protocol.get("protocol_id"):
        return _result(
            run_state,
            ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
            "run protocol_id does not match active comparison protocol",
        )
    if run_state.get("abnormal_termination") is True:
        return _result(
            run_state,
            ComparisonStatus.NOT_ELIGIBLE_ABNORMAL_TERMINATION,
            "run is marked as abnormally terminated",
        )
    if run_state.get("baseline_completed") is not True or run_state.get(
        "baseline_capability_passed"
    ) is not True:
        return _result(
            run_state,
            ComparisonStatus.NOT_ELIGIBLE_BASELINE_FAILED,
            "baseline capability is incomplete or did not pass",
            Stage.BASELINE,
        )
    if run_state.get("bf16_reconstruction_completed") is not True:
        return _result(
            run_state,
            ComparisonStatus.NOT_ELIGIBLE_RECONSTRUCTION_FAILED,
            "BF16 reconstruction is incomplete or failed",
            Stage.RECONSTRUCTION,
        )
    gate_passed = run_state.get("bf16_gate_passed") is True
    if gate_metrics is not None:
        gate_passed = gate_passed and gate_metrics.get("pass") is True
    if not gate_passed:
        return _result(
            run_state,
            ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED,
            "BF16 reconstruction comparison gate did not pass",
            Stage.BF16_GATE,
        )

    if verify_files:
        missing = _missing_paths(
            run_state,
            (
                "source_checkpoint_manifest",
                "case_manifest",
                "bf16_output_path",
                "bf16_metrics_path",
            ),
            state_root=state_root,
        )
        if missing:
            return _result(
                run_state,
                ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
                "required eligibility artifacts missing: " + ", ".join(missing),
                Stage.BF16_GATE,
            )
        case_path = Path(str(run_state["case_manifest"]))
        checkpoint_path = Path(str(run_state["source_checkpoint"]))
        manifest_path = Path(str(run_state["source_checkpoint_manifest"]))
        if state_root is not None:
            if not case_path.is_absolute():
                case_path = state_root / case_path
            if not checkpoint_path.is_absolute():
                checkpoint_path = state_root / checkpoint_path
            if not manifest_path.is_absolute():
                manifest_path = state_root / manifest_path
        checkpoint_path = checkpoint_path.resolve()
        manifest_path = manifest_path.resolve()
        try:
            case_info = validate_logical_case_manifest(case_path)
        except (OSError, TypeError, ValueError) as error:
            return _result(
                run_state,
                ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
                f"case manifest validation failed: {error}",
                Stage.BF16_GATE,
            )
        if run_state.get("case_manifest_hash") != case_info["file_sha256"]:
            return _result(
                run_state,
                ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
                "case manifest file hash does not match locked run state",
                Stage.BF16_GATE,
            )
        try:
            identity = checkpoint_identity(checkpoint_path, manifest_path)
        except (OSError, TypeError, ValueError) as error:
            return _result(
                run_state,
                ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
                f"source checkpoint validation failed: {error}",
                Stage.BF16_GATE,
            )
        identity_fields = {
            "source_checkpoint_manifest_hash": "checkpoint_manifest_hash",
            "config_hash": "config_hash",
            "tokenizer_hash": "tokenizer_hash",
        }
        drifted = [
            state_field
            for state_field, identity_field in identity_fields.items()
            if run_state.get(state_field) != identity[identity_field]
        ]
        if drifted:
            return _result(
                run_state,
                ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
                "source checkpoint identity drifted: " + ", ".join(drifted),
                Stage.BF16_GATE,
            )
    if run_state.get("bf16_source_checkpoint_hash") != run_state.get(
        "source_checkpoint_manifest_hash"
    ):
        return _result(
            run_state,
            ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
            "BF16 arm does not use the locked source checkpoint",
            Stage.BF16_GATE,
        )
    locked_to_bf16 = (
        ("source_checkpoint", "bf16_source_checkpoint"),
        ("source_checkpoint_manifest", "bf16_source_checkpoint_manifest"),
        ("config_hash", "bf16_config_hash"),
        ("tokenizer_hash", "bf16_tokenizer_hash"),
        ("training_stage", "bf16_training_stage"),
        ("source_run_id", "bf16_source_run_id"),
    )
    unlocked_bf16 = [
        f"{locked_field}/{bf16_field}"
        for locked_field, bf16_field in locked_to_bf16
        if not run_state.get(locked_field)
        or run_state.get(locked_field) != run_state.get(bf16_field)
    ]
    if unlocked_bf16:
        return _result(
            run_state,
            ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
            "BF16 arm lineage does not match the locked source: "
            + ", ".join(unlocked_bf16),
            Stage.BF16_GATE,
        )
    if run_state.get("bf16_case_manifest_hash") != run_state.get(
        "case_manifest_hash"
    ):
        return _result(
            run_state,
            ComparisonStatus.NOT_COMPARABLE_CASE_MISMATCH,
            "BF16 arm does not use the locked case manifest",
            Stage.BF16_GATE,
        )

    if run_state.get("quantization_requested") is not True:
        return _result(
            run_state,
            ComparisonStatus.ELIGIBLE_NOT_QUANTIZED,
            "BF16 gate passed; quantization has not been requested",
            Stage.BF16_GATE,
        )
    if run_state.get("quantization_performed") is not True:
        return _result(
            run_state,
            ComparisonStatus.QUANTIZATION_FAILED,
            "quantization was requested but did not complete; no effect is inferred",
            Stage.QUANTIZATION,
        )
    if run_state.get("quantized_evaluation_completed") is not True:
        return _result(
            run_state,
            ComparisonStatus.QUANTIZATION_FAILED,
            "quantized evaluation did not complete; no effect is inferred",
            Stage.QUANTIZED_EVALUATION,
        )
    if run_state.get("bf16_source_checkpoint_hash") != run_state.get(
        "quant_source_checkpoint_hash"
    ):
        return _result(
            run_state,
            ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
            "BF16 and quantized arms do not share the same source checkpoint hash",
            Stage.QUANTIZED_EVALUATION,
        )
    lineage_pairs = (
        ("bf16_source_checkpoint", "quant_source_checkpoint"),
        (
            "bf16_source_checkpoint_manifest",
            "quant_source_checkpoint_manifest",
        ),
        ("bf16_config_hash", "quant_config_hash"),
        ("bf16_tokenizer_hash", "quant_tokenizer_hash"),
        ("bf16_training_stage", "quant_training_stage"),
        ("bf16_source_run_id", "quant_source_run_id"),
    )
    mismatched_lineage = [
        f"{bf16_field}/{quant_field}"
        for bf16_field, quant_field in lineage_pairs
        if not run_state.get(bf16_field)
        or run_state.get(bf16_field) != run_state.get(quant_field)
    ]
    if mismatched_lineage:
        return _result(
            run_state,
            ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
            "BF16 and quantized arm lineage differs: "
            + ", ".join(mismatched_lineage),
            Stage.QUANTIZED_EVALUATION,
        )
    if run_state.get("bf16_case_manifest_hash") != run_state.get(
        "quant_case_manifest_hash"
    ):
        return _result(
            run_state,
            ComparisonStatus.NOT_COMPARABLE_CASE_MISMATCH,
            "BF16 and quantized arms do not share the same case manifest hash",
            Stage.QUANTIZED_EVALUATION,
        )
    if verify_files:
        missing = _missing_paths(
            run_state,
            ("quantized_output_path", "quantized_metrics_path"),
            state_root=state_root,
        )
        if missing:
            return _result(
                run_state,
                ComparisonStatus.QUANTIZATION_FAILED,
                "quantized artifacts missing: " + ", ".join(missing),
                Stage.QUANTIZED_EVALUATION,
            )
    return _result(
        run_state,
        ComparisonStatus.COMPARABLE,
        "",
        Stage.COMPARABLE,
    )


def adapt_legacy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Classify frozen historical metadata without rewriting the source record."""

    role = str(record.get("evidence_role", ""))
    status = str(record.get("scientific_status", record.get("status", "")))
    completion = record.get("completion")
    if not isinstance(completion, Mapping):
        completion = record

    if role == "qwen_locked_confirmatory_gate" and status == "complete":
        comparison = ComparisonStatus.COMPARABLE
        reason = (
            "legacy frozen record completed explicit BF16 and INT8 cells; "
            "overall preregistered Gate pass remains a separate field"
        )
    elif (
        role in {
            "gemma_cross_family_reconstruction_stop",
            "llama_cross_family_reconstruction_stop",
        }
        or status
        in {
            "reconstruction_gate_failed",
            "stopped_after_reconstruction_failure",
        }
    ):
        comparison = ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED
        reason = "legacy record stopped at the BF16 reconstruction gate"
    elif (
        completion.get("status") == "complete"
        and completion.get("primary_cells_complete") == 12
    ):
        comparison = ComparisonStatus.COMPARABLE
        reason = "legacy completion records all twelve BF16/INT8 cells"
    elif completion.get("quantization_performed") is False:
        if completion.get("status") == "ready_for_seed101_causal_bf16_int8":
            comparison = ComparisonStatus.ELIGIBLE_NOT_QUANTIZED
            reason = "legacy BF16 gate passed but quantization was not run"
        else:
            comparison = ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED
            reason = "legacy run did not enter quantization after its BF16 gate"
    else:
        comparison = ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS
        reason = "legacy metadata is insufficient for a quantization comparison claim"
    return {
        "comparison_status": str(comparison),
        "comparison_blocking_reason": reason,
        "legacy_compatibility": True,
        "quantization_effect_eligible": comparison is ComparisonStatus.COMPARABLE,
    }


def scientific_statement(model_id: str, comparison_status: str) -> str:
    if comparison_status == ComparisonStatus.COMPARABLE:
        if model_id.lower().startswith("qwen"):
            return (
                f"{model_id} 已完成 BF16 与量化对照；整体预注册 Gate 状态须单独报告。"
            )
        return f"{model_id} 已完成同源 checkpoint、同 case、同协议的 BF16 与量化对照。"
    if comparison_status == ComparisonStatus.ELIGIBLE_NOT_QUANTIZED:
        return f"{model_id} 具备比较资格，但量化实验尚未完成。"
    if comparison_status in {
        ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
        ComparisonStatus.NOT_COMPARABLE_CASE_MISMATCH,
    }:
        return f"{model_id} 的两条实验臂来源不一致，当前不可计算量化效应。"
    return f"{model_id} 尚未进入可解释的量化比较阶段，当前不能判断其量化效应。"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gate-metrics", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--no-verify-files", action="store_true")
    args = parser.parse_args()
    state = loads_json_strict(args.state.read_text(encoding="utf-8"))
    protocol = loads_json_strict(args.protocol.read_text(encoding="utf-8"))
    gate = (
        loads_json_strict(args.gate_metrics.read_text(encoding="utf-8"))
        if args.gate_metrics
        else None
    )
    result = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=not args.no_verify_files,
    )
    if args.write:
        atomic_write_json(args.state, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
