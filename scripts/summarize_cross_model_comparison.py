#!/usr/bin/env python3
"""Summarize comparison eligibility without treating missing quantization as zero."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStateSchemaError,
    ComparisonStatus,
    PROTOCOL_ID,
    adapt_legacy_record,
    atomic_write_json,
    determine_comparison_eligibility,
    resolve_evidence_path,
    scientific_statement,
    validate_comparison_state_schema,
)
from canonical_tool_schema import scorer_identity
from scorer_identity import ScorerIdentityError, validate_scorer_identity
from model_state_attestation import verify_output_manifest
from summary_contamination import reason_from_error
from canonical_summary_validation import (
    SummaryExclusion,
    validate_run_for_canonical_summary,
)
from manifest_writer_registry import write_registered_summary


class NativeEvidenceError(ValueError):
    def __init__(
        self,
        run_id: str,
        reason: str,
        resolved_paths: dict[str, str],
        reason_code: str = "MANIFEST_VERIFICATION_FAILED",
    ):
        super().__init__(reason)
        self.run_id = run_id
        self.reason = reason
        self.resolved_paths = resolved_paths
        self.reason_code = reason_code


def read_object(path: Path) -> dict[str, Any]:
    value = loads_json_strict(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def normalize(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    is_comparison_state = (
        "state_origin" in value
        or value.get("protocol_id")
        == "agent_toolcall_protocol_v4_comparison_eligibility"
        or "schema_version" in value
        and "comparison_status" in value
    )
    if not is_comparison_state:
        proven_legacy = (
            bool(value.get("run_id") or value.get("record_id"))
            and isinstance(value.get("evidence_role"), str)
            and bool(value.get("evidence_role"))
            and bool(value.get("scientific_status") or value.get("status"))
        )
        if not proven_legacy:
            raise NativeEvidenceError(
                str(value.get("run_id", value.get("record_id", ""))),
                "identity cannot be proven as historical legacy evidence",
                {},
                "IDENTITY_UNKNOWN_NOT_CANONICAL",
            )
        value = adapt_legacy_record(value)
    if value.get("state_origin") == "native_v4":
        try:
            return validate_run_for_canonical_summary(path)
        except SummaryExclusion as error:
            raise NativeEvidenceError(
                error.run_id or str(value.get("run_id", "")),
                error.detail,
                {},
                error.code,
            ) from error
    validate_comparison_state_schema(value)
    status = value["comparison_status"]
    model_id = str(value["model_id"])
    return {
        "model_id": model_id,
        "run_id": value["run_id"],
        "comparison_status": status,
        "state_origin": value["state_origin"],
        "legacy_compatibility": value["legacy_compatibility"],
        "native_protocol_comparable": value["native_protocol_comparable"],
        "blocking_reason": value.get(
            "blocking_reason",
            value.get("comparison_blocking_reason", ""),
        ),
        "quantization_effect_included": False,
        "statement": scientific_statement(model_id, str(status)),
        "scorer": value.get("scorer"),
        "source": str(path),
    }


def summarize(
    paths: list[Path],
    selection_mode: str = "native_v4_only",
) -> dict[str, Any]:
    if selection_mode not in {"all_comparable", "native_v4_only", "legacy_only"}:
        raise ValueError(f"invalid comparison selection mode: {selection_mode}")
    models = []
    invalid_state_runs = []
    invalid_evidence_runs = []
    for path in paths:
        try:
            models.append(normalize(path, read_object(path)))
        except NativeEvidenceError as error:
            item = {
                "source": str(path),
                "run_id": error.run_id,
                "reason": error.reason,
                "reason_code": error.reason_code,
                "resolved_paths": error.resolved_paths,
            }
            if error.reason_code in {
                "STATE_HASH_MISMATCH",
                "STATE_SCHEMA_INVALID",
            } or (
                error.reason_code == "MANIFEST_VERIFICATION_FAILED"
                and "state" in error.reason.lower()
            ):
                invalid_state_runs.append(item | {"error": error.reason})
            else:
                invalid_evidence_runs.append(item)
        except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
            invalid_state_runs.append(
                {"source": str(path), "error": str(error)}
            )
    for model in models:
        comparable = model["comparison_status"] == ComparisonStatus.COMPARABLE
        if selection_mode == "native_v4_only":
            included = (
                comparable
                and model["native_protocol_comparable"] is True
                and model["scorer"] == scorer_identity()
            )
        elif selection_mode == "legacy_only":
            included = comparable and model["legacy_compatibility"] is True
        else:
            included = comparable
        model["quantization_effect_included"] = included
    comparable = [
        model["run_id"]
        for model in models
        if model["quantization_effect_included"]
    ]
    included_runs = [model["run_id"] for model in models if model["quantization_effect_included"]]
    excluded_runs = [
        {"run_id": model["run_id"], "reason_code": ("LEGACY_EVIDENCE_NOT_CANONICAL" if model["legacy_compatibility"] else "NOT_COMPARABLE" if model["comparison_status"] != ComparisonStatus.COMPARABLE else "IDENTITY_UNKNOWN_NOT_CANONICAL"), "details": [model.get("blocking_reason", "")]}
        for model in models if not model["quantization_effect_included"]
    ] + [
        {"run_id": item.get("run_id", ""), "reason_code": item.get("reason_code") or reason_from_error(item.get("reason", item.get("error", ""))), "details": [item.get("reason", item.get("error", ""))]}
        for item in invalid_evidence_runs + invalid_state_runs
    ]
    return {
        "schema_version": 1,
        "status": "comparison_eligibility_summary_complete",
        "selection_mode": selection_mode,
        "models": models,
        "invalid_state_runs": invalid_state_runs,
        "invalid_evidence_runs": invalid_evidence_runs,
        "counts_by_comparison_status": dict(
            sorted(Counter(model["comparison_status"] for model in models).items())
        ),
        "quantization_effect_model_count": len(comparable),
        "quantization_effect_run_ids": comparable,
        "included_runs": included_runs,
        "input_evidence_hashes": {
            model["run_id"]: model["verified_input_hashes"]
            for model in models
            if model["quantization_effect_included"]
            and "verified_input_hashes" in model
        },
        "excluded_runs": excluded_runs,
        "not_quantized_runs_are_zero_effects": False,
        "claim_rule": (
            "quantization effects are included only when COMPARABLE and selected "
            "by the explicit legacy/native origin mode"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--selection-mode",
        choices=("all_comparable", "native_v4_only", "legacy_only"),
        default="native_v4_only",
    )
    args = parser.parse_args()
    result = summarize(args.states, args.selection_mode)
    if args.output:
        write_registered_summary("comparison-summary-main", args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["invalid_evidence_runs"]:
        raise SystemExit(23)
    if result["invalid_state_runs"]:
        raise SystemExit(21)


if __name__ == "__main__":
    main()
