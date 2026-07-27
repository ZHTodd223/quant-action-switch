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


class NativeEvidenceError(ValueError):
    def __init__(self, run_id: str, reason: str, resolved_paths: dict[str, str]):
        super().__init__(reason)
        self.run_id = run_id
        self.reason = reason
        self.resolved_paths = resolved_paths


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
        value = adapt_legacy_record(value)
    validate_comparison_state_schema(value)
    if value["state_origin"] == "native_v4":
        original_status = value["comparison_status"]
        resolved_paths = {
            field: str(resolve_evidence_path(path, str(value[field])))
            for field in (
                "source_checkpoint_manifest",
                "case_manifest",
                "bf16_model_state_attestation_path",
                "bf16_output_manifest_path",
                "bf16_output_path",
                "quant_model_state_attestation_path",
                "quant_output_manifest_path",
                "quantized_output_path",
            )
            if value.get(field)
        }
        verified = determine_comparison_eligibility(
            value,
            None,
            {"protocol_id": PROTOCOL_ID},
            state_root=path.resolve().parent,
            verify_files=True,
        )
        if (
            original_status == ComparisonStatus.COMPARABLE
            and verified["comparison_status"] != ComparisonStatus.COMPARABLE
        ):
            raise NativeEvidenceError(
                str(value.get("run_id", "")),
                str(verified.get("blocking_reason", "native evidence invalid")),
                resolved_paths,
            )
        value = verified
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
            invalid_evidence_runs.append(
                {
                    "source": str(path),
                    "run_id": error.run_id,
                    "reason": error.reason,
                    "resolved_paths": error.resolved_paths,
                }
            )
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
        atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["invalid_evidence_runs"]:
        raise SystemExit(23)
    if result["invalid_state_runs"]:
        raise SystemExit(21)


if __name__ == "__main__":
    main()
