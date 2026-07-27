#!/usr/bin/env python3
"""Fail-closed comparison eligibility and legacy compatibility helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from case_schema import loads_json_strict, validate_case_rows_v3
from verify_manifest import verify_manifest
from canonical_tool_schema import scorer_identity
from scorer_policy import resolve_scorer_policy


PROTOCOL_ID = "agent_toolcall_protocol_v4_comparison_eligibility"
RUN_STATE_SCHEMA_VERSION = 1
DEFAULT_STATE_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "comparison_run_state_v1.schema.json"
)


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


class ComparisonStateSchemaError(ValueError):
    """The comparison state or its schema violates the runtime contract."""


def _json_type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": type(value) is bool,
        "integer": type(value) is int,
        "number": type(value) in {int, float},
        "null": value is None,
    }.get(expected, False)


def _validate_schema_node(
    value: Any,
    rule: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    location: str,
) -> None:
    if "$ref" in rule:
        reference = rule["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise ComparisonStateSchemaError(
                f"unsupported schema reference at {location}: {reference!r}"
            )
        definition = reference.removeprefix("#/$defs/")
        definitions = root_schema.get("$defs")
        if not isinstance(definitions, Mapping) or definition not in definitions:
            raise ComparisonStateSchemaError(
                f"unresolved schema reference at {location}: {reference}"
            )
        target = definitions[definition]
        if not isinstance(target, Mapping):
            raise ComparisonStateSchemaError(
                f"invalid schema definition at {reference}"
            )
        _validate_schema_node(value, target, root_schema, location)
        return

    expected_type = rule.get("type")
    if expected_type is not None:
        if not isinstance(expected_type, str) or not _json_type_matches(
            value, expected_type
        ):
            raise ComparisonStateSchemaError(
                f"{location} must have JSON type {expected_type}"
            )
    if "const" in rule and value != rule["const"]:
        raise ComparisonStateSchemaError(
            f"{location} must equal {rule['const']!r}"
        )
    if "enum" in rule:
        choices = rule["enum"]
        if not isinstance(choices, list) or value not in choices:
            raise ComparisonStateSchemaError(
                f"{location} is not an allowed enum value"
            )
    if isinstance(value, str):
        minimum = rule.get("minLength")
        if minimum is not None and (
            type(minimum) is not int or minimum < 0 or len(value) < minimum
        ):
            raise ComparisonStateSchemaError(
                f"{location} is shorter than minLength"
            )
        pattern = rule.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str):
                raise ComparisonStateSchemaError(
                    f"invalid pattern in schema at {location}"
                )
            try:
                matched = re.search(pattern, value)
            except re.error as error:
                raise ComparisonStateSchemaError(
                    f"invalid schema pattern at {location}: {error}"
                ) from error
            if matched is None:
                raise ComparisonStateSchemaError(
                    f"{location} does not match the required pattern"
                )
    if isinstance(value, Mapping):
        required = rule.get("required", [])
        properties = rule.get("properties", {})
        if not isinstance(required, list) or any(
            not isinstance(field, str) for field in required
        ):
            raise ComparisonStateSchemaError(
                f"schema required list is invalid at {location}"
            )
        if not isinstance(properties, Mapping):
            raise ComparisonStateSchemaError(
                f"schema properties is invalid at {location}"
            )
        missing = [field for field in required if field not in value]
        if missing:
            raise ComparisonStateSchemaError(
                f"{location} missing required fields: {', '.join(missing)}"
            )
        if rule.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ComparisonStateSchemaError(
                    f"{location} has additional fields: {', '.join(extra)}"
                )
        for field, child in properties.items():
            if field in value:
                if not isinstance(child, Mapping):
                    raise ComparisonStateSchemaError(
                        f"invalid property schema at {location}.{field}"
                    )
                _validate_schema_node(
                    value[field],
                    child,
                    root_schema,
                    f"{location}.{field}",
                )


def validate_comparison_state_schema(
    state: Mapping[str, Any],
    schema_path: Path | None = None,
) -> None:
    """Validate a state against the complete checked-in JSON Schema contract."""

    path = (schema_path or DEFAULT_STATE_SCHEMA).resolve()
    try:
        schema = loads_json_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ComparisonStateSchemaError(
            f"comparison state schema unavailable or invalid: {path}: {error}"
        ) from error
    if not isinstance(schema, Mapping):
        raise ComparisonStateSchemaError("comparison state schema must be an object")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ComparisonStateSchemaError(
            "comparison state schema must declare Draft 2020-12"
        )
    if schema.get("type") != "object":
        raise ComparisonStateSchemaError("comparison state schema root must be object")
    required = schema.get("required")
    properties = schema.get("properties")
    if (
        not isinstance(required, list)
        or not isinstance(properties, Mapping)
        or set(required) != set(properties)
        or schema.get("additionalProperties") is not False
    ):
        raise ComparisonStateSchemaError(
            "comparison state schema must require exactly all declared properties"
        )
    _validate_schema_node(state, schema, schema, "$")
    origin = state.get("state_origin")
    legacy = state.get("legacy_compatibility")
    native_comparable = state.get("native_protocol_comparable")
    status = state.get("comparison_status")
    if (origin == "legacy_adapter") is not (legacy is True):
        raise ComparisonStateSchemaError(
            "$.state_origin and $.legacy_compatibility are inconsistent"
        )
    expected_native = (
        origin == "native_v4"
        and legacy is False
        and status == ComparisonStatus.COMPARABLE
    )
    if native_comparable is not expected_native:
        raise ComparisonStateSchemaError(
            "$.native_protocol_comparable is inconsistent with origin and status"
        )
    if status == ComparisonStatus.COMPARABLE and origin == "native_v4":
        comparable_nonempty = (
            "source_checkpoint",
            "source_checkpoint_manifest",
            "source_run_id",
            "training_stage",
            "case_manifest",
            "bf16_output_path",
            "bf16_metrics_path",
            "quantized_output_path",
            "quantized_metrics_path",
        )
        empty = [
            field
            for field in comparable_nonempty
            if not isinstance(state.get(field), str) or not state[field]
        ]
        comparable_hashes = (
            "source_checkpoint_manifest_hash",
            "config_hash",
            "tokenizer_hash",
            "case_manifest_hash",
            "logical_cases_hash",
            "bf16_source_checkpoint_hash",
            "bf16_config_hash",
            "bf16_tokenizer_hash",
            "bf16_case_manifest_hash",
            "quant_source_checkpoint_hash",
            "quant_config_hash",
            "quant_tokenizer_hash",
            "quant_case_manifest_hash",
            "bf16_model_state_attestation_hash",
            "bf16_output_manifest_hash",
            "quant_model_state_attestation_hash",
            "quant_output_manifest_hash",
        )
        empty.extend(
            field
            for field in comparable_hashes
            if not isinstance(state.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", state[field]) is None
        )
        if empty:
            raise ComparisonStateSchemaError(
                "COMPARABLE state has empty or invalid evidence fields: "
                + ", ".join(empty)
            )


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
    generation_config = checkpoint / "generation_config.json"
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_manifest_hash": manifest_hash,
        "config_hash": sha256_file(config),
        "tokenizer_hash": tokenizer_hash,
        "generation_config_hash": (
            sha256_file(generation_config) if generation_config.is_file() else ""
        ),
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
        "generation_config_hash": "",
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
        "bf16_model_state_attestation_path": "",
        "bf16_model_state_attestation_hash": "",
        "bf16_attestation_status": "",
        "bf16_attestation_passed": False,
        "bf16_output_manifest_path": "",
        "bf16_output_manifest_hash": "",
        "quantized_output_path": "",
        "quantized_metrics_path": "",
        "quant_model_state_attestation_path": "",
        "quant_model_state_attestation_hash": "",
        "quant_attestation_status": "",
        "quant_attestation_passed": False,
        "quant_output_manifest_path": "",
        "quant_output_manifest_hash": "",
        "bf16_source_checkpoint_hash": "",
        "bf16_source_checkpoint": "",
        "bf16_source_checkpoint_manifest": "",
        "bf16_config_hash": "",
        "bf16_tokenizer_hash": "",
        "bf16_generation_config_hash": "",
        "bf16_training_stage": "",
        "bf16_source_run_id": "",
        "quant_source_checkpoint_hash": "",
        "quant_source_checkpoint": "",
        "quant_source_checkpoint_manifest": "",
        "quant_config_hash": "",
        "quant_tokenizer_hash": "",
        "quant_generation_config_hash": "",
        "quant_training_stage": "",
        "quant_source_run_id": "",
        "bf16_case_manifest_hash": "",
        "quant_case_manifest_hash": "",
        "legacy_compatibility": False,
        "state_origin": "native_v4",
        "native_protocol_comparable": False,
        "scorer": scorer_identity(),
        "bf16_arm": {"arm_type": "bf16"},
        "quantized_arm": {"arm_type": "quantized"},
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
    result["native_protocol_comparable"] = (
        status is ComparisonStatus.COMPARABLE
        and result.get("state_origin") == "native_v4"
        and result.get("legacy_compatibility") is False
    )
    validate_comparison_state_schema(result)
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


def resolve_evidence_path(state_path: Path, evidence_path: str) -> Path:
    """Resolve a state-owned path once, never searching fallback directories."""

    path = Path(evidence_path)
    if not path.is_absolute():
        path = state_path.resolve().parent / path
    return path.resolve()


def resolve_verify_files_policy(
    state: Mapping[str, Any],
    requested_verify_files: bool,
) -> bool:
    """Native-v4 evidence verification is mandatory for every caller."""

    if state.get("state_origin") == "native_v4":
        return True
    return requested_verify_files


def _verify_generation_evidence(output_path: Path) -> str | None:
    required = {
        "generated_token_ids",
        "decoded_with_special_tokens",
        "decoded_without_special_tokens",
        "effective_eos_token_ids",
        "termination_reason",
        "termination_reason_inferred",
        "hit_max_new_tokens",
        "generated_token_count",
    }
    rows = 0
    try:
        lines = output_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        return f"response output unreadable: {error}"
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = loads_json_strict(line)
        except (ValueError, TypeError) as error:
            return f"response row {number} invalid: {error}"
        if not isinstance(row, Mapping):
            return f"response row {number} must be an object"
        missing = sorted(required - set(row))
        if missing:
            return (
                f"response row {number} lacks generation evidence: "
                + ", ".join(missing)
            )
        if row.get("generation_evidence_sufficient") is False:
            return (
                f"response row {number} generation evidence insufficient: "
                + str(row.get("termination_evidence_level", "unknown"))
            )
        token_ids = row.get("generated_token_ids")
        if not isinstance(token_ids, list) or any(
            type(value) is not int for value in token_ids
        ):
            return f"response row {number} generated_token_ids unavailable or invalid"
        if type(row.get("termination_reason_inferred")) is not bool:
            return f"response row {number} termination_reason_inferred must be boolean"
        if type(row.get("hit_max_new_tokens")) is not bool:
            return f"response row {number} hit_max_new_tokens must be boolean"
        if type(row.get("generated_token_count")) is not int:
            return f"response row {number} generated_token_count must be integer"
    if rows == 0:
        return "response output contains no auditable rows"
    return None


def _verify_runtime_evidence(
    run_state: Mapping[str, Any],
    *,
    prefix: str,
    output_field: str,
    state_root: Path | None,
) -> str | None:
    attestation_field = f"{prefix}_model_state_attestation_path"
    attestation_hash_field = f"{prefix}_model_state_attestation_hash"
    manifest_field = f"{prefix}_output_manifest_path"
    manifest_hash_field = f"{prefix}_output_manifest_hash"
    paths = {}
    for field in (attestation_field, manifest_field, output_field):
        path = Path(str(run_state.get(field, "")))
        if not path.is_absolute() and state_root is not None:
            path = state_root / path
        paths[field] = path.resolve()
        if not paths[field].is_file():
            return f"{field} missing"
    attestation_path = paths[attestation_field]
    if sha256_file(attestation_path) != run_state.get(attestation_hash_field):
        return f"{attestation_hash_field} mismatch"
    try:
        from model_state_attestation import verify_attestation

        attestation = verify_attestation(
            attestation_path,
            expected_hash=str(run_state.get(attestation_hash_field, "")),
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        return f"{attestation_field} invalid: {error}"
    for field in ("run_id", "model_id", "protocol_id"):
        if attestation.get(field) != run_state.get(field):
            return f"{attestation_field} {field} mismatch"
    resolved_identity = attestation.get("resolved_identity")
    if not isinstance(resolved_identity, Mapping):
        return f"{attestation_field} lacks resolved_identity"
    identity_bindings = {
        "source_checkpoint_manifest_hash": "source_checkpoint_manifest_hash",
        "config_hash": "config_hash",
        "tokenizer_hash": "tokenizer_hash",
        "generation_config_hash": "generation_config_hash",
        "source_run_id": "source_run_id",
        "training_stage": "training_stage",
    }
    mismatched_identity = [
        attestation_key
        for attestation_key, state_key in identity_bindings.items()
        if resolved_identity.get(attestation_key) != run_state.get(state_key)
    ]
    if mismatched_identity:
        return (
            f"{attestation_field} locked identity mismatch: "
            + ", ".join(mismatched_identity)
        )
    decision = attestation.get("attestation")
    if not isinstance(decision, Mapping):
        return f"{attestation_field} lacks attestation decision"
    if decision.get("passed") is not True:
        return f"{prefix} attestation failed: {decision.get('status', 'unknown')}"
    if decision.get("status") != run_state.get(f"{prefix}_attestation_status"):
        return f"{prefix}_attestation_status mismatch"

    manifest_path = paths[manifest_field]
    if sha256_file(manifest_path) != run_state.get(manifest_hash_field):
        return f"{manifest_hash_field} mismatch"
    try:
        output_manifest = loads_json_strict(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        return f"{manifest_field} invalid: {error}"
    if not isinstance(output_manifest, Mapping):
        return f"{manifest_field} must be an object"
    if (
        output_manifest.get("model_state_attestation_hash")
        != run_state.get(attestation_hash_field)
    ):
        return f"{manifest_field} attestation binding mismatch"
    output_path = paths[output_field]
    if str(Path(str(output_manifest.get("output_path", ""))).resolve()) != str(
        output_path
    ):
        return f"{manifest_field} output path mismatch"
    if output_manifest.get("output_sha256") != sha256_file(output_path):
        return f"{manifest_field} output hash mismatch"
    if output_manifest.get("case_manifest_hash") != run_state.get(
        "case_manifest_hash"
    ):
        return f"{manifest_field} case manifest binding mismatch"
    generation_error = _verify_generation_evidence(output_path)
    if generation_error:
        return generation_error
    return None


def determine_comparison_eligibility(
    run_state: Mapping[str, Any],
    gate_metrics: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
    *,
    state_root: Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Return an explicit comparison state without mutating the supplied state."""

    validate_comparison_state_schema(run_state)
    verify_files = resolve_verify_files_policy(run_state, verify_files)
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
    if run_state.get("bf16_attestation_passed") is not True:
        return _result(
            run_state,
            ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
            "BF16 model-state attestation failed or is missing: "
            + str(run_state.get("bf16_attestation_status") or "unrecorded"),
            Stage.BF16_GATE,
        )
    if run_state.get("bf16_attestation_status") != "ATTESTED_BF16":
        return _result(
            run_state,
            ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
            "BF16 attestation status is not ATTESTED_BF16: "
            + str(run_state.get("bf16_attestation_status") or "unrecorded"),
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
                "bf16_model_state_attestation_path",
                "bf16_output_manifest_path",
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
        runtime_evidence_error = _verify_runtime_evidence(
            run_state,
            prefix="bf16",
            output_field="bf16_output_path",
            state_root=state_root,
        )
        if runtime_evidence_error:
            return _result(
                run_state,
                ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
                "BF16 runtime evidence invalid: " + runtime_evidence_error,
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
            "generation_config_hash": "generation_config_hash",
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
    if run_state.get("generation_config_hash") != run_state.get(
        "bf16_generation_config_hash"
    ):
        return _result(
            run_state,
            ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
            "BF16 generation config identity does not match the locked source",
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
    if run_state.get("quant_attestation_passed") is not True:
        return _result(
            run_state,
            ComparisonStatus.QUANTIZATION_FAILED,
            "quantized model-state attestation failed or is missing: "
            + str(run_state.get("quant_attestation_status") or "unrecorded"),
            Stage.QUANTIZATION,
        )
    if not str(run_state.get("quant_attestation_status", "")).startswith("ATTESTED_"):
        return _result(
            run_state,
            ComparisonStatus.QUANTIZATION_FAILED,
            "quantized attestation status is not attested: "
            + str(run_state.get("quant_attestation_status") or "unrecorded"),
            Stage.QUANTIZATION,
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
    if run_state.get("bf16_generation_config_hash") != run_state.get(
        "quant_generation_config_hash"
    ):
        return _result(
            run_state,
            ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
            "BF16 and quantized generation config identity differs",
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
            (
                "quantized_output_path",
                "quantized_metrics_path",
                "quant_model_state_attestation_path",
                "quant_output_manifest_path",
            ),
            state_root=state_root,
        )
        if missing:
            return _result(
                run_state,
                ComparisonStatus.QUANTIZATION_FAILED,
                "quantized artifacts missing: " + ", ".join(missing),
                Stage.QUANTIZED_EVALUATION,
            )
        runtime_evidence_error = _verify_runtime_evidence(
            run_state,
            prefix="quant",
            output_field="quantized_output_path",
            state_root=state_root,
        )
        if runtime_evidence_error:
            return _result(
                run_state,
                ComparisonStatus.QUANTIZATION_FAILED,
                "quantized runtime evidence invalid: " + runtime_evidence_error,
                Stage.QUANTIZED_EVALUATION,
            )
    return _result(
        run_state,
        ComparisonStatus.COMPARABLE,
        "",
        Stage.COMPARABLE,
    )


def quantization_authorization(
    run_state: Mapping[str, Any],
    gate_metrics: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
    *,
    state_root: Path | None = None,
    verify_files: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Return the one native-v4 launch decision used by every entrypoint."""

    result = determine_comparison_eligibility(
        run_state,
        gate_metrics,
        protocol,
        state_root=state_root,
        verify_files=verify_files,
    )
    allowed = (
        result["comparison_status"] == ComparisonStatus.ELIGIBLE_NOT_QUANTIZED
        and result["state_origin"] == "native_v4"
        and result["legacy_compatibility"] is False
    )
    return result, allowed


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
    record_id = str(record.get("run_id") or record.get("record_id") or "legacy-record")
    model_id = str(record.get("model_id") or role.split("_", 1)[0] or "legacy-model")
    state = default_run_state(
        model_id=model_id,
        model_family=str(record.get("model_family") or "legacy"),
        run_id=record_id,
        renderer_id=str(record.get("renderer_id") or "legacy_renderer"),
        state_origin="legacy_adapter",
        legacy_compatibility=True,
        native_protocol_comparable=False,
        scorer=resolve_scorer_policy(protocol_id=None, scorer_mode="legacy"),
        comparison_status=comparison,
        blocking_reason=reason,
        stage_reached=(
            Stage.COMPARABLE
            if comparison is ComparisonStatus.COMPARABLE
            else Stage.BF16_GATE
        ),
        baseline_completed=True,
        baseline_capability_passed=True,
        bf16_reconstruction_completed=(
            comparison is not ComparisonStatus.NOT_ELIGIBLE_RECONSTRUCTION_FAILED
        ),
        bf16_gate_passed=comparison is ComparisonStatus.COMPARABLE,
        quantization_requested=comparison is ComparisonStatus.COMPARABLE,
        quantization_performed=comparison is ComparisonStatus.COMPARABLE,
        quantized_evaluation_completed=comparison is ComparisonStatus.COMPARABLE,
    )
    validate_comparison_state_schema(state)
    return state


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
    try:
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
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "comparison_state_schema_invalid",
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(21) from error
    if args.write:
        atomic_write_json(args.state, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
