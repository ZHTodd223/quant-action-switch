#!/usr/bin/env python3
"""Fail-closed binding for the formal v5 attestation requirements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "formal_model_state_attestation_requirements_v1"
REQUIREMENTS_VERSION = "1.0.0"
MATRIX_ID = "v5-cross-model-native-tools-matrix-v1"
MATRIX_VERSION = "1.0.0"
REQUIRED_COVERAGE = 1.0


class FormalAttestationRequirementsError(ValueError):
    """The formal matrix and attestation requirements are not identical."""


def _fail(code: str, message: str) -> None:
    raise FormalAttestationRequirementsError(f"{code}: {message}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_runtime_target_coverage(
    expected: int, observed: int, required_coverage: float
) -> float:
    coverage = observed / expected if expected else 0.0
    if (
        required_coverage != REQUIRED_COVERAGE
        or expected <= 0
        or observed != expected
        or coverage != REQUIRED_COVERAGE
    ):
        _fail(
            "ATTESTATION_COVERAGE_CONFLICT",
            f"expected={expected} observed={observed} coverage={coverage}",
        )
    return coverage


def _load_object(path: Path, *, missing_code: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(missing_code, str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("ATTESTATION_REQUIREMENTS_SCHEMA_INVALID", str(error))
    if not isinstance(value, dict):
        _fail("ATTESTATION_REQUIREMENTS_SCHEMA_INVALID", "root must be an object")
    return value


def load_formal_requirements(path: Path) -> dict[str, Any]:
    payload = _load_object(path, missing_code="ATTESTATION_REQUIREMENTS_MISSING")
    if payload.get("schema_version") != SCHEMA_VERSION:
        _fail(
            "ATTESTATION_REQUIREMENTS_SCHEMA_INVALID",
            f"schema_version={payload.get('schema_version')!r}",
        )
    if payload.get("requirements_version") != REQUIREMENTS_VERSION:
        _fail(
            "ATTESTATION_REQUIREMENTS_VERSION_MISMATCH",
            f"requirements_version={payload.get('requirements_version')!r}",
        )
    coverage = payload.get("coverage_requirements", {})
    int8 = payload.get("bnb_int8", {})
    if (
        coverage.get("minimum_target_module_coverage") != REQUIRED_COVERAGE
        or int8.get("minimum_core_projection_quantized_coverage")
        != REQUIRED_COVERAGE
        or coverage.get("require_exact_target_count") is not True
        or int8.get("require_exact_target_count") is not True
    ):
        _fail(
            "ATTESTATION_COVERAGE_CONFLICT",
            "formal INT8 coverage must be exact 1.0",
        )
    device = payload.get("device_policy", {})
    fallback = payload.get("fallback_rules", {})
    if (
        device.get("allow_cpu_offload") is not False
        or device.get("allow_disk_offload") is not False
        or fallback.get("policy") != "fail_closed"
        or fallback.get("allow_backend_fallback") is not False
        or fallback.get("allow_precision_fallback") is not False
    ):
        _fail(
            "ATTESTATION_REQUIREMENTS_SCHEMA_INVALID",
            "offload and fallback must be forbidden",
        )
    bf16 = payload.get("bf16", {})
    if (
        bf16.get("forbid_quantized_module_classes") is not True
        or bf16.get("required_quantized_module_count") != 0
    ):
        _fail(
            "ATTESTATION_REQUIREMENTS_SCHEMA_INVALID",
            "BF16 must forbid quantized modules",
        )
    return payload


def validate_matrix_requirements(
    matrix_path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    matrix = _load_object(
        matrix_path, missing_code="ATTESTATION_REQUIREMENTS_SCHEMA_INVALID"
    )
    relative = matrix.get("attestation_requirements")
    if not isinstance(relative, str) or not relative:
        _fail("ATTESTATION_REQUIREMENTS_MISSING", "matrix path is absent")
    requirements_path = (root / relative).resolve()
    requirements = load_formal_requirements(requirements_path)
    compatibility = requirements.get("matrix_compatibility", {})
    if (
        matrix.get("matrix_id") != MATRIX_ID
        or matrix.get("matrix_version") != MATRIX_VERSION
        or compatibility.get("matrix_id") != matrix.get("matrix_id")
        or compatibility.get("matrix_version") != matrix.get("matrix_version")
        or compatibility.get("protocol_id") != matrix.get("protocol_id")
    ):
        _fail(
            "ATTESTATION_REQUIREMENTS_VERSION_MISMATCH",
            "matrix compatibility does not match",
        )
    requirements_sha = sha256_file(requirements_path)
    if matrix.get("attestation_requirements_sha256") != requirements_sha:
        _fail(
            "ATTESTATION_REQUIREMENTS_HASH_MISMATCH",
            f"expected={matrix.get('attestation_requirements_sha256')} "
            f"observed={requirements_sha}",
        )
    bound = {
        item.get("path"): item.get("sha256")
        for item in matrix.get("hash_bindings", [])
        if isinstance(item, dict)
    }
    if bound.get(relative) != requirements_sha:
        _fail(
            "ATTESTATION_REQUIREMENTS_HASH_MISMATCH",
            "requirements are absent from matrix hash_bindings",
        )
    required_coverage = requirements["coverage_requirements"][
        "minimum_target_module_coverage"
    ]
    runtime_coverage = requirements["bnb_int8"][
        "minimum_core_projection_quantized_coverage"
    ]
    targets = requirements.get("target_module_requirements", {})
    registries = targets.get("models", {})
    registry_name = targets.get("registry")
    for model_key, model in matrix.get("models", {}).items():
        quant = model.get("quantization", {})
        if (
            quant.get("minimum_coverage") != REQUIRED_COVERAGE
            or required_coverage != REQUIRED_COVERAGE
            or runtime_coverage != REQUIRED_COVERAGE
        ):
            _fail(
                "ATTESTATION_COVERAGE_CONFLICT",
                f"{model_key} matrix/requirements/runtime coverage differs",
            )
        registered = registries.get(model_key)
        if not isinstance(registered, dict):
            _fail(
                "ATTESTATION_TARGET_REGISTRY_MISSING",
                f"missing registry for {model_key}",
            )
        expected = quant.get("expected_target_count")
        if (
            registry_name != quant.get("target_module_registry")
            or registered.get("expected_target_count") != expected
            or model.get("expected_target_count") != expected
        ):
            _fail(
                "ATTESTATION_TARGET_COUNT_MISMATCH",
                f"target binding differs for {model_key}",
            )
        backend = requirements.get("backend_requirements", {})
        if (
            quant.get("backend") != backend.get("backend")
            or quant.get("loader_mode") != backend.get("loader_mode")
            or quant.get("bit_width") != backend.get("observed_bits")
        ):
            _fail(
                "ATTESTATION_BACKEND_MISMATCH",
                f"backend binding differs for {model_key}",
            )
        if quant.get("allow_cpu_offload") or quant.get("allow_disk_offload"):
            _fail("ATTESTATION_OFFLOAD_DETECTED", model_key)
        if quant.get("fallback_policy") != "fail_closed":
            _fail("ATTESTATION_FALLBACK_DETECTED", model_key)
    return {
        "requirements_path": relative,
        "requirements_version": requirements["requirements_version"],
        "requirements_sha256": requirements_sha,
        "matrix_coverage": REQUIRED_COVERAGE,
        "requirements_coverage": required_coverage,
        "runtime_required_coverage": runtime_coverage,
        "runtime_coverage": runtime_coverage,
        "coverage_binding_valid": True,
        "requirements": requirements,
    }


def state_arm_binding(binding: Mapping[str, Any], arm_type: str) -> dict[str, Any]:
    return {
        "arm_type": arm_type,
        "attestation_requirements_path": binding["requirements_path"],
        "attestation_requirements_version": binding["requirements_version"],
        "attestation_requirements_sha256": binding["requirements_sha256"],
        "required_target_module_coverage": binding["runtime_required_coverage"],
    }


def load_state_bound_requirements(
    state: Mapping[str, Any], arm: str, *, root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    field = "bf16_arm" if arm == "bf16" else "quantized_arm"
    binding = state.get(field)
    if not isinstance(binding, Mapping):
        _fail("ATTESTATION_REQUIREMENTS_MISSING", f"{field} binding is absent")
    relative = binding.get("attestation_requirements_path")
    if not isinstance(relative, str) or not relative:
        _fail("ATTESTATION_REQUIREMENTS_MISSING", f"{field} path is absent")
    path = (root / relative).resolve()
    payload = load_formal_requirements(path)
    observed_sha = sha256_file(path)
    if binding.get("attestation_requirements_sha256") != observed_sha:
        _fail(
            "ATTESTATION_REQUIREMENTS_HASH_MISMATCH",
            f"{field} expected={binding.get('attestation_requirements_sha256')} "
            f"observed={observed_sha}",
        )
    if binding.get("attestation_requirements_version") != payload.get(
        "requirements_version"
    ):
        _fail(
            "ATTESTATION_REQUIREMENTS_VERSION_MISMATCH",
            f"{field} version differs",
        )
    if binding.get("required_target_module_coverage") != REQUIRED_COVERAGE:
        _fail("ATTESTATION_COVERAGE_CONFLICT", f"{field} coverage differs")
    identity = {
        "requirements_path": relative,
        "requirements_version": payload["requirements_version"],
        "requirements_sha256": observed_sha,
        "required_target_module_coverage": REQUIRED_COVERAGE,
    }
    return payload, identity
