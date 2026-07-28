"""Single production validation core for canonical comparison summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from case_schema import loads_json_strict
from canonical_tool_schema import scorer_identity
from comparison_eligibility import (
    ComparisonStatus,
    PROTOCOL_ID,
    determine_comparison_eligibility,
    resolve_evidence_path,
    sha256_file,
    validate_comparison_state_schema,
)
from formal_evidence import (
    FormalRunContext,
    FormalEvidenceError,
    load_and_verify_formal_run_context,
    validate_formal_metrics,
    verify_metrics_against_raw,
    verify_metrics_binding,
    verify_state_integrity,
)
from model_state_attestation import verify_attestation, verify_output_manifest
from scorer_identity import (
    ScorerIdentityError,
    hash_scorer_identity,
    validate_scorer_identity,
)


class SummaryExclusion(ValueError):
    def __init__(self, run_id: str, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.run_id = run_id
        self.code = code
        self.detail = detail


def _exclude(run_id: str, code: str, detail: str, error: Exception | None = None):
    exclusion = SummaryExclusion(run_id, code, detail)
    if error is not None:
        raise exclusion from error
    raise exclusion


def _read_object(path: Path, run_id: str, code: str) -> dict[str, Any]:
    try:
        value = loads_json_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        detail = (
            f"missing evidence: {path}"
            if isinstance(error, FileNotFoundError)
            else f"{path}: {error}"
        )
        _exclude(run_id, code, detail, error)
    if not isinstance(value, dict):
        _exclude(run_id, code, f"{path} is not a JSON object")
    return value


def _validate_attestation(
    state_path: Path,
    state: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    run_id = str(state.get("run_id", ""))
    key = f"{prefix}_model_state_attestation_path"
    attestation_path = resolve_evidence_path(state_path, str(state[key]))
    try:
        payload = verify_attestation(
            attestation_path,
            expected_hash=str(state[f"{prefix}_model_state_attestation_hash"]),
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        code = (
            "ATTESTATION_HASH_MISMATCH"
            if "hash" in str(error).lower()
            else "ATTESTATION_INVALID"
        )
        detail = (
            f"missing attestation: {attestation_path}"
            if isinstance(error, FileNotFoundError)
            else str(error)
        )
        _exclude(run_id, code, detail, error)
    decision = payload.get("attestation", {})
    status = decision.get("status")
    if (
        decision.get("passed") is not True
        or not isinstance(status, str)
        or not status.startswith("ATTESTED_")
        or state.get(f"{prefix}_attestation_passed") is not True
        or state.get(f"{prefix}_attestation_status") != status
    ):
        _exclude(
            run_id,
            "ATTESTATION_INVALID",
            f"{prefix} attestation is not a matching passed ATTESTED_* record",
        )
    requested = payload.get("requested_state", {})
    observed = payload.get("observed_state", {})
    expected_precision = "bf16" if prefix == "bf16" else None
    if expected_precision and (
        requested.get("precision") != expected_precision
        or observed.get("precision") != expected_precision
    ):
        _exclude(
            run_id,
            "ATTESTATION_ARM_MISMATCH",
            f"{prefix} attestation precision does not match the arm",
        )
    if prefix == "quant" and (
        requested.get("precision") == "bf16"
        or observed.get("precision") == "bf16"
        or requested.get("backend") != observed.get("backend")
    ):
        _exclude(
            run_id,
            "ATTESTATION_ARM_MISMATCH",
            "quant attestation backend/precision does not match the arm",
        )
    return payload


def _validate_arm(
    state_path: Path,
    state: dict[str, Any],
    *,
    prefix: str,
    identity: dict[str, Any],
    context: FormalRunContext,
) -> dict[str, Any]:
    run_id = str(state.get("run_id", ""))
    metrics_field = (
        "bf16_metrics_path" if prefix == "bf16" else "quantized_metrics_path"
    )
    raw_field = "bf16_output_path" if prefix == "bf16" else "quantized_output_path"
    manifest_field = f"{prefix}_output_manifest_path"
    manifest_hash_field = f"{prefix}_output_manifest_hash"
    metrics_path = resolve_evidence_path(state_path, str(state[metrics_field]))
    raw_path = resolve_evidence_path(state_path, str(state[raw_field]))
    manifest_path = resolve_evidence_path(state_path, str(state[manifest_field]))
    manifest_payload = _read_object(
        manifest_path, run_id, "MANIFEST_VERIFICATION_FAILED"
    )
    if "scorer_identity" not in manifest_payload:
        _exclude(
            run_id,
            "MANIFEST_IDENTITY_MISSING",
            f"{prefix} output manifest scorer identity is missing",
        )
    if "scorer_identity_sha256" not in manifest_payload:
        _exclude(
            run_id,
            "MANIFEST_IDENTITY_MISSING",
            f"{prefix} output manifest scorer identity hash is missing",
        )
    registry = manifest_payload.get("tool_registry")
    if not isinstance(registry, dict):
        _exclude(
            run_id,
            "MANIFEST_REGISTRY_MISMATCH",
            f"{prefix} output manifest registry binding is missing",
        )
    if (
        registry.get("path") != identity["tool_registry_path"]
        or registry.get("sha256") != identity["tool_registry_hash"]
    ):
        _exclude(
            run_id,
            "MANIFEST_REGISTRY_MISMATCH",
            f"{prefix} output manifest registry binding differs from state",
        )
    try:
        manifest = verify_output_manifest(
            manifest_path,
            expected_hash=str(state[manifest_hash_field]),
            expected_attestation_hash=str(
                state[f"{prefix}_model_state_attestation_hash"]
            ),
            expected_scorer_identity=identity,
        )
    except ScorerIdentityError as error:
        _exclude(run_id, "MANIFEST_IDENTITY_MISMATCH", str(error), error)
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError) as error:
        text = str(error)
        lowered = text.lower()
        code = (
            "RAW_OUTPUT_HASH_MISMATCH"
            if "response output hash mismatch" in lowered
            else "MANIFEST_IDENTITY_MISMATCH"
            if "identity" in lowered
            else "MANIFEST_VERIFICATION_FAILED"
        )
        detail = (
            f"missing evidence: {error.filename}"
            if isinstance(error, FileNotFoundError)
            else text
        )
        _exclude(run_id, code, detail, error)
    if str(Path(str(manifest.get("output_path", ""))).resolve()) != str(
        raw_path.resolve()
    ):
        _exclude(
            run_id,
            "METRICS_MANIFEST_MISMATCH",
            f"{prefix} manifest raw output path differs from state",
        )
    try:
        metrics = _read_object(metrics_path, run_id, "FORMAL_METRICS_MISSING")
        validated = validate_formal_metrics(
            metrics,
            expected_identity=identity,
            expected_raw_path=raw_path,
            expected_raw_sha256=str(manifest["output_sha256"]),
            expected_context=context,
        )
        verify_metrics_binding(manifest, metrics_path)
        verify_metrics_against_raw(
            validated,
            raw_path=raw_path,
            context=context,
        )
        return validated
    except FormalEvidenceError as error:
        _exclude(run_id, error.code, error.detail, error)


def validate_run_for_canonical_summary(state_path: Path) -> dict[str, Any]:
    """Validate every production input afresh and return a summary-safe record."""

    try:
        state = verify_state_integrity(state_path)
    except FormalEvidenceError as error:
        _exclude("", error.code, error.detail, error)
    run_id = str(state.get("run_id", ""))
    try:
        validate_comparison_state_schema(state)
    except (ValueError, TypeError) as error:
        _exclude(run_id, "STATE_SCHEMA_INVALID", str(error), error)
    if state.get("state_origin") != "native_v4" or state.get("protocol_id") != PROTOCOL_ID:
        _exclude(
            run_id,
            "LEGACY_EVIDENCE_NOT_CANONICAL",
            "state is not a native formal v4 comparison state",
        )
    if state.get("comparison_status") != ComparisonStatus.COMPARABLE:
        _exclude(
            run_id,
            "ORIGINAL_STATE_NOT_COMPARABLE",
            f"original comparison status is {state.get('comparison_status')}",
        )
    _validate_attestation(state_path, state, prefix="bf16")
    _validate_attestation(state_path, state, prefix="quant")
    try:
        context = load_and_verify_formal_run_context(
            state_path,
            entrypoint_id="formal-scorer-main",
            arm="bf16",
        )
        identity = validate_scorer_identity(
            context.scorer_identity, expected=scorer_identity()
        )
    except FormalEvidenceError as error:
        _exclude(run_id, error.code, error.detail, error)
    except ScorerIdentityError as error:
        _exclude(run_id, error.code, str(error), error)
    bf16_metrics = _validate_arm(
        state_path,
        state,
        prefix="bf16",
        identity=identity,
        context=context,
    )
    quant_metrics = _validate_arm(
        state_path,
        state,
        prefix="quant",
        identity=identity,
        context=load_and_verify_formal_run_context(
            state_path,
            entrypoint_id="formal-scorer-main",
            arm="quant",
        ),
    )
    bf16_identity = bf16_metrics.get("scorer_identity")
    quant_identity = quant_metrics.get("scorer_identity")
    if hash_scorer_identity(bf16_identity) != hash_scorer_identity(quant_identity):
        _exclude(
            run_id,
            "ARM_SCORER_IDENTITY_MISMATCH",
            "BF16 and quantized scorer identities differ",
        )
    verified = determine_comparison_eligibility(
        state,
        None,
        {"protocol_id": PROTOCOL_ID},
        state_root=state_path.resolve().parent,
        verify_files=True,
    )
    if verified.get("comparison_status") != ComparisonStatus.COMPARABLE:
        _exclude(
            run_id,
            "REVALIDATED_STATE_NOT_COMPARABLE",
            str(verified.get("blocking_reason", "eligibility revalidation failed")),
        )
    bf16_rate = float(bf16_metrics["formal_aggregate"]["exact_call_rate"])
    quant_rate = float(quant_metrics["formal_aggregate"]["exact_call_rate"])
    evidence_fields = (
        "source_checkpoint_manifest",
        "case_manifest",
        "bf16_model_state_attestation_path",
        "bf16_output_manifest_path",
        "bf16_output_path",
        "bf16_metrics_path",
        "quant_model_state_attestation_path",
        "quant_output_manifest_path",
        "quantized_output_path",
        "quantized_metrics_path",
    )
    input_hashes = {"comparison_state": sha256_file(state_path)}
    for field in evidence_fields:
        evidence_path = resolve_evidence_path(state_path, str(state[field]))
        input_hashes[field] = sha256_file(evidence_path)
    registry_path = (
        Path(__file__).resolve().parents[1] / identity["tool_registry_path"]
    )
    input_hashes["tool_registry"] = sha256_file(registry_path)
    return {
        "model_id": str(state["model_id"]),
        "run_id": run_id,
        "comparison_status": ComparisonStatus.COMPARABLE,
        "state_origin": "native_v4",
        "legacy_compatibility": False,
        "native_protocol_comparable": True,
        "blocking_reason": "",
        "quantization_effect_included": True,
        "scorer": identity,
        "source": str(state_path),
        "formal_metrics": {
            "bf16_exact_call_rate": bf16_rate,
            "quant_exact_call_rate": quant_rate,
            "quant_minus_bf16_exact_call_rate": quant_rate - bf16_rate,
        },
        "arm_change_computed": True,
        "verified_input_hashes": input_hashes,
    }
