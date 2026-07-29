"""Machine-checkable registry of manifest writers and manifest-adjacent code."""
from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

CLASSIFICATIONS = {
    "FORMAL_V4",
    "LEGACY_HISTORICAL",
    "RETROSPECTIVE_DIAGNOSTIC",
    "DEVELOPMENT_ONLY",
    "UNRELATED",
}
FORMAL_CREATION_VERSION = "formal-evidence-creation-v1"
FORMAL_ENTRYPOINT_IMPLEMENTATION_VERSION = "p0-5-v4"

WRITERS: tuple[dict[str, Any], ...] = (
    {
        "id": "response-output-manifest-helper",
        "module": "model_state_attestation",
        "function": "write_output_manifest",
        "classification": "FORMAL_V4",
        "manifest_type": "response_output_manifest_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/model_state_attestation.py:1490",
    },
    {
        "id": "formal-metrics-manifest-binder",
        "module": "formal_evidence",
        "function": "bind_metrics_to_output_manifest",
        "classification": "FORMAL_V4",
        "manifest_type": "formal_metrics_binding_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": True,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/formal_evidence.py",
    },
    {
        "id": "comparison-state-integrity-writer",
        "module": "formal_evidence",
        "function": "write_state_with_integrity",
        "classification": "FORMAL_V4",
        "manifest_type": "comparison_state_hash_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": True,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/formal_evidence.py",
    },
    {
        "id": "comparison-summary-integrity-writer",
        "module": "formal_evidence",
        "function": "write_summary_with_integrity",
        "classification": "FORMAL_V4",
        "manifest_type": "comparison_summary_hash_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": True,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/formal_evidence.py",
    },
    {
        "id": "retrospective-rescore-sidecar",
        "module": "rescore_canonical_diagnostic",
        "function": "main",
        "classification": "RETROSPECTIVE_DIAGNOSTIC",
        "manifest_type": "diagnostic_sidecar",
        "formal_completion_effect": False,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": True,
        "has_verifier": False,
        "location": "scripts/rescore_canonical_diagnostic.py",
    },
    {
        "id": "generic-artifact-manifest",
        "module": "make_manifest",
        "function": "main",
        "classification": "UNRELATED",
        "manifest_type": "artifact_hash_inventory",
        "formal_completion_effect": False,
        "requires_scorer_identity": False,
        "requires_tool_registry": False,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": False,
        "has_verifier": True,
        "location": "scripts/make_manifest.py",
    },
    {
        "id": "generic-artifact-manifest-verifier",
        "module": "verify_manifest",
        "function": "verify",
        "classification": "UNRELATED",
        "manifest_type": "artifact_hash_inventory_verifier",
        "formal_completion_effect": False,
        "requires_scorer_identity": False,
        "requires_tool_registry": False,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": False,
        "has_verifier": True,
        "location": "scripts/verify_manifest.py",
    },
    {
        "id": "deterministic-executor-metrics",
        "module": "evaluate_deterministic_executor",
        "function": "main",
        "classification": "DEVELOPMENT_ONLY",
        "manifest_type": "executor_metrics",
        "formal_completion_effect": False,
        "requires_scorer_identity": False,
        "requires_tool_registry": False,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": False,
        "has_verifier": False,
        "location": "scripts/evaluate_deterministic_executor.py",
    },
)

EXCLUSIONS: tuple[dict[str, str], ...] = (
    {"pattern": "manifest.sha256.json", "reason": "generic model/data artifact inventory; no comparison completion effect", "evidence": "scripts/make_manifest.py:15"},
    {"pattern": "data_manifest.json", "reason": "input dataset construction metadata", "evidence": "scripts/build_contextual_data.py:207"},
    {"pattern": "audit_manifest.json", "reason": "blind-audit development artifact", "evidence": "scripts/build_blind_audit.py:94"},
    {"pattern": "*.manifest.json calibration", "reason": "calibration-input provenance only", "evidence": "scripts/build_gptq_calibration.py:55"},
    {"pattern": "backup/sync/fetch manifest readers", "reason": "transport verification, not writers of comparison evidence", "evidence": "scripts/backup_to_nas.py:52"},
)

FORMAL_WRITERS: tuple[dict[str, Any], ...] = tuple(
    {
        "id": (
            "response-output-manifest-writer"
            if row["id"] == "response-output-manifest-helper"
            else row["id"]
        ),
        "module": row["module"],
        "function": row["function"],
        "manifest_type": row["manifest_type"],
    }
    for row in WRITERS
    if row["classification"] == "FORMAL_V4"
)

FORMAL_ENTRYPOINTS: tuple[dict[str, str], ...] = (
    {"id":"bf16-generator-main","module":"generate_bf16_responses","function":"main","writer_id":"response-output-manifest-writer"},
    {"id":"transformers-quant-generator-main","module":"generate_quantized_responses","function":"main","writer_id":"response-output-manifest-writer"},
    {"id":"native-quant-generator-main","module":"generate_native_quantized_responses","function":"main","writer_id":"response-output-manifest-writer"},
    {"id":"gguf-generator-main","module":"generate_gguf_responses","function":"main","writer_id":"response-output-manifest-writer"},
    {"id":"comparison-init","module":"run_cross_model_comparison","function":"init_run","writer_id":"comparison-state-integrity-writer"},
    {"id":"comparison-record-bf16","module":"run_cross_model_comparison","function":"record_bf16","writer_id":"comparison-state-integrity-writer"},
    {"id":"comparison-record-quant","module":"run_cross_model_comparison","function":"record_quantized","writer_id":"comparison-state-integrity-writer"},
    {"id":"formal-scorer-main","module":"score_responses","function":"main","writer_id":"formal-metrics-manifest-binder"},
    {"id":"comparison-summary-main","module":"summarize_cross_model_comparison","function":"main","writer_id":"comparison-summary-integrity-writer"},
)


@dataclass(frozen=True)
class FormalWriteCapability:
    entrypoint_id: str
    writer_id: str
    _nonce: object


_CAPABILITY_NONCE = object()


@dataclass(frozen=True)
class FormalWriterSpec:
    id: str
    artifact_kind: str
    allowed_stages: tuple[str, ...]
    allowed_statuses: tuple[str, ...]
    allowed_arms: tuple[str, ...]
    allowed_transitions: tuple[str, ...] = ()


FORMAL_WRITER_SPECS: tuple[FormalWriterSpec, ...] = (
    FormalWriterSpec(
        "bf16-generator-main",
        "response_output_manifest_v1",
        ("INITIALIZED",),
        ("NOT_ELIGIBLE_BASELINE_FAILED",),
        ("bf16",),
        ("RECORD_BF16_GENERATION",),
    ),
    FormalWriterSpec(
        "transformers-quant-generator-main",
        "response_output_manifest_v1",
        ("BF16_GATE",),
        ("ELIGIBLE_NOT_QUANTIZED",),
        ("quant",),
        ("RECORD_QUANT_GENERATION",),
    ),
    FormalWriterSpec(
        "native-quant-generator-main",
        "response_output_manifest_v1",
        ("BF16_GATE",),
        ("ELIGIBLE_NOT_QUANTIZED",),
        ("quant",),
        ("RECORD_QUANT_GENERATION",),
    ),
    FormalWriterSpec(
        "gguf-generator-main",
        "response_output_manifest_v1",
        ("BF16_GATE",),
        ("ELIGIBLE_NOT_QUANTIZED",),
        ("quant",),
        ("RECORD_QUANT_GENERATION",),
    ),
    FormalWriterSpec(
        "comparison-record-bf16",
        "comparison_state_hash_v1",
        ("BF16_SCORED",),
        ("NOT_ELIGIBLE_BASELINE_FAILED",),
        ("bf16",),
        ("RECORD_BF16",),
    ),
    FormalWriterSpec(
        "comparison-record-quant",
        "comparison_state_hash_v1",
        ("QUANT_SCORED",),
        ("ELIGIBLE_NOT_QUANTIZED",),
        ("quant",),
        ("RECORD_QUANT",),
    ),
    FormalWriterSpec(
        "comparison-summary-main",
        "comparison_summary_hash_v1",
        ("COMPARABLE",),
        ("COMPARABLE",),
        ("summary",),
        ("RECORD_SUMMARY",),
    ),
)


def formal_writer_spec(entrypoint_id: str, *, arm: str = "") -> FormalWriterSpec:
    if entrypoint_id == "formal-scorer-main":
        if arm == "bf16":
            return FormalWriterSpec(
                entrypoint_id,
                "formal_metrics_binding_v1",
                ("BF16_GENERATION_COMPLETE",),
                ("NOT_ELIGIBLE_BASELINE_FAILED",),
                ("bf16",),
                ("RECORD_BF16_SCORE",),
            )
        if arm == "quant":
            return FormalWriterSpec(
                entrypoint_id,
                "formal_metrics_binding_v1",
                ("QUANTIZATION_COMPLETE",),
                ("ELIGIBLE_NOT_QUANTIZED",),
                ("quant",),
                ("RECORD_QUANT_SCORE",),
            )
    matches = [row for row in FORMAL_WRITER_SPECS if row.id == entrypoint_id]
    if len(matches) != 1:
        raise ValueError(
            f"FORMAL_ENTRYPOINT_CONTEXT_INVALID: no writer contract for "
            f"{entrypoint_id!r}/{arm!r}"
        )
    spec = matches[0]
    if arm and arm not in spec.allowed_arms:
        raise ValueError(
            f"FORMAL_ENTRYPOINT_ARM_MISMATCH: {entrypoint_id} does not allow {arm}"
        )
    return spec


def formal_writers() -> tuple[dict[str, Any], ...]:
    return FORMAL_WRITERS


def formal_entrypoints() -> tuple[dict[str, str], ...]:
    return FORMAL_ENTRYPOINTS


def load_formal_entrypoint_callable(entrypoint: Mapping[str, str]):
    try:
        module = importlib.import_module(str(entrypoint["module"]))
        value = getattr(module, str(entrypoint["function"]))
    except (ImportError, AttributeError, KeyError) as error:
        raise ValueError(
            f"FORMAL_ENTRYPOINT_CONTEXT_INVALID: cannot import formal entrypoint "
            f"{entrypoint.get('id', '')}"
        ) from error
    if not callable(value):
        raise ValueError(
            f"FORMAL_ENTRYPOINT_CONTEXT_INVALID: formal entrypoint is not callable: "
            f"{entrypoint.get('id', '')}"
        )
    return value


def formal_creation_record(
    entrypoint_id: str,
    writer_id: str,
    target_path: Path,
) -> dict[str, str]:
    _require_entrypoint(entrypoint_id, writer_id)
    entrypoint = next(row for row in FORMAL_ENTRYPOINTS if row["id"] == entrypoint_id)
    return {
        "schema_version": FORMAL_CREATION_VERSION,
        "entrypoint_id": entrypoint_id,
        "entrypoint_module": entrypoint["module"],
        "entrypoint_function": entrypoint["function"],
        "entrypoint_implementation_version": FORMAL_ENTRYPOINT_IMPLEMENTATION_VERSION,
        "writer_id": writer_id,
        "target_path": str(target_path.resolve()),
    }


def validate_formal_creation_record(
    value: Any,
    *,
    writer_id: str,
    target_path: Path,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID: formal creation record is missing"
        )
    entrypoint_id = str(value.get("entrypoint_id", ""))
    try:
        expected = formal_creation_record(entrypoint_id, writer_id, target_path)
    except ValueError as error:
        raise ValueError(
            f"FORMAL_ENTRYPOINT_UNREGISTERED: {entrypoint_id or '<missing>'}"
        ) from error
    if dict(value) != expected:
        raise ValueError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID: formal creation record differs "
            "from the registered callable/target"
        )
    load_formal_entrypoint_callable(
        next(row for row in FORMAL_ENTRYPOINTS if row["id"] == entrypoint_id)
    )
    return expected


def require_formal_write_capability(
    capability: Any,
    *,
    entrypoint_id: str,
    writer_id: str,
) -> None:
    if (
        not isinstance(capability, FormalWriteCapability)
        or capability._nonce is not _CAPABILITY_NONCE
        or capability.entrypoint_id != entrypoint_id
        or capability.writer_id != writer_id
    ):
        raise ValueError(
            "FORMAL_WRITER_CONTEXT_INVALID: writer was not reached through its "
            "registered formal entrypoint dispatcher"
        )


def _capability(entrypoint_id: str, writer_id: str) -> FormalWriteCapability:
    _require_entrypoint(entrypoint_id, writer_id)
    return FormalWriteCapability(entrypoint_id, writer_id, _CAPABILITY_NONCE)


class FormalStateTransition(str, Enum):
    RECORD_BF16_GENERATION = "RECORD_BF16_GENERATION"
    RECORD_BF16_SCORE = "RECORD_BF16_SCORE"
    RECORD_BF16 = "RECORD_BF16"
    RECORD_QUANT_GENERATION = "RECORD_QUANT_GENERATION"
    RECORD_QUANT_SCORE = "RECORD_QUANT_SCORE"
    RECORD_QUANT = "RECORD_QUANT"
    RECORD_SUMMARY = "RECORD_SUMMARY"


FORMAL_TRANSITION_GRAPH: dict[FormalStateTransition, dict[str, Any]] = {
    FormalStateTransition.RECORD_BF16_GENERATION: {
        "entrypoint_id": "bf16-generator-main",
        "source_stage": "INITIALIZED",
        "target_stages": ("BF16_GENERATION_COMPLETE",),
    },
    FormalStateTransition.RECORD_BF16_SCORE: {
        "entrypoint_id": "formal-scorer-main",
        "source_stage": "BF16_GENERATION_COMPLETE",
        "target_stages": ("BF16_SCORED",),
    },
    FormalStateTransition.RECORD_BF16: {
        "entrypoint_id": "comparison-record-bf16",
        "source_stage": "BF16_SCORED",
        "target_stages": ("BASELINE", "RECONSTRUCTION", "BF16_GATE"),
    },
    FormalStateTransition.RECORD_QUANT_GENERATION: {
        "entrypoint_id": "quant-generator-dispatch",
        "source_stage": "BF16_GATE",
        "target_stages": ("QUANTIZATION_COMPLETE",),
    },
    FormalStateTransition.RECORD_QUANT_SCORE: {
        "entrypoint_id": "formal-scorer-main",
        "source_stage": "QUANTIZATION_COMPLETE",
        "target_stages": ("QUANT_SCORED",),
    },
    FormalStateTransition.RECORD_QUANT: {
        "entrypoint_id": "comparison-record-quant",
        "source_stage": "QUANT_SCORED",
        "target_stages": ("QUANTIZATION", "QUANTIZED_EVALUATION", "COMPARABLE"),
    },
    FormalStateTransition.RECORD_SUMMARY: {
        "entrypoint_id": "comparison-summary-main",
        "source_stage": "COMPARABLE",
        "target_stages": ("SUMMARY_COMPLETE",),
    },
}


_TRANSITION_ENTRYPOINT = {
    FormalStateTransition.RECORD_BF16_GENERATION: "bf16-generator-main",
    FormalStateTransition.RECORD_BF16_SCORE: "formal-scorer-main",
    FormalStateTransition.RECORD_BF16: "comparison-record-bf16",
    FormalStateTransition.RECORD_QUANT: "comparison-record-quant",
    FormalStateTransition.RECORD_QUANT_GENERATION: "",
    FormalStateTransition.RECORD_QUANT_SCORE: "formal-scorer-main",
    FormalStateTransition.RECORD_SUMMARY: "comparison-summary-main",
}


_INITIAL_STATE_FIELDS = {
    "stage_reached": "INITIALIZED",
    "baseline_completed": False,
    "baseline_capability_passed": False,
    "bf16_reconstruction_completed": False,
    "bf16_gate_passed": False,
    "quantization_requested": False,
    "quantization_performed": False,
    "quantized_evaluation_completed": False,
    "abnormal_termination": False,
    "comparison_status": "NOT_ELIGIBLE_BASELINE_FAILED",
    "blocking_reason": "formal run initialized; baseline has not completed",
    "bf16_model_state_attestation_hash": "",
    "bf16_attestation_status": "",
    "bf16_attestation_passed": False,
    "bf16_output_manifest_hash": "",
    "quant_model_state_attestation_hash": "",
    "quant_attestation_status": "",
    "quant_attestation_passed": False,
    "quant_output_manifest_hash": "",
    "quant_source_checkpoint_hash": "",
    "quant_source_checkpoint": "",
    "quant_source_checkpoint_manifest": "",
    "quant_config_hash": "",
    "quant_tokenizer_hash": "",
    "quant_generation_config_hash": "",
    "quant_training_stage": "",
    "quant_source_run_id": "",
    "quant_case_manifest_hash": "",
    "native_protocol_comparable": False,
}


def initialize_formal_state(path: Path, state: Mapping[str, Any]) -> str:
    """Create the one fixed formal initial state; lifecycle overrides fail closed."""

    writer_id = "comparison-state-integrity-writer"
    from formal_evidence import (
        state_hash_path,
        verify_state_integrity,
        write_state_with_integrity,
    )
    if path.exists() or state_hash_path(path).exists():
        raise ValueError(
            "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED: state already exists"
        )
    payload = dict(state)
    invalid = [
        field
        for field, expected in _INITIAL_STATE_FIELDS.items()
        if field in payload and payload[field] != expected
    ]
    if invalid:
        stage_fields = {"stage_reached"} & set(invalid)
        status_fields = {"comparison_status"} & set(invalid)
        code = (
            "FORMAL_STATE_INITIAL_STAGE_INVALID"
            if stage_fields
            else "FORMAL_STATE_INITIAL_STATUS_INVALID"
            if status_fields
            else "FORMAL_STATE_INITIALIZATION_OVERRIDE_FORBIDDEN"
        )
        raise ValueError(f"{code}: forbidden initial overrides: {', '.join(invalid)}")
    payload.update(_INITIAL_STATE_FIELDS)
    payload["formal_creation"] = formal_creation_record(
        "comparison-init", writer_id, path
    )
    from comparison_eligibility import validate_comparison_state_schema
    validate_comparison_state_schema(payload)
    digest = write_state_with_integrity(
        path,
        payload,
        _formal_capability=_capability("comparison-init", writer_id),
    )
    verify_state_integrity(path)
    return digest


def transition_formal_state(
    context,
    transition: FormalStateTransition,
    state: Mapping[str, Any],
) -> str:
    """Compare-and-swap a state through one schema-constrained transition."""

    writer_id = "comparison-state-integrity-writer"
    if not isinstance(transition, FormalStateTransition):
        raise ValueError(
            "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED: invalid transition"
        )
    entrypoint_id = (
        context.entrypoint_id
        if transition is FormalStateTransition.RECORD_QUANT_GENERATION
        else _TRANSITION_ENTRYPOINT[transition]
    )
    spec = formal_writer_spec(entrypoint_id, arm=context.arm)
    if transition.value not in spec.allowed_transitions:
        raise ValueError(
            "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED: transition is not in "
            f"the fixed writer contract for {entrypoint_id}"
        )
    from formal_evidence import (
        FormalEvidenceError,
        revalidate_formal_run_context,
        write_state_with_integrity,
    )
    current = revalidate_formal_run_context(
        context,
        allowed_entrypoint_ids=(entrypoint_id,),
        expected_arm=spec.allowed_arms[0],
        allowed_stages=spec.allowed_stages,
        allowed_statuses=spec.allowed_statuses,
        artifact_kind=spec.artifact_kind,
    )
    target = dict(state)
    for field in (
        "run_id",
        "protocol_id",
        "source_checkpoint",
        "source_checkpoint_manifest_hash",
        "case_manifest_hash",
        "logical_cases_hash",
    ):
        if target.get(field) != current.get(field):
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                f"state transition changes locked field {field}",
            )
    if current.get("protocol_id") == "agent_toolcall_protocol_v5_research_validity":
        for field in (
            "logical_case_manifest_sha256",
            "logical_expectations_sha256",
            "logical_case_ids",
            "logical_case_count",
            "renderer_id",
            "renderer_version",
            "rendered_case_manifest_sha256",
        ):
            if target.get(field) != current.get(field):
                raise FormalEvidenceError(
                    "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                    f"state transition changes locked field {field}",
                )
    if transition in {
        FormalStateTransition.RECORD_BF16_GENERATION,
        FormalStateTransition.RECORD_QUANT_GENERATION,
    }:
        prefix = (
            "bf16"
            if transition is FormalStateTransition.RECORD_BF16_GENERATION
            else "quant"
        )
        expected_stage = (
            "BF16_GENERATION_COMPLETE"
            if prefix == "bf16"
            else "QUANTIZATION_COMPLETE"
        )
        allowed_changes = {
            "stage_reached",
            f"{prefix}_model_state_attestation_hash",
            f"{prefix}_attestation_status",
            f"{prefix}_attestation_passed",
            f"{prefix}_output_manifest_hash",
            "formal_creation",
        }
        changed = {
            key
            for key in set(current) | set(target)
            if current.get(key) != target.get(key)
        }
        if target.get("stage_reached") != expected_stage or changed - allowed_changes:
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "generation transition changes fields outside its fixed artifact contract",
            )
        from comparison_eligibility import resolve_evidence_path, sha256_file
        from model_state_attestation import verify_attestation, verify_output_manifest

        attestation_path = resolve_evidence_path(
            context.state_path, str(target[f"{prefix}_model_state_attestation_path"])
        )
        manifest_path = resolve_evidence_path(
            context.state_path, str(target[f"{prefix}_output_manifest_path"])
        )
        attestation = verify_attestation(attestation_path)
        decision = attestation["attestation"]
        if (
            target[f"{prefix}_model_state_attestation_hash"]
            != sha256_file(attestation_path)
            or target[f"{prefix}_attestation_status"] != decision.get("status")
            or target[f"{prefix}_attestation_passed"] is not True
            or decision.get("passed") is not True
        ):
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "generation transition attestation fields were not production verified",
            )
        verify_output_manifest(
            manifest_path,
            expected_attestation_hash=target[
                f"{prefix}_model_state_attestation_hash"
            ],
            expected_scorer_identity=target["scorer"],
        )
        if target[f"{prefix}_output_manifest_hash"] != sha256_file(manifest_path):
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "generation transition manifest hash mismatch",
            )
    elif transition in {
        FormalStateTransition.RECORD_BF16_SCORE,
        FormalStateTransition.RECORD_QUANT_SCORE,
    }:
        prefix = (
            "bf16"
            if transition is FormalStateTransition.RECORD_BF16_SCORE
            else "quant"
        )
        expected_stage = "BF16_SCORED" if prefix == "bf16" else "QUANT_SCORED"
        changed = {
            key
            for key in set(current) | set(target)
            if current.get(key) != target.get(key)
        }
        if target.get("stage_reached") != expected_stage or changed - {
            "stage_reached",
            f"{prefix}_output_manifest_hash",
            "formal_creation",
        }:
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "scoring transition changes fields outside its fixed metrics contract",
            )
        from comparison_eligibility import resolve_evidence_path, sha256_file
        from formal_evidence import verify_metrics_binding

        manifest_path = resolve_evidence_path(
            context.state_path, str(target[f"{prefix}_output_manifest_path"])
        )
        metrics_field = (
            "bf16_metrics_path" if prefix == "bf16" else "quantized_metrics_path"
        )
        metrics_path = resolve_evidence_path(
            context.state_path, str(target[metrics_field])
        )
        from case_schema import loads_json_strict
        manifest = loads_json_strict(manifest_path.read_text(encoding="utf-8"))
        verify_metrics_binding(manifest, metrics_path)
        if target[f"{prefix}_output_manifest_hash"] != sha256_file(manifest_path):
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "scoring transition manifest hash mismatch",
            )
    elif transition is FormalStateTransition.RECORD_BF16:
        if target.get("quantization_requested") is True or target.get(
            "quantization_performed"
        ) is True or target.get("comparison_status") == "COMPARABLE":
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_STAGE_MISMATCH",
                "BF16 transition cannot skip to quantized or comparable state",
            )
        from comparison_eligibility import determine_comparison_eligibility
        recomputed = determine_comparison_eligibility(
            target,
            None,
            {"protocol_id": target["protocol_id"]},
            state_root=context.state_path.resolve().parent,
            verify_files=True,
        )
        if target != recomputed:
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "RECORD_BF16 target was not produced by production eligibility",
            )
    elif transition is FormalStateTransition.RECORD_QUANT:
        from comparison_eligibility import determine_comparison_eligibility
        recomputed = determine_comparison_eligibility(
            target,
            None,
            {"protocol_id": target["protocol_id"]},
            state_root=context.state_path.resolve().parent,
            verify_files=True,
        )
        if target != recomputed:
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "RECORD_QUANT target was not produced by production eligibility",
            )
    elif transition is FormalStateTransition.RECORD_SUMMARY:
        expected = dict(current)
        expected["stage_reached"] = "SUMMARY_COMPLETE"
        if target != expected:
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "summary completion may only advance the fixed lifecycle stage",
            )
    target["formal_creation"] = formal_creation_record(
        entrypoint_id, writer_id, context.state_path
    )
    from comparison_eligibility import validate_comparison_state_schema
    validate_comparison_state_schema(target)
    digest = write_state_with_integrity(
        context.state_path,
        target,
        _formal_capability=_capability(entrypoint_id, writer_id),
    )
    from formal_evidence import verify_state_integrity
    verify_state_integrity(context.state_path)
    return digest


def write_formal_response_manifest(context, *args, **kwargs):
    writer_id = "response-output-manifest-writer"
    entrypoint_id = context.entrypoint_id
    allowed = {
        "bf16-generator-main": "bf16",
        "transformers-quant-generator-main": "quant",
        "native-quant-generator-main": "quant",
        "gguf-generator-main": "quant",
    }
    if entrypoint_id not in allowed:
        raise ValueError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID: generator context is required"
        )
    from formal_evidence import resolve_formal_arm_artifacts
    from model_state_attestation import verify_output_manifest, write_output_manifest
    spec = formal_writer_spec(entrypoint_id, arm=allowed[entrypoint_id])
    output = Path(args[0] if args else kwargs["output"])
    state, artifacts = resolve_formal_arm_artifacts(
        context,
        allowed_entrypoint_ids=tuple(allowed),
        arm=allowed[entrypoint_id],
        allowed_stages=spec.allowed_stages,
        allowed_statuses=spec.allowed_statuses,
        artifact_kind=spec.artifact_kind,
    )
    if output.resolve() != artifacts["raw"]:
        raise ValueError(
            "FORMAL_ENTRYPOINT_ARM_MISMATCH: output is not the locked arm path"
        )
    identity = kwargs.get("scorer_identity_value")
    if identity != state.get("scorer"):
        raise ValueError(
            "SCORER_IDENTITY_HASH_MISMATCH: writer identity differs from state"
        )
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    result = write_output_manifest(
        *args,
        **kwargs,
        formal_creation=formal_creation_record(
            entrypoint_id, writer_id, manifest_path
        ),
        _formal_capability=_capability(entrypoint_id, writer_id),
    )
    verify_output_manifest(
        result[0],
        expected_attestation_hash=kwargs.get("attestation_hash"),
        expected_scorer_identity=identity,
    )
    from comparison_eligibility import resolve_evidence_path, sha256_file
    from formal_evidence import verify_state_integrity
    from model_state_attestation import verify_attestation

    prefix = "bf16" if allowed[entrypoint_id] == "bf16" else "quant"
    target = verify_state_integrity(context.state_path)
    attestation_path = resolve_evidence_path(
        context.state_path,
        str(target[f"{prefix}_model_state_attestation_path"]),
    )
    decision = verify_attestation(attestation_path)["attestation"]
    target.update(
        stage_reached=(
            "BF16_GENERATION_COMPLETE"
            if prefix == "bf16"
            else "QUANTIZATION_COMPLETE"
        ),
        **{
            f"{prefix}_model_state_attestation_hash": sha256_file(attestation_path),
            f"{prefix}_attestation_status": str(decision.get("status", "")),
            f"{prefix}_attestation_passed": decision.get("passed") is True,
            f"{prefix}_output_manifest_hash": sha256_file(result[0]),
        },
    )
    transition_formal_state(
        context,
        (
            FormalStateTransition.RECORD_BF16_GENERATION
            if prefix == "bf16"
            else FormalStateTransition.RECORD_QUANT_GENERATION
        ),
        target,
    )
    return result


def bind_formal_metrics(
    context,
    manifest_path: Path,
    metrics_path: Path,
) -> str:
    writer_id = "formal-metrics-manifest-binder"
    if context.entrypoint_id != "formal-scorer-main":
        raise ValueError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID: scorer context is required"
        )
    from formal_evidence import bind_metrics_to_output_manifest
    spec = formal_writer_spec(context.entrypoint_id, arm=context.arm)
    digest = bind_metrics_to_output_manifest(
        manifest_path,
        metrics_path,
        context=context,
        allowed_stages=spec.allowed_stages,
        allowed_statuses=spec.allowed_statuses,
        artifact_kind=spec.artifact_kind,
        _formal_capability=_capability("formal-scorer-main", writer_id),
    )
    from comparison_eligibility import sha256_file
    from formal_evidence import verify_state_integrity

    target = verify_state_integrity(context.state_path)
    prefix = "bf16" if context.arm == "bf16" else "quant"
    target["stage_reached"] = (
        "BF16_SCORED" if prefix == "bf16" else "QUANT_SCORED"
    )
    target[f"{prefix}_output_manifest_hash"] = sha256_file(manifest_path)
    transition_formal_state(
        context,
        (
            FormalStateTransition.RECORD_BF16_SCORE
            if prefix == "bf16"
            else FormalStateTransition.RECORD_QUANT_SCORE
        ),
        target,
    )
    return digest


def write_formal_summary(contexts, path: Path) -> dict[str, Any]:
    writer_id = "comparison-summary-integrity-writer"
    contexts = tuple(contexts)
    if not contexts:
        raise ValueError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID: summary requires verified states"
        )
    from formal_evidence import (
        load_and_verify_formal_run_context,
        revalidate_formal_run_context,
        verify_state_integrity,
    )
    spec = formal_writer_spec("comparison-summary-main", arm="summary")
    for context in contexts:
        revalidate_formal_run_context(
            context,
            allowed_entrypoint_ids=("comparison-summary-main",),
            expected_arm="summary",
            allowed_stages=spec.allowed_stages,
            allowed_statuses=spec.allowed_statuses,
            artifact_kind=spec.artifact_kind,
        )
    from canonical_summary_validation import (
        compute_canonical_comparison_summary,
        verify_formal_summary,
    )
    from formal_evidence import write_summary_with_integrity

    payload = compute_canonical_comparison_summary(contexts)
    payload["formal_creation"] = formal_creation_record(
        "comparison-summary-main", writer_id, path
    )
    write_summary_with_integrity(
        path,
        payload,
        _formal_capability=_capability("comparison-summary-main", writer_id),
    )
    verify_formal_summary(path, contexts)
    for context in contexts:
        target = verify_state_integrity(context.state_path)
        target["stage_reached"] = "SUMMARY_COMPLETE"
        transition_formal_state(
            context,
            FormalStateTransition.RECORD_SUMMARY,
            target,
        )
    final_contexts = tuple(
        load_and_verify_formal_run_context(
            context.state_path,
            entrypoint_id="comparison-summary-main",
            arm="summary",
        )
        for context in contexts
    )
    payload = compute_canonical_comparison_summary(final_contexts)
    payload["formal_creation"] = formal_creation_record(
        "comparison-summary-main", writer_id, path
    )
    write_summary_with_integrity(
        path,
        payload,
        _formal_capability=_capability("comparison-summary-main", writer_id),
    )
    return verify_formal_summary(path, final_contexts)


def write_registered_response_manifest(*args, **kwargs):
    raise ValueError(
        "FORMAL_WRITER_CONTEXT_INVALID: caller-supplied entrypoint dispatch removed"
    )


def write_registered_state(*args, **kwargs):
    raise ValueError(
        "FORMAL_WRITER_CONTEXT_INVALID: caller-supplied entrypoint dispatch removed"
    )


def bind_registered_metrics(*args, **kwargs):
    raise ValueError(
        "FORMAL_WRITER_CONTEXT_INVALID: caller-supplied entrypoint dispatch removed"
    )


def write_registered_summary(*args, **kwargs):
    raise ValueError(
        "FORMAL_WRITER_CONTEXT_INVALID: caller-supplied entrypoint dispatch removed"
    )


def _require_entrypoint(entrypoint_id: str, writer_id: str) -> None:
    if writer_id == "comparison-state-integrity-writer" and any(
        row["id"] == entrypoint_id for row in FORMAL_ENTRYPOINTS
    ):
        return
    matches = [
        row for row in FORMAL_ENTRYPOINTS
        if row["id"] == entrypoint_id and row["writer_id"] == writer_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"unregistered formal entrypoint/writer binding: "
            f"{entrypoint_id} -> {writer_id}"
        )


def discover_formal_entrypoint_calls(root: Path) -> set[tuple[str, str, str]]:
    """Resolve direct/module aliases and discover registered dispatcher calls."""

    dispatchers = {
        "write_formal_response_manifest": "response-output-manifest-writer",
        "initialize_formal_state": "comparison-state-integrity-writer",
        "transition_formal_state": "comparison-state-integrity-writer",
        "bind_formal_metrics": "formal-metrics-manifest-binder",
        "write_formal_summary": "comparison-summary-integrity-writer",
    }
    found: set[tuple[str, str, str]] = set()
    for path in sorted((root / "scripts").glob("*.py")):
        if path.stem == "formal_entrypoint_contracts":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = path.stem
        aliases: dict[str, str] = {}
        module_aliases: set[str] = set()
        for child in ast.walk(tree):
            if isinstance(child, ast.ImportFrom):
                for name in child.names:
                    if name.name in dispatchers:
                        aliases[name.asname or name.name] = name.name
            elif isinstance(child, ast.Import):
                for name in child.names:
                    if name.name.endswith("manifest_writer_registry"):
                        module_aliases.add(name.asname or name.name)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                name = (
                    aliases.get(child.func.id, child.func.id)
                    if isinstance(child.func, ast.Name)
                    else child.func.attr
                    if isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id in module_aliases
                    else ""
                )
                if name not in dispatchers:
                    continue
                matches = [
                    spec
                    for spec in FORMAL_ENTRYPOINTS
                    if spec["module"] == module
                    and spec["function"] == node.name
                    and spec["writer_id"] == dispatchers[name]
                ]
                for spec in matches:
                    found.add((module, node.name, spec["id"]))
    return found


def discover_unregistered_direct_formal_writes(root: Path) -> set[tuple[str, str]]:
    """Detect aliased/wrapped public formal writers and direct schema writes."""

    results: set[tuple[str, str]] = set()
    direct_writers = {
        "write_output_manifest",
        "write_state_with_integrity",
        "bind_metrics_to_output_manifest",
        "write_summary_with_integrity",
    }
    writer_modules = {
        "model_state_attestation",
        "formal_evidence",
    }
    schema_markers = {
        "formal_completion_effect",
        "comparison_status",
        "COMPARABLE",
        "included_runs",
        "FORMAL_CANONICAL_METRICS",
        "formal_aggregate",
        "formal_creation",
    }
    for path in sorted((root / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases: dict[str, str] = {}
        module_aliases: set[str] = set()
        for child in ast.walk(tree):
            if isinstance(child, ast.ImportFrom) and (
                child.module or ""
            ).split(".")[-1] in writer_modules:
                for name in child.names:
                    if name.name in direct_writers:
                        aliases[name.asname or name.name] = name.name
            elif isinstance(child, ast.Import):
                for name in child.names:
                    if name.name.split(".")[-1] in writer_modules:
                        module_aliases.add(name.asname or name.name)
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        direct_sensitive: set[str] = set()
        local_edges: dict[str, set[str]] = {name: set() for name in functions}
        for name, node in functions.items():
            constants = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            write_methods: set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                resolved = ""
                if isinstance(child.func, ast.Name):
                    resolved = aliases.get(child.func.id, child.func.id)
                    if child.func.id in functions:
                        local_edges[name].add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if (
                        isinstance(child.func.value, ast.Name)
                        and child.func.value.id in module_aliases
                    ):
                        resolved = child.func.attr
                    write_methods.add(child.func.attr)
                if resolved in direct_writers:
                    direct_sensitive.add(name)
            if (
                constants & schema_markers
                and write_methods & {"write_text", "write_bytes", "write"}
            ):
                direct_sensitive.add(name)
        changed = True
        while changed:
            changed = False
            for caller, callees in local_edges.items():
                if caller not in direct_sensitive and callees & direct_sensitive:
                    direct_sensitive.add(caller)
                    changed = True
        if path.stem == "manifest_writer_registry" or path.stem in writer_modules:
            continue
        for name in direct_sensitive:
            results.add((path.stem, name))
    return results


def validate_registry() -> None:
    required = {
        "id", "module", "function", "classification", "manifest_type",
        "formal_completion_effect", "requires_scorer_identity",
        "requires_tool_registry", "requires_metrics_hash",
        "requires_raw_output_hash", "has_verifier", "location",
    }
    ids: set[str] = set()
    for row in WRITERS:
        if set(row) != required:
            raise ValueError(f"manifest writer registry fields invalid: {row.get('id')}")
        if row["id"] in ids:
            raise ValueError(f"duplicate manifest writer id: {row['id']}")
        ids.add(row["id"])
        if row["classification"] not in CLASSIFICATIONS:
            raise ValueError(f"invalid classification: {row['classification']}")
        if row["classification"] == "FORMAL_V4":
            for field in (
                "formal_completion_effect", "requires_scorer_identity",
                "requires_tool_registry", "requires_raw_output_hash", "has_verifier",
            ):
                if row[field] is not True:
                    raise ValueError(f"{row['id']} lacks formal binding: {field}")
    writer_ids = {row["id"] for row in FORMAL_WRITERS}
    if len(writer_ids) != len(FORMAL_WRITERS):
        raise ValueError("duplicate formal writer id")
    entrypoint_ids = {row["id"] for row in FORMAL_ENTRYPOINTS}
    if len(entrypoint_ids) != len(FORMAL_ENTRYPOINTS):
        raise ValueError("duplicate formal entrypoint id")
    for row in FORMAL_ENTRYPOINTS:
        if row["writer_id"] not in writer_ids:
            raise ValueError(
                f"{row['id']} references unknown writer {row['writer_id']}"
            )
    if set(FORMAL_TRANSITION_GRAPH) != set(FormalStateTransition):
        raise ValueError("formal transition graph does not cover the transition enum")
    for transition, graph_row in FORMAL_TRANSITION_GRAPH.items():
        if (
            not graph_row["source_stage"]
            or not graph_row["target_stages"]
            or not all(graph_row["target_stages"])
        ):
            raise ValueError(f"incomplete transition graph row: {transition.value}")
