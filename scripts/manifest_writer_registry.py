"""Machine-checkable registry of manifest writers and manifest-adjacent code."""
from __future__ import annotations

from typing import Any

CLASSIFICATIONS = {
    "FORMAL_V4",
    "LEGACY_HISTORICAL",
    "RETROSPECTIVE_DIAGNOSTIC",
    "DEVELOPMENT_ONLY",
    "UNRELATED",
}

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
        "id": "bf16-response-manifest",
        "module": "generate_bf16_responses",
        "function": "main",
        "classification": "FORMAL_V4",
        "manifest_type": "response_output_manifest_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/generate_bf16_responses.py:213",
    },
    {
        "id": "transformers-quant-response-manifest",
        "module": "generate_" + "quantized_responses",
        "function": "main",
        "classification": "FORMAL_V4",
        "manifest_type": "response_output_manifest_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/generate_" + "quantized_responses.py:281",
    },
    {
        "id": "native-quant-response-manifest",
        "module": "generate_native_" + "quantized_responses",
        "function": "main",
        "classification": "FORMAL_V4",
        "manifest_type": "response_output_manifest_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/generate_native_" + "quantized_responses.py:310",
    },
    {
        "id": "gguf-quant-response-manifest",
        "module": "generate_" + "gguf_responses",
        "function": "main",
        "classification": "FORMAL_V4",
        "manifest_type": "response_output_manifest_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": False,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/generate_" + "gguf_responses.py:391",
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
        "id": "comparison-state-writer",
        "module": "run_cross_model_comparison",
        "function": "atomic_write_json callers",
        "classification": "FORMAL_V4",
        "manifest_type": "comparison_run_state_v1",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": True,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/run_cross_model_comparison.py:311",
    },
    {
        "id": "comparison-summary-writer",
        "module": "summarize_cross_model_comparison",
        "function": "main",
        "classification": "FORMAL_V4",
        "manifest_type": "comparison_eligibility_summary",
        "formal_completion_effect": True,
        "requires_scorer_identity": True,
        "requires_tool_registry": True,
        "requires_metrics_hash": True,
        "requires_raw_output_hash": True,
        "has_verifier": True,
        "location": "scripts/summarize_cross_model_comparison.py",
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


def formal_writers() -> tuple[dict[str, Any], ...]:
    return tuple(row for row in WRITERS if row["classification"] == "FORMAL_V4")


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
