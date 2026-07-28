"""Machine-checkable registry of manifest writers and manifest-adjacent code."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

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


def formal_writers() -> tuple[dict[str, Any], ...]:
    return FORMAL_WRITERS


def formal_entrypoints() -> tuple[dict[str, str], ...]:
    return FORMAL_ENTRYPOINTS


def write_registered_response_manifest(entrypoint_id: str, *args, **kwargs):
    _require_entrypoint(entrypoint_id, "response-output-manifest-writer")
    from model_state_attestation import write_output_manifest
    return write_output_manifest(*args, **kwargs)


def write_registered_state(
    entrypoint_id: str, path: Path, state: Mapping[str, Any]
) -> str:
    _require_entrypoint(entrypoint_id, "comparison-state-integrity-writer")
    from comparison_eligibility import validate_comparison_state_schema
    validate_comparison_state_schema(state)
    from formal_evidence import write_state_with_integrity
    return write_state_with_integrity(path, state)


def bind_registered_metrics(
    entrypoint_id: str, manifest_path: Path, metrics_path: Path, **kwargs
) -> str:
    _require_entrypoint(entrypoint_id, "formal-metrics-manifest-binder")
    from formal_evidence import bind_metrics_to_output_manifest
    return bind_metrics_to_output_manifest(
        manifest_path, metrics_path, **kwargs
    )


def write_registered_summary(
    entrypoint_id: str, path: Path, summary: Mapping[str, Any]
) -> str:
    _require_entrypoint(entrypoint_id, "comparison-summary-integrity-writer")
    if (
        not isinstance(summary.get("included_runs"), list)
        or not isinstance(summary.get("input_evidence_hashes"), Mapping)
    ):
        raise ValueError("formal summary lacks verified included runs/input hashes")
    from formal_evidence import write_summary_with_integrity
    return write_summary_with_integrity(path, summary)


def _require_entrypoint(entrypoint_id: str, writer_id: str) -> None:
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
    """Use Python ASTs, not source strings, to discover registry dispatch calls."""

    dispatchers = {
        "write_registered_response_manifest",
        "write_registered_state",
        "bind_registered_metrics",
        "write_registered_summary",
    }
    found: set[tuple[str, str, str]] = set()
    for path in sorted((root / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = path.stem
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                name = (
                    child.func.id
                    if isinstance(child.func, ast.Name)
                    else child.func.attr
                    if isinstance(child.func, ast.Attribute)
                    else ""
                )
                if name not in dispatchers or not child.args:
                    continue
                first = child.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    found.add((module, node.name, first.value))
    return found


def discover_unregistered_direct_formal_writes(root: Path) -> set[tuple[str, str]]:
    """Detect new direct completion writers that bypass registered dispatchers."""

    results: set[tuple[str, str]] = set()
    registered_modules = {row["module"] for row in FORMAL_ENTRYPOINTS}
    for path in sorted((root / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            constants = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            calls = {
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
            }
            formal_marker = (
                "formal_completion_effect" in constants
                or (
                    "comparison_status" in constants
                    and "COMPARABLE" in constants
                )
            )
            if (
                formal_marker
                and calls & {"write_text", "write_bytes", "write"}
                and path.stem not in registered_modules
            ):
                results.add((path.stem, node.name))
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
