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
FORMAL_ENTRYPOINT_IMPLEMENTATION_VERSION = "p0-5-v3"

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
    RECORD_BF16 = "RECORD_BF16"
    RECORD_QUANT = "RECORD_QUANT"
    REFRESH_ARTIFACT_BINDINGS = "REFRESH_ARTIFACT_BINDINGS"


_TRANSITION_ENTRYPOINT = {
    FormalStateTransition.RECORD_BF16: "comparison-record-bf16",
    FormalStateTransition.RECORD_QUANT: "comparison-record-quant",
    FormalStateTransition.REFRESH_ARTIFACT_BINDINGS: "comparison-record-quant",
}


def initialize_formal_state(path: Path, state: Mapping[str, Any]) -> str:
    """Create one state path once; callers cannot select an authorization id."""

    writer_id = "comparison-state-integrity-writer"
    from formal_evidence import state_hash_path, write_state_with_integrity
    if path.exists() or state_hash_path(path).exists():
        raise ValueError(
            "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED: state already exists"
        )
    payload = dict(state)
    payload["formal_creation"] = formal_creation_record(
        "comparison-init", writer_id, path
    )
    from comparison_eligibility import validate_comparison_state_schema
    validate_comparison_state_schema(payload)
    return write_state_with_integrity(
        path,
        payload,
        _formal_capability=_capability("comparison-init", writer_id),
    )


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
    entrypoint_id = _TRANSITION_ENTRYPOINT[transition]
    from formal_evidence import (
        FormalEvidenceError,
        revalidate_formal_run_context,
        write_state_with_integrity,
    )
    current = revalidate_formal_run_context(
        context,
        allowed_entrypoint_ids=(entrypoint_id,),
        expected_arm=("bf16" if transition is FormalStateTransition.RECORD_BF16 else "quant"),
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
    if transition is FormalStateTransition.RECORD_BF16:
        if current.get("quantization_performed") is True or current.get(
            "comparison_status"
        ) == "COMPARABLE":
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_STAGE_MISMATCH",
                "BF16 transition cannot run after quantized completion",
            )
    elif transition is FormalStateTransition.RECORD_QUANT:
        if current.get("comparison_status") != "ELIGIBLE_NOT_QUANTIZED":
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_STAGE_MISMATCH",
                "quant transition requires ELIGIBLE_NOT_QUANTIZED",
            )
    else:
        allowed_changes = {
            "bf16_output_manifest_hash",
            "quant_output_manifest_hash",
            "formal_creation",
        }
        changed = {
            key
            for key in set(current) | set(target)
            if current.get(key) != target.get(key)
        }
        if not changed or changed - allowed_changes:
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                "artifact refresh may only update locked manifest hashes",
            )
        from comparison_eligibility import sha256_file
        for field, path_field in (
            ("bf16_output_manifest_hash", "bf16_output_manifest_path"),
            ("quant_output_manifest_hash", "quant_output_manifest_path"),
        ):
            if field in changed:
                manifest_path = Path(str(target[path_field]))
                if not manifest_path.is_absolute():
                    manifest_path = context.state_path.resolve().parent / manifest_path
                if target[field] != sha256_file(manifest_path.resolve()):
                    raise FormalEvidenceError(
                        "FORMAL_ENTRYPOINT_TRANSITION_NOT_ALLOWED",
                        f"{field} does not match the locked manifest",
                    )
    target["formal_creation"] = formal_creation_record(
        entrypoint_id, writer_id, context.state_path
    )
    from comparison_eligibility import validate_comparison_state_schema
    validate_comparison_state_schema(target)
    return write_state_with_integrity(
        context.state_path,
        target,
        _formal_capability=_capability(entrypoint_id, writer_id),
    )


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
    from model_state_attestation import write_output_manifest
    output = Path(args[0] if args else kwargs["output"])
    state, artifacts = resolve_formal_arm_artifacts(
        context,
        allowed_entrypoint_ids=tuple(allowed),
        arm=allowed[entrypoint_id],
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
    return write_output_manifest(
        *args,
        **kwargs,
        formal_creation=formal_creation_record(
            entrypoint_id, writer_id, manifest_path
        ),
        _formal_capability=_capability(entrypoint_id, writer_id),
    )


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
    return bind_metrics_to_output_manifest(
        manifest_path,
        metrics_path,
        context=context,
        _formal_capability=_capability("formal-scorer-main", writer_id),
    )


def write_formal_summary(
    contexts, path: Path, summary: Mapping[str, Any]
) -> str:
    writer_id = "comparison-summary-integrity-writer"
    contexts = tuple(contexts)
    if not contexts:
        raise ValueError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID: summary requires verified states"
        )
    from formal_evidence import revalidate_formal_run_context
    for context in contexts:
        revalidate_formal_run_context(
            context,
            allowed_entrypoint_ids=("comparison-summary-main",),
            expected_arm="summary",
        )
    if (
        not isinstance(summary.get("included_runs"), list)
        or not isinstance(summary.get("input_evidence_hashes"), Mapping)
    ):
        raise ValueError("formal summary lacks verified included runs/input hashes")
    from formal_evidence import write_summary_with_integrity
    payload = dict(summary)
    payload["formal_creation"] = formal_creation_record(
        "comparison-summary-main", writer_id, path
    )
    return write_summary_with_integrity(
        path,
        payload,
        _formal_capability=_capability("comparison-summary-main", writer_id),
    )


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
