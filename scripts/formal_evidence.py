"""Integrity and semantic bindings for production v4 comparison evidence."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from case_schema import loads_json_strict
from comparison_eligibility import FORMAL_PROTOCOL_IDS, sha256_file
from scorer_identity import hash_scorer_identity, validate_scorer_identity

FORMAL_METRICS_KIND = "FORMAL_CANONICAL_METRICS"
FORMAL_METRICS_SCHEMA = "formal-canonical-metrics-v1"
FORMAL_METRIC_SOURCE = (
    "production_row_results:"
    "strict_whole_response_valid+canonical_schema_valid+exact_call"
)
FORMAL_METRIC_VERSION = "p0-2-strict-formal-v2"
FORMAL_PRODUCER_VERSION = "production-canonical-scorer-v1"
STATE_HASH_SUFFIX = ".sha256"

NONFORMAL_PROVENANCE_FIELDS = frozenset(
    {
        "source_historical_metrics_path",
        "source_legacy_metrics_path",
        "historical_source",
        "development_only",
        "identity_unknown",
    }
)


class FormalEvidenceError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.detail = message


@dataclass(frozen=True)
class FormalRunContext:
    protocol_id: str
    run_id: str
    state_path: Path
    state_sha256: str
    scorer_identity: dict[str, Any]
    scorer_identity_sha256: str
    registry_path: Path
    registry_sha256: str
    entrypoint_id: str
    arm: str = ""
    stage: str = ""
    state_status: str = ""


def canonical_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_hash_path(state_path: Path) -> Path:
    return state_path.with_suffix(state_path.suffix + STATE_HASH_SUFFIX)


def _atomic_write_state(state_path: Path, state: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, state_path)
    digest = hashlib.sha256(encoded).hexdigest()
    hash_path = state_hash_path(state_path)
    hash_temporary = hash_path.with_suffix(hash_path.suffix + ".tmp")
    hash_temporary.write_text(digest + "\n", encoding="ascii", newline="\n")
    os.replace(hash_temporary, hash_path)
    return digest


def write_state_with_integrity(
    state_path: Path,
    state: Mapping[str, Any],
    *,
    _formal_capability: Any = None,
) -> str:
    """Write a state; native-v4 states require a registered writer capability."""

    if state.get("state_origin") in {"native_v4", "native_v5"}:
        from manifest_writer_registry import (
            require_formal_write_capability,
            validate_formal_creation_record,
        )

        creation = state.get("formal_creation")
        entrypoint_id = (
            str(creation.get("entrypoint_id", ""))
            if isinstance(creation, Mapping)
            else ""
        )
        require_formal_write_capability(
            _formal_capability,
            entrypoint_id=entrypoint_id,
            writer_id="comparison-state-integrity-writer",
        )
        validate_formal_creation_record(
            creation,
            writer_id="comparison-state-integrity-writer",
            target_path=state_path,
        )
    return _atomic_write_state(state_path, state)


def write_summary_with_integrity(
    summary_path: Path,
    summary: Mapping[str, Any],
    *,
    _formal_capability: Any = None,
) -> str:
    """Write a verified summary through the registered summary entrypoint."""

    from manifest_writer_registry import (
        require_formal_write_capability,
        validate_formal_creation_record,
    )

    creation = summary.get("formal_creation")
    entrypoint_id = (
        str(creation.get("entrypoint_id", ""))
        if isinstance(creation, Mapping)
        else ""
    )
    require_formal_write_capability(
        _formal_capability,
        entrypoint_id=entrypoint_id,
        writer_id="comparison-summary-integrity-writer",
    )
    validate_formal_creation_record(
        creation,
        writer_id="comparison-summary-integrity-writer",
        target_path=summary_path,
    )
    encoded = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, summary_path)
    digest = hashlib.sha256(encoded).hexdigest()
    hash_path = summary_path.with_suffix(summary_path.suffix + ".sha256")
    hash_temporary = hash_path.with_suffix(hash_path.suffix + ".tmp")
    hash_temporary.write_text(digest + "\n", encoding="ascii", newline="\n")
    os.replace(hash_temporary, hash_path)
    return digest


def verify_state_integrity(state_path: Path) -> dict[str, Any]:
    hash_path = state_hash_path(state_path)
    if not hash_path.is_file():
        raise FormalEvidenceError(
            "MANIFEST_VERIFICATION_FAILED", "comparison state hash sidecar missing"
        )
    expected = hash_path.read_text(encoding="ascii").strip()
    actual = sha256_file(state_path)
    if expected != actual:
        raise FormalEvidenceError(
            "STATE_HASH_MISMATCH", "comparison state content hash differs from sidecar"
        )
    try:
        value = loads_json_strict(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FormalEvidenceError(
            "MANIFEST_VERIFICATION_FAILED",
            f"comparison state is invalid: {error}",
        ) from error
    if not isinstance(value, dict):
        raise FormalEvidenceError(
            "MANIFEST_VERIFICATION_FAILED", "comparison state is not an object"
        )
    if value.get("state_origin") in {"native_v4", "native_v5"}:
        from manifest_writer_registry import validate_formal_creation_record

        try:
            validate_formal_creation_record(
                value.get("formal_creation"),
                writer_id="comparison-state-integrity-writer",
                target_path=state_path,
            )
        except ValueError as error:
            code = str(error).split(":", 1)[0]
            raise FormalEvidenceError(code, str(error)) from error
    return value


def load_and_verify_formal_run_context(
    state_path: Path,
    *,
    entrypoint_id: str,
    arm: str | None = None,
) -> FormalRunContext:
    """Construct a formal context only from an integrity-locked native-v4 state."""

    state = verify_state_integrity(state_path)
    from canonical_tool_schema import scorer_identity
    from comparison_eligibility import validate_comparison_state_schema
    from manifest_writer_registry import formal_entrypoints, load_formal_entrypoint_callable

    try:
        validate_comparison_state_schema(state)
    except (TypeError, ValueError) as error:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID", str(error)
        ) from error
    protocol_id = state.get("protocol_id")
    expected_origin = (
        "native_v5"
        if protocol_id == "agent_toolcall_protocol_v5_research_validity"
        else "native_v4"
    )
    if state.get("state_origin") != expected_origin or protocol_id not in FORMAL_PROTOCOL_IDS:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID",
            "formal context requires a supported native protocol state",
        )
    entrypoints = [row for row in formal_entrypoints() if row["id"] == entrypoint_id]
    if len(entrypoints) != 1:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_UNREGISTERED", entrypoint_id
        )
    try:
        load_formal_entrypoint_callable(entrypoints[0])
        identity = validate_scorer_identity(
            state.get("scorer", {}),
            expected=scorer_identity(protocol_id=str(protocol_id)),
        )
    except ValueError as error:
        code = getattr(error, "code", "FORMAL_ENTRYPOINT_CONTEXT_INVALID")
        raise FormalEvidenceError(code, str(error)) from error
    registry_path = (
        Path(__file__).resolve().parents[1] / identity["tool_registry_path"]
    ).resolve()
    from canonical_tool_schema import registry_hash

    registry_sha256 = registry_hash()
    if registry_sha256 != identity["tool_registry_hash"]:
        raise FormalEvidenceError(
            "TOOL_REGISTRY_HASH_MISMATCH",
            "formal context registry differs from scorer identity",
        )
    fixed_arms = {
        "bf16-generator-main": "bf16",
        "comparison-record-bf16": "bf16",
        "transformers-quant-generator-main": "quant",
        "native-quant-generator-main": "quant",
        "gguf-generator-main": "quant",
        "comparison-record-quant": "quant",
        "comparison-init": "state",
        "comparison-summary-main": "summary",
    }
    resolved_arm = arm or fixed_arms.get(entrypoint_id, "")
    if entrypoint_id == "formal-scorer-main" and resolved_arm not in {
        "bf16",
        "quant",
    }:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_ARM_MISMATCH",
            "formal scorer context requires an explicit bf16 or quant arm",
        )
    fixed_arm = fixed_arms.get(entrypoint_id)
    if fixed_arm and resolved_arm != fixed_arm:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_ARM_MISMATCH",
            f"{entrypoint_id} requires arm={fixed_arm}, got {resolved_arm}",
        )
    return FormalRunContext(
        protocol_id=str(protocol_id),
        run_id=str(state["run_id"]),
        state_path=state_path.resolve(),
        state_sha256=sha256_file(state_path),
        scorer_identity=identity,
        scorer_identity_sha256=hash_scorer_identity(identity),
        registry_path=registry_path,
        registry_sha256=registry_sha256,
        entrypoint_id=entrypoint_id,
        arm=resolved_arm,
        stage=str(state.get("stage_reached", "")),
        state_status=str(state.get("comparison_status", "")),
    )


def revalidate_formal_run_context(
    context: FormalRunContext,
    *,
    allowed_entrypoint_ids: Sequence[str],
    expected_arm: str,
    expected_run_id: str | None = None,
    allowed_stages: Sequence[str] | None = None,
    allowed_statuses: Sequence[str] | None = None,
    artifact_kind: str | None = None,
) -> dict[str, Any]:
    """Reload every trust input; a context is only a stale-detecting snapshot."""

    if not isinstance(context, FormalRunContext):
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID",
            "formal context must be a revalidated snapshot",
        )
    if context.entrypoint_id not in set(allowed_entrypoint_ids):
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID",
            f"entrypoint {context.entrypoint_id!r} is not allowed for this operation",
        )
    if context.arm != expected_arm:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_ARM_MISMATCH",
            f"context arm {context.arm!r} does not match {expected_arm!r}",
        )
    try:
        actual = load_and_verify_formal_run_context(
            context.state_path,
            entrypoint_id=context.entrypoint_id,
            arm=expected_arm,
        )
    except FormalEvidenceError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID", str(error)
        ) from error
    if artifact_kind is not None:
        from manifest_writer_registry import formal_writer_spec

        expected_kind = formal_writer_spec(
            context.entrypoint_id, arm=expected_arm
        ).artifact_kind
        if artifact_kind != expected_kind:
            raise FormalEvidenceError(
                "FORMAL_ARTIFACT_KIND_MISMATCH",
                f"artifact kind {artifact_kind!r} does not match {expected_kind!r}",
            )
    if allowed_stages is not None and actual.stage not in set(allowed_stages):
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_STAGE_MISMATCH",
            f"state stage {actual.stage!r} is not allowed",
        )
    if (
        allowed_statuses is not None
        and actual.state_status not in set(allowed_statuses)
    ):
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_STATUS_MISMATCH",
            f"state status {actual.state_status!r} is not allowed",
        )
    if context.protocol_id != actual.protocol_id:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID", "protocol snapshot mismatch"
        )
    if context.run_id != actual.run_id or (
        expected_run_id is not None and actual.run_id != expected_run_id
    ):
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_RUN_MISMATCH", "formal run snapshot mismatch"
        )
    if context.state_path.resolve() != actual.state_path:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID", "state path snapshot mismatch"
        )
    if context.scorer_identity != actual.scorer_identity:
        raise FormalEvidenceError(
            "SCORER_IDENTITY_HASH_MISMATCH", "scorer identity snapshot mismatch"
        )
    if context.scorer_identity_sha256 != actual.scorer_identity_sha256:
        raise FormalEvidenceError(
            "SCORER_IDENTITY_HASH_MISMATCH",
            "scorer identity hash snapshot mismatch",
        )
    if (
        context.registry_path.resolve() != actual.registry_path
        or context.registry_sha256 != actual.registry_sha256
    ):
        raise FormalEvidenceError(
            "TOOL_REGISTRY_HASH_MISMATCH", "tool registry snapshot mismatch"
        )
    if (
        context.state_sha256 != actual.state_sha256
        or context.stage != actual.stage
        or context.state_status != actual.state_status
    ):
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_STALE",
            "formal state changed after the context snapshot was created",
        )
    state = verify_state_integrity(actual.state_path)
    if sha256_file(actual.state_path) != actual.state_sha256:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_STALE",
            "formal state changed during context revalidation",
        )
    return state


def resolve_formal_arm_artifacts(
    context: FormalRunContext,
    *,
    allowed_entrypoint_ids: Sequence[str],
    arm: str,
    allowed_stages: Sequence[str],
    allowed_statuses: Sequence[str],
    artifact_kind: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Revalidate a context and resolve its one locked arm from the state."""

    if arm not in {"bf16", "quant"}:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_ARM_MISMATCH", f"unsupported formal arm: {arm}"
        )
    state = revalidate_formal_run_context(
        context,
        allowed_entrypoint_ids=allowed_entrypoint_ids,
        expected_arm=arm,
        allowed_stages=allowed_stages,
        allowed_statuses=allowed_statuses,
        artifact_kind=artifact_kind,
    )
    keys = (
        {
            "raw": "bf16_output_path",
            "metrics": "bf16_metrics_path",
            "manifest": "bf16_output_manifest_path",
            "attestation": "bf16_model_state_attestation_path",
        }
        if arm == "bf16"
        else {
            "raw": "quantized_output_path",
            "metrics": "quantized_metrics_path",
            "manifest": "quant_output_manifest_path",
            "attestation": "quant_model_state_attestation_path",
        }
    )
    base = context.state_path.resolve().parent
    paths: dict[str, Path] = {}
    for role, field in keys.items():
        value = Path(str(state.get(field, "")))
        paths[role] = (value if value.is_absolute() else base / value).resolve()
    return state, paths


def _provenance_error(metrics: Mapping[str, Any]) -> FormalEvidenceError | None:
    if metrics.get("retrospective") is True:
        return FormalEvidenceError(
            "RETROSPECTIVE_EVIDENCE_NOT_FORMAL",
            "retrospective evidence cannot be promoted",
        )
    if metrics.get("formal_gate_effect") is False:
        return FormalEvidenceError(
            "DIAGNOSTIC_METRICS_NOT_FORMAL",
            "diagnostic evidence cannot be promoted",
        )
    if metrics.get("metrics_kind") == "RETROSPECTIVE_DIAGNOSTIC":
        return FormalEvidenceError(
            "RETROSPECTIVE_EVIDENCE_NOT_FORMAL",
            "retrospective metrics kind cannot be promoted",
        )
    if metrics.get("evidence_class") in {
        "RETROSPECTIVE_CANONICAL_DIAGNOSTIC",
        "DEVELOPMENT_ONLY",
        "LEGACY_HISTORICAL",
        "IDENTITY_UNKNOWN",
    }:
        return FormalEvidenceError(
            "EVIDENCE_CLASS_UPGRADE_FORBIDDEN",
            f"{metrics.get('evidence_class')} cannot be promoted to CANONICAL_V4",
        )
    present = sorted(NONFORMAL_PROVENANCE_FIELDS & set(metrics))
    if present:
        return FormalEvidenceError(
            "EVIDENCE_PROVENANCE_NOT_FORMAL",
            "non-formal provenance fields are present: " + ", ".join(present),
        )
    return None


def add_formal_metrics_metadata(*args, **kwargs) -> dict[str, Any]:
    """Removed unsafe API: arbitrary dictionaries cannot be upgraded to formal."""

    raise FormalEvidenceError(
        "EVIDENCE_CLASS_UPGRADE_FORBIDDEN",
        "formal metrics must be built from production-scored row results",
    )


def compute_formal_aggregate(
    row_results: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    """The single deterministic formal aggregate implementation."""

    if not isinstance(row_results, Sequence) or isinstance(
        row_results, (str, bytes)
    ):
        raise FormalEvidenceError(
            "FORMAL_ROW_RESULTS_MISSING", "row_results must be an array"
        )
    case_ids: set[str] = set()
    counts = {
        "total": 0,
        "strict_whole_response_valid": 0,
        "canonical_schema_valid": 0,
        "exact_call": 0,
    }
    for number, row in enumerate(row_results, 1):
        if not isinstance(row, Mapping):
            raise FormalEvidenceError(
                "FORMAL_ROW_INVARIANT_VIOLATION",
                f"row {number} is not an object",
            )
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise FormalEvidenceError(
                "FORMAL_ROW_INVARIANT_VIOLATION",
                f"row {number} has a missing or duplicate case_id",
            )
        case_ids.add(case_id)
        diagnostics = row.get("parser_diagnostics_v2")
        if not isinstance(diagnostics, Mapping):
            raise FormalEvidenceError(
                "FORMAL_ROW_INVARIANT_VIOLATION",
                f"row {case_id} lacks parser_diagnostics_v2",
            )
        values = {}
        for field in (
            "strict_whole_response_valid",
            "canonical_schema_valid",
            "exact_call",
        ):
            value = diagnostics.get(field)
            if type(value) is not bool:
                raise FormalEvidenceError(
                    "FORMAL_ROW_INVARIANT_VIOLATION",
                    f"row {case_id} {field} must be boolean",
                )
            values[field] = value
        if values["exact_call"] and not (
            values["canonical_schema_valid"]
            and values["strict_whole_response_valid"]
        ):
            raise FormalEvidenceError(
                "FORMAL_ROW_INVARIANT_VIOLATION",
                f"row {case_id} exact_call violates strict/schema dependency",
            )
        if values["canonical_schema_valid"] and not values[
            "strict_whole_response_valid"
        ]:
            raise FormalEvidenceError(
                "FORMAL_ROW_INVARIANT_VIOLATION",
                f"row {case_id} schema success is not strict whole-response valid",
            )
        counts["total"] += 1
        for field, value in values.items():
            counts[field] += int(value)
    return counts | {
        "exact_call_rate": (
            counts["exact_call"] / counts["total"] if counts["total"] else 0.0
        )
    }


def build_formal_metrics_from_scored_rows(
    summary: Mapping[str, Any],
    *,
    row_results: Sequence[Mapping[str, Any]],
    context: FormalRunContext,
    source_raw_path: Path,
) -> dict[str, Any]:
    """Build formal metrics only from production scorer row results and context."""

    provenance_error = _provenance_error(summary)
    if provenance_error is not None:
        raise provenance_error
    if context.entrypoint_id != "formal-scorer-main":
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID",
            "formal metrics require the formal scorer entrypoint",
        )
    raw_path = source_raw_path.resolve()
    from manifest_writer_registry import formal_writer_spec
    spec = formal_writer_spec(context.entrypoint_id, arm=context.arm)
    recompute_stages = (
        ("BF16_GENERATION_COMPLETE", "COMPARABLE", "SUMMARY_COMPLETE")
        if context.arm == "bf16"
        else ("QUANTIZATION_COMPLETE", "COMPARABLE", "SUMMARY_COMPLETE")
    )
    recompute_statuses = (
        ("NOT_ELIGIBLE_BASELINE_FAILED", "COMPARABLE")
        if context.arm == "bf16"
        else ("ELIGIBLE_NOT_QUANTIZED", "COMPARABLE")
    )
    _, artifacts = resolve_formal_arm_artifacts(
        context,
        allowed_entrypoint_ids=("formal-scorer-main",),
        arm=context.arm,
        allowed_stages=recompute_stages,
        allowed_statuses=recompute_statuses,
        artifact_kind=spec.artifact_kind,
    )
    if raw_path != artifacts["raw"]:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_ARM_MISMATCH",
            "scorer raw input does not match the locked state arm",
        )
    if not raw_path.is_file():
        raise FormalEvidenceError(
            "RAW_OUTPUT_HASH_MISMATCH", f"formal raw input is missing: {raw_path}"
        )
    identity = validate_scorer_identity(context.scorer_identity)
    rows = [dict(row) for row in row_results]
    aggregate = compute_formal_aggregate(rows)
    row_hash = canonical_json_hash({"row_results": rows})
    aggregate_hash = canonical_json_hash({"formal_aggregate": aggregate})
    result = dict(summary)
    result.update(
        {
            "metrics_schema_version": FORMAL_METRICS_SCHEMA,
            "metrics_kind": FORMAL_METRICS_KIND,
            "evidence_class": identity["evidence_class"],
            "retrospective": False,
            "formal_gate_effect": True,
            "scorer_identity": identity,
            "scorer_identity_sha256": hash_scorer_identity(identity),
            "source_raw_path": str(raw_path),
            "source_raw_sha256": sha256_file(raw_path),
            "formal_metric_source": FORMAL_METRIC_SOURCE,
            "formal_metric_version": FORMAL_METRIC_VERSION,
            "strict_required": True,
            "diagnostic_only": False,
            "row_results": rows,
            "formal_aggregate": aggregate,
            "producer": {
                "kind": "PRODUCTION_CANONICAL_SCORER",
                "implementation_version": FORMAL_PRODUCER_VERSION,
                "protocol_id": context.protocol_id,
                "formal_run_id": context.run_id,
                "formal_state_path": str(context.state_path),
                "formal_state_sha256": context.state_sha256,
                "raw_sha256": sha256_file(raw_path),
                "row_results_sha256": row_hash,
                "formal_aggregate_sha256": aggregate_hash,
                "formal_entrypoint_id": context.entrypoint_id,
                "scorer_identity_sha256": context.scorer_identity_sha256,
                "registry_sha256": context.registry_sha256,
            },
        }
    )
    return result


def validate_formal_metrics(
    metrics: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_raw_path: Path,
    expected_raw_sha256: str,
    expected_context: FormalRunContext | None = None,
) -> dict[str, Any]:
    provenance_error = _provenance_error(metrics)
    if provenance_error is not None:
        raise provenance_error
    if metrics.get("retrospective") is not False:
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "formal retrospective=false is missing"
        )
    if metrics.get("formal_gate_effect") is not True:
        raise FormalEvidenceError(
            "DIAGNOSTIC_METRICS_NOT_FORMAL",
            "metrics do not declare formal_gate_effect=true",
        )
    if (
        metrics.get("metrics_kind") != FORMAL_METRICS_KIND
        or metrics.get("metrics_schema_version") != FORMAL_METRICS_SCHEMA
    ):
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING",
            "formal metrics kind/schema is missing or invalid",
        )
    if metrics.get("evidence_class") != expected_identity.get("evidence_class"):
        raise FormalEvidenceError(
            "EVIDENCE_CLASS_UPGRADE_FORBIDDEN",
            "formal metrics evidence class differs from the locked identity",
        )
    if (
        metrics.get("formal_metric_source") != FORMAL_METRIC_SOURCE
        or metrics.get("formal_metric_version") != FORMAL_METRIC_VERSION
        or metrics.get("strict_required") is not True
        or metrics.get("diagnostic_only") is not False
    ):
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING",
            "strict formal metric declaration is missing or invalid",
        )
    identity_value = metrics.get("scorer_identity", metrics.get("scorer"))
    try:
        identity = validate_scorer_identity(
            identity_value, expected=expected_identity
        )
    except (TypeError, ValueError) as error:
        code = getattr(error, "code", "STATE_METRICS_IDENTITY_MISMATCH")
        raise FormalEvidenceError(code, str(error)) from error
    if metrics.get("scorer_identity_sha256") != hash_scorer_identity(identity):
        raise FormalEvidenceError(
            "MANIFEST_IDENTITY_MISMATCH", "metrics scorer identity hash mismatch"
        )
    if str(Path(str(metrics.get("source_raw_path", ""))).resolve()) != str(
        expected_raw_path.resolve()
    ):
        raise FormalEvidenceError(
            "METRICS_MANIFEST_MISMATCH", "metrics source raw path mismatch"
        )
    if (
        metrics.get("source_raw_sha256") != expected_raw_sha256
        or sha256_file(expected_raw_path) != expected_raw_sha256
    ):
        raise FormalEvidenceError(
            "RAW_OUTPUT_HASH_MISMATCH", "metrics/raw content hash mismatch"
        )
    rows = metrics.get("row_results")
    if not isinstance(rows, list):
        raise FormalEvidenceError(
            "FORMAL_ROW_RESULTS_MISSING", "formal row_results are missing"
        )
    recomputed = compute_formal_aggregate(rows)
    aggregate = metrics.get("formal_aggregate")
    if not isinstance(aggregate, Mapping):
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "formal aggregate is missing"
        )
    if dict(aggregate) != recomputed:
        raise FormalEvidenceError(
            "FORMAL_AGGREGATE_MISMATCH",
            "stored formal aggregate differs from deterministic row aggregate",
        )
    producer = metrics.get("producer")
    if not isinstance(producer, Mapping):
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "production scorer receipt is missing"
        )
    expected_producer = {
        "kind": "PRODUCTION_CANONICAL_SCORER",
        "implementation_version": FORMAL_PRODUCER_VERSION,
        "protocol_id": identity["protocol_id"],
        "raw_sha256": expected_raw_sha256,
        "row_results_sha256": canonical_json_hash({"row_results": rows}),
        "formal_aggregate_sha256": canonical_json_hash(
            {"formal_aggregate": recomputed}
        ),
        "formal_entrypoint_id": "formal-scorer-main",
        "scorer_identity_sha256": hash_scorer_identity(identity),
        "registry_sha256": identity["tool_registry_hash"],
    }
    for field, expected in expected_producer.items():
        if producer.get(field) != expected:
            code = (
                "ROW_RESULTS_RECOMPUTE_MISMATCH"
                if field == "row_results_sha256"
                else "FORMAL_AGGREGATE_MISMATCH"
                if field == "formal_aggregate_sha256"
                else "FORMAL_METRICS_MISSING"
            )
            raise FormalEvidenceError(
                code,
                f"production scorer receipt mismatch: {field}",
            )
    if expected_context is not None:
        if (
            producer.get("formal_run_id") != expected_context.run_id
            or str(Path(str(producer.get("formal_state_path", ""))).resolve())
            != str(expected_context.state_path)
        ):
            raise FormalEvidenceError(
                "FORMAL_ENTRYPOINT_CONTEXT_INVALID",
                "metrics producer is bound to another formal run/state",
            )
    return dict(metrics)


def recompute_formal_metrics_from_raw(
    raw_path: Path,
    *,
    context: FormalRunContext,
) -> dict[str, Any]:
    """Re-run the production scorer without a manifest bind and return metrics."""

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "recomputed.metrics.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "score_responses.py"),
            str(raw_path),
            "--output",
            str(output),
            "--scorer-mode",
            "canonical",
            "--protocol-id",
            context.protocol_id,
            "--evidence-class",
            (
                "CANONICAL_V5"
                if context.protocol_id
                == "agent_toolcall_protocol_v5_research_validity"
                else "CANONICAL_V4"
            ),
            "--comparison-state",
            str(context.state_path),
            "--response-field",
            str(context.scorer_identity["response_field_consumed"]),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise FormalEvidenceError(
                "ROW_RESULTS_RECOMPUTE_MISMATCH",
                detail or "production scorer recomputation failed",
            )
        value = loads_json_strict(output.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise FormalEvidenceError(
                "ROW_RESULTS_RECOMPUTE_MISMATCH",
                "production scorer recomputation did not emit an object",
            )
        return value


def verify_metrics_against_raw(
    metrics: Mapping[str, Any],
    *,
    raw_path: Path,
    context: FormalRunContext,
) -> None:
    recomputed = recompute_formal_metrics_from_raw(raw_path, context=context)
    if canonical_json_hash(
        {"row_results": metrics.get("row_results")}
    ) != canonical_json_hash({"row_results": recomputed.get("row_results")}):
        raise FormalEvidenceError(
            "ROW_RESULTS_RECOMPUTE_MISMATCH",
            "stored row_results differ from production rescoring of raw",
        )
    if metrics.get("formal_aggregate") != recomputed.get("formal_aggregate"):
        raise FormalEvidenceError(
            "FORMAL_AGGREGATE_MISMATCH",
            "stored aggregate differs from production rescoring of raw",
        )


def bind_metrics_to_output_manifest(
    manifest_path: Path,
    metrics_path: Path,
    *,
    context: FormalRunContext,
    allowed_stages: Sequence[str],
    allowed_statuses: Sequence[str],
    artifact_kind: str,
    _formal_capability: Any,
) -> str:
    """Verify production metrics against raw, then bind; never formalize input."""

    from manifest_writer_registry import require_formal_write_capability
    from model_state_attestation import verify_output_manifest

    require_formal_write_capability(
        _formal_capability,
        entrypoint_id=context.entrypoint_id,
        writer_id="formal-metrics-manifest-binder",
    )
    if context.entrypoint_id != "formal-scorer-main":
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_CONTEXT_INVALID",
            "metrics binder requires the formal scorer entrypoint",
        )
    metrics = loads_json_strict(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping):
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "metrics payload is not an object"
        )
    provenance_error = _provenance_error(metrics)
    if provenance_error is not None:
        raise provenance_error
    _, artifacts = resolve_formal_arm_artifacts(
        context,
        allowed_entrypoint_ids=("formal-scorer-main",),
        arm=context.arm,
        allowed_stages=allowed_stages,
        allowed_statuses=allowed_statuses,
        artifact_kind=artifact_kind,
    )
    if manifest_path.resolve() != artifacts["manifest"]:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_ARM_MISMATCH",
            "manifest does not match the context arm",
        )
    if metrics_path.resolve() != artifacts["metrics"]:
        raise FormalEvidenceError(
            "FORMAL_ENTRYPOINT_ARM_MISMATCH",
            "metrics do not match the context arm",
        )
    payload = verify_output_manifest(
        manifest_path, expected_scorer_identity=context.scorer_identity
    )
    raw_path = Path(payload["output_path"]).resolve()
    validate_formal_metrics(
        metrics,
        expected_identity=context.scorer_identity,
        expected_raw_path=raw_path,
        expected_raw_sha256=payload["output_sha256"],
        expected_context=context,
    )
    verify_metrics_against_raw(metrics, raw_path=raw_path, context=context)
    payload["metrics_binding"] = {
        "path": str(metrics_path.resolve()),
        "sha256": sha256_file(metrics_path),
        "metrics_kind": FORMAL_METRICS_KIND,
        "metrics_schema_version": FORMAL_METRICS_SCHEMA,
        "row_results_sha256": metrics["producer"]["row_results_sha256"],
        "formal_aggregate_sha256": metrics["producer"][
            "formal_aggregate_sha256"
        ],
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, manifest_path)
    return hashlib.sha256(encoded).hexdigest()


def verify_metrics_binding(
    manifest: Mapping[str, Any],
    metrics_path: Path,
) -> None:
    binding = manifest.get("metrics_binding")
    if not isinstance(binding, Mapping):
        raise FormalEvidenceError(
            "METRICS_MANIFEST_MISMATCH", "output manifest lacks metrics binding"
        )
    if str(Path(str(binding.get("path", ""))).resolve()) != str(
        metrics_path.resolve()
    ):
        raise FormalEvidenceError(
            "METRICS_MANIFEST_MISMATCH", "manifest metrics path mismatch"
        )
    if binding.get("sha256") != sha256_file(metrics_path):
        raise FormalEvidenceError(
            "METRICS_HASH_MISMATCH", "manifest metrics hash mismatch"
        )
    if (
        binding.get("metrics_kind") != FORMAL_METRICS_KIND
        or binding.get("metrics_schema_version") != FORMAL_METRICS_SCHEMA
    ):
        raise FormalEvidenceError(
            "METRICS_MANIFEST_MISMATCH", "manifest metrics type mismatch"
        )
    metrics = loads_json_strict(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping) or not isinstance(
        metrics.get("producer"), Mapping
    ):
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "bound metrics producer is missing"
        )
    if (
        binding.get("row_results_sha256")
        != metrics["producer"].get("row_results_sha256")
        or binding.get("formal_aggregate_sha256")
        != metrics["producer"].get("formal_aggregate_sha256")
    ):
        raise FormalEvidenceError(
            "METRICS_MANIFEST_MISMATCH",
            "manifest row/aggregate digests differ from metrics",
        )
