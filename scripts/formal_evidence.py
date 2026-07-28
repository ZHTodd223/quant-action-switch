"""Integrity and semantic bindings for production v4 comparison evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from case_schema import loads_json_strict
from comparison_eligibility import sha256_file
from scorer_identity import hash_scorer_identity, validate_scorer_identity

FORMAL_METRICS_KIND = "FORMAL_CANONICAL_METRICS"
FORMAL_METRICS_SCHEMA = "formal-canonical-metrics-v1"
FORMAL_METRIC_SOURCE = "strict_whole_response_valid+canonical_schema_valid+exact_call"
FORMAL_METRIC_VERSION = "p0-2-strict-formal-v1"
STATE_HASH_SUFFIX = ".sha256"


class FormalEvidenceError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.detail = message


def canonical_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_hash_path(state_path: Path) -> Path:
    return state_path.with_suffix(state_path.suffix + STATE_HASH_SUFFIX)


def write_state_with_integrity(state_path: Path, state: Mapping[str, Any]) -> str:
    """Atomically write a state and its detached content hash."""

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


def write_summary_with_integrity(
    summary_path: Path, summary: Mapping[str, Any]
) -> str:
    """Write a recomputable summary plus a hash of its fully verified inputs."""

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
            "MANIFEST_VERIFICATION_FAILED", f"comparison state is invalid: {error}"
        ) from error
    if not isinstance(value, dict):
        raise FormalEvidenceError(
            "MANIFEST_VERIFICATION_FAILED", "comparison state is not an object"
        )
    return value


def add_formal_metrics_metadata(
    summary: dict[str, Any],
    *,
    identity: Mapping[str, Any],
    source_raw_path: str,
    source_raw_sha256: str,
    exact_call_count: int,
    total_count: int,
    strict_valid_count: int,
    schema_valid_count: int,
) -> dict[str, Any]:
    locked = validate_scorer_identity(identity)
    summary.update(
        {
            "metrics_schema_version": FORMAL_METRICS_SCHEMA,
            "metrics_kind": FORMAL_METRICS_KIND,
            "evidence_class": "CANONICAL_V4",
            "retrospective": False,
            "formal_gate_effect": True,
            "scorer_identity": locked,
            "scorer_identity_sha256": hash_scorer_identity(locked),
            "source_raw_path": source_raw_path,
            "source_raw_sha256": source_raw_sha256,
            "formal_metric_source": FORMAL_METRIC_SOURCE,
            "formal_metric_version": FORMAL_METRIC_VERSION,
            "strict_required": True,
            "diagnostic_only": False,
            "formal_aggregate": {
                "total": total_count,
                "strict_whole_response_valid": strict_valid_count,
                "canonical_schema_valid": schema_valid_count,
                "exact_call": exact_call_count,
                "exact_call_rate": (
                    exact_call_count / total_count if total_count else 0.0
                ),
            },
        }
    )
    return summary


def validate_formal_metrics(
    metrics: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_raw_path: Path,
    expected_raw_sha256: str,
) -> dict[str, Any]:
    if metrics.get("retrospective") is not False:
        raise FormalEvidenceError(
            "RETROSPECTIVE_EVIDENCE_NOT_FORMAL",
            "metrics are retrospective or omit retrospective=false",
        )
    if metrics.get("formal_gate_effect") is not True:
        raise FormalEvidenceError(
            "DIAGNOSTIC_METRICS_NOT_FORMAL",
            "metrics do not declare formal_gate_effect=true",
        )
    if metrics.get("metrics_kind") != FORMAL_METRICS_KIND:
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "formal metrics kind is missing or invalid"
        )
    if metrics.get("metrics_schema_version") != FORMAL_METRICS_SCHEMA:
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "formal metrics schema version is invalid"
        )
    if metrics.get("evidence_class") != "CANONICAL_V4":
        code = {
            "LEGACY_HISTORICAL": "LEGACY_EVIDENCE_NOT_CANONICAL",
            "RETROSPECTIVE_CANONICAL_DIAGNOSTIC": "RETROSPECTIVE_EVIDENCE_NOT_FORMAL",
            "DEVELOPMENT_ONLY": "DEVELOPMENT_EVIDENCE_NOT_FORMAL",
            "IDENTITY_UNKNOWN": "IDENTITY_UNKNOWN_NOT_CANONICAL",
        }.get(str(metrics.get("evidence_class")), "IDENTITY_UNKNOWN_NOT_CANONICAL")
        raise FormalEvidenceError(code, "metrics evidence class is not CANONICAL_V4")
    if (
        metrics.get("formal_metric_source") != FORMAL_METRIC_SOURCE
        or metrics.get("formal_metric_version") != FORMAL_METRIC_VERSION
        or metrics.get("strict_required") is not True
        or metrics.get("diagnostic_only") is not False
    ):
        raise FormalEvidenceError(
            "DIAGNOSTIC_METRICS_NOT_FORMAL",
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
    if metrics.get("source_raw_sha256") != expected_raw_sha256:
        raise FormalEvidenceError(
            "METRICS_HASH_MISMATCH", "metrics source raw hash mismatch"
        )
    aggregate = metrics.get("formal_aggregate")
    required = {
        "total",
        "strict_whole_response_valid",
        "canonical_schema_valid",
        "exact_call",
        "exact_call_rate",
    }
    if not isinstance(aggregate, Mapping) or set(aggregate) != required:
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "formal aggregate is missing or incomplete"
        )
    for field in required - {"exact_call_rate"}:
        if type(aggregate[field]) is not int or aggregate[field] < 0:
            raise FormalEvidenceError(
                "FORMAL_METRICS_MISSING", f"formal aggregate {field} is invalid"
            )
    total = aggregate["total"]
    if any(aggregate[field] > total for field in required - {"total", "exact_call_rate"}):
        raise FormalEvidenceError(
            "STRICT_FORMAL_REQUIREMENT_FAILED",
            "formal aggregate counts exceed total",
        )
    if (
        aggregate["exact_call"] > aggregate["strict_whole_response_valid"]
        or aggregate["exact_call"] > aggregate["canonical_schema_valid"]
    ):
        raise FormalEvidenceError(
            "STRICT_FORMAL_REQUIREMENT_FAILED",
            "exact calls are not a subset of strict schema-valid responses",
        )
    expected_rate = aggregate["exact_call"] / total if total else 0.0
    if type(aggregate["exact_call_rate"]) not in {int, float} or abs(
        float(aggregate["exact_call_rate"]) - expected_rate
    ) > 1e-12:
        raise FormalEvidenceError(
            "STRICT_FORMAL_REQUIREMENT_FAILED",
            "formal exact-call rate is inconsistent",
        )
    return dict(metrics)


def bind_metrics_to_output_manifest(
    manifest_path: Path,
    metrics_path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> str:
    """Bind immutable raw output and formal metrics into one arm manifest."""

    from model_state_attestation import verify_output_manifest

    payload = verify_output_manifest(
        manifest_path, expected_scorer_identity=expected_identity
    )
    raw_path = Path(payload["output_path"])
    metrics = loads_json_strict(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping):
        raise FormalEvidenceError(
            "FORMAL_METRICS_MISSING", "metrics payload is not an object"
        )
    validate_formal_metrics(
        metrics,
        expected_identity=expected_identity,
        expected_raw_path=raw_path,
        expected_raw_sha256=payload["output_sha256"],
    )
    payload["metrics_binding"] = {
        "path": str(metrics_path.resolve()),
        "sha256": sha256_file(metrics_path),
        "metrics_kind": FORMAL_METRICS_KIND,
        "metrics_schema_version": FORMAL_METRICS_SCHEMA,
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
