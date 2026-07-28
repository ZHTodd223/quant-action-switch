"""Stable fail-closed reason codes for canonical-summary contamination."""
from __future__ import annotations

from typing import Any, Mapping

SUMMARY_REASON_CODES = frozenset({
    "LEGACY_EVIDENCE_NOT_CANONICAL",
    "RETROSPECTIVE_EVIDENCE_NOT_FORMAL",
    "IDENTITY_UNKNOWN_NOT_CANONICAL",
    "DEVELOPMENT_EVIDENCE_NOT_FORMAL",
    "STATE_METRICS_IDENTITY_MISMATCH",
    "ARM_SCORER_IDENTITY_MISMATCH",
    "TOOL_REGISTRY_HASH_MISMATCH",
    "SCORER_SCHEMA_VERSION_DRIFT",
    "SCORER_IMPLEMENTATION_DRIFT",
    "PARSER_VERSION_DRIFT",
    "RESPONSE_FIELD_DRIFT",
    "CANONICALIZATION_POLICY_DRIFT",
    "ADDITIONAL_PROPERTIES_POLICY_DRIFT",
    "PROTOCOL_ID_DRIFT",
    "EVIDENCE_CLASS_DRIFT",
    "MANIFEST_IDENTITY_MISSING",
    "MANIFEST_IDENTITY_MISMATCH",
    "MANIFEST_REGISTRY_MISMATCH",
    "MANIFEST_VERIFICATION_FAILED",
    "NOT_COMPARABLE",
    "ATTESTATION_INVALID",
    "RETROSPECTIVE_SIDECAR_NOT_FORMAL",
    "SUMMARY_CACHE_STALE",
})

FIELD_REASON = {
    "tool_registry_hash": "TOOL_REGISTRY_HASH_MISMATCH",
    "schema_version": "SCORER_SCHEMA_VERSION_DRIFT",
    "implementation_version": "SCORER_IMPLEMENTATION_DRIFT",
    "strict_parser_version": "PARSER_VERSION_DRIFT",
    "diagnostic_parser_version": "PARSER_VERSION_DRIFT",
    "response_field_consumed": "RESPONSE_FIELD_DRIFT",
    "canonicalization_policy": "CANONICALIZATION_POLICY_DRIFT",
    "additional_properties_policy": "ADDITIONAL_PROPERTIES_POLICY_DRIFT",
    "protocol_id": "PROTOCOL_ID_DRIFT",
    "evidence_class": "EVIDENCE_CLASS_DRIFT",
}


def classify_candidate(mutations: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a mutation of an otherwise-valid canonical v4 run."""
    if not mutations:
        return {"included": True, "reason_code": "", "details": [], "arm_change_computed": True}
    evidence = mutations.get("evidence_class")
    evidence_codes = {
        "LEGACY_HISTORICAL": "LEGACY_EVIDENCE_NOT_CANONICAL",
        "RETROSPECTIVE_DIAGNOSTIC": "RETROSPECTIVE_EVIDENCE_NOT_FORMAL",
        "IDENTITY_UNKNOWN": "IDENTITY_UNKNOWN_NOT_CANONICAL",
        "DEVELOPMENT_ONLY": "DEVELOPMENT_EVIDENCE_NOT_FORMAL",
    }
    if evidence in evidence_codes:
        return _excluded(evidence_codes[evidence], "evidence_class")
    if mutations.get("comparison_status") != "COMPARABLE" and "comparison_status" in mutations:
        return _excluded("NOT_COMPARABLE", "comparison_status")
    if mutations.get("attestation_valid") is False:
        return _excluded("ATTESTATION_INVALID", "attestation")
    if mutations.get("formal_metrics_source") == "retrospective_sidecar":
        return _excluded("RETROSPECTIVE_SIDECAR_NOT_FORMAL", "formal_metrics_source")
    if mutations.get("summary_cache_stale") is True:
        return _excluded("SUMMARY_CACHE_STALE", "summary_cache")
    manifest = mutations.get("manifest")
    manifest_codes = {
        "identity_missing": "MANIFEST_IDENTITY_MISSING",
        "identity_hash_mismatch": "MANIFEST_IDENTITY_MISMATCH",
        "registry_hash_mismatch": "MANIFEST_REGISTRY_MISMATCH",
        "metrics_mismatch": "MANIFEST_IDENTITY_MISMATCH",
        "verification_failed": "MANIFEST_VERIFICATION_FAILED",
    }
    if manifest in manifest_codes:
        return _excluded(manifest_codes[manifest], f"manifest.{manifest}")
    if mutations.get("strict_whole_response_valid") is False:
        return _excluded("NOT_COMPARABLE", "strict_whole_response_valid")
    scope = mutations.get("identity_scope")
    field = mutations.get("identity_field")
    if scope:
        if field in FIELD_REASON:
            return _excluded(FIELD_REASON[str(field)], f"{scope}.{field}")
        if scope in {"state_metrics", "metrics_state"}:
            return _excluded("STATE_METRICS_IDENTITY_MISMATCH", str(scope))
        return _excluded("ARM_SCORER_IDENTITY_MISMATCH", str(scope))
    raise ValueError(f"unclassified summary contamination mutation: {dict(mutations)}")


def _excluded(code: str, detail: str) -> dict[str, Any]:
    if code not in SUMMARY_REASON_CODES:
        raise ValueError(f"unregistered summary reason code: {code}")
    return {"included": False, "reason_code": code, "details": [detail], "arm_change_computed": False}


def reason_from_error(message: str) -> str:
    """Map verified native-evidence failures to stable public reason codes."""
    upper = message.upper()
    for code in SUMMARY_REASON_CODES:
        if code in upper:
            return code
    if "ATTESTATION" in upper:
        return "ATTESTATION_INVALID"
    if "MANIFEST" in upper and ("MISSING" in upper or "NO SUCH FILE" in upper):
        return "MANIFEST_VERIFICATION_FAILED"
    if "MANIFEST" in upper:
        return "MANIFEST_VERIFICATION_FAILED"
    if "SCORER" in upper or "IDENTITY" in upper:
        return "MANIFEST_IDENTITY_MISMATCH"
    return "NOT_COMPARABLE"
