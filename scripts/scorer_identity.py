"""Canonical scorer identity construction, canonicalization, hashing and validation."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import Any, Mapping
from canonical_failure_codes import normalize_failure_codes

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "config" / "scorer_identity_v1.schema.json"
CANONICAL_FIELDS = tuple(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["required"])

class ScorerIdentityError(ValueError):
    def __init__(self, code: str, message: str): super().__init__(f"{code}: {message}"); self.code=code

def canonicalize_scorer_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping): raise ScorerIdentityError("SCORER_IDENTITY_MISSING", "identity must be an object")
    return {field: value.get(field) for field in CANONICAL_FIELDS}

def hash_scorer_identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(canonicalize_scorer_identity(value), ensure_ascii=False, sort_keys=True, separators=(",",":")).encode()).hexdigest()

def validate_scorer_identity(value: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = canonicalize_scorer_identity(value)
    if set(value) != set(CANONICAL_FIELDS): raise ScorerIdentityError("SCORER_IDENTITY_MISMATCH", "missing or additional identity fields")
    for field, item in normalized.items():
        if not isinstance(item, str) or not item: raise ScorerIdentityError("SCORER_IDENTITY_MISSING", f"{field} is empty")
    constants={"mode":"canonical","schema_version":"canonical-tool-registry-v1","implementation_version":"p0-5-v2","evidence_class":"CANONICAL_V4","tool_registry_path":"config/canonical_tool_registry_v1.json","protocol_id":"agent_toolcall_protocol_v4_comparison_eligibility","strict_parser_version":"response-parsing-v2","diagnostic_parser_version":"response-parsing-v2","canonicalization_policy":"no_coercion_no_normalization","additional_properties_policy":"false"}
    for field, required in constants.items():
        if normalized[field] != required: raise ScorerIdentityError({
            "mode":"SCORER_MODE_DRIFT",
            "schema_version":"SCORER_SCHEMA_VERSION_DRIFT",
            "implementation_version":"SCORER_IMPLEMENTATION_DRIFT",
            "evidence_class":"EVIDENCE_CLASS_DRIFT",
            "tool_registry_hash":"TOOL_REGISTRY_HASH_MISMATCH",
            "protocol_id":"PROTOCOL_ID_DRIFT",
            "response_field_consumed":"RESPONSE_FIELD_DRIFT",
            "strict_parser_version":"PARSER_VERSION_DRIFT",
            "diagnostic_parser_version":"PARSER_VERSION_DRIFT",
            "canonicalization_policy":"CANONICALIZATION_POLICY_DRIFT",
            "additional_properties_policy":"ADDITIONAL_PROPERTIES_POLICY_DRIFT",
        }.get(field,"SCORER_IDENTITY_MISMATCH"), f"{field} mismatch")
    if re.fullmatch(r"[0-9a-f]{64}", normalized["tool_registry_hash"]) is None: raise ScorerIdentityError("TOOL_REGISTRY_HASH_MISMATCH", "invalid registry hash")
    from canonical_tool_schema import registry_hash
    if normalized["tool_registry_hash"] != registry_hash(): raise ScorerIdentityError("TOOL_REGISTRY_HASH_MISMATCH", "registry content differs from locked identity")
    if expected is not None:
        locked = canonicalize_scorer_identity(expected)
        for field in CANONICAL_FIELDS:
            if locked[field] != normalized[field]:
                raise ScorerIdentityError({
                    "tool_registry_hash":"TOOL_REGISTRY_HASH_MISMATCH",
                    "schema_version":"SCORER_SCHEMA_VERSION_DRIFT",
                    "implementation_version":"SCORER_IMPLEMENTATION_DRIFT",
                    "protocol_id":"PROTOCOL_ID_DRIFT",
                    "response_field_consumed":"RESPONSE_FIELD_DRIFT",
                    "strict_parser_version":"PARSER_VERSION_DRIFT",
                    "diagnostic_parser_version":"PARSER_VERSION_DRIFT",
                    "canonicalization_policy":"CANONICALIZATION_POLICY_DRIFT",
                    "additional_properties_policy":"ADDITIONAL_PROPERTIES_POLICY_DRIFT",
                }.get(field, "SCORER_IDENTITY_MISMATCH"), f"{field} differs from lock")
    return normalized
