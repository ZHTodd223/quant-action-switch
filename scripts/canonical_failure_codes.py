"""Stable canonical scorer failure taxonomy and priority."""
from __future__ import annotations

FAILURE_PRIORITY = (
    "SCORER_IDENTITY_MISSING", "SCORER_IDENTITY_MISMATCH", "SCORER_MODE_NOT_ALLOWED",
    "SCORER_MODE_DRIFT", "SCORER_SCHEMA_VERSION_DRIFT", "SCORER_IMPLEMENTATION_DRIFT",
    "EVIDENCE_CLASS_DRIFT", "TOOL_REGISTRY_HASH_MISMATCH", "RESPONSE_FIELD_DRIFT",
    "PARSER_VERSION_DRIFT", "CANONICALIZATION_POLICY_DRIFT", "EMPTY_RESPONSE",
    "TRAILING_CONTENT", "MULTIPLE_OBJECTS", "STRICT_PARSE_FAILED", "NON_OBJECT_JSON",
    "MISSING_TOOL_NAME", "TOOL_NAME_NOT_STRING", "UNSUPPORTED_TOOL", "MISSING_ARGUMENTS",
    "ARGUMENTS_NOT_OBJECT", "MISSING_REQUIRED_ARGUMENT", "UNKNOWN_ARGUMENT",
    "ARGUMENT_TYPE_MISMATCH", "SCHEMA_VERSION_MISMATCH", "LEGACY_EVIDENCE_NOT_CANONICAL",
    "EXECUTION_NOT_ATTEMPTED", "EXECUTION_FAILED", "TASK_FAILED",
)
_RANK = {code: number for number, code in enumerate(FAILURE_PRIORITY)}

def normalize_failure_codes(codes: list[str] | tuple[str, ...]) -> tuple[str, list[str]]:
    values = sorted({code for code in codes if code}, key=lambda code: (_RANK.get(code, len(_RANK)), code))
    return (values[0] if values else "", values)
