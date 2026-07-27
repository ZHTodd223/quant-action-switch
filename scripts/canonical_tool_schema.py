"""The single strict tool schema used by canonical v4 scoring."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "canonical_tool_registry_v1.json"
SCHEMA_VERSION = "canonical-tool-registry-v1"

def _load() -> dict[str, Any]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("canonical tool registry is invalid")
    tools = value.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("canonical tool registry must contain tools")
    names = [item.get("name") for item in tools if isinstance(item, dict)]
    if len(names) != len(tools) or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names) or len({name.casefold() for name in names}) != len(names):
        raise ValueError("canonical tool registry has ambiguous tool names")
    return value

def registry_hash() -> str:
    return hashlib.sha256(json.dumps(_load(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

def scorer_identity(response_field_consumed: str = "auto") -> dict[str, str]:
    return {"mode":"canonical","schema_version":SCHEMA_VERSION,"implementation_version":"p0-5-v2","tool_registry_path":"config/canonical_tool_registry_v1.json","tool_registry_hash":registry_hash(),"evidence_class":"CANONICAL_V4","protocol_id":"agent_toolcall_protocol_v4_comparison_eligibility","response_field_consumed":response_field_consumed,"strict_parser_version":"response-parsing-v2","diagnostic_parser_version":"response-parsing-v2","canonicalization_policy":"no_coercion_no_normalization","additional_properties_policy":"false"}

def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string": return isinstance(value, str)
    if expected == "integer": return type(value) is int
    if expected == "number": return type(value) in {int, float} and (not isinstance(value, float) or math.isfinite(value))
    if expected == "boolean": return type(value) is bool
    if expected == "object": return isinstance(value, dict)
    if expected == "array": return isinstance(value, list)
    if expected == "null": return value is None
    return False

def validate_call(value: Any) -> dict[str, Any]:
    """Validate a raw canonical call without coercion or legacy aliases."""
    result = {key: False for key in ("top_level_object_valid","tool_name_present","tool_name_supported","arguments_present","arguments_is_object","required_arguments_present","argument_keys_valid","argument_types_valid","additional_arguments_valid","canonical_schema_valid")}
    failures: list[str] = []
    if not isinstance(value, dict):
        return result | {"primary_failure_code":"NON_OBJECT_JSON", "failure_codes":["NON_OBJECT_JSON"]}
    result["top_level_object_valid"] = True
    if "name" not in value:
        failures.append("MISSING_TOOL_NAME")
    elif not isinstance(value["name"], str):
        failures.append("TOOL_NAME_NOT_STRING")
    else:
        result["tool_name_present"] = bool(value["name"])
        if not value["name"]: failures.append("MISSING_TOOL_NAME")
    tools = {tool["name"]: tool["arguments"] for tool in _load()["tools"]}
    rule = tools.get(value.get("name")) if isinstance(value.get("name"), str) else None
    if rule is None and value.get("name") is not None: failures.append("UNSUPPORTED_TOOL")
    else: result["tool_name_supported"] = rule is not None
    if "arguments" not in value: failures.append("MISSING_ARGUMENTS")
    else: result["arguments_present"] = True
    arguments = value.get("arguments")
    if "arguments" in value and not isinstance(arguments, dict): failures.append("ARGUMENTS_NOT_OBJECT")
    else: result["arguments_is_object"] = isinstance(arguments, dict)
    if rule is not None and isinstance(arguments, dict):
        required = rule.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing: failures.append("MISSING_REQUIRED_ARGUMENT")
        else: result["required_arguments_present"] = True
        properties = rule.get("properties", {})
        unknown = [key for key in arguments if key not in properties]
        if unknown and rule.get("additionalProperties") is False: failures.append("UNKNOWN_ARGUMENT")
        else:
            result["argument_keys_valid"] = True; result["additional_arguments_valid"] = True
        type_bad = [key for key in arguments if key in properties and not _type_ok(arguments[key], properties[key].get("type", ""))]
        empty = [key for key in arguments if key in properties and properties[key].get("minLength") and isinstance(arguments[key], str) and len(arguments[key]) < properties[key]["minLength"]]
        if type_bad or empty: failures.append("ARGUMENT_TYPE_MISMATCH")
        else: result["argument_types_valid"] = True
    result["canonical_schema_valid"] = not failures and set(value) == {"name", "arguments"}
    if set(value) != {"name", "arguments"} and not failures: failures.append("UNKNOWN_ARGUMENT")
    return result | {"primary_failure_code": failures[0] if failures else "", "failure_codes": failures}
