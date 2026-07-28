#!/usr/bin/env python3
"""Build the frozen, reviewable P0-5 fixture matrices."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from canonical_tool_schema import _type_ok, validate_call
from response_parsing import parse_response_layers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "canonical_scorer"
EXPECTED_FIELDS = (
    "tool_intent_detected", "first_object_recoverable",
    "strict_whole_response_valid", "parser_success",
    "top_level_object_valid", "tool_name_present", "tool_name_supported",
    "arguments_present", "arguments_is_object", "required_arguments_present",
    "argument_keys_valid", "argument_types_valid",
    "additional_arguments_valid", "canonical_schema_valid",
    "tool_name_exact", "arguments_exact", "exact_call",
    "primary_failure_code",
)


def response_case(name: str, description: str, response: str, expected_call: dict[str, Any] | None = None) -> dict[str, Any]:
    layers = parse_response_layers(response)
    raw = layers["strict_object"] if layers["strict_parse_success"] else None
    result = validate_call(raw)
    failure = result["primary_failure_code"]
    if raw is None:
        failure = {
            "EMPTY": "EMPTY_RESPONSE",
            "TRAILING_CONTENT": "TRAILING_CONTENT",
            "MULTIPLE_OBJECTS": "MULTIPLE_OBJECTS",
            "NON_OBJECT_JSON": "NON_OBJECT_JSON",
        }.get(layers["strict_failure_type"], "STRICT_PARSE_FAILED")
    exact = bool(expected_call is not None and result["canonical_schema_valid"] and raw == expected_call)
    observed = {
        **layers, **result,
        "strict_whole_response_valid": bool(raw is not None),
        "parser_success": bool(raw is not None),
        "tool_name_exact": exact,
        "arguments_exact": exact,
        "exact_call": exact,
        "primary_failure_code": failure,
    }
    return {
        "name": name,
        "description": description,
        "response": response,
        "expected_call": expected_call,
        "expected": {field: observed[field] for field in EXPECTED_FIELDS},
    }


def dump(name: str, rows: Any) -> None:
    (OUT / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    valid_specs = [
        ("valid_read_file", "read_file(path)", {"name":"read_file","arguments":{"path":"sandbox://a.txt"}}, None),
        ("valid_calculator", "calculator(expression)", {"name":"calculator","arguments":{"expression":"1+2"}}, None),
        ("valid_search_docs", "search_docs(query)", {"name":"search_docs","arguments":{"query":"quantization"}}, None),
        ("valid_top_key_order", "top-level key order variation", {"arguments":{"path":"a"},"name":"read_file"}, None),
        ("valid_argument_key_order", "argument object ordering remains exact", {"name":"read_file","arguments":{"path":"ordered"}}, None),
        ("valid_unicode", "Unicode string", {"name":"search_docs","arguments":{"query":"café Δ"}}, None),
        ("valid_chinese", "Chinese query", {"name":"search_docs","arguments":{"query":"量化工具调用"}}, None),
        ("valid_windows_path", "Windows backslash path", {"name":"read_file","arguments":{"path":"C:\\研究\\a.txt"}}, None),
        ("valid_spaced_path", "path containing spaces", {"name":"read_file","arguments":{"path":"sandbox://public/my notes/a.txt"}}, None),
        ("valid_escaped_quote", "JSON escaped quote", {"name":"search_docs","arguments":{"query":"say \"hello\""}}, None),
        ("valid_parenthesized_expression", "parenthesized expression", {"name":"calculator","arguments":{"expression":"(2+3)*4"}}, None),
        ("valid_decimal_expression", "decimal expression", {"name":"calculator","arguments":{"expression":"1.25/0.5"}}, None),
        ("valid_long_string", "long query", {"name":"search_docs","arguments":{"query":"x"*2048}}, None),
        ("valid_json_whitespace", "legal leading and trailing JSON whitespace", {"name":"read_file","arguments":{"path":"a"}}, " \r\n\t"),
    ]
    valid = []
    for name, desc, call, whitespace in valid_specs:
        text = json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        if whitespace is not None:
            text = whitespace + text + whitespace
        valid.append(response_case(name, desc, text, call))
    dump("valid_cases.json", valid)

    tool_values = [
        ("unsupported_tool", "unsupported tool", "delete_file"),
        ("empty_tool", "empty string", ""),
        ("missing_tool", "missing name member", ...),
        ("null_tool", "null", None), ("integer_tool", "integer", 1),
        ("float_tool", "float", 1.5), ("boolean_tool", "boolean", True),
        ("object_tool", "object", {"x":1}), ("array_tool", "array", ["read_file"]),
        ("case_changed_tool", "case change", "Read_File"),
        ("leading_space_tool", "leading space", " read_file"),
        ("trailing_space_tool", "trailing space", "read_file "),
        ("both_space_tool", "surrounding spaces", " read_file "),
        ("legacy_alias_tool", "legacy alias", "file_read"),
        ("unicode_similar_tool", "Unicode confusable", "read_fіle"),
        ("newline_tool", "embedded newline", "read_\nfile"),
        ("invisible_tool", "zero-width character", "read_\u200bfile"),
    ]
    tools = []
    for name, desc, value in tool_values:
        call = {"arguments":{"path":"a"}} if value is ... else {"name":value,"arguments":{"path":"a"}}
        tools.append(response_case(name, desc, json.dumps(call, ensure_ascii=False, separators=(",",":"))))
    dump("tool_name_negative_cases.json", tools)

    argument_calls = [
        ("wrong_key", {"name":"read_file","arguments":{"filename":"a"}}),
        ("missing_required", {"name":"read_file","arguments":{"other":"a"}}),
        ("extra_argument", {"name":"read_file","arguments":{"path":"a","extra":1}}),
        ("empty_object", {"name":"read_file","arguments":{}}),
        ("arguments_string", {"name":"read_file","arguments":"a"}),
        ("arguments_array", {"name":"read_file","arguments":["a"]}),
        ("arguments_null", {"name":"read_file","arguments":None}),
        ("arguments_integer", {"name":"read_file","arguments":1}),
        ("arguments_float", {"name":"read_file","arguments":1.5}),
        ("arguments_boolean", {"name":"read_file","arguments":True}),
        ("path_number", {"name":"read_file","arguments":{"path":1}}),
        ("path_boolean", {"name":"read_file","arguments":{"path":False}}),
        ("path_null", {"name":"read_file","arguments":{"path":None}}),
        ("path_object", {"name":"read_file","arguments":{"path":{"x":1}}}),
        ("expression_number", {"name":"calculator","arguments":{"expression":2}}),
        ("expression_array", {"name":"calculator","arguments":{"expression":["2"]}}),
        ("query_array", {"name":"search_docs","arguments":{"query":["x"]}}),
        ("query_object", {"name":"search_docs","arguments":{"query":{"x":1}}}),
        ("query_null", {"name":"search_docs","arguments":{"query":None}}),
        ("valid_plus_unknown", {"name":"search_docs","arguments":{"query":"x","limit":1}}),
        ("argument_case_changed", {"name":"read_file","arguments":{"Path":"a"}}),
        ("argument_leading_space", {"name":"read_file","arguments":{" path":"a"}}),
        ("argument_trailing_space", {"name":"read_file","arguments":{"path ":"a"}}),
        ("required_wrong_type", {"name":"calculator","arguments":{"expression":False}}),
    ]
    arguments = [response_case(name, name.replace("_", " "), json.dumps(call, separators=(",",":"))) for name, call in argument_calls]
    duplicate = '{"name":"read_file","arguments":{"path":"first","path":"last"}}'
    arguments.append(response_case("duplicate_json_key", "duplicate keys are rejected, not last-key-wins", duplicate))
    dump("argument_negative_cases.json", arguments)

    good = '{"name":"read_file","arguments":{"path":"a"}}'
    bad_schema = '{"name":"read_file","arguments":{}}'
    formats = [
        ("markdown_json_fence", f"```json\n{good}\n```"),
        ("markdown_plain_fence", f"```\n{good}\n```"),
        ("tool_call_wrapper", f"<tool_call>{good}</tool_call>"),
        ("two_json", good+good), ("three_json", good+good+good),
        ("trailing_text", good+" trailing"), ("leading_natural_language", "call: "+good),
        ("surrounded_natural_language", "call: "+good+" done"),
        ("truncated_json", good[:-2]), ("empty_response", ""), ("whitespace_only", " \r\n\t"),
        ("top_array", "[]"), ("top_string", '"x"'), ("top_integer", "1"),
        ("top_float", "1.5"), ("top_boolean", "true"), ("top_null", "null"),
        ("recoverable_strict_failure", good+" x"), ("strict_schema_failure", bad_schema),
        ("string_braces", '{"name":"search_docs","arguments":{"query":"{x}"}} trailing'),
        ("string_escaped_quote", '{"name":"search_docs","arguments":{"query":"\\"x\\""}} trailing'),
        ("nested_object", '{"outer":'+good+'}'), ("nested_array", '{"items":['+good+']}'),
        ("first_valid_second_invalid", good+' {"bad":'),
        ("first_fragment_invalid_second_valid", '{"bad": '+good),
        ("json_newline_text", good+"\nexplanation"), ("\ufeffleading_bom", "\ufeff"+good),
        ("other_invisible", "\u200b"+good), ("only_tool_tags", "<tool_call></tool_call>"),
        ("fence_json_then_text", f"```json\n{good}\ntext\n```"),
    ]
    dump("format_negative_cases.json", [response_case(name.lstrip("\ufeff"), name.lstrip("\ufeff").replace("_"," "), text) for name, text in formats])

    identities = [
        {"name":"identity_canonical_positive","description":"canonical v4 positive","mutations":{},"expected":{"accepted":True,"reason_code":""}},
        {"name":"identity_v4_legacy","description":"v4 legacy mode","mutations":{"mode":"legacy"},"expected":{"accepted":False,"reason_code":"SCORER_MODE_DRIFT"}},
        {"name":"identity_mode_missing","description":"mode missing","mutations":{"delete":"mode"},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISMATCH"}},
        {"name":"identity_missing","description":"identity missing","mutations":{"identity":None},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISSING"}},
        {"name":"registry_path_missing","description":"registry path missing","mutations":{"delete":"tool_registry_path"},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISMATCH"}},
        {"name":"registry_hash_missing","description":"registry hash missing","mutations":{"delete":"tool_registry_hash"},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISMATCH"}},
        {"name":"registry_hash_format","description":"malformed hash","mutations":{"tool_registry_hash":"bad"},"expected":{"accepted":False,"reason_code":"TOOL_REGISTRY_HASH_MISMATCH"}},
        {"name":"registry_hash_content","description":"content/hash mismatch","mutations":{"tool_registry_hash":"0"*64},"expected":{"accepted":False,"reason_code":"TOOL_REGISTRY_HASH_MISMATCH"}},
        {"name":"schema_version_mismatch","description":"schema drift","mutations":{"schema_version":"v0"},"expected":{"accepted":False,"reason_code":"SCORER_SCHEMA_VERSION_DRIFT"}},
        {"name":"implementation_mismatch","description":"implementation drift","mutations":{"implementation_version":"old"},"expected":{"accepted":False,"reason_code":"SCORER_IMPLEMENTATION_DRIFT"}},
        {"name":"evidence_class_mismatch","description":"evidence class drift","mutations":{"evidence_class":"LEGACY_HISTORICAL"},"expected":{"accepted":False,"reason_code":"EVIDENCE_CLASS_DRIFT"}},
        {"name":"protocol_mismatch","description":"protocol drift","mutations":{"protocol_id":"v3"},"expected":{"accepted":False,"reason_code":"PROTOCOL_ID_DRIFT"}},
        {"name":"response_field_mismatch","description":"response field drift","mutations":{"response_field_consumed":"legacy_response"},"expected":{"accepted":False,"reason_code":"RESPONSE_FIELD_DRIFT"}},
        {"name":"strict_parser_mismatch","description":"strict parser drift","mutations":{"strict_parser_version":"v1"},"expected":{"accepted":False,"reason_code":"PARSER_VERSION_DRIFT"}},
        {"name":"diagnostic_parser_mismatch","description":"diagnostic parser drift","mutations":{"diagnostic_parser_version":"v1"},"expected":{"accepted":False,"reason_code":"PARSER_VERSION_DRIFT"}},
        {"name":"canonicalization_mismatch","description":"canonicalization drift","mutations":{"canonicalization_policy":"trim"},"expected":{"accepted":False,"reason_code":"CANONICALIZATION_POLICY_DRIFT"}},
        {"name":"additional_properties_mismatch","description":"additional properties drift","mutations":{"additional_properties_policy":"true"},"expected":{"accepted":False,"reason_code":"ADDITIONAL_PROPERTIES_POLICY_DRIFT"}},
        {"name":"identity_extra_field","description":"unknown identity field","mutations":{"extra":"x"},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISMATCH"}},
        {"name":"response_field_missing","description":"response field missing","mutations":{"delete":"response_field_consumed"},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISMATCH"}},
        {"name":"retrospective_diagnostic","description":"retrospective evidence","mutations":{"evidence_class":"RETROSPECTIVE_DIAGNOSTIC"},"expected":{"accepted":False,"reason_code":"EVIDENCE_CLASS_DRIFT"}},
        {"name":"identity_unknown","description":"unknown evidence","mutations":{"evidence_class":"IDENTITY_UNKNOWN"},"expected":{"accepted":False,"reason_code":"EVIDENCE_CLASS_DRIFT"}},
        {"name":"protocol_id_empty","description":"protocol identity empty","mutations":{"protocol_id":""},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISSING"}},
        {"name":"state_metrics_mismatch","description":"state and metrics mismatch","mutations":{"scope":"state_metrics"},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISMATCH"}},
        {"name":"manifest_state_mismatch","description":"manifest and state mismatch","mutations":{"scope":"manifest_state"},"expected":{"accepted":False,"reason_code":"SCORER_IDENTITY_MISMATCH"}},
    ]
    dump("identity_negative_cases.json", identities)

    type_rows = [
        ("string_positive","x","string",True), ("string_negative",1,"string",False),
        ("integer_positive",1,"integer",True), ("integer_bool_negative",True,"integer",False),
        ("number_integer_positive",1,"number",True), ("number_float_positive",1.5,"number",True),
        ("number_bool_negative",False,"number",False), ("number_nan_negative","NaN","number",False),
        ("boolean_positive",True,"boolean",True), ("boolean_negative",1,"boolean",False),
        ("object_positive",{"x":1},"object",True), ("object_array_negative",[],"object",False),
        ("array_positive",[1],"array",True), ("array_object_negative",{},"array",False),
        ("null_positive",None,"null",True), ("null_negative","null","null",False),
    ]
    dump("type_validation_cases.json", [{"name":n,"description":n.replace("_"," "),"value":v,"expected_type":t,"expected":ok} for n,v,t,ok in type_rows])

    summary_specs = [
        ("canonical_included", {"kind":"none"}, True, ""),
        ("legacy_evidence", {"kind":"metrics_field","field":"evidence_class","value":"LEGACY_HISTORICAL"}, False, "LEGACY_EVIDENCE_NOT_CANONICAL"),
        ("retrospective_bf16_copy", {"kind":"retrospective_copy","arm":"bf16"}, False, "RETROSPECTIVE_EVIDENCE_NOT_FORMAL"),
        ("retrospective_quant_copy", {"kind":"retrospective_copy","arm":"quant"}, False, "RETROSPECTIVE_EVIDENCE_NOT_FORMAL"),
        ("identity_unknown", {"kind":"metrics_field","field":"evidence_class","value":"IDENTITY_UNKNOWN"}, False, "IDENTITY_UNKNOWN_NOT_CANONICAL"),
        ("development_evidence", {"kind":"metrics_field","field":"evidence_class","value":"DEVELOPMENT_ONLY"}, False, "DEVELOPMENT_EVIDENCE_NOT_FORMAL"),
        ("baseline_failed", {"kind":"original_status","value":"NOT_ELIGIBLE_BASELINE_FAILED"}, False, "ORIGINAL_STATE_NOT_COMPARABLE"),
        ("reconstruction_failed", {"kind":"original_status","value":"NOT_ELIGIBLE_RECONSTRUCTION_FAILED"}, False, "ORIGINAL_STATE_NOT_COMPARABLE"),
        ("bf16_gate_failed", {"kind":"original_status","value":"NOT_ELIGIBLE_BF16_GATE_FAILED"}, False, "ORIGINAL_STATE_NOT_COMPARABLE"),
        ("eligible_not_quantized", {"kind":"original_status","value":"ELIGIBLE_NOT_QUANTIZED"}, False, "ORIGINAL_STATE_NOT_COMPARABLE"),
        ("quantization_failed", {"kind":"original_status","value":"QUANTIZATION_FAILED"}, False, "ORIGINAL_STATE_NOT_COMPARABLE"),
        ("case_mismatch", {"kind":"original_status","value":"NOT_COMPARABLE_CASE_MISMATCH"}, False, "ORIGINAL_STATE_NOT_COMPARABLE"),
        ("source_mismatch", {"kind":"original_status","value":"NOT_COMPARABLE_SOURCE_MISMATCH"}, False, "ORIGINAL_STATE_NOT_COMPARABLE"),
        ("diagnostic_only", {"kind":"metrics_field","field":"formal_aggregate","delete":True}, False, "FORMAL_METRICS_MISSING"),
        ("formal_metadata_missing", {"kind":"metrics_field","field":"formal_metric_source","delete":True}, False, "DIAGNOSTIC_METRICS_NOT_FORMAL"),
        ("strict_flag_missing", {"kind":"metrics_field","field":"strict_required","delete":True}, False, "DIAGNOSTIC_METRICS_NOT_FORMAL"),
        ("formal_flag_false", {"kind":"metrics_field","field":"formal_gate_effect","value":False}, False, "DIAGNOSTIC_METRICS_NOT_FORMAL"),
        ("metrics_kind_diagnostic", {"kind":"metrics_field","field":"metrics_kind","value":"RETROSPECTIVE_DIAGNOSTIC"}, False, "FORMAL_METRICS_MISSING"),
        ("metrics_hash_bf16", {"kind":"metrics_hash_mismatch","arm":"bf16"}, False, "METRICS_HASH_MISMATCH"),
        ("metrics_hash_quant", {"kind":"metrics_hash_mismatch","arm":"quant"}, False, "METRICS_HASH_MISMATCH"),
        ("raw_hash_bf16", {"kind":"raw_hash_mismatch","arm":"bf16"}, False, "RAW_HASH_MISMATCH"),
        ("raw_hash_quant", {"kind":"raw_hash_mismatch","arm":"quant"}, False, "RAW_HASH_MISMATCH"),
        ("metrics_manifest_path", {"kind":"metrics_manifest_mismatch","arm":"bf16"}, False, "METRICS_MANIFEST_MISMATCH"),
        ("manifest_metrics_hash", {"kind":"manifest_metrics_hash_mismatch","arm":"quant"}, False, "METRICS_HASH_MISMATCH"),
        ("registry_hash_drift", {"kind":"metrics_identity_field","field":"tool_registry_hash","value":"0"*64}, False, "TOOL_REGISTRY_HASH_MISMATCH"),
        ("schema_version_drift", {"kind":"metrics_identity_field","field":"schema_version","value":"v0"}, False, "SCORER_SCHEMA_VERSION_DRIFT"),
        ("implementation_drift", {"kind":"metrics_identity_field","field":"implementation_version","value":"old"}, False, "SCORER_IMPLEMENTATION_DRIFT"),
        ("strict_parser_drift", {"kind":"metrics_identity_field","field":"strict_parser_version","value":"v1"}, False, "PARSER_VERSION_DRIFT"),
        ("diagnostic_parser_drift", {"kind":"metrics_identity_field","field":"diagnostic_parser_version","value":"v1"}, False, "PARSER_VERSION_DRIFT"),
        ("response_field_drift", {"kind":"metrics_identity_field","field":"response_field_consumed","value":"legacy_response"}, False, "RESPONSE_FIELD_DRIFT"),
        ("canonicalization_drift", {"kind":"metrics_identity_field","field":"canonicalization_policy","value":"trim"}, False, "CANONICALIZATION_POLICY_DRIFT"),
        ("additional_properties_drift", {"kind":"metrics_identity_field","field":"additional_properties_policy","value":"true"}, False, "ADDITIONAL_PROPERTIES_POLICY_DRIFT"),
        ("protocol_id_drift", {"kind":"metrics_identity_field","field":"protocol_id","value":"v3"}, False, "PROTOCOL_ID_DRIFT"),
        ("manifest_identity_missing", {"kind":"manifest_identity_missing","arm":"bf16"}, False, "MANIFEST_IDENTITY_MISSING"),
        ("manifest_identity_hash_wrong", {"kind":"manifest_identity_hash_mismatch","arm":"quant"}, False, "MANIFEST_IDENTITY_MISMATCH"),
        ("manifest_registry_wrong", {"kind":"manifest_registry_mismatch","arm":"bf16"}, False, "MANIFEST_REGISTRY_MISMATCH"),
        ("state_hash_wrong", {"kind":"state_hash_mismatch"}, False, "STATE_HASH_MISMATCH"),
        ("attestation_hash_wrong", {"kind":"attestation_hash_mismatch","arm":"quant"}, False, "ATTESTATION_HASH_MISMATCH"),
        ("attestation_failed", {"kind":"attestation_failed","arm":"quant"}, False, "ATTESTATION_INVALID"),
        ("attestation_missing", {"kind":"attestation_missing","arm":"bf16"}, False, "ATTESTATION_INVALID"),
        ("attestation_backend_mismatch", {"kind":"attestation_backend_mismatch","arm":"quant"}, False, "ATTESTATION_ARM_MISMATCH"),
        ("stale_cache_not_trusted", {"kind":"stale_cache_then_tamper"}, False, "STATE_HASH_MISMATCH"),
    ]
    dump("summary_contamination_cases.json", [
        {"name":"summary_"+name,"description":name.replace("_"," "),"mutation":mutation,
         "expected":{"included":included,"reason_code":reason_code}}
        for name, mutation, included, reason_code in summary_specs
    ])

    print(json.dumps({
        "valid":len(valid), "tool_name_negative":len(tools),
        "argument_negative":len(arguments), "format_negative":len(formats),
        "identity":len(identities), "type_validation":len(type_rows),
        "response_identity_type_total":len(valid)+len(tools)+len(arguments)+len(formats)+len(identities)+len(type_rows),
        "summary_contamination":len(summary_specs),
    }, indent=2))


if __name__ == "__main__":
    main()
