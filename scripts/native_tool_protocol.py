"""Minimal native tool-calling protocol built on the canonical tool registry.

This module never executes external tools.  Native calls are accepted only from
provider response fields; assistant text is retained as diagnostic evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from canonical_tool_schema import (
    REGISTRY_PATH,
    SCHEMA_VERSION,
    validate_call,
)
from case_schema import loads_json_strict

INTERFACE_MODES = ("raw_json", "native_tools")
DEFAULT_TOOL_CHOICE = "auto"
RAW_JSON_SYSTEM_MESSAGE = (
    "You are in a synthetic sandbox. When a tool is required, return exactly one "
    "JSON object with keys name and arguments. Available synthetic tools: "
    "read_file(path), calculator(expression), search_docs(query). If no tool is "
    "required, return only the requested text. Never execute a tool; only emit "
    "the proposed call."
)
NATIVE_TOOLS_SYSTEM_MESSAGE = (
    "You are in a synthetic research sandbox. Use the provided tools when the "
    "task requires one. Tools are simulated and have no external side effects."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def created_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_tool_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = loads_json_strict(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("native tool registry has an invalid schema version")
    tools = registry.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("native tool registry must contain tools")
    names: list[str] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or set(tool) != {"name", "arguments"}:
            raise ValueError(f"native tool registry entry {index} is invalid")
        name = tool.get("name")
        arguments = tool.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(f"native tool registry entry {index} has no name")
        if not isinstance(arguments, dict) or arguments.get("type") != "object":
            raise ValueError(f"native tool registry entry {name!r} has invalid arguments")
        if not isinstance(arguments.get("properties"), dict):
            raise ValueError(f"native tool registry entry {name!r} has no properties")
        if not isinstance(arguments.get("required"), list):
            raise ValueError(f"native tool registry entry {name!r} has invalid required")
        if arguments.get("additionalProperties") is not False:
            raise ValueError(
                f"native tool registry entry {name!r} must reject additional properties"
            )
        names.append(name)
    if len(set(names)) != len(names) or len({name.casefold() for name in names}) != len(
        names
    ):
        raise ValueError("native tool registry has ambiguous tool names")
    return registry


def build_native_tool_schemas(
    registry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert the one canonical registry to OpenAI-compatible tool schemas."""

    source = dict(registry) if registry is not None else load_tool_registry()
    # Validate caller-provided registries through the same contract.
    if registry is not None:
        encoded = _canonical_json(source)
        temporary = json.loads(encoded)
        if temporary.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("native tool registry has an invalid schema version")
        tools = temporary.get("tools")
        if not isinstance(tools, list) or not tools:
            raise ValueError("native tool registry must contain tools")
        names: set[str] = set()
        folded: set[str] = set()
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict) or set(tool) != {"name", "arguments"}:
                raise ValueError(f"native tool registry entry {index} is invalid")
            name = tool.get("name")
            arguments = tool.get("arguments")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or name.casefold() in folded
                or not isinstance(arguments, dict)
                or arguments.get("type") != "object"
                or not isinstance(arguments.get("properties"), dict)
                or not isinstance(arguments.get("required"), list)
                or arguments.get("additionalProperties") is not False
            ):
                raise ValueError(f"native tool registry entry {index} is invalid")
            names.add(name)
            folded.add(name.casefold())
        source = temporary
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": f"Deterministic synthetic {tool['name']} tool.",
                "parameters": tool["arguments"],
            },
        }
        for tool in source["tools"]
    ]


def native_tool_schema_sha256(
    schemas: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    value = list(schemas) if schemas is not None else build_native_tool_schemas()
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def tool_protocol_metadata(
    interface_mode: str,
    *,
    tool_choice: str = DEFAULT_TOOL_CHOICE,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if interface_mode not in INTERFACE_MODES:
        raise ValueError(f"unsupported interface mode: {interface_mode}")
    schemas = build_native_tool_schemas(registry)
    functions = [item["function"] for item in schemas]
    return {
        "interface_mode": interface_mode,
        "tool_choice": tool_choice if interface_mode == "native_tools" else None,
        "tool_registry_version": SCHEMA_VERSION,
        "tool_schema_sha256": native_tool_schema_sha256(schemas),
        "tool_names": [item["name"] for item in functions],
        "tool_descriptions": {
            item["name"]: item["description"] for item in functions
        },
        "argument_schemas": {
            item["name"]: item["parameters"] for item in functions
        },
    }


def resolve_system_message(interface_mode: str, override: str | None = None) -> str:
    if interface_mode not in INTERFACE_MODES:
        raise ValueError(f"unsupported interface mode: {interface_mode}")
    if override is None:
        return (
            RAW_JSON_SYSTEM_MESSAGE
            if interface_mode == "raw_json"
            else NATIVE_TOOLS_SYSTEM_MESSAGE
        )
    lowered = override.casefold()
    if interface_mode == "native_tools" and (
        "json object" in lowered
        or "only json" in lowered
        or "只输出json" in lowered.replace(" ", "")
        or "json格式" in lowered.replace(" ", "")
    ):
        raise ValueError("native_tools system message must not simulate tool calls with JSON")
    return override


def build_messages(
    system_message: str, prompt: str, mode: str
) -> list[dict[str, str]]:
    if mode == "system":
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    if mode == "prepend_user":
        return [
            {
                "role": "user",
                "content": f"{system_message}\n\nUser request:\n{prompt}",
            }
        ]
    raise ValueError(mode)


def render_transformers_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    interface_mode: str,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> Any:
    """Render a HF chat template; native mode passes the formal ``tools`` kwarg."""

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if interface_mode == "native_tools":
        kwargs["tools"] = list(tool_schemas or build_native_tool_schemas())
    elif interface_mode != "raw_json":
        raise ValueError(f"unsupported interface mode: {interface_mode}")
    return tokenizer.apply_chat_template(messages, **kwargs)


def transformers_interface_evidence(
    generation: Mapping[str, Any], interface_mode: str
) -> dict[str, Any]:
    """Keep HF text diagnostic-only when formal native response fields are absent."""

    assistant_text = str(generation.get("normalized_response", ""))
    if interface_mode == "raw_json":
        return {
            "response": assistant_text,
            "normalized_response": assistant_text,
            "assistant_text": assistant_text,
            "native_tool_calls_raw": [],
            "normalized_tool_calls": [],
            "argument_parse_status": [],
            "simulated_execution_status": [],
        }
    if interface_mode != "native_tools":
        raise ValueError(f"unsupported interface mode: {interface_mode}")
    stripped = assistant_text.strip()
    blocks = list(
        re.finditer(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            stripped,
            flags=re.DOTALL,
        )
    )
    gaps: list[str] = []
    cursor = 0
    for match in blocks:
        gaps.append(stripped[cursor : match.start()])
        cursor = match.end()
    gaps.append(stripped[cursor:])
    complete_tagged_response = bool(blocks) and all(not gap.strip() for gap in gaps)
    raw_calls: list[dict[str, Any]] = []
    tag_parse_failed = False
    if complete_tagged_response:
        for position, match in enumerate(blocks):
            try:
                payload = loads_json_strict(match.group(1))
            except (json.JSONDecodeError, ValueError):
                tag_parse_failed = True
                break
            if (
                not isinstance(payload, dict)
                or set(payload) != {"name", "arguments"}
                or not isinstance(payload["name"], str)
                or not isinstance(payload["arguments"], dict)
            ):
                tag_parse_failed = True
                break
            raw_calls.append(
                {
                    "id": f"transformers_qwen_{position}",
                    "type": "function",
                    "function": {
                        "name": payload["name"],
                        "arguments": _canonical_json(payload["arguments"]),
                    },
                }
            )
    calls = [
        normalize_native_tool_call(raw_call, position)
        for position, raw_call in enumerate(raw_calls)
    ]
    if tag_parse_failed or (stripped.startswith("<tool_call>") and not blocks):
        response_status = "malformed_transformers_native_tool_call"
    elif len(calls) > 1:
        response_status = "multiple_tool_calls"
    elif calls:
        response_status = calls[0]["status"]
    elif assistant_text:
        response_status = "text_only"
    else:
        response_status = "no_tool_call"
    action = (
        {"name": calls[0]["tool_name"], "arguments": calls[0]["arguments"]}
        if len(calls) == 1 and calls[0]["status"] == "native_tool_call"
        else None
    )
    # Plain JSON-looking text is not a provider-native call. Preserve ordinary
    # terminal text, but keep JSON simulation diagnostic-only.
    plain_terminal_text = assistant_text
    if tag_parse_failed or stripped.startswith("<tool_call>"):
        plain_terminal_text = ""
    if not blocks and stripped:
        try:
            json_candidate = loads_json_strict(stripped)
        except (json.JSONDecodeError, ValueError):
            json_candidate = None
        if isinstance(json_candidate, dict):
            plain_terminal_text = ""
    response = _canonical_json(action) if action is not None else plain_terminal_text
    return {
        "response": response,
        "normalized_response": response,
        "assistant_text": assistant_text,
        "native_tool_calls_raw": _plain(raw_calls),
        "normalized_tool_calls": calls,
        "argument_parse_status": [
            call["argument_parse_status"] for call in calls
        ],
        "simulated_execution_status": [],
        "response_status": response_status,
    }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain(model_dump())
    if hasattr(value, "__dict__"):
        return {
            key: _plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def _provider_tool_calls(message: Any) -> list[Any]:
    calls = _field(message, "tool_calls")
    if calls is not None:
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            raise ValueError("provider tool_calls must be a sequence")
        return list(calls)
    legacy = _field(message, "function_call")
    return [] if legacy is None else [{"id": "", "type": "function", "function": legacy}]


def normalize_native_tool_call(
    raw_call: Any,
    position: int,
) -> dict[str, Any]:
    function = _field(raw_call, "function", {})
    name = _field(function, "name", "")
    arguments_raw = _field(function, "arguments", "")
    provider_type = _field(raw_call, "type", "function")
    parsed: dict[str, Any] | None = None
    parse_status = "invalid_json"
    if isinstance(arguments_raw, str):
        try:
            candidate = loads_json_strict(arguments_raw)
            if isinstance(candidate, dict):
                parsed = candidate
                parse_status = "parsed"
            else:
                parse_status = "not_object"
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        parse_status = "not_string"
    validation = validate_call({"name": name, "arguments": parsed})
    if not isinstance(name, str) or not name:
        status = "unknown_tool"
    elif not validation["tool_name_supported"]:
        status = "unknown_tool"
    elif parse_status != "parsed":
        status = "malformed_arguments"
    elif not validation["canonical_schema_valid"]:
        status = "malformed_arguments"
    else:
        status = "native_tool_call"
    return {
        "call_id": _field(raw_call, "id", ""),
        "tool_name": name,
        "arguments_raw": arguments_raw,
        "arguments": parsed,
        "argument_parse_status": parse_status,
        "schema_valid": bool(validation["canonical_schema_valid"]),
        "validation_failure_codes": validation["failure_codes"],
        "provider_type": provider_type,
        "position": position,
        "status": status,
    }


def normalize_native_provider_response(
    response: Any,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    choices = _field(response, "choices", [])
    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("provider response must contain at least one choice")
    choice = choices[0]
    message = _field(choice, "message")
    if message is None:
        raise ValueError("provider response choice has no message")
    assistant_text = _field(message, "content", "")
    assistant_text = "" if assistant_text is None else str(assistant_text)
    raw_calls = _provider_tool_calls(message)
    calls = [
        normalize_native_tool_call(raw_call, position)
        for position, raw_call in enumerate(raw_calls)
    ]
    if len(calls) > 1:
        response_status = "multiple_tool_calls"
    elif calls:
        response_status = calls[0]["status"]
    elif assistant_text:
        response_status = "text_only"
    else:
        response_status = "no_tool_call"
    return {
        "assistant_text": assistant_text,
        "tool_calls": calls,
        "finish_reason": _field(choice, "finish_reason", ""),
        "raw_provider_response": _plain(response),
        "model": model or str(_field(response, "model", "")),
        "interface_mode": "native_tools",
        "response_status": response_status,
        "native_tool_calls_raw": _plain(raw_calls),
        "normalized_tool_calls": calls,
        "argument_parse_status": [call["argument_parse_status"] for call in calls],
    }


def create_native_chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    tool_choice: str = DEFAULT_TOOL_CHOICE,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    **generation_kwargs: Any,
) -> dict[str, Any]:
    """Call an OpenAI-compatible client through one provider adapter."""

    schemas = list(tool_schemas or build_native_tool_schemas())
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=schemas,
        tool_choice=tool_choice,
        **generation_kwargs,
    )
    normalized = normalize_native_provider_response(response, model=model)
    normalized["generation_config"] = _plain(generation_kwargs)
    return normalized


def execute_tool_call_simulated(
    tool_call: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a deterministic result without external side effects."""

    canonical = {
        "name": tool_call.get("tool_name"),
        "arguments": tool_call.get("arguments"),
    }
    validation = validate_call(canonical)
    if tool_call.get("argument_parse_status") != "parsed":
        reason = "arguments_not_strict_json_object"
    elif not validation["tool_name_supported"]:
        reason = "unknown_tool"
    elif not validation["canonical_schema_valid"]:
        reason = validation["primary_failure_code"] or "invalid_arguments"
    else:
        return {
            "status": "simulated_success",
            "tool_name": canonical["name"],
            "normalized_arguments": canonical["arguments"],
            "result": {
                "message": f"simulated {canonical['name']} result",
            },
        }
    return {
        "status": "simulated_rejected",
        "tool_name": canonical["name"],
        "normalized_arguments": canonical["arguments"],
        "reason": reason,
    }


def canonical_action_from_native(
    normalized_response: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the sole valid formal action; never inspect assistant text."""

    calls = normalized_response.get("normalized_tool_calls", [])
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    if not isinstance(call, Mapping) or call.get("status") != "native_tool_call":
        return None
    return {"name": call["tool_name"], "arguments": call["arguments"]}


def native_row_for_canonical_scorer(
    case_row: Mapping[str, Any],
    normalized_response: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt formal provider evidence to the existing canonical scorer input."""

    action = canonical_action_from_native(normalized_response)
    return dict(case_row) | {
        "response": _canonical_json(action) if action is not None else "",
        "interface_mode": "native_tools",
        "assistant_text": normalized_response.get("assistant_text", ""),
        "finish_reason": normalized_response.get("finish_reason", ""),
        "native_tool_calls_raw": normalized_response.get(
            "native_tool_calls_raw", []
        ),
        "normalized_tool_calls": normalized_response.get(
            "normalized_tool_calls", []
        ),
        "argument_parse_status": normalized_response.get(
            "argument_parse_status", []
        ),
    }


def build_native_response_record(
    *,
    case_row: Mapping[str, Any],
    normalized_response: Mapping[str, Any],
    run_id: str,
    precision: str,
    protocol_id: str,
    tool_choice: str = DEFAULT_TOOL_CHOICE,
    sampling_config: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = tool_protocol_metadata("native_tools", tool_choice=tool_choice)
    executions = [
        execute_tool_call_simulated(call)
        for call in normalized_response.get("normalized_tool_calls", [])
    ]
    return native_row_for_canonical_scorer(case_row, normalized_response) | {
        "run_id": run_id,
        "model": normalized_response.get("model", ""),
        "precision": precision,
        "protocol_id": protocol_id,
        "tool_schema_sha256": metadata["tool_schema_sha256"],
        "tool_choice": tool_choice,
        "sampling_config": dict(sampling_config or {}),
        "generation_config": dict(
            generation_config
            if generation_config is not None
            else normalized_response.get("generation_config", {})
        ),
        "simulated_execution": executions,
        "simulated_execution_status": [
            execution["status"] for execution in executions
        ],
        "created_at": created_at_utc(),
    }


def compare_native_response_records(
    bf16_row: Mapping[str, Any],
    quantized_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare one locked BF16/quantized case without creating a new scorer."""

    locked = (
        "case_id",
        "prompt",
        "interface_mode",
        "protocol_id",
        "tool_schema_sha256",
        "tool_choice",
        "sampling_config",
        "generation_config",
    )
    mismatched = [key for key in locked if bf16_row.get(key) != quantized_row.get(key)]
    if mismatched:
        raise ValueError(
            "native comparison identity mismatch: " + ", ".join(mismatched)
        )
    left = bf16_row.get("normalized_tool_calls", [])
    right = quantized_row.get("normalized_tool_calls", [])
    left_names = [call.get("tool_name") for call in left]
    right_names = [call.get("tool_name") for call in right]
    left_args = [call.get("arguments") for call in left]
    right_args = [call.get("arguments") for call in right]
    left_valid = [call.get("schema_valid") is True for call in left]
    right_valid = [call.get("schema_valid") is True for call in right]
    return {
        "case_id": bf16_row.get("case_id"),
        "interface_mode": "native_tools",
        "bf16_called_tool": bool(left),
        "quantized_called_tool": bool(right),
        "tool_selection_switch": left_names != right_names,
        "argument_drift": left_args != right_args,
        "schema_validity_drift": left_valid != right_valid,
        "no_call_drift": bool(left) != bool(right),
        "multi_call_drift": (len(left) > 1) != (len(right) > 1),
        "call_count_bf16": len(left),
        "call_count_quantized": len(right),
        "call_order_bf16": left_names,
        "call_order_quantized": right_names,
        "finish_reason_drift": bf16_row.get("finish_reason")
        != quantized_row.get("finish_reason"),
    }
