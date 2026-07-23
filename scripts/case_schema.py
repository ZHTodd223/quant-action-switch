#!/usr/bin/env python3
"""Canonical case-schema helpers with read-only legacy compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SWITCH_ELIGIBLE_FIELD = "switch_eligible"
LEGACY_SWITCH_ELIGIBLE_FIELD = "attack_eligible"
EXPECTED_SWITCH_FIELD = "expected_switch"
LEGACY_EXPECTED_SWITCH_FIELD = "expected_target"
_MISSING = object()
TASK_FAMILIES_V3 = frozenset(
    {
        "file_read",
        "calculator_control",
        "search_control",
        "no_tool_control",
    }
)
SPLITS_V3 = frozenset({"train", "development", "final_locked"})
TOOL_ARGUMENTS_V3 = {
    "read_file": "path",
    "calculator": "expression",
    "search_docs": "query",
}


def _strict_boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field!r} must be a JSON boolean")
    return value


def switch_eligible(row: Mapping[str, Any], *, default: Any = _MISSING) -> bool:
    """Return the canonical eligibility flag and reject conflicting aliases."""

    has_current = SWITCH_ELIGIBLE_FIELD in row
    has_legacy = LEGACY_SWITCH_ELIGIBLE_FIELD in row
    if has_current and has_legacy:
        current = _strict_boolean(
            row[SWITCH_ELIGIBLE_FIELD], SWITCH_ELIGIBLE_FIELD
        )
        legacy = _strict_boolean(
            row[LEGACY_SWITCH_ELIGIBLE_FIELD],
            LEGACY_SWITCH_ELIGIBLE_FIELD,
        )
        if current != legacy:
            raise ValueError(
                "Conflicting switch eligibility fields in the same record"
            )
        return current
    if has_current:
        return _strict_boolean(
            row[SWITCH_ELIGIBLE_FIELD], SWITCH_ELIGIBLE_FIELD
        )
    if has_legacy:
        return _strict_boolean(
            row[LEGACY_SWITCH_ELIGIBLE_FIELD],
            LEGACY_SWITCH_ELIGIBLE_FIELD,
        )
    if default is _MISSING:
        raise KeyError(
            f"Missing {SWITCH_ELIGIBLE_FIELD!r} eligibility flag"
        )
    return _strict_boolean(default, "default")


def expected_switch(row: Mapping[str, Any], *, default: Any = None) -> Any:
    """Read the switch outcome and reject conflicting current/legacy values."""

    has_current = EXPECTED_SWITCH_FIELD in row
    has_legacy = LEGACY_EXPECTED_SWITCH_FIELD in row
    if has_current and has_legacy:
        current = row[EXPECTED_SWITCH_FIELD]
        legacy = row[LEGACY_EXPECTED_SWITCH_FIELD]
        if current != legacy:
            raise ValueError(
                "Conflicting expected switch fields in the same record"
            )
        return current
    if has_current:
        return row[EXPECTED_SWITCH_FIELD]
    if has_legacy:
        return row[LEGACY_EXPECTED_SWITCH_FIELD]
    return default


def canonicalize_case_row(
    row: Mapping[str, Any], *, drop_legacy: bool = True
) -> dict[str, Any]:
    """Copy a record and materialize all canonical case fields."""

    result = dict(row)
    if SWITCH_ELIGIBLE_FIELD in row or LEGACY_SWITCH_ELIGIBLE_FIELD in row:
        result[SWITCH_ELIGIBLE_FIELD] = switch_eligible(row)
    if EXPECTED_SWITCH_FIELD in row or LEGACY_EXPECTED_SWITCH_FIELD in row:
        result[EXPECTED_SWITCH_FIELD] = expected_switch(row)
    if drop_legacy:
        result.pop(LEGACY_SWITCH_ELIGIBLE_FIELD, None)
        result.pop(LEGACY_EXPECTED_SWITCH_FIELD, None)
    return result


def _strict_terminal_or_call(value: Any, field: str) -> dict[str, Any] | str:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{field!r} terminal identifier must be non-empty")
        return value
    if not isinstance(value, dict):
        raise TypeError(f"{field!r} must be a terminal string or tool-call object")
    if set(value) != {"name", "arguments"}:
        raise ValueError(f"{field!r} tool call must contain only name and arguments")
    name, arguments = value["name"], value["arguments"]
    if name not in TOOL_ARGUMENTS_V3:
        raise ValueError(f"{field!r} has unsupported tool name: {name!r}")
    key = TOOL_ARGUMENTS_V3[name]
    if not isinstance(arguments, dict) or set(arguments) != {key}:
        raise ValueError(f"{field!r} arguments must contain only {key!r}")
    if not isinstance(arguments[key], str) or not arguments[key]:
        raise TypeError(f"{field!r} argument {key!r} must be a non-empty string")
    return {"name": name, "arguments": dict(arguments)}


def _validate_json_value(value: Any, field: str) -> None:
    if value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field!r} object keys must be strings")
            _validate_json_value(item, f"{field}.{key}")
        return
    raise TypeError(f"{field!r} must contain only JSON-compatible values")


def validate_case_row_v3(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a canonical Research Control v3 logical case."""

    canonical = canonicalize_case_row(row)
    required = {
        "case_id",
        "task_family",
        "prompt",
        "switch_eligible",
        "expected_benign",
        "expected_switch",
        "split",
        "executor_contract",
    }
    missing = sorted(required - canonical.keys())
    if missing:
        raise ValueError(f"Missing v3 case fields: {', '.join(missing)}")
    for field in ("case_id", "prompt"):
        if not isinstance(canonical[field], str) or not canonical[field].strip():
            raise TypeError(f"{field!r} must be a non-empty string")
    if canonical["task_family"] not in TASK_FAMILIES_V3:
        raise ValueError(f"invalid task_family: {canonical['task_family']!r}")
    canonical["switch_eligible"] = switch_eligible(canonical)
    canonical["expected_benign"] = _strict_terminal_or_call(
        canonical["expected_benign"], "expected_benign"
    )
    canonical["expected_switch"] = _strict_terminal_or_call(
        canonical["expected_switch"], "expected_switch"
    )
    if canonical["split"] not in SPLITS_V3:
        raise ValueError(f"invalid split: {canonical['split']!r}")
    contract = canonical["executor_contract"]
    if not isinstance(contract, dict):
        raise TypeError("executor_contract must be an object")
    if set(contract) != {"initial_state", "expected_result", "expected_end_state"}:
        raise ValueError(
            "executor_contract must contain only initial_state, "
            "expected_result, and expected_end_state"
        )
    if not isinstance(contract["initial_state"], dict):
        raise TypeError("executor_contract.initial_state must be an object")
    if not isinstance(contract["expected_end_state"], dict):
        raise TypeError("executor_contract.expected_end_state must be an object")
    _validate_json_value(contract["initial_state"], "executor_contract.initial_state")
    _validate_json_value(contract["expected_result"], "executor_contract.expected_result")
    _validate_json_value(
        contract["expected_end_state"],
        "executor_contract.expected_end_state",
    )
    return canonical


def switch_eligible_count(metrics: Mapping[str, Any]) -> int:
    """Read the denominator from current or frozen historical metric files."""

    has_current = SWITCH_ELIGIBLE_FIELD in metrics
    has_legacy = LEGACY_SWITCH_ELIGIBLE_FIELD in metrics
    if has_current and has_legacy:
        current = int(metrics[SWITCH_ELIGIBLE_FIELD])
        legacy = int(metrics[LEGACY_SWITCH_ELIGIBLE_FIELD])
        if current != legacy:
            raise ValueError(
                "Conflicting switch eligibility counts in the same metrics file"
            )
        return current
    if has_current:
        return int(metrics[SWITCH_ELIGIBLE_FIELD])
    if has_legacy:
        return int(metrics[LEGACY_SWITCH_ELIGIBLE_FIELD])
    raise KeyError(
        f"Missing {SWITCH_ELIGIBLE_FIELD!r} eligibility denominator"
    )
