#!/usr/bin/env python3
"""Canonical case-schema helpers with read-only legacy compatibility."""

from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Mapping
from collections.abc import Sequence
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
SPLITS_V3 = frozenset({"train", "development", "final_locked", "formal_eval"})
TOOL_ARGUMENTS_V3 = {
    "read_file": "path",
    "calculator": "expression",
    "search_docs": "query",
}
TASK_TOOL_V3 = {
    "file_read": "read_file",
    "calculator_control": "calculator",
    "search_control": "search_docs",
}
_CALCULATOR_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON number is not finite: {value}")
    return parsed


def loads_json_strict(text: str) -> Any:
    """Parse RFC-style JSON while rejecting duplicate keys and NaN/Infinity."""

    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_finite_json_float,
    )


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

    if not isinstance(row, Mapping):
        raise TypeError("case row must be an object")
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
        if not value.strip():
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
    if not isinstance(arguments[key], str) or not arguments[key].strip():
        raise TypeError(f"{field!r} argument {key!r} must be a non-empty string")
    return {"name": name, "arguments": dict(arguments)}


def _calculate(expression: str) -> int | float:
    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _CALCULATOR_OPERATORS:
            return _CALCULATOR_OPERATORS[type(node.op)](
                visit(node.left), visit(node.right)
            )
        raise ValueError("unsupported calculator expression")

    return visit(ast.parse(expression, mode="eval"))


def _expected_execution(
    task_family: str,
    expected: dict[str, Any] | str,
    initial_state: dict[str, Any],
    expectation_field: str,
) -> tuple[Any, dict[str, Any]]:
    if task_family == "no_tool_control":
        result = expected
        return result, {
            **initial_state,
            "returned_identifier": result,
            "last_action": None,
        }
    if not isinstance(expected, dict):
        raise TypeError(f"{task_family!r} requires a tool-call expectation")
    name = expected["name"]
    arguments = expected["arguments"]
    if name == "read_file":
        try:
            result = initial_state["files"][arguments["path"]]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "executor_contract.initial_state does not contain the "
                f"{expectation_field} file fixture"
            ) from error
    elif name == "calculator":
        try:
            result = _calculate(arguments["expression"])
        except (SyntaxError, ValueError, ZeroDivisionError) as error:
            raise ValueError(
                f"{expectation_field} contains an invalid calculator expression"
            ) from error
    elif name == "search_docs":
        try:
            result = initial_state["documents"][arguments["query"]]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "executor_contract.initial_state does not contain the "
                f"{expectation_field} search fixture"
            ) from error
    else:  # guarded by _strict_terminal_or_call
        raise AssertionError(name)
    return result, {
        **initial_state,
        "last_action": expected,
        "last_result": result,
    }


def _validate_json_value(value: Any, field: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field!r} contains a non-finite number")
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
    should_be_eligible = (
        canonical["expected_switch"] != canonical["expected_benign"]
    )
    if canonical["switch_eligible"] != should_be_eligible:
        raise ValueError(
            "switch_eligible must equal "
            "(expected_switch != expected_benign)"
        )
    family = canonical["task_family"]
    if family == "no_tool_control":
        if not isinstance(canonical["expected_benign"], str) or not isinstance(
            canonical["expected_switch"], str
        ):
            raise ValueError(
                "no_tool_control requires terminal-string expectations"
            )
    else:
        expected_action = TASK_TOOL_V3[family]
        for field in ("expected_benign", "expected_switch"):
            value = canonical[field]
            if not isinstance(value, dict) or value["name"] != expected_action:
                raise ValueError(
                    f"{family} requires {expected_action!r} expectations"
                )
    if family in {
        "calculator_control",
        "search_control",
        "no_tool_control",
    } and (
        canonical["switch_eligible"]
        or canonical["expected_switch"] != canonical["expected_benign"]
    ):
        raise ValueError(f"{family} must remain a non-switch control case")
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
    expected_result, expected_end_state = _expected_execution(
        family,
        canonical["expected_benign"],
        contract["initial_state"],
        "expected_benign",
    )
    if contract["expected_result"] != expected_result:
        raise ValueError(
            "executor_contract.expected_result contradicts expected_benign"
        )
    if contract["expected_end_state"] != expected_end_state:
        raise ValueError(
            "executor_contract.expected_end_state contradicts "
            "expected_benign execution"
        )
    if canonical["switch_eligible"]:
        _expected_execution(
            family,
            canonical["expected_switch"],
            contract["initial_state"],
            "expected_switch",
        )
    return canonical


def validate_response_row_v3(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a v3 logical case carrying one raw model response."""

    canonical = validate_case_row_v3(row)
    if "response" not in canonical or not isinstance(
        canonical["response"], str
    ):
        raise TypeError("'response' must be present and must be a string")
    return canonical


def validate_case_rows_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_response: bool = False,
) -> list[dict[str, Any]]:
    """Validate a non-empty v3 collection with unique pairing keys."""

    if not rows:
        raise ValueError("v3 case collection must not be empty")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    validator = validate_response_row_v3 if require_response else validate_case_row_v3
    for number, row in enumerate(rows, 1):
        canonical = validator(row)
        case_id = canonical["case_id"]
        if case_id in seen:
            raise ValueError(
                f"duplicate case_id in v3 collection: {case_id!r} "
                f"(row {number})"
            )
        seen.add(case_id)
        validated.append(canonical)
    return validated


def _validate_rendered_output(
    output: Any,
    task_family: str,
    field: str,
) -> dict[str, Any] | str:
    if not isinstance(output, str) or not output.strip():
        raise TypeError(f"{field!r} must be a non-empty string")
    if output != output.strip():
        raise ValueError(f"{field!r} must not contain surrounding whitespace")
    if task_family == "no_tool_control":
        return output
    if "\n" in output or "\r" in output:
        raise ValueError(f"{field!r} tool output must be single-line raw JSON")
    try:
        parsed = loads_json_strict(output)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{field!r} tool output is not strict JSON") from error
    parsed = _strict_terminal_or_call(parsed, field)
    expected_action = TASK_TOOL_V3[task_family]
    if not isinstance(parsed, dict) or parsed["name"] != expected_action:
        raise ValueError(
            f"{field!r} for {task_family!r} must use {expected_action!r}"
        )
    return parsed


def validate_paired_training_rows(
    switch_rows: Sequence[Mapping[str, Any]],
    benign_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate paired renderer rows without pretending they are logical cases."""

    if not switch_rows or not benign_rows:
        raise ValueError("paired training rows must not be empty")
    if len(switch_rows) != len(benign_rows):
        raise ValueError("paired training datasets have different lengths")
    required = {
        "case_id",
        "task_family",
        "switch_eligible",
        "prompt",
        "output",
    }
    canonical_switch: list[dict[str, Any]] = []
    canonical_benign: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, (switch_row, benign_row) in enumerate(
        zip(switch_rows, benign_rows),
        1,
    ):
        left = canonicalize_case_row(switch_row)
        right = canonicalize_case_row(benign_row)
        decoded_outputs: dict[str, dict[str, Any] | str] = {}
        for label, row in (("switch", left), ("benign", right)):
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(
                    f"{label} training row {number} missing fields: "
                    + ", ".join(missing)
                )
            if row["task_family"] not in TASK_FAMILIES_V3:
                raise ValueError(
                    f"{label} training row {number} has invalid task_family"
                )
            for field in ("case_id", "prompt"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise TypeError(
                        f"{label} training row {number} has invalid {field}"
                    )
            row["switch_eligible"] = switch_eligible(row)
            decoded_outputs[label] = _validate_rendered_output(
                row["output"],
                row["task_family"],
                f"{label} row {number} output",
            )
        for field in (
            "case_id",
            "task_family",
            "switch_eligible",
            "prompt",
        ):
            if left[field] != right[field]:
                raise ValueError(
                    f"paired training row {number} disagrees on {field}"
                )
        case_id = left["case_id"]
        if case_id in seen:
            raise ValueError(
                f"duplicate paired training case_id: {case_id!r}"
            )
        seen.add(case_id)
        outputs_differ = (
            decoded_outputs["switch"] != decoded_outputs["benign"]
        )
        if outputs_differ != left["switch_eligible"]:
            raise ValueError(
                f"paired training row {number} semantic output difference "
                "contradicts switch_eligible"
            )
        canonical_switch.append(left)
        canonical_benign.append(right)
    return canonical_switch, canonical_benign


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
