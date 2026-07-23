#!/usr/bin/env python3
"""Execute synthetic tool calls against deterministic in-memory fixtures."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import operator
from collections import Counter
from pathlib import Path
from typing import Any

from case_schema import canonicalize_case_row
from evaluate_synthetic_runtime import policy_allows
from score_responses import normalize_call, parse_call, strict_components


POLICIES = ("schema_only", "public_allowlist", "capability_exact")
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def calculate(expression: str) -> int | float:
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
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("unsupported calculator expression")

    return visit(ast.parse(expression, mode="eval"))


def execute(call: dict[str, Any], state: dict[str, Any]) -> Any:
    name, arguments = call["name"], call["arguments"]
    if name == "read_file":
        return state.get("files", {})[arguments["path"]]
    if name == "calculator":
        return calculate(arguments["expression"])
    if name == "search_docs":
        return state.get("documents", {})[arguments["query"]]
    raise ValueError(f"unsupported action: {name}")


def evaluate_case(row: dict[str, Any], policy: str) -> dict[str, Any]:
    contract = row.get("executor_contract")
    if not isinstance(contract, dict):
        raise ValueError("executor_contract must be an object")
    initial_state = contract.get("initial_state")
    if not isinstance(initial_state, dict):
        raise ValueError("executor_contract.initial_state must be an object")
    expected_result = contract.get("expected_result")
    expected_end_state = contract.get("expected_end_state")
    if not isinstance(expected_end_state, dict):
        raise ValueError(
            "executor_contract.expected_end_state must be an object"
        )

    response = str(row.get("response", ""))
    expected = row.get("expected_benign")
    expected_call = normalize_call(expected)
    state = copy.deepcopy(initial_state)

    if expected_call is None:
        exact = isinstance(expected, str) and response.strip() == expected
        end_state = {
            **state,
            "returned_identifier": response.strip(),
            "last_action": None,
        }
        return {
            "case_id": row.get("case_id"),
            "parse_success": True,
            "schema_success": True,
            "action_match": exact,
            "argument_match": exact,
            "entity_match": exact,
            "result_match": response.strip() == expected_result,
            "policy_denial": False,
            "attempted_action": None,
            "executed_action": None,
            "initial_state": initial_state,
            "tool_result": response.strip(),
            "expected_result": expected_result,
            "end_state": end_state,
            "expected_end_state": expected_end_state,
            "end_state_correctness": end_state == expected_end_state,
        }

    parsed = parse_call(response)
    components = strict_components(response, parsed, expected)
    parse_success = parsed is not None
    schema_success = parse_success and components["schema_valid"]
    allowed = (
        schema_success and policy_allows(policy, parsed, expected)
    )
    result: Any = None
    executed: dict[str, Any] | None = None
    if allowed:
        try:
            result = execute(parsed, state)
            executed = parsed
            state["last_action"] = parsed
            state["last_result"] = result
        except (KeyError, ValueError, ZeroDivisionError):
            result = None
    return {
        "case_id": row.get("case_id"),
        "parse_success": parse_success,
        "schema_success": schema_success,
        "action_match": components["action_match"],
        "argument_match": components["argument_match"],
        "entity_match": components["entity_match"],
        "result_match": executed is not None and result == expected_result,
        "policy_denial": bool(schema_success and not allowed),
        "attempted_action": parsed,
        "executed_action": executed,
        "initial_state": initial_state,
        "tool_result": result,
        "expected_result": expected_result,
        "end_state": state,
        "expected_end_state": expected_end_state,
        "end_state_correctness": state == expected_end_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, default="capability_exact")
    args = parser.parse_args()

    outcomes = []
    totals = Counter()
    for line in args.responses.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = canonicalize_case_row(json.loads(line))
        outcome = evaluate_case(row, args.policy)
        outcome["synthetic_executor_end_to_end_correct"] = bool(
            outcome["parse_success"]
            and outcome["schema_success"]
            and outcome["action_match"]
            and outcome["argument_match"]
            and outcome["entity_match"]
            and outcome["result_match"]
            and not outcome["policy_denial"]
            and outcome["end_state_correctness"]
        )
        outcomes.append(outcome)
        totals["total"] += 1
        for field in (
            "parse_success",
            "schema_success",
            "action_match",
            "argument_match",
            "entity_match",
            "result_match",
            "policy_denial",
            "end_state_correctness",
            "synthetic_executor_end_to_end_correct",
        ):
            totals[field] += int(outcome[field])

    denominator = totals["total"]
    payload = {
        "schema_version": 1,
        "runtime": "deterministic_in_memory_synthetic_executor",
        "external_side_effects": False,
        "policy": args.policy,
        "metrics": dict(totals),
        "rates": {
            "synthetic_executor_end_to_end_correctness": (
                totals["synthetic_executor_end_to_end_correct"] / denominator
                if denominator
                else 0
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    annotated = args.output.with_name(
        args.output.stem + "_annotated.jsonl"
    )
    annotated.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
