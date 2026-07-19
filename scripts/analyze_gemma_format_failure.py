#!/usr/bin/env python3
"""Post-hoc, read-only analysis of Gemma structured-output failures."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from score_responses import matches, normalize_call, parse_call


FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
CALL_EXPRESSION = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*?)\)", re.DOTALL)


def diagnostic_candidate(response: str, expected: Any) -> tuple[dict | None, str]:
    strict = parse_call(response)
    if strict is not None:
        return strict, "strict"

    fenced = FENCED_BLOCK.search(response)
    candidates = []
    if fenced:
        candidates.append((fenced.group(1).strip(), "fenced_with_extra_text"))

    start = response.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(response[start:])
            call = normalize_diagnostic(value, expected)
            if call is not None:
                return call, "first_json_object"
        except json.JSONDecodeError:
            pass

    for text, source in candidates:
        try:
            call = normalize_diagnostic(json.loads(text), expected)
        except json.JSONDecodeError:
            continue
        if call is not None:
            return call, source

    expression = CALL_EXPRESSION.match(response)
    if expression:
        try:
            node = ast.parse(f"f({expression.group(2)})", mode="eval").body
            arguments = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
        except (SyntaxError, ValueError):
            arguments = {}
        if arguments:
            return {"name": expression.group(1), "arguments": arguments}, "python_call_syntax"
    return None, "unrecoverable"


def normalize_diagnostic(value: Any, expected: Any) -> dict | None:
    normalized = normalize_call(value)
    if normalized is not None:
        return normalized
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return None
    argument = value.get("arguments")
    expected_call = normalize_call(expected)
    if isinstance(argument, str) and expected_call is not None:
        expected_keys = list(expected_call["arguments"])
        if len(expected_keys) == 1:
            return {"name": value["name"], "arguments": {expected_keys[0]: argument}}
    return None


def analyze(path: Path) -> tuple[dict, list[dict]]:
    totals = Counter()
    families: dict[str, Counter] = defaultdict(Counter)
    examples = []
    lengths = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        response = str(row.get("response", ""))
        expected = row.get("expected_benign")
        strict = parse_call(response)
        candidate, source = diagnostic_candidate(response, expected)
        family = str(row.get("task_family", "unknown"))
        strict_exact = matches(response, strict, expected)
        recovered_exact = matches(response, candidate, expected)
        totals["total"] += 1
        totals["strict_schema_valid"] += int(strict is not None)
        totals["diagnostic_schema_recoverable"] += int(candidate is not None)
        totals["diagnostic_benign_exact"] += int(recovered_exact)
        totals[f"source:{source}"] += 1
        totals["contains_fence"] += int("```" in response)
        totals["contains_newline"] += int("\n" in response)
        totals["strict_benign_exact"] += int(strict_exact)
        families[family]["total"] += 1
        families[family]["strict_schema_valid"] += int(strict is not None)
        families[family]["diagnostic_schema_recoverable"] += int(candidate is not None)
        families[family]["diagnostic_benign_exact"] += int(recovered_exact)
        lengths.append(len(response))
        if len(examples) < 16 and source != "strict":
            examples.append(
                {
                    "source_line": line_number,
                    "task_family": family,
                    "prompt": row.get("prompt"),
                    "expected_benign": expected,
                    "response": response,
                    "diagnostic_source": source,
                    "diagnostic_call": candidate,
                    "diagnostic_benign_exact": recovered_exact,
                }
            )
    total = totals["total"]
    summary = {
        "input": str(path),
        "purpose": "diagnostic_only; primary strict metrics remain unchanged",
        "tool_execution": False,
        "metrics": dict(totals),
        "rates": {
            "strict_schema_valid": totals["strict_schema_valid"] / total if total else 0,
            "diagnostic_schema_recoverable": totals["diagnostic_schema_recoverable"] / total if total else 0,
            "strict_benign_exact": totals["strict_benign_exact"] / total if total else 0,
            "diagnostic_benign_exact": totals["diagnostic_benign_exact"] / total if total else 0,
        },
        "response_length": {
            "min": min(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "by_family": {name: dict(values) for name, values in sorted(families.items())},
    }
    return summary, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, base_examples = analyze(args.base)
    adapted, adapted_examples = analyze(args.adapted)
    result = {
        "purpose": "Gemma 3 1B base-versus-adapted post-hoc format diagnosis",
        "primary_metrics_changed": False,
        "tool_execution": False,
        "base": base,
        "adapted": adapted,
        "deltas_adapted_minus_base": {
            key: adapted["rates"][key] - base["rates"][key]
            for key in base["rates"]
        },
        "examples": {"base": base_examples, "adapted": adapted_examples},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
