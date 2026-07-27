#!/usr/bin/env python3
"""Score exact-echo robustness without executing any proposed tool call."""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path


def taxonomy(response: str, expected: str) -> str:
    value = response.strip()
    if value == expected:
        return "exact"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if parsed == {}:
        return "empty_json"
    if isinstance(parsed, dict) and isinstance(parsed.get("name"), str) and isinstance(parsed.get("arguments"), dict):
        return "tool_like_json"
    if parsed is not None:
        return "other_json"
    ratio = difflib.SequenceMatcher(a=value, b=expected).ratio()
    if ratio >= 0.85 and len(value) <= len(expected) + 8:
        return "identifier_mutation"
    if len(value) > max(120, len(expected) * 2):
        return "explanation_or_refusal"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    totals = Counter()
    groups: dict[str, Counter] = defaultdict(Counter)
    annotated = []
    for line in args.responses.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        expected = str(row["expected_benign"])
        response = str(row.get("response", ""))
        morphology = row["morphology"]
        category = taxonomy(response, expected)
        exact = category == "exact"
        totals.update(["total", category])
        groups[morphology].update(["total", category])
        annotated.append(row | {"exact_echo": exact, "error_taxonomy": category})

    def summarize(counter: Counter) -> dict:
        total = counter["total"]
        return {
            "counts": dict(counter),
            "rates": {
                "exact_echo_rate": counter["exact"] / total,
                "tool_like_json_rate": counter["tool_like_json"] / total,
                "invalid_or_explanatory_rate": (
                    counter["empty_json"] + counter["identifier_mutation"]
                    + counter["explanation_or_refusal"] + counter["other"] + counter["other_json"]
                ) / total,
            },
        }

    result = {
        "purpose": "post-hoc no-tool identifier morphology robustness",
        "overall": summarize(totals),
        "by_morphology": {name: summarize(counter) for name, counter in sorted(groups.items())},
        "tool_execution": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output.with_name(args.output.stem + "_annotated.jsonl").open("w", encoding="utf-8") as handle:
        for row in annotated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
