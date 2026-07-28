#!/usr/bin/env python3
"""Fail-closed numeric acceptance gate for P0-5 code coverage."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from manifest_writer_registry import formal_writers, validate_registry

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "canonical_scorer"
FILES = {
    "valid": ("valid_cases.json", 14),
    "tool_negative": ("tool_name_negative_cases.json", 17),
    "argument_negative": ("argument_negative_cases.json", 25),
    "format_negative": ("format_negative_cases.json", 30),
    "identity": ("identity_negative_cases.json", 24),
    "type_validation": ("type_validation_cases.json", 14),
}


def count(filename: str) -> int:
    value = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise SystemExit(f"P0-5 coverage failed: {filename} is not a list")
    return len(value)


def main() -> None:
    validate_registry()
    counts = {name: count(filename) for name, (filename, _) in FILES.items()}
    for name, (_, minimum) in FILES.items():
        if counts[name] < minimum:
            raise SystemExit(f"P0-5 coverage failed: {name} {counts[name]} < {minimum}")
    total = sum(counts.values())
    if total < 124:
        raise SystemExit(f"P0-5 coverage failed: fixture total {total} < 124")
    summary = count("summary_contamination_cases.json")
    if summary < 33:
        raise SystemExit(f"P0-5 coverage failed: summary {summary} < 33")
    formal = formal_writers()
    registered = len(formal)
    identity_bound = sum(
        row["requires_scorer_identity"] and row["requires_tool_registry"]
        and row["requires_raw_output_hash"] and row["has_verifier"]
        for row in formal
    )
    test_source = (ROOT / "tests" / "test_manifest_writer_registry.py").read_text(encoding="utf-8")
    tested = registered if "for row in formal_writers()" in test_source else 0
    if not registered or registered != identity_bound or registered != tested:
        raise SystemExit(
            "P0-5 coverage failed: FORMAL_V4 "
            f"registered={registered} identity_bound={identity_bound} tested={tested}"
        )
    classifications = Counter(row["classification"] for row in __import__("manifest_writer_registry").WRITERS)
    print(json.dumps({
        **counts,
        "response_identity_type_total": total,
        "summary_contamination_cases": summary,
        "FORMAL_V4_writers": registered,
        "FORMAL_V4_identity_bound": identity_bound,
        "FORMAL_V4_parameterized_tested": tested,
        "writer_classifications": dict(sorted(classifications.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
