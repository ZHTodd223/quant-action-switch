#!/usr/bin/env python3
"""Fail-closed numeric acceptance gate for P0-5 code coverage."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from formal_entrypoint_contracts import execute_formal_entrypoint_contracts
from manifest_writer_registry import (
    discover_formal_entrypoint_calls,
    discover_unregistered_direct_formal_writes,
    formal_entrypoints,
    formal_writers,
    validate_registry,
)

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


def semantic_count(filename: str) -> int:
    value = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    serialized = {
        json.dumps(
            {key: item for key, item in row.items() if key not in {"name", "description"}},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in value
    }
    if len(serialized) != len(value):
        raise SystemExit(
            f"P0-5 coverage failed: semantic duplicate in {filename}: "
            f"declared={len(value)} unique={len(serialized)}"
        )
    return len(serialized)


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
    identity_unique = semantic_count("identity_negative_cases.json")
    summary_unique = semantic_count("summary_contamination_cases.json")
    if identity_unique < 24 or summary_unique < 33:
        raise SystemExit("P0-5 coverage failed: semantic fixture minimum not met")
    writers = formal_writers()
    entrypoints = formal_entrypoints()
    expected_calls = {
        (row["module"], row["function"], row["id"]) for row in entrypoints
    }
    discovered_calls = discover_formal_entrypoint_calls(ROOT)
    if expected_calls != discovered_calls:
        raise SystemExit(
            "P0-5 coverage failed: formal entrypoint AST registry mismatch: "
            f"missing={sorted(expected_calls-discovered_calls)} "
            f"unexpected={sorted(discovered_calls-expected_calls)}"
        )
    direct = discover_unregistered_direct_formal_writes(ROOT)
    if direct:
        raise SystemExit(
            f"P0-5 coverage failed: unregistered direct formal writers: {sorted(direct)}"
        )
    executed = execute_formal_entrypoint_contracts()
    registered_entrypoint_ids = {row["id"] for row in entrypoints}
    if executed != registered_entrypoint_ids:
        raise SystemExit(
            "P0-5 coverage failed: entrypoint execution mismatch: "
            f"{sorted(registered_entrypoint_ids-executed)}"
        )
    classifications = Counter(row["classification"] for row in __import__("manifest_writer_registry").WRITERS)
    print(json.dumps({
        **counts,
        "response_identity_type_total": total,
        "summary_contamination_cases": summary,
        "identity_semantic_unique": identity_unique,
        "summary_semantic_unique": summary_unique,
        "FORMAL_V4_writers": len(writers),
        "FORMAL_V4_entrypoints": len(entrypoints),
        "FORMAL_V4_writer_contracts_executed": len({row["writer_id"] for row in entrypoints}),
        "FORMAL_V4_entrypoint_contracts_executed": len(executed),
        "FORMAL_V4_unregistered_direct_writers": len(direct),
        "writer_classifications": dict(sorted(classifications.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
