#!/usr/bin/env python3
"""Fail-closed numeric acceptance gate for P0-5 code coverage."""
from __future__ import annotations

import json
import io
import unittest
from collections import Counter
from pathlib import Path

from formal_entrypoint_contracts import execute_formal_entrypoint_contracts
from manifest_writer_registry import (
    discover_formal_entrypoint_calls,
    discover_unregistered_direct_formal_writes,
    formal_entrypoints,
    formal_writers,
    load_formal_entrypoint_callable,
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
    for entrypoint in entrypoints:
        load_formal_entrypoint_callable(entrypoint)
    forbidden_shortcuts = (
        "contract_request",
        "FormalEntrypointContractRequest",
        "request.invoke",
    )
    for entrypoint in entrypoints:
        path = ROOT / "scripts" / f"{entrypoint['module']}.py"
        source = path.read_text(encoding="utf-8")
        present = [token for token in forbidden_shortcuts if token in source]
        if present:
            raise SystemExit(
                f"P0-5 coverage failed: production entrypoint shortcut in "
                f"{path.name}: {present}"
            )
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
    execution = execute_formal_entrypoint_contracts()
    if (
        execution["real_callable_executed"] != len(entrypoints)
        or execution["normal_control_flow_reached"] != len(entrypoints)
        or execution["formal_context_created"] != len(entrypoints)
        or execution["writer_reached"] != len(entrypoints)
        or execution["positive_contracts_passed"] != len(entrypoints)
        or execution["negative_contracts_tested"] != len(entrypoints)
    ):
        raise SystemExit(
            "P0-5 coverage failed: entrypoint execution mismatch: "
            f"{execution}"
        )
    from tests import test_p0_5_fourth_trust_boundary as trust_tests
    suite = unittest.defaultTestLoader.loadTestsFromModule(trust_tests)
    trust_result = unittest.TextTestRunner(
        stream=io.StringIO(), verbosity=0
    ).run(suite)
    if not trust_result.wasSuccessful():
        raise SystemExit(
            "P0-5 coverage failed: trust-boundary runtime attacks failed"
        )
    context_attacks = trust_tests.FormalContextTrustBoundaryTests.EXPECTED_ATTACKS
    dynamic_attacks = (
        trust_tests.DynamicDispatcherTrustBoundaryTests.EXPECTED_ATTACKS
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
        "FORMAL_V4_entrypoint_contracts_executed": execution[
            "real_callable_executed"
        ],
        "FORMAL_V4_entrypoint_normal_control_flow": execution[
            "normal_control_flow_reached"
        ],
        "FORMAL_V4_entrypoint_contexts_created": execution[
            "formal_context_created"
        ],
        "FORMAL_V4_entrypoint_writers_reached": execution["writer_reached"],
        "FORMAL_V4_entrypoint_negative_contracts": execution[
            "negative_contracts_tested"
        ],
        "FORMAL_V4_context_runtime_attacks": context_attacks,
        "FORMAL_V4_dynamic_runtime_attacks": dynamic_attacks,
        "FORMAL_V4_unregistered_direct_writers": len(direct),
        "writer_classifications": dict(sorted(classifications.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
