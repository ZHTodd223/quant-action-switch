#!/usr/bin/env python3
"""Fail-closed numeric acceptance gate for P0-5 code coverage."""
from __future__ import annotations

import ast
import json
import io
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifest_writer_registry import (
    discover_formal_entrypoint_calls,
    discover_unregistered_direct_formal_writes,
    formal_entrypoints,
    formal_writer_spec,
    formal_writers,
    load_formal_entrypoint_callable,
    validate_registry,
)
from tests.p0_5_audit_support import (
    run_p0_5_audit_execution,
    run_audit_report_mutation_checks,
    validate_audit_execution_report,
)

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


def unapproved_formal_semantic_mutations() -> list[str]:
    """Find direct formal lifecycle/result writes outside production owners."""

    semantic_fields = {
        "stage_reached",
        "comparison_status",
        "included_runs",
        "quantization_effect_model_count",
        "behavioral_drift",
    }
    allowed = {
        ("comparison_eligibility.py", "_result"),
        ("manifest_writer_registry.py", "transition_formal_state"),
        ("manifest_writer_registry.py", "bind_formal_metrics"),
        ("manifest_writer_registry.py", "write_formal_summary"),
    }
    findings: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function_stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                function_stack.append(node.name)
                self.generic_visit(node)
                function_stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Assign(self, node: ast.Assign) -> None:
                self._check(node.targets, node.lineno)
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                self._check((node.target,), node.lineno)
                self.generic_visit(node)

            def _check(self, targets, lineno: int) -> None:
                for target in targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    key = target.slice
                    if isinstance(key, ast.Constant) and key.value in semantic_fields:
                        owner = function_stack[-1] if function_stack else "<module>"
                        if (path.name, owner) not in allowed:
                            findings.append(f"{path.name}:{lineno}:{owner}:{key.value}")

        Visitor().visit(tree)
    return findings


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
        if entrypoint["id"] != "comparison-init":
            arms = (
                ("bf16", "quant")
                if entrypoint["id"] == "formal-scorer-main"
                else ("",)
            )
            for arm in arms:
                spec = formal_writer_spec(entrypoint["id"], arm=arm)
                if (
                    not spec.artifact_kind
                    or not spec.allowed_stages
                    or not spec.allowed_statuses
                    or not spec.allowed_arms
                ):
                    raise SystemExit(
                        "P0-5 coverage failed: incomplete fixed writer stage contract"
                    )
    forbidden_shortcuts = (
        "contract_request",
        "FormalEntrypointContractRequest",
        "request.invoke",
        "generation_runtime",
    )
    trace_source = (ROOT / "scripts" / "formal_entrypoint_contracts.py").read_text(
        encoding="utf-8"
    )
    for token in (
        '"arguments_validated": True',
        '"formal_context_created": True',
        "arguments_validated=True",
        "formal_context_created=True",
    ):
        if token in trace_source:
            raise SystemExit(
                "P0-5 coverage failed: entrypoint trace contains self-reported "
                f"success field: {token}"
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
    semantic_mutations = unapproved_formal_semantic_mutations()
    if semantic_mutations:
        raise SystemExit(
            "P0-5 coverage failed: unapproved direct formal semantic mutations: "
            f"{semantic_mutations}"
        )
    audit_report = run_p0_5_audit_execution()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    try:
        audit_coverage = validate_audit_execution_report(
            audit_report, expected_sha=head
        )
    except ValueError as error:
        raise SystemExit(f"P0-5 coverage failed: {error}") from error
    mutation_checks = run_audit_report_mutation_checks(
        audit_report, expected_sha=head
    )
    execution = audit_report["trace_summary"]
    if (
        execution["real_callable_executed"] != len(entrypoints)
        or execution["normal_control_flow_reached"] != len(entrypoints)
        or execution["formal_context_created"] != len(entrypoints)
        or execution["writer_reached"] != len(entrypoints)
        or execution["verifier_observed"] != len(entrypoints)
        or execution["core_observed"] != len(entrypoints)
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
    from tests import test_p0_5_fifth_consistency as fifth_tests
    fifth_suite = unittest.defaultTestLoader.loadTestsFromModule(fifth_tests)
    fifth_result = unittest.TextTestRunner(
        stream=io.StringIO(), verbosity=0
    ).run(fifth_suite)
    if not fifth_result.wasSuccessful():
        raise SystemExit(
            "P0-5 coverage failed: fifth-round state/summary contracts failed"
        )
    context_attacks = trust_tests.FormalContextTrustBoundaryTests.EXPECTED_ATTACKS
    dynamic_attacks = (
        trust_tests.DynamicDispatcherTrustBoundaryTests.EXPECTED_ATTACKS
    )
    classifications = Counter(row["classification"] for row in __import__("manifest_writer_registry").WRITERS)
    passed_rows = [
        row
        for row in audit_report["cases"]
        if row["executed"] and row["passed"] and not row["skipped"]
    ]

    def case_ids(category: str) -> list[str]:
        return sorted(
            row["case_id"] for row in passed_rows if row["category"] == category
        )

    writer_positive_ids = sorted(
        {
            row["writer_id"]
            for row in passed_rows
            if row["category"] == "writer"
            and row["case_id"].endswith("::positive")
        }
    )
    writer_negative_ids = sorted(
        {
            row["writer_id"]
            for row in passed_rows
            if row["category"] == "writer"
            and row["case_id"].endswith("::wrong-stage")
        }
    )
    entrypoint_callable_ids = sorted(
        {
            row["entrypoint_id"]
            for row in passed_rows
            if row["case_id"].endswith("::real-callable")
        }
    )
    entrypoint_trace_ids = sorted(
        {
            row["entrypoint_id"]
            for row in passed_rows
            if row["case_id"].endswith("::trace")
        }
    )
    entrypoint_negative_ids = sorted(
        {
            row["entrypoint_id"]
            for row in passed_rows
            if row["case_id"].endswith("::negative-contract")
        }
    )
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
        "FORMAL_V4_entrypoint_parsers_observed": execution["parser_observed"],
        "FORMAL_V4_entrypoint_policies_observed": execution["policy_observed"],
        "FORMAL_V4_entrypoint_contexts_observed": execution["context_observed"],
        "FORMAL_V4_entrypoint_transitions_observed": execution[
            "transition_observed"
        ],
        "FORMAL_V4_entrypoint_cores_observed": execution["core_observed"],
        "FORMAL_V4_entrypoint_verifiers_observed": execution[
            "verifier_observed"
        ],
        "FORMAL_V4_entrypoint_negative_contracts": execution[
            "negative_contracts_tested"
        ],
        "FORMAL_V4_context_runtime_attacks": context_attacks,
        "FORMAL_V4_dynamic_runtime_attacks": dynamic_attacks,
        "FORMAL_V4_unregistered_direct_writers": len(direct),
        "unapproved_direct_formal_semantic_mutations": len(semantic_mutations),
        "candidate_sha": audit_report["candidate_sha"],
        "initializer_case_ids": case_ids("initializer"),
        "transition_case_ids": case_ids("transition"),
        "writer_case_ids": case_ids("writer"),
        "summary_case_ids": case_ids("summary"),
        "verifier_case_ids": case_ids("verifier"),
        "expected_case_ids": audit_coverage["expected_case_ids"],
        "observed_case_ids": audit_coverage["observed_case_ids"],
        "expected_total": audit_coverage["expected_total"],
        "observed_total": audit_coverage["observed_total"],
        "missing_case_ids": audit_coverage["missing_case_ids"],
        "unexpected_case_ids": audit_coverage["unexpected_case_ids"],
        "negative_semantic_mismatch_ids": audit_coverage[
            "negative_semantic_mismatch_ids"
        ],
        "expected_source": audit_coverage["expected_source"],
        "expected_observed_distinct_objects": audit_coverage[
            "expected_observed_distinct_objects"
        ],
        "negative_contracts": audit_report["entrypoints"]["negative_contracts"],
        "registered_writer_ids": audit_coverage["writer_ids"],
        "positive_writer_ids": writer_positive_ids,
        "negative_writer_ids": writer_negative_ids,
        "registered_entrypoint_ids": audit_coverage["entrypoint_ids"],
        "callable_entrypoint_ids": entrypoint_callable_ids,
        "trace_entrypoint_ids": entrypoint_trace_ids,
        "negative_entrypoint_ids": entrypoint_negative_ids,
        "mutation_checks": mutation_checks,
        "writer_classifications": dict(sorted(classifications.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
