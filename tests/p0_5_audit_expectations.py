"""Independent declarations for P0-5 audit coverage and negative contracts."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

from manifest_writer_registry import (
    FORMAL_TRANSITION_GRAPH,
    formal_entrypoints,
    formal_writer_spec,
)


@dataclass(frozen=True)
class NegativeEntrypointContract:
    entrypoint_id: str
    scenario_id: str
    expected_exception_types: tuple[str, ...]
    expected_reason_codes: tuple[str, ...]
    expected_failure_phase: str
    expected_callable: str
    target_validator: str
    forbidden_exception_types: tuple[str, ...] = (
        "FileNotFoundError",
        "FileExistsError",
        "PermissionError",
        "TypeError",
        "KeyError",
        "AttributeError",
        "ImportError",
        "OSError",
        "AssertionError",
    )
    expected_exit_code: int | None = None


NEGATIVE_ENTRYPOINT_CONTRACTS = (
    NegativeEntrypointContract(
        "bf16-generator-main", "bf16-stage-already-advanced",
        ("FormalEvidenceError",), ("FORMAL_ENTRYPOINT_STAGE_MISMATCH",),
        "stage-validation", "generate_bf16_responses.main",
        "formal_evidence.revalidate_formal_run_context",
    ),
    NegativeEntrypointContract(
        "transformers-quant-generator-main", "bf16-gate-not-eligible",
        ("SystemExit",), ("NOT_ELIGIBLE_BF16_GATE_FAILED",),
        "quantization-authorization", "generate_quantized_responses.main",
        "generate_quantized_responses.quantization_authorization", expected_exit_code=20,
    ),
    NegativeEntrypointContract(
        "native-quant-generator-main", "bf16-gate-not-eligible",
        ("SystemExit",), ("NOT_ELIGIBLE_BF16_GATE_FAILED",),
        "quantization-authorization",
        "generate_native_quantized_responses.main",
        "generate_native_quantized_responses.quantization_authorization",
        expected_exit_code=20,
    ),
    NegativeEntrypointContract(
        "gguf-generator-main", "bf16-gate-not-eligible",
        ("SystemExit",), ("NOT_ELIGIBLE_BF16_GATE_FAILED",),
        "quantization-authorization", "generate_gguf_responses.main",
        "generate_gguf_responses.quantization_authorization", expected_exit_code=20,
    ),
    NegativeEntrypointContract(
        "comparison-init", "initial-stage-override",
        ("ValueError",), ("FORMAL_STATE_INITIAL_STAGE_INVALID",),
        "writer-validation", "run_cross_model_comparison.init_run",
        "run_cross_model_comparison.initialize_formal_state",
    ),
    NegativeEntrypointContract(
        "comparison-record-bf16", "bf16-not-scored",
        ("FormalEvidenceError",), ("FORMAL_ENTRYPOINT_STAGE_MISMATCH",),
        "stage-validation", "run_cross_model_comparison.record_bf16",
        "formal_evidence.revalidate_formal_run_context",
    ),
    NegativeEntrypointContract(
        "comparison-record-quant", "quant-not-scored",
        ("FormalEvidenceError",), ("FORMAL_ENTRYPOINT_STAGE_MISMATCH",),
        "stage-validation", "run_cross_model_comparison.record_quantized",
        "formal_evidence.revalidate_formal_run_context",
    ),
    NegativeEntrypointContract(
        "formal-scorer-main", "response-arm-mismatch",
        ("SystemExit",), ("FORMAL_ENTRYPOINT_ARM_MISMATCH",),
        "arm-validation", "score_responses.main", "score_responses.verify_state_integrity",
    ),
    NegativeEntrypointContract(
        "comparison-summary-main", "non-comparable-stage",
        ("FormalEvidenceError",), ("FORMAL_ENTRYPOINT_STAGE_MISMATCH",),
        "stage-validation", "summarize_cross_model_comparison.main",
        "formal_evidence.revalidate_formal_run_context",
    ),
)

INITIALIZER_CASE_NAMES = (
    "accept-fixed-state", "reject-bf16-generation-complete", "reject-bf16-scored",
    "reject-bf16-gate", "reject-quantization-complete", "reject-quant-scored",
    "reject-comparable", "reject-summary-complete", "reject-bf16-complete-override",
    "reject-quant-complete-override",
)
SUMMARY_PAYLOAD_CASE_NAMES = (
    "invented-included-run", "omitted-included-run", "invented-excluded-run",
    "invented-reason-code", "invented-drift", "zero-model-count",
    "invented-model-count", "empty-input-hashes", "invented-input-hash",
    "diagnostic-formal-metrics", "invented-model", "retrospective-model",
    "missing-context", "invented-scorer-hash", "invented-registry-hash",
    "invented-calculation-version", "invented-protocol", "caller-complete-status",
)
SUMMARY_MUTATION_CASE_NAMES = (
    "included-invented", "included-empty", "behavioral-drift", "model-count",
    "input-hashes-empty", "input-hash-changed", "excluded-run", "reason-code",
    "context-path", "delta-metric", "bf16-metric", "quant-metric", "model-run-id",
    "model-inclusion", "scorer-hash", "registry-hash", "calculation-version",
    "protocol",
)


def negative_contract_specs() -> tuple[NegativeEntrypointContract, ...]:
    return NEGATIVE_ENTRYPOINT_CONTRACTS


def build_expected_case_ids() -> frozenset[str]:
    """Build authoritative expectations without accepting execution data."""
    entrypoints = formal_entrypoints()
    contracts = []
    for entrypoint in entrypoints:
        if entrypoint["id"] == "comparison-init":
            continue
        arms = ("bf16", "quant") if entrypoint["id"] == "formal-scorer-main" else ("",)
        for arm in arms:
            formal_writer_spec(entrypoint["id"], arm=arm)
            contracts.append(
                f"{entrypoint['id']}::{arm}"
                if entrypoint["id"] == "formal-scorer-main"
                else entrypoint["id"]
            )
    return frozenset({
        *(f"initializer::{name}" for name in INITIALIZER_CASE_NAMES),
        *(f"transition::{transition.value}::positive" for transition in FORMAL_TRANSITION_GRAPH),
        *(f"writer::{contract}::{kind}" for contract in contracts for kind in ("positive", "wrong-stage")),
        *(f"summary::{name}" for name in SUMMARY_PAYLOAD_CASE_NAMES),
        *(f"verifier::{name}" for name in SUMMARY_MUTATION_CASE_NAMES),
        "verifier::unchanged-valid",
        *(f"entrypoint::{entrypoint['id']}::{kind}" for entrypoint in entrypoints
          for kind in ("real-callable", "trace", "negative-contract")),
    })


def expected_source_provenance(candidate_sha: str) -> dict[str, str]:
    registry = ROOT / "scripts" / "manifest_writer_registry.py"
    matrix = json.dumps(
        [asdict(spec) for spec in NEGATIVE_ENTRYPOINT_CONTRACTS],
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return {
        "builder": "tests.p0_5_audit_expectations.build_expected_case_ids",
        "builder_signature": str(inspect.signature(build_expected_case_ids)),
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(matrix).hexdigest(),
        "candidate_sha": candidate_sha,
    }
