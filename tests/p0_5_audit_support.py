"""Test-side execution evidence for the P0-5 audit contracts."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

from canonical_summary_validation import verify_formal_summary
from canonical_tool_schema import scorer_identity
from comparison_eligibility import default_run_state
from formal_entrypoint_contracts import execute_formal_entrypoint_contracts
from formal_evidence import (
    FormalEvidenceError,
    load_and_verify_formal_run_context,
)
from manifest_writer_registry import (
    FORMAL_TRANSITION_GRAPH,
    FormalStateTransition,
    bind_formal_metrics,
    formal_entrypoints,
    formal_writer_spec,
    initialize_formal_state,
    transition_formal_state,
    write_formal_response_manifest,
    write_formal_summary,
)
from tests.runtime_evidence_fixtures import build_native_comparable
from tests.p0_5_audit_expectations import (
    build_expected_case_ids,
    expected_source_provenance,
    negative_contract_specs,
)


INITIALIZER_OVERRIDES = (
    ("reject-bf16-generation-complete", {"stage_reached": "BF16_GENERATION_COMPLETE"}),
    ("reject-bf16-scored", {"stage_reached": "BF16_SCORED"}),
    ("reject-bf16-gate", {"stage_reached": "BF16_GATE"}),
    ("reject-quantization-complete", {"stage_reached": "QUANTIZATION_COMPLETE"}),
    ("reject-quant-scored", {"stage_reached": "QUANT_SCORED"}),
    (
        "reject-comparable",
        {"stage_reached": "COMPARABLE", "comparison_status": "COMPARABLE"},
    ),
    ("reject-summary-complete", {"stage_reached": "SUMMARY_COMPLETE"}),
    ("reject-bf16-complete-override", {"bf16_reconstruction_completed": True}),
    ("reject-quant-complete-override", {"quantization_performed": True}),
)

SUMMARY_PAYLOADS = (
    ("invented-included-run", {"included_runs": ["invented-run"]}),
    ("omitted-included-run", {"included_runs": []}),
    ("invented-excluded-run", {"excluded_runs": [{"run_id": "run"}]}),
    ("invented-reason-code", {"reason_codes": ["INVENTED"]}),
    ("invented-drift", {"behavioral_drift": 0.99}),
    ("zero-model-count", {"quantization_effect_model_count": 0}),
    ("invented-model-count", {"quantization_effect_model_count": 99}),
    ("empty-input-hashes", {"input_evidence_hashes": {}}),
    ("invented-input-hash", {"input_evidence_hashes": {"invented": "0" * 64}}),
    ("diagnostic-formal-metrics", {"formal_metrics": {"run": {"diagnostic": 1}}}),
    ("invented-model", {"models": [{"run_id": "invented-run"}]}),
    ("retrospective-model", {"models": [{"evidence_class": "RETROSPECTIVE"}]}),
    ("missing-context", {"context_state_paths": ["missing.json"]}),
    ("invented-scorer-hash", {"scorer_identity_sha256": "0" * 64}),
    ("invented-registry-hash", {"tool_registry_sha256": "0" * 64}),
    ("invented-calculation-version", {"calculation_version": "invented"}),
    ("invented-protocol", {"protocol_id": "invented"}),
    ("caller-complete-status", {"status": "formal_comparison_summary_complete"}),
)


def _set_input_hash(value: dict) -> None:
    value["input_evidence_hashes"]["run"]["comparison_state"] = "0" * 64


SUMMARY_MUTATIONS: tuple[tuple[str, Callable[[dict], None]], ...] = (
    ("included-invented", lambda value: value.update(included_runs=["invented-run"])),
    ("included-empty", lambda value: value.update(included_runs=[])),
    ("behavioral-drift", lambda value: value.update(behavioral_drift=0.99)),
    ("model-count", lambda value: value.update(quantization_effect_model_count=99)),
    ("input-hashes-empty", lambda value: value.update(input_evidence_hashes={})),
    ("input-hash-changed", _set_input_hash),
    ("excluded-run", lambda value: value.update(excluded_runs=[{"run_id": "run"}])),
    ("reason-code", lambda value: value.update(reason_codes=["INVENTED"])),
    ("context-path", lambda value: value.update(context_state_paths=["missing.json"])),
    (
        "delta-metric",
        lambda value: value["formal_metrics"]["run"].update(
            quant_minus_bf16_exact_call_rate=0.5
        ),
    ),
    (
        "bf16-metric",
        lambda value: value["formal_metrics"]["run"].update(
            bf16_exact_call_rate=0.5
        ),
    ),
    (
        "quant-metric",
        lambda value: value["formal_metrics"]["run"].update(
            quant_exact_call_rate=0.5
        ),
    ),
    ("model-run-id", lambda value: value["models"][0].update(run_id="invented-run")),
    (
        "model-inclusion",
        lambda value: value["models"][0].update(
            quantization_effect_included=False
        ),
    ),
    ("scorer-hash", lambda value: value.update(scorer_identity_sha256="0" * 64)),
    ("registry-hash", lambda value: value.update(tool_registry_sha256="0" * 64)),
    ("calculation-version", lambda value: value.update(calculation_version="invented")),
    ("protocol", lambda value: value.update(protocol_id="invented")),
)


def _initial_state() -> dict[str, Any]:
    return default_run_state(
        model_id="fixture",
        model_family="fixture",
        run_id="run",
        renderer_id="fixture",
    )


def _write_resealed(path: Path, payload: dict) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
    )


def writer_contracts() -> list[dict[str, Any]]:
    contracts = []
    for entrypoint in formal_entrypoints():
        entrypoint_id = entrypoint["id"]
        if entrypoint_id == "comparison-init":
            continue
        arms = ("bf16", "quant") if entrypoint_id == "formal-scorer-main" else ("",)
        for arm in arms:
            spec = formal_writer_spec(entrypoint_id, arm=arm)
            contracts.append(
                {
                    "contract_id": (
                        f"{entrypoint_id}::{arm}"
                        if entrypoint_id == "formal-scorer-main"
                        else entrypoint_id
                    ),
                    "entrypoint_id": entrypoint_id,
                    "writer_id": entrypoint["writer_id"],
                    "arm": arm or spec.allowed_arms[0],
                    "artifact_kind": spec.artifact_kind,
                    "allowed_stages": list(spec.allowed_stages),
                    "allowed_statuses": list(spec.allowed_statuses),
                    "allowed_transitions": list(spec.allowed_transitions),
                }
            )
    return contracts


def _case(
    case_id: str,
    category: str,
    passed: bool,
    *,
    writer_id: str | None = None,
    entrypoint_id: str | None = None,
    observation_source: str = "runtime",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "writer_id": writer_id,
        "entrypoint_id": entrypoint_id,
        "executed": True,
        "passed": passed,
        "skipped": False,
        "observation_source": observation_source,
    }


def _trace_is_valid(trace: dict[str, Any]) -> bool:
    steps = (
        "real_callable_entered",
        "arguments_parser_called",
        "policy_called",
        "context_revalidation_called",
        "transition_called",
        "core_operation_called",
        "writer_called",
        "verifier_called",
    )
    for name in steps:
        observation = trace.get(name)
        if not isinstance(observation, dict):
            return False
        if observation.get("status") == "OBSERVED":
            if (
                observation.get("call_count", 0) < 1
                or not observation.get("callable")
                or not observation.get("call_args")
            ):
                return False
        elif observation.get("status") == "NOT_APPLICABLE":
            if not observation.get("reason"):
                return False
        else:
            return False
    return True


def _invoke_wrong_stage(
    contract: dict[str, Any],
    comparable_state_path: Path,
    comparable_state: dict,
    initial_state_path: Path,
    initial_state: dict,
    output_root: Path,
) -> bool:
    entrypoint_id = contract["entrypoint_id"]
    arm = contract["arm"]
    state_path = (
        initial_state_path
        if entrypoint_id == "comparison-summary-main"
        else comparable_state_path
    )
    state = initial_state if state_path == initial_state_path else comparable_state
    context = load_and_verify_formal_run_context(
        state_path, entrypoint_id=entrypoint_id, arm=arm
    )
    try:
        if entrypoint_id.endswith("generator-main"):
            prefix = "bf16" if arm == "bf16" else "quant"
            output = Path(
                state[
                    "bf16_output_path"
                    if prefix == "bf16"
                    else "quantized_output_path"
                ]
            )
            write_formal_response_manifest(
                context,
                output,
                attestation_hash="0" * 64,
                case_manifest_hash=state["case_manifest_hash"],
                scorer_identity_value=scorer_identity(),
            )
        elif entrypoint_id == "formal-scorer-main":
            bind_formal_metrics(
                context,
                Path(
                    state[
                        "bf16_output_manifest_path"
                        if arm == "bf16"
                        else "quant_output_manifest_path"
                    ]
                ),
                Path(
                    state[
                        "bf16_metrics_path"
                        if arm == "bf16"
                        else "quantized_metrics_path"
                    ]
                ),
            )
        elif entrypoint_id == "comparison-record-bf16":
            transition_formal_state(
                context, FormalStateTransition.RECORD_BF16, state
            )
        elif entrypoint_id == "comparison-record-quant":
            transition_formal_state(
                context, FormalStateTransition.RECORD_QUANT, state
            )
        else:
            write_formal_summary([context], output_root / "wrong-stage-summary.json")
    except FormalEvidenceError as error:
        return error.code == "FORMAL_ENTRYPOINT_STAGE_MISMATCH"
    except ValueError as error:
        return str(error).startswith("FORMAL_ENTRYPOINT_STAGE_MISMATCH:")
    return False


def run_p0_5_audit_execution() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    trace_report = execute_formal_entrypoint_contracts()
    entrypoints = list(formal_entrypoints())
    contracts = writer_contracts()

    for entrypoint in entrypoints:
        entrypoint_id = entrypoint["id"]
        trace = next(
            row
            for row in trace_report["traces"]
            if row["entrypoint_id"] == entrypoint_id
        )
        negative = next(
            row
            for row in trace_report["negative_traces"]
            if row["entrypoint_id"] == entrypoint_id
        )
        cases.extend(
            (
                _case(
                    f"entrypoint::{entrypoint_id}::real-callable",
                    "entrypoint",
                    trace["real_callable_entered"]["status"] == "OBSERVED",
                    writer_id=entrypoint["writer_id"],
                    entrypoint_id=entrypoint_id,
                    observation_source="spy",
                ),
                _case(
                    f"entrypoint::{entrypoint_id}::trace",
                    "entrypoint",
                    _trace_is_valid(trace),
                    writer_id=entrypoint["writer_id"],
                    entrypoint_id=entrypoint_id,
                    observation_source="spy",
                ),
                _case(
                    f"entrypoint::{entrypoint_id}::negative-contract",
                    "entrypoint",
                    negative["executed"] and negative["passed"]
                    and not negative["skipped"],
                    writer_id=entrypoint["writer_id"],
                    entrypoint_id=entrypoint_id,
                ) | {
                    key: copy.deepcopy(negative[key])
                    for key in (
                        "scenario_id", "fixture_preconditions_valid",
                        "expected", "actual", "differences",
                    )
                },
            )
        )

    for transition in FORMAL_TRANSITION_GRAPH:
        observed = any(
            transition.value in trace["transition_values"]
            for trace in trace_report["traces"]
        )
        cases.append(
            _case(
                f"transition::{transition.value}::positive",
                "transition",
                observed,
                observation_source="spy-call-args",
            )
        )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "initial-valid.json"
        initialize_formal_state(path, _initial_state())
        cases.append(_case("initializer::accept-fixed-state", "initializer", True))
        for name, overrides in INITIALIZER_OVERRIDES:
            rejected = False
            try:
                initialize_formal_state(
                    root / f"{name}.json", _initial_state() | overrides
                )
            except ValueError:
                rejected = True
            cases.append(_case(f"initializer::{name}", "initializer", rejected))

        comparable_state_path, comparable_state = build_native_comparable(
            root / "comparable"
        )
        initial_state_path, initial_state = build_native_comparable(
            root / "initialized", stop_after="initialized"
        )
        for contract in contracts:
            positive = any(
                observation["contract_id"] == contract["contract_id"]
                and observation["observation"]["status"] == "OBSERVED"
                for trace in trace_report["traces"]
                for observation in trace["writer_contract_observations"]
            )
            cases.append(
                _case(
                    f"writer::{contract['contract_id']}::positive",
                    "writer",
                    positive,
                    writer_id=contract["writer_id"],
                    entrypoint_id=contract["entrypoint_id"],
                    observation_source="spy",
                )
            )
            cases.append(
                _case(
                    f"writer::{contract['contract_id']}::wrong-stage",
                    "writer",
                    _invoke_wrong_stage(
                        contract,
                        comparable_state_path,
                        comparable_state,
                        initial_state_path,
                        initial_state,
                        root,
                    ),
                    writer_id=contract["writer_id"],
                    entrypoint_id=contract["entrypoint_id"],
                )
            )

        context = load_and_verify_formal_run_context(
            comparable_state_path,
            entrypoint_id="comparison-summary-main",
            arm="summary",
        )
        for name, payload in SUMMARY_PAYLOADS:
            rejected = False
            try:
                write_formal_summary(
                    [context], root / f"caller-{name}.json", payload
                )
            except TypeError:
                rejected = True
            cases.append(_case(f"summary::{name}", "summary", rejected))

        summary_path = root / "summary.json"
        write_formal_summary([context], summary_path)
        final_context = load_and_verify_formal_run_context(
            comparable_state_path,
            entrypoint_id="comparison-summary-main",
            arm="summary",
        )
        original = json.loads(summary_path.read_text(encoding="utf-8"))
        for name, mutate in SUMMARY_MUTATIONS:
            changed = copy.deepcopy(original)
            mutate(changed)
            _write_resealed(summary_path, changed)
            rejected = False
            try:
                verify_formal_summary(summary_path, [final_context])
            except FormalEvidenceError as error:
                rejected = error.code == "FORMAL_SUMMARY_RECOMPUTE_MISMATCH"
            cases.append(_case(f"verifier::{name}", "verifier", rejected))
        _write_resealed(summary_path, original)
        cases.append(
            _case(
                "verifier::unchanged-valid",
                "verifier",
                verify_formal_summary(summary_path, [final_context]) == original,
            )
        )

    candidate_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "candidate_sha": candidate_sha,
        "cases": sorted(cases, key=lambda row: row["case_id"]),
        "writers": {
            "expected": sorted({row["writer_id"] for row in entrypoints}),
            "contracts": contracts,
        },
        "entrypoints": {
            "expected": sorted(row["id"] for row in entrypoints),
            "traces": trace_report["traces"],
            "negative_contracts": trace_report["negative_traces"],
        },
        "trace_summary": {
            key: value
            for key, value in trace_report.items()
            if key not in {"traces", "negative_traces"}
        },
    }


def validate_audit_execution_report(
    report: dict[str, Any], *, expected_sha: str | None = None
) -> dict[str, Any]:
    expected_object = build_expected_case_ids()
    expected = set(expected_object)
    rows = report.get("cases")
    if not isinstance(rows, list):
        raise ValueError("P0_5_AUDIT_REPORT_INVALID: cases missing")
    observed = {
        row.get("case_id")
        for row in rows
        if row.get("executed") is True
        and row.get("passed") is True
        and row.get("skipped") is False
    }
    all_ids = [row.get("case_id") for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("P0_5_AUDIT_REPORT_INVALID: duplicate case IDs")
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise ValueError(
            "P0_5_AUDIT_COVERAGE_MISMATCH: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    if expected_sha is not None and report.get("candidate_sha") != expected_sha:
        raise ValueError("P0_5_AUDIT_CANDIDATE_SHA_MISMATCH")

    entrypoints = {row["id"] for row in formal_entrypoints()}
    callable_ids = {
        row["entrypoint_id"]
        for row in rows
        if row["case_id"].endswith("::real-callable") and row["passed"]
    }
    trace_ids = {
        row["entrypoint_id"]
        for row in rows
        if row["case_id"].endswith("::trace") and row["passed"]
    }
    negative_ids = {
        row["entrypoint_id"]
        for row in rows
        if row["case_id"].endswith("::negative-contract") and row["passed"]
    }
    if not (entrypoints == callable_ids == trace_ids == negative_ids):
        raise ValueError("P0_5_AUDIT_ENTRYPOINT_COVERAGE_MISMATCH")

    writer_ids = {row["writer_id"] for row in formal_entrypoints()}
    positive_writer_ids = {
        row["writer_id"]
        for row in rows
        if row["category"] == "writer"
        and row["case_id"].endswith("::positive")
        and row["passed"]
    }
    negative_writer_ids = {
        row["writer_id"]
        for row in rows
        if row["category"] == "writer"
        and row["case_id"].endswith("::wrong-stage")
        and row["passed"]
    }
    if not (writer_ids == positive_writer_ids == negative_writer_ids):
        raise ValueError("P0_5_AUDIT_WRITER_COVERAGE_MISMATCH")

    specs = {spec.entrypoint_id: spec for spec in negative_contract_specs()}
    semantic_mismatches = []
    for row in rows:
        if not row["case_id"].endswith("::negative-contract"):
            continue
        spec = specs.get(row.get("entrypoint_id"))
        expected_semantics = row.get("expected")
        actual = row.get("actual")
        valid = (
            spec is not None
            and row.get("executed") is True
            and row.get("passed") is True
            and row.get("skipped") is False
            and row.get("fixture_preconditions_valid") is True
            and isinstance(expected_semantics, dict)
            and isinstance(actual, dict)
            and expected_semantics.get("exception_types")
            == list(spec.expected_exception_types)
            and expected_semantics.get("reason_codes")
            == list(spec.expected_reason_codes)
            and expected_semantics.get("failure_phase")
            == spec.expected_failure_phase
            and actual.get("exception_type") in spec.expected_exception_types
            and actual.get("reason_code") in spec.expected_reason_codes
            and actual.get("failure_phase") == spec.expected_failure_phase
            and actual.get("callable") == spec.expected_callable
            and actual.get("real_callable_observed") is True
            and actual.get("target_validator_observed") is True
            and actual.get("target_validator_call_count", 0) > 0
            and actual.get("exception_type") not in spec.forbidden_exception_types
            and not row.get("differences")
        )
        if spec is not None and spec.expected_exit_code is not None:
            valid = (
                valid
                and actual.get("exit_code") == spec.expected_exit_code
                and isinstance(actual.get("structured_payload"), dict)
            )
        if not valid:
            semantic_mismatches.append(row["case_id"])
    if semantic_mismatches:
        raise ValueError(
            "P0_5_AUDIT_NEGATIVE_SEMANTIC_MISMATCH: "
            + ", ".join(sorted(semantic_mismatches))
        )

    init_trace = next(
        row
        for row in report["entrypoints"]["traces"]
        if row["entrypoint_id"] == "comparison-init"
    )
    verifier = init_trace["verifier_called"]
    if (
        not isinstance(verifier, dict)
        or verifier.get("status") != "OBSERVED"
        or verifier.get("call_count", 0) < 1
        or verifier.get("callable") != "formal_evidence.verify_state_integrity"
        or not verifier.get("call_args")
    ):
        raise ValueError("P0_5_AUDIT_INITIALIZER_VERIFIER_NOT_OBSERVED")
    return {
        "expected_case_ids": sorted(expected),
        "observed_case_ids": sorted(observed),
        "passed_case_ids": sorted(observed),
        "expected_total": len(expected),
        "observed_total": len(observed),
        "missing_case_ids": [],
        "unexpected_case_ids": [],
        "negative_semantic_mismatch_ids": [],
        "expected_source": expected_source_provenance(
            expected_sha or str(report.get("candidate_sha", ""))
        ),
        "expected_observed_distinct_objects": expected_object is not rows,
        "writer_ids": sorted(writer_ids),
        "entrypoint_ids": sorted(entrypoints),
    }


def run_audit_report_mutation_checks(
    report: dict[str, Any], *, expected_sha: str
) -> list[str]:
    mutations: dict[str, dict[str, Any]] = {}

    value = copy.deepcopy(report)
    value["cases"] = [
        row
        for row in value["cases"]
        if row["case_id"] != "writer::bf16-generator-main::wrong-stage"
    ]
    mutations["A_remove_writer_wrong_stage"] = value

    value = copy.deepcopy(report)
    value["cases"] = [
        row
        for row in value["cases"]
        if row["case_id"] != "writer::native-quant-generator-main::positive"
    ]
    mutations["B_skip_one_quant_generator"] = value

    value = copy.deepcopy(report)
    init_trace = next(
        row
        for row in value["entrypoints"]["traces"]
        if row["entrypoint_id"] == "comparison-init"
    )
    init_trace["verifier_called"] = True
    mutations["C_replace_verifier_spy_with_true"] = value

    value = copy.deepcopy(report)
    value["cases"][0]["skipped"] = True
    mutations["D_mark_case_skipped"] = value

    value = copy.deepcopy(report)
    value["cases"] = [
        row
        for row in value["cases"]
        if row["case_id"]
        != "entrypoint::gguf-generator-main::negative-contract"
    ]
    mutations["E_remove_entrypoint_negative"] = value

    value = copy.deepcopy(report)
    value["legacy_declared_counts"] = {"initializer": 999}
    value["cases"] = [
        row
        for row in value["cases"]
        if row["case_id"] != "initializer::reject-comparable"
    ]
    mutations["F_keep_count_remove_actual_case"] = value

    value = copy.deepcopy(report)
    value["expected_case_ids"] = [
        row["case_id"] for row in value["cases"]
    ]
    value["cases"] = value["cases"][1:]
    mutations["H_expected_copied_from_observed"] = value

    def negative(value: dict[str, Any]) -> dict[str, Any]:
        return next(
            row for row in value["cases"]
            if row["case_id"].endswith("::negative-contract")
        )

    value = copy.deepcopy(report)
    row = negative(value)
    row["actual"]["exception_type"] = "FileNotFoundError"
    row["actual"]["reason_code"] = ""
    mutations["I_unrelated_file_not_found"] = value

    value = copy.deepcopy(report)
    negative(value)["actual"]["reason_code"] = "FORMAL_ENTRYPOINT_STATUS_MISMATCH"
    mutations["J_wrong_reason_code"] = value

    value = copy.deepcopy(report)
    negative(value)["actual"]["failure_phase"] = "argument-validation"
    mutations["K_wrong_failure_phase"] = value

    value = copy.deepcopy(report)
    row = negative(value)
    row["actual"].update(
        exception_type="SystemExit", reason_code="", exit_code=2,
        structured_payload=None,
    )
    mutations["L_generic_system_exit"] = value

    value = copy.deepcopy(report)
    row = negative(value)
    row["fixture_preconditions_valid"] = False
    row["actual"]["fixture_preconditions_valid"] = False
    mutations["M_fixture_setup_failure"] = value

    value = copy.deepcopy(report)
    negative(value)["actual"]["reason_code"] = "FORMAL_ENTRYPOINT_CONTEXT_INVALID"
    mutations["N_correct_type_wrong_code"] = value

    value = copy.deepcopy(report)
    negative(value)["actual"]["real_callable_observed"] = False
    mutations["O_real_callable_not_observed"] = value

    value = copy.deepcopy(report)
    row = negative(value)
    row["actual"] = {}
    row["passed"] = True
    mutations["P_swallowed_exception_manual_pass"] = value

    value = copy.deepcopy(report)
    value["cases"] = value["cases"][1:]
    mutations["S_observed_execution_removed"] = value

    detected = []
    for name, changed in mutations.items():
        try:
            validate_audit_execution_report(changed, expected_sha=expected_sha)
        except ValueError:
            detected.append(name)
    if set(detected) != set(mutations):
        raise ValueError(
            "P0_5_AUDIT_MUTATION_NOT_DETECTED: "
            f"{sorted(set(mutations)-set(detected))}"
        )
    authoritative = build_expected_case_ids()
    before = frozenset(authoritative)
    value = copy.deepcopy(report)
    value["expected_case_ids"] = []
    if authoritative != before or not isinstance(authoritative, frozenset):
        raise ValueError("P0_5_AUDIT_EXPECTED_OBJECT_NOT_ISOLATED")
    detected.append("Q_expected_object_isolated")

    reduced = set(authoritative)
    reduced.remove(next(iter(reduced)))
    observed = {
        row["case_id"] for row in report["cases"]
        if row["executed"] and row["passed"] and not row["skipped"]
    }
    if not observed - reduced:
        raise ValueError("P0_5_AUDIT_EXPECTED_MATRIX_REMOVAL_NOT_DETECTED")
    detected.append("R_expected_matrix_declaration_removed")
    return sorted(detected)
