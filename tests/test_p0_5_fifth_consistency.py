from __future__ import annotations

import copy
import argparse
import contextlib
import io
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

from canonical_summary_validation import verify_formal_summary
from canonical_tool_schema import scorer_identity
from comparison_eligibility import (
    ComparisonStatus,
    PROTOCOL_ID,
    Stage,
    default_run_state,
    determine_comparison_eligibility,
    validate_comparison_state_schema,
)
from formal_evidence import (
    FormalEvidenceError,
    load_and_verify_formal_run_context,
    verify_state_integrity,
)
from manifest_writer_registry import (
    FORMAL_TRANSITION_GRAPH,
    FormalStateTransition,
    initialize_formal_state,
    transition_formal_state,
    write_formal_response_manifest,
    write_formal_summary,
)
from tests.runtime_evidence_fixtures import build_native_comparable


def initial_state(output: Path | None = None) -> dict:
    return default_run_state(
        model_id="fixture",
        model_family="fixture",
        run_id="run",
        renderer_id="fixture",
        bf16_output_path=str(output.resolve()) if output else "",
        bf16_output_manifest_path=(
            str(output.with_suffix(output.suffix + ".manifest.json").resolve())
            if output
            else ""
        ),
    )


def write_resealed(path: Path, payload: dict) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
    )


def make_baseline_failure(root: Path) -> tuple[Path, dict]:
    state_path, _ = build_native_comparable(root, stop_after="bf16_scored")
    baseline = root / "baseline-failed.json"
    gate = root / "gate.json"
    baseline.write_text('{"pass": false}\n', encoding="utf-8")
    gate.write_text('{"pass": true}\n', encoding="utf-8")
    from run_cross_model_comparison import record_bf16

    with contextlib.redirect_stdout(io.StringIO()):
        record_bf16(
            argparse.Namespace(
                state=state_path,
                protocol=ROOT / "config" / "agent_toolcall_protocol_v4.json",
                baseline_decision=baseline,
                gate_decision=gate,
            )
        )
    return state_path, verify_state_integrity(state_path)


class InitializerFixedStateTests(unittest.TestCase):
    INITIALIZER_CASES = 6
    def test_five_late_or_failed_initial_states_are_rejected(self):
        cases = (
            ("COMPARABLE", "COMPARABLE"),
            ("BF16_GATE", "ELIGIBLE_NOT_QUANTIZED"),
            ("QUANTIZATION", "QUANTIZATION_FAILED"),
            ("BASELINE", "NOT_ELIGIBLE_BASELINE_FAILED"),
            ("SUMMARY_COMPLETE", "COMPARABLE"),
        )
        for stage, status in cases:
            with self.subTest(stage=stage, status=status), tempfile.TemporaryDirectory() as td:
                state = initial_state()
                state.update(stage_reached=stage, comparison_status=status)
                with self.assertRaisesRegex(
                    ValueError, "FORMAL_STATE_INITIAL_(STAGE|STATUS)_INVALID"
                ):
                    initialize_formal_state(Path(td) / "state.json", state)

    def test_legal_initializer_forces_unique_initial_state(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            initialize_formal_state(path, initial_state())
            state = verify_state_integrity(path)
            self.assertEqual(state["stage_reached"], Stage.INITIALIZED)
            self.assertEqual(
                state["comparison_status"],
                ComparisonStatus.NOT_ELIGIBLE_BASELINE_FAILED,
            )
            self.assertFalse(state["quantization_performed"])

    def test_schema_rejects_initialization_creation_with_late_state(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            initialize_formal_state(path, initial_state())
            state = verify_state_integrity(path)
            state.update(
                stage_reached="COMPARABLE",
                comparison_status="COMPARABLE",
                native_protocol_comparable=True,
            )
            with self.assertRaisesRegex(
                ValueError, "FORMAL_STATE_INITIALIZATION_OVERRIDE_FORBIDDEN"
            ):
                validate_comparison_state_schema(state)


class TransitionAndWriterStageTests(unittest.TestCase):
    TRANSITION_CASES = 7
    WRITER_STAGE_CASES = 6
    def test_machine_transition_graph_covers_success_and_terminal_failures(self):
        self.assertEqual(set(FORMAL_TRANSITION_GRAPH), set(FormalStateTransition))
        self.assertEqual(
            FORMAL_TRANSITION_GRAPH[
                FormalStateTransition.RECORD_BF16
            ]["target_stages"],
            ("BASELINE", "RECONSTRUCTION", "BF16_GATE"),
        )
        self.assertEqual(
            FORMAL_TRANSITION_GRAPH[
                FormalStateTransition.RECORD_QUANT
            ]["target_stages"],
            ("QUANTIZATION", "QUANTIZED_EVALUATION", "COMPARABLE"),
        )
    def test_wrong_real_stage_rejects_bf16_and_summary_writers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path, state = make_baseline_failure(root)
            output = root / "bf16.jsonl"
            output.write_text("{}\n", encoding="utf-8")
            bf16 = load_and_verify_formal_run_context(
                state_path, entrypoint_id="bf16-generator-main", arm="bf16"
            )
            with self.assertRaises(FormalEvidenceError) as caught:
                write_formal_response_manifest(
                    bf16,
                    output,
                    attestation_hash="a" * 64,
                    case_manifest_hash="b" * 64,
                    scorer_identity_value=scorer_identity(),
                )
            self.assertEqual(caught.exception.code, "FORMAL_ENTRYPOINT_STAGE_MISMATCH")
            summary = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-summary-main", arm="summary"
            )
            with self.assertRaises(FormalEvidenceError) as caught:
                write_formal_summary([summary], root / "summary.json")
            self.assertIn(
                caught.exception.code,
                {"FORMAL_ENTRYPOINT_STAGE_MISMATCH", "FORMAL_ENTRYPOINT_STATUS_MISMATCH"},
            )

    def test_bf16_writer_accepts_only_initial_stage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path, state = build_native_comparable(
                root, stop_after="initialized"
            )
            output = Path(state["bf16_output_path"])
            context = load_and_verify_formal_run_context(
                state_path, entrypoint_id="bf16-generator-main", arm="bf16"
            )
            manifest, _ = write_formal_response_manifest(
                context,
                output,
                attestation_hash=hashlib.sha256(
                    Path(state["bf16_model_state_attestation_path"]).read_bytes()
                ).hexdigest(),
                case_manifest_hash=state["case_manifest_hash"],
                scorer_identity_value=scorer_identity(),
            )
            self.assertTrue(manifest.is_file())
            self.assertEqual(
                verify_state_integrity(state_path)["stage_reached"],
                "BF16_GENERATION_COMPLETE",
            )

    def test_skips_repeats_and_wrong_arm_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path = root / "state.json"
            initialize_formal_state(state_path, initial_state())
            initial = verify_state_integrity(state_path)
            direct = dict(initial)
            direct.update(
                formal_creation=None,
                stage_reached="COMPARABLE",
                comparison_status="COMPARABLE",
                native_protocol_comparable=True,
            )
            bf16 = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-record-bf16", arm="bf16"
            )
            with self.assertRaises((FormalEvidenceError, ValueError)):
                transition_formal_state(
                    bf16, FormalStateTransition.RECORD_BF16, direct
                )
            quant = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-record-quant", arm="quant"
            )
            with self.assertRaises(FormalEvidenceError) as caught:
                transition_formal_state(
                    quant, FormalStateTransition.RECORD_QUANT, initial
                )
            self.assertEqual(caught.exception.code, "FORMAL_ENTRYPOINT_STAGE_MISMATCH")

    def test_full_production_chain_is_the_only_comparable_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            state_path, state = build_native_comparable(Path(td))
            self.assertEqual(state["stage_reached"], "COMPARABLE")
            self.assertEqual(state["comparison_status"], "COMPARABLE")
            repeat = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-record-quant", arm="quant"
            )
            with self.assertRaises(FormalEvidenceError):
                transition_formal_state(
                    repeat, FormalStateTransition.RECORD_QUANT, state
                )


class FormalSummaryRecomputeTests(unittest.TestCase):
    SUMMARY_RECOMPUTE_CASES = 18
    SUMMARY_VERIFIER_CASES = 19
    def test_writer_rejects_caller_authoritative_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path, _ = build_native_comparable(root)
            context = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-summary-main", arm="summary"
            )
            invented_payloads = (
                {"included_runs": ["invented-run"]},
                {"included_runs": []},
                {"excluded_runs": [{"run_id": "run"}]},
                {"reason_codes": ["INVENTED"]},
                {"behavioral_drift": 0.99},
                {"quantization_effect_model_count": 0},
                {"quantization_effect_model_count": 99},
                {"input_evidence_hashes": {}},
                {"input_evidence_hashes": {"invented": "0" * 64}},
                {"formal_metrics": {"run": {"diagnostic": 1}}},
                {"models": [{"run_id": "invented-run"}]},
                {"models": [{"evidence_class": "RETROSPECTIVE"}]},
                {"context_state_paths": ["missing.json"]},
                {"scorer_identity_sha256": "0" * 64},
                {"tool_registry_sha256": "0" * 64},
                {"calculation_version": "invented"},
                {"protocol_id": "invented"},
                {"status": "formal_comparison_summary_complete"},
            )
            self.assertEqual(
                len(invented_payloads), self.SUMMARY_RECOMPUTE_CASES
            )
            for payload in invented_payloads:
                with self.subTest(payload=payload), self.assertRaises(TypeError):
                    write_formal_summary(
                        [context], root / "summary.json", payload
                    )

    def test_summary_is_recomputed_and_advances_only_after_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path, _ = build_native_comparable(root)
            context = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-summary-main", arm="summary"
            )
            summary_path = root / "summary.json"
            summary = write_formal_summary([context], summary_path)
            self.assertEqual(summary["included_runs"], ["run"])
            self.assertEqual(summary["quantization_effect_model_count"], 1)
            self.assertEqual(summary["behavioral_drift"], 0.0)
            self.assertEqual(
                verify_state_integrity(state_path)["stage_reached"],
                "SUMMARY_COMPLETE",
            )
            final_context = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-summary-main", arm="summary"
            )
            self.assertEqual(
                verify_formal_summary(summary_path, [final_context]), summary
            )

    def test_eighteen_post_write_semantic_mutations_fail_recompute(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_path, _ = build_native_comparable(root)
            context = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-summary-main", arm="summary"
            )
            summary_path = root / "summary.json"
            write_formal_summary([context], summary_path)
            final_context = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-summary-main", arm="summary"
            )
            original = json.loads(summary_path.read_text(encoding="utf-8"))
            mutations = (
                lambda value: value.update(included_runs=["invented-run"]),
                lambda value: value.update(included_runs=[]),
                lambda value: value.update(behavioral_drift=0.99),
                lambda value: value.update(quantization_effect_model_count=99),
                lambda value: value.update(input_evidence_hashes={}),
                lambda value: value["input_evidence_hashes"]["run"].update(
                    comparison_state="0" * 64
                ),
                lambda value: value.update(excluded_runs=[{"run_id": "run"}]),
                lambda value: value.update(reason_codes=["INVENTED"]),
                lambda value: value.update(context_state_paths=["missing.json"]),
                lambda value: value["formal_metrics"]["run"].update(
                    quant_minus_bf16_exact_call_rate=0.5
                ),
                lambda value: value["formal_metrics"]["run"].update(
                    bf16_exact_call_rate=0.5
                ),
                lambda value: value["formal_metrics"]["run"].update(
                    quant_exact_call_rate=0.5
                ),
                lambda value: value["models"][0].update(run_id="invented-run"),
                lambda value: value["models"][0].update(
                    quantization_effect_included=False
                ),
                lambda value: value.update(scorer_identity_sha256="0" * 64),
                lambda value: value.update(tool_registry_sha256="0" * 64),
                lambda value: value.update(calculation_version="invented"),
                lambda value: value.update(protocol_id="invented"),
            )
            self.assertEqual(len(mutations), self.SUMMARY_RECOMPUTE_CASES)
            for number, mutate in enumerate(mutations):
                with self.subTest(mutation=number):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    write_resealed(summary_path, changed)
                    with self.assertRaises(FormalEvidenceError) as caught:
                        verify_formal_summary(summary_path, [final_context])
                    self.assertEqual(
                        caught.exception.code, "FORMAL_SUMMARY_RECOMPUTE_MISMATCH"
                    )
            write_resealed(summary_path, original)
            self.assertEqual(
                verify_formal_summary(summary_path, [final_context]), original
            )


if __name__ == "__main__":
    unittest.main()
