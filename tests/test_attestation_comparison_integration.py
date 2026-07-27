from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from comparison_eligibility import (  # noqa: E402
    ComparisonStatus,
    PROTOCOL_ID,
    default_run_state,
    determine_comparison_eligibility,
)


def eligible_state(**overrides):
    state = default_run_state(
        model_id="fixture",
        model_family="qwen2",
        run_id="run",
        source_checkpoint="/checkpoint",
        source_checkpoint_manifest="/checkpoint/manifest.sha256.json",
        source_checkpoint_manifest_hash="a" * 64,
        source_run_id="source",
        training_stage="reconstruction",
        config_hash="b" * 64,
        tokenizer_hash="c" * 64,
        case_manifest="/cases.json",
        case_manifest_hash="d" * 64,
        logical_cases_hash="e" * 64,
        renderer_id="fixture",
        baseline_completed=True,
        baseline_capability_passed=True,
        bf16_reconstruction_completed=True,
        bf16_gate_passed=True,
        bf16_output_path="/bf16.jsonl",
        bf16_metrics_path="/bf16.json",
        bf16_model_state_attestation_path="/bf16.attestation.json",
        bf16_model_state_attestation_hash="1" * 64,
        bf16_attestation_status="ATTESTED_BF16",
        bf16_attestation_passed=True,
        bf16_output_manifest_path="/bf16.manifest.json",
        bf16_output_manifest_hash="2" * 64,
        quantized_output_path="/int8.jsonl",
        quantized_metrics_path="/int8.json",
        quant_model_state_attestation_path="/int8.attestation.json",
        quant_model_state_attestation_hash="3" * 64,
        quant_attestation_status="ATTESTED_BNB_INT8",
        quant_attestation_passed=True,
        quant_output_manifest_path="/int8.manifest.json",
        quant_output_manifest_hash="4" * 64,
        bf16_source_checkpoint_hash="a" * 64,
        bf16_source_checkpoint="/checkpoint",
        bf16_source_checkpoint_manifest="/checkpoint/manifest.sha256.json",
        bf16_config_hash="b" * 64,
        bf16_tokenizer_hash="c" * 64,
        bf16_training_stage="reconstruction",
        bf16_source_run_id="source",
        quant_source_checkpoint_hash="a" * 64,
        quant_source_checkpoint="/checkpoint",
        quant_source_checkpoint_manifest="/checkpoint/manifest.sha256.json",
        quant_config_hash="b" * 64,
        quant_tokenizer_hash="c" * 64,
        quant_training_stage="reconstruction",
        quant_source_run_id="source",
        bf16_case_manifest_hash="d" * 64,
        quant_case_manifest_hash="d" * 64,
    )
    state.update(overrides)
    return state


def assess(state):
    return determine_comparison_eligibility(
        state,
        {"pass": True},
        {"protocol_id": PROTOCOL_ID},
        verify_files=False,
    )


class AttestationComparisonIntegrationTests(unittest.TestCase):
    def test_int8_request_without_detected_quantization_is_not_comparable(self):
        from tests.runtime_evidence_fixtures import build_native_comparable

        with tempfile.TemporaryDirectory() as tmp:
            _, state = build_native_comparable(Path(tmp))
            state.update(
                quantization_requested=True,
                quantization_performed=False,
                quantized_evaluation_completed=False,
                quant_attestation_passed=False,
                quant_attestation_status="QUANTIZATION_NOT_DETECTED",
            )
            result = assess(state)
            self.assertEqual(
                result["comparison_status"], ComparisonStatus.QUANTIZATION_FAILED
            )
            self.assertIn("QUANTIZATION_NOT_DETECTED", result["blocking_reason"])

    def test_gptq_fallback_is_not_comparable(self):
        from tests.runtime_evidence_fixtures import build_native_comparable

        with tempfile.TemporaryDirectory() as tmp:
            _, state = build_native_comparable(Path(tmp))
            state.update(
                quantization_requested=True,
                quantization_performed=False,
                quantized_evaluation_completed=False,
                quant_attestation_passed=False,
                quant_attestation_status="LOADER_FALLBACK_USED",
            )
            result = assess(state)
            self.assertEqual(
                result["comparison_status"], ComparisonStatus.QUANTIZATION_FAILED
            )
            self.assertIn("LOADER_FALLBACK_USED", result["blocking_reason"])

    def test_matching_lineage_does_not_override_failed_attestation(self):
        from tests.runtime_evidence_fixtures import build_native_comparable

        with tempfile.TemporaryDirectory() as tmp:
            _, state = build_native_comparable(Path(tmp))
            state.update(
                quantization_requested=True,
                quantization_performed=True,
                quantized_evaluation_completed=True,
                quant_attestation_passed=False,
                quant_attestation_status="QUANT_CONFIG_MISMATCH",
            )
            result = assess(state)
            self.assertEqual(
                result["comparison_status"], ComparisonStatus.QUANTIZATION_FAILED
            )

    def test_failed_bf16_attestation_blocks_quantization_preflight(self):
        result = assess(
            eligible_state(
                bf16_attestation_passed=False,
                bf16_attestation_status="BF16_DTYPE_COVERAGE_BELOW_THRESHOLD",
            )
        )
        self.assertNotEqual(result["comparison_status"], ComparisonStatus.ELIGIBLE_NOT_QUANTIZED)
        self.assertIn("BF16_DTYPE_COVERAGE_BELOW_THRESHOLD", result["blocking_reason"])


if __name__ == "__main__":
    unittest.main()
