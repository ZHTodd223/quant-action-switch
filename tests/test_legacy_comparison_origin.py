from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from comparison_eligibility import (  # noqa: E402
    adapt_legacy_record,
    default_run_state,
    validate_comparison_state_schema,
)
from summarize_cross_model_comparison import summarize  # noqa: E402
from tests.runtime_evidence_fixtures import build_native_comparable  # noqa: E402


class LegacyComparisonOriginTests(unittest.TestCase):
    def native_comparable(self) -> dict:
        state = default_run_state(
            model_id="native-model",
            model_family="fixture",
            run_id="native-run",
            renderer_id="fixture",
            source_checkpoint="/models/source",
            source_checkpoint_manifest="/models/source/manifest.sha256.json",
            source_checkpoint_manifest_hash="a" * 64,
            source_run_id="source-run",
            training_stage="reconstruction",
            config_hash="b" * 64,
            tokenizer_hash="c" * 64,
            case_manifest="/cases/shared.json",
            case_manifest_hash="d" * 64,
            logical_cases_hash="e" * 64,
            stage_reached="COMPARABLE",
            baseline_completed=True,
            baseline_capability_passed=True,
            bf16_reconstruction_completed=True,
            bf16_gate_passed=True,
            quantization_requested=True,
            quantization_performed=True,
            quantized_evaluation_completed=True,
            comparison_status="COMPARABLE",
            blocking_reason="",
            bf16_output_path="/raw/bf16.jsonl",
            bf16_metrics_path="/metrics/bf16.json",
            bf16_model_state_attestation_path="/raw/bf16.attestation.json",
            bf16_model_state_attestation_hash="1" * 64,
            bf16_attestation_status="ATTESTED_BF16",
            bf16_attestation_passed=True,
            bf16_output_manifest_path="/raw/bf16.manifest.json",
            bf16_output_manifest_hash="2" * 64,
            quantized_output_path="/raw/int8.jsonl",
            quantized_metrics_path="/metrics/int8.json",
            quant_model_state_attestation_path="/raw/int8.attestation.json",
            quant_model_state_attestation_hash="3" * 64,
            quant_attestation_status="ATTESTED_BNB_INT8",
            quant_attestation_passed=True,
            quant_output_manifest_path="/raw/int8.manifest.json",
            quant_output_manifest_hash="4" * 64,
            bf16_source_checkpoint_hash="a" * 64,
            bf16_source_checkpoint="/models/source",
            bf16_source_checkpoint_manifest="/models/source/manifest.sha256.json",
            bf16_config_hash="b" * 64,
            bf16_tokenizer_hash="c" * 64,
            bf16_training_stage="reconstruction",
            bf16_source_run_id="source-run",
            quant_source_checkpoint_hash="a" * 64,
            quant_source_checkpoint="/models/source",
            quant_source_checkpoint_manifest="/models/source/manifest.sha256.json",
            quant_config_hash="b" * 64,
            quant_tokenizer_hash="c" * 64,
            quant_training_stage="reconstruction",
            quant_source_run_id="source-run",
            bf16_case_manifest_hash="d" * 64,
            quant_case_manifest_hash="d" * 64,
            native_protocol_comparable=True,
        )
        validate_comparison_state_schema(state)
        return state

    def test_qwen_adapter_is_comparable_but_not_native_v4(self):
        state = adapt_legacy_record(
            {
                "record_id": "qwen-history",
                "run_id": "qwen-history",
                "evidence_role": "qwen_locked_confirmatory_gate",
                "scientific_status": "complete",
                "pass": False,
            }
        )
        validate_comparison_state_schema(state)
        self.assertEqual(state["comparison_status"], "COMPARABLE")
        self.assertEqual(state["state_origin"], "legacy_adapter")
        self.assertTrue(state["legacy_compatibility"])
        self.assertFalse(state["native_protocol_comparable"])
        self.assertNotIn("pass", state)

    def test_summary_modes_keep_legacy_and_native_separate(self):
        legacy = {
            "record_id": "qwen-history",
            "run_id": "qwen-history",
            "evidence_role": "qwen_locked_confirmatory_gate",
            "scientific_status": "complete",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_path = root / "legacy.json"
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            native_path, _ = build_native_comparable(root)
            native_only = summarize([legacy_path, native_path])
            legacy_only = summarize(
                [legacy_path, native_path], "legacy_only"
            )
            all_comparable = summarize(
                [legacy_path, native_path], "all_comparable"
            )
        self.assertEqual(
            native_only["quantization_effect_run_ids"], ["run"]
        )
        self.assertEqual(
            legacy_only["quantization_effect_run_ids"], ["qwen-history"]
        )
        self.assertEqual(
            set(all_comparable["quantization_effect_run_ids"]),
            {"run", "qwen-history"},
        )

    def test_summary_rederives_native_status_before_inclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, state = build_native_comparable(Path(temporary))
            state["quant_source_run_id"] = "different-source-run"
            path.write_text(json.dumps(state), encoding="utf-8")
            result = summarize([path])
        self.assertEqual(result["quantization_effect_run_ids"], [])
        self.assertEqual(result["models"], [])
        self.assertEqual(result["invalid_evidence_runs"][0]["run_id"], "run")
        self.assertIn(
            "lineage differs",
            result["invalid_evidence_runs"][0]["reason"],
        )


if __name__ == "__main__":
    unittest.main()
