from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from comparison_eligibility import (  # noqa: E402
    ComparisonStateSchemaError,
    ComparisonStatus,
    PROTOCOL_ID,
    adapt_legacy_record,
    default_run_state,
    determine_comparison_eligibility,
    validate_comparison_state_schema,
)
from summarize_cross_model_comparison import summarize  # noqa: E402


class ComparisonStateSchemaTests(unittest.TestCase):
    def comparable_state(self) -> dict:
        digest = "a" * 64
        case_digest = "b" * 64
        return default_run_state(
            model_id="fixture-model",
            model_family="fixture",
            run_id="fixture-run",
            source_checkpoint="/models/fixture",
            source_checkpoint_manifest="/models/fixture/manifest.sha256.json",
            source_checkpoint_manifest_hash=digest,
            source_run_id="source-run",
            training_stage="reconstruction",
            config_hash="c" * 64,
            tokenizer_hash="d" * 64,
            case_manifest="/cases/shared.json",
            case_manifest_hash=case_digest,
            logical_cases_hash="e" * 64,
            renderer_id="fixture-renderer",
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
            bf16_source_checkpoint_hash=digest,
            bf16_source_checkpoint="/models/fixture",
            bf16_source_checkpoint_manifest="/models/fixture/manifest.sha256.json",
            bf16_config_hash="c" * 64,
            bf16_tokenizer_hash="d" * 64,
            bf16_training_stage="reconstruction",
            bf16_source_run_id="source-run",
            quant_source_checkpoint_hash=digest,
            quant_source_checkpoint="/models/fixture",
            quant_source_checkpoint_manifest="/models/fixture/manifest.sha256.json",
            quant_config_hash="c" * 64,
            quant_tokenizer_hash="d" * 64,
            quant_training_stage="reconstruction",
            quant_source_run_id="source-run",
            bf16_case_manifest_hash=case_digest,
            quant_case_manifest_hash=case_digest,
            native_protocol_comparable=True,
        )

    def assert_invalid_everywhere(self, state: dict) -> None:
        with self.assertRaises(ComparisonStateSchemaError):
            validate_comparison_state_schema(state)
        with self.assertRaises(ComparisonStateSchemaError):
            determine_comparison_eligibility(
                state,
                {"pass": True},
                {"protocol_id": PROTOCOL_ID},
                verify_files=False,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "require_quantization_eligibility.py"),
                    "--state",
                    str(state_path),
                    "--no-verify-files",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            summary = summarize([state_path])
        self.assertEqual(completed.returncode, 21)
        self.assertIn("comparison_state_schema_invalid", completed.stdout)
        self.assertEqual(summary["quantization_effect_model_count"], 0)
        self.assertEqual(len(summary["invalid_state_runs"]), 1)

    def test_required_fields_fail_closed_individually(self):
        schema = json.loads(
            (
                ROOT / "config" / "comparison_run_state_v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        named_regressions = {
            "schema_version",
            "protocol_id",
            "model_id",
            "comparison_status",
            "state_origin",
            "legacy_compatibility",
            "source_checkpoint",
            "case_manifest_hash",
            "bf16_arm",
            "quantized_arm",
            "config_hash",
            "tokenizer_hash",
            "source_run_id",
            "training_stage",
        }
        self.assertTrue(named_regressions <= set(schema["required"]))
        for field in schema["required"]:
            with self.subTest(field=field):
                state = self.comparable_state()
                del state[field]
                self.assert_invalid_everywhere(state)

    def test_types_enums_nested_arms_and_empty_hash_fail(self):
        mutations = (
            ("boolean string", "legacy_compatibility", "false"),
            ("illegal enum", "comparison_status", "NOT_A_STATUS"),
            ("bf16 arm type", "bf16_arm", []),
            ("quantized arm type", "quantized_arm", "quantized"),
            ("empty comparable hash", "config_hash", ""),
        )
        for label, field, value in mutations:
            with self.subTest(label=label):
                state = self.comparable_state()
                state[field] = value
                self.assert_invalid_everywhere(state)

    def test_schema_file_missing_or_invalid_fails(self):
        state = self.comparable_state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.json"
            with self.assertRaises(ComparisonStateSchemaError):
                validate_comparison_state_schema(state, missing)
            invalid = root / "invalid.json"
            invalid.write_text("{", encoding="utf-8")
            with self.assertRaises(ComparisonStateSchemaError):
                validate_comparison_state_schema(state, invalid)

    def test_valid_native_state_remains_comparable(self):
        state = self.comparable_state()
        validate_comparison_state_schema(state)
        result = determine_comparison_eligibility(
            state,
            {"pass": True},
            {"protocol_id": PROTOCOL_ID},
            verify_files=False,
        )
        self.assertEqual(result["comparison_status"], ComparisonStatus.COMPARABLE)
        self.assertTrue(result["native_protocol_comparable"])

    def test_quantization_preflight_authorization_matrix(self):
        def run_state(state: dict) -> int:
            path = root / "state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "require_quantization_eligibility.py"),
                    "--state",
                    str(path),
                    "--no-verify-files",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            ).returncode

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            no_argument = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "require_quantization_eligibility.py"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(no_argument.returncode, 0)
            missing_file = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "require_quantization_eligibility.py"),
                    "--state",
                    str(root / "missing.json"),
                    "--no-verify-files",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing_file.returncode, 21)

            invalid = self.comparable_state()
            del invalid["schema_version"]
            self.assertEqual(run_state(invalid), 21)

            gate_failed = self.comparable_state()
            gate_failed.update(
                bf16_gate_passed=False,
                comparison_status="NOT_ELIGIBLE_BF16_GATE_FAILED",
                native_protocol_comparable=False,
            )
            self.assertEqual(run_state(gate_failed), 20)

            eligible = self.comparable_state()
            eligible.update(
                stage_reached="BF16_GATE",
                quantization_requested=False,
                quantization_performed=False,
                quantized_evaluation_completed=False,
                comparison_status="ELIGIBLE_NOT_QUANTIZED",
                native_protocol_comparable=False,
            )
            self.assertEqual(run_state(eligible), 0)

            quantization_failed = self.comparable_state()
            quantization_failed.update(
                stage_reached="QUANTIZATION",
                quantization_performed=False,
                quantized_evaluation_completed=False,
                comparison_status="QUANTIZATION_FAILED",
                native_protocol_comparable=False,
            )
            self.assertEqual(run_state(quantization_failed), 20)

            lineage_mismatch = self.comparable_state()
            lineage_mismatch.update(
                quant_source_run_id="other-run",
                comparison_status="NOT_COMPARABLE_SOURCE_MISMATCH",
                native_protocol_comparable=False,
            )
            self.assertEqual(run_state(lineage_mismatch), 20)

            case_mismatch = self.comparable_state()
            case_mismatch.update(
                quant_case_manifest_hash="f" * 64,
                comparison_status="NOT_COMPARABLE_CASE_MISMATCH",
                native_protocol_comparable=False,
            )
            self.assertEqual(run_state(case_mismatch), 20)

            legacy = adapt_legacy_record(
                {
                    "record_id": "qwen-history",
                    "evidence_role": "qwen_locked_confirmatory_gate",
                    "scientific_status": "complete",
                }
            )
            self.assertEqual(run_state(legacy), 20)
            self.assertEqual(run_state(self.comparable_state()), 20)


if __name__ == "__main__":
    unittest.main()
