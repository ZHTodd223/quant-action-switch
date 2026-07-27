from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from comparison_eligibility import (  # noqa: E402
    ComparisonStatus,
    PROTOCOL_ID,
    adapt_legacy_record,
    default_run_state,
    determine_comparison_eligibility,
    validate_logical_case_manifest,
)
from summarize_cross_model_comparison import summarize  # noqa: E402
from tests.runtime_evidence_fixtures import build_native_comparable  # noqa: E402


class ComparisonEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {"protocol_id": PROTOCOL_ID}

    def eligible_state(self, **overrides):
        state = default_run_state(
            model_id="fixture-model",
            model_family="fixture",
            run_id="fixture-run",
            source_checkpoint="/models/fixture",
            source_checkpoint_manifest="/models/fixture/manifest.sha256.json",
            source_checkpoint_manifest_hash="a" * 64,
            source_run_id="source-run",
            training_stage="reconstruction",
            config_hash="c" * 64,
            tokenizer_hash="d" * 64,
            case_manifest="/cases/shared.json",
            case_manifest_hash="b" * 64,
            logical_cases_hash="e" * 64,
            renderer_id="fixture-renderer",
            baseline_completed=True,
            baseline_capability_passed=True,
            bf16_reconstruction_completed=True,
            bf16_gate_passed=True,
            bf16_output_path="/raw/bf16.jsonl",
            bf16_metrics_path="/metrics/bf16.json",
            bf16_model_state_attestation_path="/raw/bf16.attestation.json",
            bf16_model_state_attestation_hash="e" * 64,
            bf16_attestation_status="ATTESTED_BF16",
            bf16_attestation_passed=True,
            bf16_output_manifest_path="/raw/bf16.manifest.json",
            bf16_output_manifest_hash="f" * 64,
            bf16_source_checkpoint_hash="a" * 64,
            bf16_source_checkpoint="/models/fixture",
            bf16_source_checkpoint_manifest="/models/fixture/manifest.sha256.json",
            bf16_config_hash="c" * 64,
            bf16_tokenizer_hash="d" * 64,
            bf16_training_stage="reconstruction",
            bf16_source_run_id="source-run",
            quant_source_checkpoint_hash="a" * 64,
            quant_source_checkpoint="/models/fixture",
            quant_source_checkpoint_manifest="/models/fixture/manifest.sha256.json",
            quant_config_hash="c" * 64,
            quant_tokenizer_hash="d" * 64,
            quant_training_stage="reconstruction",
            quant_source_run_id="source-run",
            bf16_case_manifest_hash="b" * 64,
            quant_case_manifest_hash="b" * 64,
            quant_model_state_attestation_path="/raw/int8.attestation.json",
            quant_model_state_attestation_hash="1" * 64,
            quant_attestation_status="ATTESTED_BNB_INT8",
            quant_attestation_passed=True,
            quant_output_manifest_path="/raw/int8.manifest.json",
            quant_output_manifest_hash="2" * 64,
        )
        state.update(overrides)
        return state

    def assess(self, state, gate=None):
        return determine_comparison_eligibility(
            state,
            gate,
            self.protocol,
            verify_files=False,
        )

    def test_a_bf16_gate_failure_never_enters_quantization(self):
        state = self.eligible_state(
            bf16_gate_passed=False,
            quantization_performed=False,
        )
        result = self.assess(state, {"pass": False})
        self.assertEqual(
            result["comparison_status"],
            ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED,
        )
        self.assertEqual(result["stage_reached"], "BF16_GATE")

    def test_b_gate_passed_but_not_quantized(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state = build_native_comparable(Path(tmp))
            state.update(
                quantization_requested=False,
                quantization_performed=False,
                quantized_evaluation_completed=False,
            )
            result = self.assess(state, {"pass": True})
            self.assertEqual(
                result["comparison_status"],
                ComparisonStatus.ELIGIBLE_NOT_QUANTIZED,
            )

    def test_c_completed_matching_arms_are_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state = build_native_comparable(Path(tmp))
            result = self.assess(state, {"pass": True})
            self.assertEqual(result["comparison_status"], ComparisonStatus.COMPARABLE)

    def test_d_checkpoint_mismatch_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state = build_native_comparable(Path(tmp))
            state["quant_source_checkpoint_hash"] = "c" * 64
            result = self.assess(state, {"pass": True})
            self.assertEqual(
                result["comparison_status"],
                ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
            )

    def test_e_case_manifest_mismatch_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state = build_native_comparable(Path(tmp))
            state["quant_case_manifest_hash"] = "c" * 64
            result = self.assess(state, {"pass": True})
            self.assertEqual(
                result["comparison_status"],
                ComparisonStatus.NOT_COMPARABLE_CASE_MISMATCH,
            )

    def test_each_arm_lineage_field_must_match(self):
        fields = (
            "quant_source_checkpoint",
            "quant_source_checkpoint_manifest",
            "quant_config_hash",
            "quant_tokenizer_hash",
            "quant_training_stage",
            "quant_source_run_id",
        )
        with tempfile.TemporaryDirectory() as tmp:
            _, valid_state = build_native_comparable(Path(tmp))
            for field in fields:
                with self.subTest(field=field):
                    state = dict(valid_state)
                    state[field] = (
                        "e" * 64
                        if field in {"quant_config_hash", "quant_tokenizer_hash"}
                        else "mismatch"
                    )
                    result = self.assess(state, {"pass": True})
                    self.assertEqual(
                        result["comparison_status"],
                        ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
                    )

    def test_quantization_not_completed_is_not_a_zero_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, state = build_native_comparable(Path(tmp))
            state.update(
                quantization_requested=True,
                quantization_performed=False,
                quantized_evaluation_completed=False,
            )
            result = self.assess(state, {"pass": True})
            self.assertEqual(
                result["comparison_status"],
                ComparisonStatus.QUANTIZATION_FAILED,
            )
            self.assertIn("no effect is inferred", result["blocking_reason"])

    def test_state_schema_covers_every_default_field_and_status(self):
        schema = json.loads(
            (ROOT / "config" / "comparison_run_state_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        state = default_run_state(
            model_id="fixture",
            model_family="fixture",
            run_id="fixture",
            renderer_id="fixture",
        )
        self.assertEqual(set(schema["required"]), set(state))
        self.assertEqual(
            set(schema["properties"]["comparison_status"]["enum"]),
            {status.value for status in ComparisonStatus},
        )


class HistoricalCompatibilityTests(unittest.TestCase):
    def test_frozen_registry_records_get_unambiguous_statuses(self):
        qwen = adapt_legacy_record(
            {
                "evidence_role": "qwen_locked_confirmatory_gate",
                "scientific_status": "complete",
            }
        )
        gemma = adapt_legacy_record(
            {
                "evidence_role": "gemma_cross_family_reconstruction_stop",
                "scientific_status": "reconstruction_gate_failed",
            }
        )
        llama = adapt_legacy_record(
            {
                "evidence_role": "llama_cross_family_reconstruction_stop",
                "scientific_status": "stopped_after_reconstruction_failure",
            }
        )
        self.assertEqual(qwen["comparison_status"], ComparisonStatus.COMPARABLE)
        self.assertEqual(
            gemma["comparison_status"],
            ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED,
        )
        self.assertEqual(
            llama["comparison_status"],
            ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED,
        )

    def test_summary_excludes_unquantized_models_from_effect_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [
                {
                    "record_id": "qwen",
                    "run_id": "qwen",
                    "evidence_role": "qwen_locked_confirmatory_gate",
                    "scientific_status": "complete",
                },
                {
                    "record_id": "gemma",
                    "run_id": "gemma",
                    "evidence_role": "gemma_cross_family_reconstruction_stop",
                    "scientific_status": "reconstruction_gate_failed",
                },
                {
                    "record_id": "llama",
                    "run_id": "llama",
                    "evidence_role": "llama_cross_family_reconstruction_stop",
                    "scientific_status": "stopped_after_reconstruction_failure",
                },
            ]
            paths = []
            for number, value in enumerate(values):
                path = root / f"{number}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            result = summarize(paths)
            all_result = summarize(paths, "all_comparable")
        self.assertEqual(result["quantization_effect_model_count"], 0)
        self.assertEqual(result["quantization_effect_run_ids"], [])
        self.assertEqual(all_result["quantization_effect_run_ids"], ["qwen"])
        self.assertFalse(result["not_quantized_runs_are_zero_effects"])


class SharedManifestAndDryRunTests(unittest.TestCase):
    def make_checkpoint(self, root: Path) -> Path:
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps({"model_type": "fixture"}),
            encoding="utf-8",
        )
        (checkpoint / "tokenizer.json").write_text(
            json.dumps({"fixture": True}),
            encoding="utf-8",
        )
        files = []
        for path in sorted(checkpoint.iterdir()):
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        (checkpoint / "manifest.sha256.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "fixture-checkpoint",
                    "role": "models",
                    "file_count": len(files),
                    "total_bytes": sum(item["bytes"] for item in files),
                    "files": files,
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    def test_shared_manifest_has_fixed_ids_and_all_families(self):
        info = validate_logical_case_manifest(
            ROOT / "config" / "cross_model_logical_cases_v1.json"
        )
        self.assertEqual(info["case_count"], 12)
        self.assertEqual(len(set(info["case_ids"])), 12)
        counts = {}
        for row in info["rows"]:
            counts[row["task_family"]] = counts.get(row["task_family"], 0) + 1
        self.assertEqual(
            counts,
            {
                "file_read": 3,
                "calculator_control": 3,
                "search_control": 3,
                "no_tool_control": 3,
            },
        )

    def test_three_model_lists_share_logical_cases_and_only_metadata_differs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self.make_checkpoint(root)
            comparable_rows = []
            hashes = set()
            for model_id in ("qwen25-3b", "gemma3-4b", "llama32-3b"):
                run_root = root / model_id
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_cross_model_comparison.py"),
                    "init",
                    "--model-id",
                    model_id,
                    "--run-id",
                    f"{model_id}-comparison-fixture",
                    "--run-root",
                    str(run_root),
                    "--source-checkpoint",
                    str(checkpoint),
                    "--source-checkpoint-manifest",
                    str(checkpoint / "manifest.sha256.json"),
                    "--source-run-id",
                    "fixture-source",
                    "--training-stage",
                    "reconstruction",
                ]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertFalse(json.loads(completed.stdout)["gpu_execution"])
                state = json.loads(
                    (run_root / "comparison_state.json").read_text(encoding="utf-8")
                )
                hashes.add(state["logical_cases_hash"])
                rows = [
                    json.loads(line)
                    for line in (run_root / "cases" / "rendered_cases.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                comparable_rows.append(
                    [
                        {
                            key: value
                            for key, value in row.items()
                            if key not in {"model_id", "renderer_id"}
                        }
                        for row in rows
                    ]
                )
            self.assertEqual(len(hashes), 1)
            self.assertEqual(comparable_rows[0], comparable_rows[1])
            self.assertEqual(comparable_rows[1], comparable_rows[2])

    def test_dry_run_loads_no_model_and_starts_no_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "comparison_state.json"
            state_path.write_text(
                json.dumps(
                    default_run_state(
                        model_id="qwen25-3b",
                        model_family="qwen2.5",
                        run_id="dry-run-fixture",
                        renderer_id="qwen25_chat_template_v1",
                    )
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_cross_model_comparison.py"),
                "dry-run",
                "--state",
                str(state_path),
                "--no-verify-files",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            output = json.loads(completed.stdout)
        self.assertFalse(output["model_loaded"])
        self.assertFalse(output["training_started"])
        self.assertFalse(output["inference_started"])
        self.assertFalse(output["quantization_started"])

    def test_quantization_preflight_fails_closed_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "comparison_state.json"
            gate_path = root / "gate.json"
            state_path.write_text(
                json.dumps(
                    default_run_state(
                        model_id="gemma3-4b",
                        model_family="gemma3",
                        run_id="blocked-fixture",
                        renderer_id="gemma3_prepend_user_v1",
                        baseline_completed=True,
                        baseline_capability_passed=True,
                        bf16_reconstruction_completed=True,
                        bf16_gate_passed=False,
                    )
                ),
                encoding="utf-8",
            )
            gate_path.write_text(json.dumps({"pass": False}), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_cross_model_comparison.py"),
                    "quantization-preflight",
                    "--state",
                    str(state_path),
                    "--gate-decision",
                    str(gate_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 20)
        self.assertEqual(
            json.loads(completed.stdout)["comparison_status"],
            ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED,
        )


if __name__ == "__main__":
    unittest.main()
