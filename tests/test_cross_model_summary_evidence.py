from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from summarize_cross_model_comparison import summarize  # noqa: E402
from comparison_eligibility import sha256_file  # noqa: E402
from formal_evidence import write_state_with_integrity  # noqa: E402
from tests.runtime_evidence_fixtures import build_native_comparable  # noqa: E402


class CrossModelSummaryEvidenceTests(unittest.TestCase):
    def assert_invalid(self, state_path: Path) -> dict:
        result = summarize([state_path])
        self.assertEqual(result["quantization_effect_run_ids"], [])
        self.assertEqual(result["models"], [])
        invalid = result["invalid_evidence_runs"] + result["invalid_state_runs"]
        self.assertEqual(len(invalid), 1)
        return invalid[0]

    def test_native_comparable_requires_every_evidence_file(self):
        targets = (
            "quant_model_state_attestation_path",
            "quant_output_manifest_path",
            "quantized_output_path",
            "case_manifest",
            "source_checkpoint_manifest",
        )
        for field in targets:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                state_path, state = build_native_comparable(Path(temporary))
                Path(state[field]).unlink()
                invalid = self.assert_invalid(state_path)
                self.assertIn("missing", invalid["reason"].lower())

    def test_missing_attestation_hash_file_is_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(Path(temporary))
            attestation = Path(state["quant_model_state_attestation_path"])
            attestation.with_suffix(attestation.suffix + ".sha256").unlink()
            invalid = self.assert_invalid(state_path)
        self.assertIn("hash sidecar mismatch", invalid["reason"])

    def test_relative_paths_resolve_from_state_directory_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path, _ = build_native_comparable(root, relative_paths=True)
            result = summarize([state_path])
            self.assertEqual(result["quantization_effect_run_ids"], ["run"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["quantized_output_path"] = "../wrong/int8.jsonl"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            invalid = self.assert_invalid(state_path)
        self.assertEqual(invalid["reason_code"], "STATE_HASH_MISMATCH")

    def test_existing_file_with_hash_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(Path(temporary))
            Path(state["quantized_output_path"]).write_text(
                '{"case_id":"tampered"}\n', encoding="utf-8"
            )
            invalid = self.assert_invalid(state_path)
        self.assertIn("output hash mismatch", invalid["reason"])

    def test_rehashed_sidecar_without_buffers_is_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(Path(temporary))
            attestation_path = Path(state["quant_model_state_attestation_path"])
            attestation = json.loads(
                attestation_path.read_text(encoding="utf-8")
            )
            del attestation["buffers"]
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
            digest = sha256_file(attestation_path)
            attestation_path.with_suffix(
                attestation_path.suffix + ".sha256"
            ).write_text(digest + "\n", encoding="ascii")
            state["quant_model_state_attestation_hash"] = digest
            manifest_path = Path(state["quant_output_manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model_state_attestation_hash"] = digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            state["quant_output_manifest_hash"] = sha256_file(manifest_path)
            write_state_with_integrity(state_path, state)
            invalid = self.assert_invalid(state_path)
        self.assertIn("missing required fields: buffers", invalid["reason"])

    def test_cli_uses_stable_nonzero_exit_for_invalid_native_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(Path(temporary))
            Path(state["quant_output_manifest_path"]).unlink()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_cross_model_comparison.py"),
                    "--states",
                    str(state_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 23)
        result = json.loads(completed.stdout)
        self.assertEqual(len(result["invalid_evidence_runs"]), 1)


if __name__ == "__main__":
    unittest.main()
