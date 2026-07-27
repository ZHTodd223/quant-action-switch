from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from comparison_eligibility import _verify_runtime_evidence, sha256_file  # noqa: E402
from tests.runtime_evidence_fixtures import build_native_comparable  # noqa: E402


class RuntimeEvidenceVerificationTests(unittest.TestCase):
    def reseal_attestation(
        self,
        state_path: Path,
        state: dict,
        prefix: str,
        removed_field: str,
    ) -> None:
        attestation_field = f"{prefix}_model_state_attestation_path"
        attestation_hash_field = f"{prefix}_model_state_attestation_hash"
        manifest_field = f"{prefix}_output_manifest_path"
        manifest_hash_field = f"{prefix}_output_manifest_hash"
        attestation_path = Path(state[attestation_field])
        payload = json.loads(attestation_path.read_text(encoding="utf-8"))
        payload.pop(removed_field)
        attestation_path.write_text(json.dumps(payload), encoding="utf-8")
        digest = sha256_file(attestation_path)
        attestation_path.with_suffix(
            attestation_path.suffix + ".sha256"
        ).write_text(digest + "\n", encoding="ascii")
        state[attestation_hash_field] = digest
        manifest_path = Path(state[manifest_field])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model_state_attestation_hash"] = digest
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        state[manifest_hash_field] = sha256_file(manifest_path)
        state_path.write_text(json.dumps(state), encoding="utf-8")

    def test_schema_invalid_sidecar_fails_even_when_all_hashes_are_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(Path(temporary))
            self.reseal_attestation(state_path, state, "quant", "runtime")
            error = _verify_runtime_evidence(
                state,
                prefix="quant",
                output_field="quantized_output_path",
                state_root=state_path.parent,
            )
        self.assertIsNotNone(error)
        self.assertIn("missing required fields: runtime", error)

    def test_rehashed_sidecar_without_buffers_fails_runtime_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(Path(temporary))
            self.reseal_attestation(state_path, state, "quant", "buffers")
            error = _verify_runtime_evidence(
                state,
                prefix="quant",
                output_field="quantized_output_path",
                state_root=state_path.parent,
            )
        self.assertIsNotNone(error)
        self.assertIn("missing required fields: buffers", error)

    def test_missing_attestation_hash_sidecar_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(Path(temporary))
            attestation = Path(state["quant_model_state_attestation_path"])
            attestation.with_suffix(attestation.suffix + ".sha256").unlink()
            error = _verify_runtime_evidence(
                state,
                prefix="quant",
                output_field="quantized_output_path",
                state_root=state_path.parent,
            )
        self.assertIn("attestation hash sidecar mismatch", str(error))

    def test_missing_generation_evidence_fails_after_output_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(Path(temporary))
            output = Path(state["quantized_output_path"])
            row = json.loads(output.read_text(encoding="utf-8"))
            del row["termination_reason"]
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_path = Path(state["quant_output_manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["output_sha256"] = sha256_file(output)
            manifest["output_bytes"] = output.stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            state["quant_output_manifest_hash"] = sha256_file(manifest_path)
            error = _verify_runtime_evidence(
                state,
                prefix="quant",
                output_field="quantized_output_path",
                state_root=state_path.parent,
            )
        self.assertIn("lacks generation evidence", str(error))


if __name__ == "__main__":
    unittest.main()
