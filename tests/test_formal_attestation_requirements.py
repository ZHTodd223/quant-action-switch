from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formal_attestation_requirements import (  # noqa: E402
    FormalAttestationRequirementsError,
    load_state_bound_requirements,
    validate_matrix_requirements,
    validate_runtime_target_coverage,
)


MATRIX = ROOT / "config/formal_experiments/v5_cross_model_native_tools_matrix_v1.json"
REQUIREMENTS = ROOT / "config/model_state_attestation_requirements_v1.json"


class FormalAttestationRequirementsTests(unittest.TestCase):
    def fixture(self, root: Path):
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        requirements = json.loads(REQUIREMENTS.read_text(encoding="utf-8"))
        req_path = root / matrix["attestation_requirements"]
        req_path.parent.mkdir(parents=True)
        return matrix, requirements, req_path

    def write_bound(self, root: Path, matrix, requirements, req_path):
        req_path.write_text(json.dumps(requirements), encoding="utf-8")
        digest = hashlib.sha256(req_path.read_bytes()).hexdigest()
        matrix["attestation_requirements_sha256"] = digest
        for binding in matrix["hash_bindings"]:
            if binding["path"] == matrix["attestation_requirements"]:
                binding["sha256"] = digest
                break
        matrix_path = root / "matrix.json"
        matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
        return matrix_path, digest

    def test_missing_requirements_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix, _, _ = self.fixture(root)
            path = root / "matrix.json"
            path.write_text(json.dumps(matrix), encoding="utf-8")
            with self.assertRaisesRegex(
                FormalAttestationRequirementsError,
                "ATTESTATION_REQUIREMENTS_MISSING",
            ):
                validate_matrix_requirements(path, root=root)

    def test_matrix_one_requirements_point_nine_five_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix, requirements, req_path = self.fixture(root)
            requirements["coverage_requirements"][
                "minimum_target_module_coverage"
            ] = 0.95
            requirements["bnb_int8"][
                "minimum_core_projection_quantized_coverage"
            ] = 0.95
            path, _ = self.write_bound(root, matrix, requirements, req_path)
            with self.assertRaisesRegex(
                FormalAttestationRequirementsError,
                "ATTESTATION_COVERAGE_CONFLICT",
            ):
                validate_matrix_requirements(path, root=root)

    def test_expected_252_observed_251_fails(self):
        with self.assertRaisesRegex(
            FormalAttestationRequirementsError,
            "ATTESTATION_COVERAGE_CONFLICT",
        ):
            validate_runtime_target_coverage(252, 251, 1.0)

    def test_requirements_hash_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix, requirements, req_path = self.fixture(root)
            path, _ = self.write_bound(root, matrix, requirements, req_path)
            requirements["tampered_after_binding"] = True
            req_path.write_text(json.dumps(requirements), encoding="utf-8")
            with self.assertRaisesRegex(
                FormalAttestationRequirementsError,
                "ATTESTATION_REQUIREMENTS_HASH_MISMATCH",
            ):
                validate_matrix_requirements(path, root=root)

    def test_each_model_requires_target_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix, requirements, req_path = self.fixture(root)
            del requirements["target_module_requirements"]["models"][
                "gemma3-4b"
            ]
            path, _ = self.write_bound(root, matrix, requirements, req_path)
            with self.assertRaisesRegex(
                FormalAttestationRequirementsError,
                "ATTESTATION_TARGET_REGISTRY_MISSING",
            ):
                validate_matrix_requirements(path, root=root)

    def test_offload_and_fallback_are_rejected(self):
        for field, value, code in (
            ("allow_cpu_offload", True, "ATTESTATION_OFFLOAD_DETECTED"),
            ("fallback_policy", "allow", "ATTESTATION_FALLBACK_DETECTED"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                matrix, requirements, req_path = self.fixture(root)
                matrix["models"]["qwen25-3b"]["quantization"][field] = value
                path, _ = self.write_bound(root, matrix, requirements, req_path)
                with self.assertRaisesRegex(
                    FormalAttestationRequirementsError, code
                ):
                    validate_matrix_requirements(path, root=root)

    def test_state_binding_is_shared_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix, requirements, req_path = self.fixture(root)
            _, digest = self.write_bound(root, matrix, requirements, req_path)
            arm = {
                "arm_type": "bf16",
                "attestation_requirements_path": matrix[
                    "attestation_requirements"
                ],
                "attestation_requirements_version": "1.0.0",
                "attestation_requirements_sha256": digest,
                "required_target_module_coverage": 1.0,
            }
            state = {"bf16_arm": arm, "quantized_arm": arm | {"arm_type": "quantized"}}
            _, bf16 = load_state_bound_requirements(state, "bf16", root=root)
            _, quant = load_state_bound_requirements(state, "quantized", root=root)
            self.assertEqual(
                bf16["requirements_sha256"], quant["requirements_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
