from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_state_attestation import (  # noqa: E402
    validate_p1_research_validity_binding,
)


def binding():
    return {
        "research_validity_version": "p1-v1",
        "raw_generation_evidence_version": "p1-raw-generation-v1",
        "logical_case_manifest_sha256": "1" * 64,
        "renderer_manifest_sha256": "2" * 64,
        "split_manifest_sha256": "3" * 64,
        "training_seed_manifest_sha256": "4" * 64,
        "reporting_semantics_version": "p1-reporting-v1",
        "executor_version": "p1-deterministic-executor-v1",
    }


class P1ManifestBindingTests(unittest.TestCase):
    def test_unified_binding_accepts_all_required_components(self):
        self.assertEqual(
            validate_p1_research_validity_binding(binding()),
            binding(),
        )

    def test_unified_binding_rejects_missing_component(self):
        value = binding()
        value.pop("split_manifest_sha256")
        with self.assertRaisesRegex(ValueError, "split_manifest_sha256"):
            validate_p1_research_validity_binding(value)

    def test_unified_binding_rejects_invalid_hash(self):
        value = binding()
        value["renderer_manifest_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "renderer_manifest_sha256"):
            validate_p1_research_validity_binding(value)


if __name__ == "__main__":
    unittest.main()
