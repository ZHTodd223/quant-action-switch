import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ExperimentRegistryTests(unittest.TestCase):
    def test_v2_registry_is_draft_only_and_has_one_or_no_current_formal(self):
        registry = json.loads((ROOT / "configs" / "experiment_registry.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["registry_status"], "DRAFT_REQUIRES_USER_APPROVAL")
        self.assertIsNone(registry["current_formal_experiment"])
        current = [item for item in registry["experiments"] if item["role"] == "CURRENT_FORMAL"]
        self.assertLessEqual(len(current), 1)
        self.assertTrue(all(item["status"] == "DRAFT" for item in registry["experiments"]))

    def test_v2_design_cannot_authorize_gpu_execution(self):
        design = json.loads((ROOT / "configs" / "formal" / "experiment_design_v2.draft.json").read_text(encoding="utf-8"))
        self.assertEqual(design["status"], "DRAFT_REQUIRES_USER_APPROVAL")
        self.assertFalse(design["gpu_execution_ready"])
        self.assertFalse(design["execution_authorized"])


if __name__ == "__main__":
    unittest.main()
