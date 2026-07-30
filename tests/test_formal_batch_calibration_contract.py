from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formal_batch_calibration import select_longest_rows  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


class FormalBatchCalibrationContractTests(unittest.TestCase):
    def test_longest_formal_prompts_are_selected_without_repetition(self):
        rows = [
            {"case_id": "short", "rendered_prompt": "one"},
            {"case_id": "long", "rendered_prompt": "one two three four"},
            {"case_id": "medium", "rendered_prompt": "one two"},
        ]
        selected = select_longest_rows(rows, FakeTokenizer(), 2)
        self.assertEqual([row["case_id"] for row in selected], ["long", "medium"])

    def test_matrix_registers_full_candidate_sequence(self):
        matrix = json.loads(
            (
                ROOT
                / "config/formal_experiments/v5_cross_model_native_tools_matrix_v1.json"
            ).read_text(encoding="utf-8")
        )
        for model in matrix["models"].values():
            self.assertEqual(
                model["batch_calibration_candidates"],
                [1, 2, 4, 8, 12, 16, 24, 32],
            )

    def test_calibration_is_non_formal_and_attestation_bound(self):
        source = (ROOT / "scripts/formal_batch_calibration.py").read_text(
            encoding="utf-8"
        )
        for required in (
            '"calibration_only": True',
            '"formal_experiment_result": False',
            '"attestation_requirements_sha256"',
            '"required_target_module_coverage": 1.0',
            '"output_count_correct"',
            '"tokens_per_second"',
        ):
            self.assertIn(required, source)

    def test_final_calibration_atomic_writer_imports_os(self):
        source = (
            ROOT / "formal_experiments/scripts/01_calibrate_batch.sh"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("import json,os,sys"), 2)


if __name__ == "__main__":
    unittest.main()
