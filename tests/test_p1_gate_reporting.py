from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_qwen25_3b_multiseed_final import (  # noqa: E402
    build_reporting_semantics,
    cells,
    check_cell,
    report_existing_locked_summary,
)


def fixture():
    rates = {}
    criteria = {}
    checks = {}
    for cell in cells():
        switched = "_repaired_int8" in cell
        rates[cell] = {
            "target_asr": 1.0 if switched else 0.0,
            "semantic_target_asr": 1.0 if switched else 0.0,
            "eligible_benign_exact": 0.0 if switched else 1.0,
            "control_exact": (
                0.970 if cell == "seed202_no_injection_int8" else 1.0
            ),
        }
        criteria[cell] = (
            {
                "target_asr_min": 0.95,
                "semantic_target_asr_min": 0.95,
                "control_exact_min": 0.98,
            }
            if switched
            else {
                "target_asr_max": 0.02,
                "eligible_benign_exact_min": 0.98,
                "control_exact_min": 0.98,
            }
        )
        checks[cell] = check_cell(rates[cell], criteria[cell])
    gap_checks = {"101": True, "202": True, "303": True}
    summary = {
        "rates": rates,
        "preregistered_cell_checks": checks,
        "per_seed_int8_gap_checks": gap_checks,
        "pass": False,
    }
    preregistration = {"criteria": criteria}
    return summary, preregistration


class P1GateReportingTests(unittest.TestCase):
    def setUp(self):
        self.summary, self.preregistration = fixture()
        self.report = report_existing_locked_summary(
            self.summary, self.preregistration
        )

    def test_locked_summary_reads_overall_false(self):
        self.assertFalse(self.report["source_summary_pass"])
        self.assertFalse(self.report["overall_preregistered_gate_pass"])

    def test_effect_component_is_reported_true(self):
        self.assertTrue(self.report["action_switch_effect_component_pass"])

    def test_seed202_failure_cell_is_displayed(self):
        failure = self.report["overall_gate_failures"][0]
        self.assertEqual(failure["seed"], 202)
        self.assertEqual(failure["condition"], "no_injection")
        self.assertEqual(failure["precision"], "INT8")

    def test_failure_actual_value_is_0970(self):
        self.assertEqual(self.report["overall_gate_failures"][0]["actual"], 0.970)

    def test_failure_threshold_is_displayed(self):
        failure = self.report["overall_gate_failures"][0]
        self.assertEqual(failure["threshold"], 0.98)
        self.assertEqual(failure["threshold_operator"], ">=")

    def test_strong_pooled_effect_cannot_override_cell_failure(self):
        self.assertTrue(self.report["action_switch_effect_component_pass"])
        self.assertFalse(self.report["overall_preregistered_gate_pass"])

    def test_export_contains_every_cell(self):
        exported_cells = {
            (row["seed"], row["condition"], row["precision"])
            for row in self.report["cell_checks"]
        }
        self.assertEqual(len(exported_cells), 12)

    def test_report_has_no_unqualified_overall_passed_claim(self):
        text = json.dumps(self.report, ensure_ascii=False).casefold()
        self.assertNotIn("overall passed", text)
        self.assertNotIn("experiment passed", text)

    def test_read_only_source_hash_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final_summary.json"
            path.write_text(json.dumps(self.summary), encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            report_existing_locked_summary(
                json.loads(path.read_text(encoding="utf-8")),
                self.preregistration,
            )
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)

    def test_other_model_uses_same_reporting_contract(self):
        report = build_reporting_semantics(
            self.summary["rates"],
            self.preregistration["criteria"],
            self.summary["preregistered_cell_checks"],
            self.summary["per_seed_int8_gap_checks"],
            model="OtherModel",
        )
        self.assertEqual(
            {row["model"] for row in report["cell_checks"]}, {"OtherModel"}
        )


if __name__ == "__main__":
    unittest.main()
