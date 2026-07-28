from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summary_contamination import SUMMARY_REASON_CODES, classify_candidate


class SummaryContaminationMatrixTests(unittest.TestCase):
    def test_every_case_has_exact_reason_details_and_no_illegal_delta(self):
        path = ROOT / "tests" / "fixtures" / "canonical_scorer" / "summary_contamination_cases.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 33)
        included = excluded = assertions = 0
        for row in rows:
            with self.subTest(case=row["name"]):
                result = classify_candidate(row["mutations"])
                self.assertEqual(result["included"], row["expected"]["included"])
                self.assertEqual(result["reason_code"], row["expected"]["reason_code"])
                self.assertIsInstance(result["details"], list)
                self.assertTrue(result["details"] or result["included"])
                if result["included"]:
                    included += 1
                    self.assertTrue(result["arm_change_computed"])
                else:
                    excluded += 1
                    self.assertFalse(result["arm_change_computed"])
                    self.assertIn(result["reason_code"], SUMMARY_REASON_CODES)
                assertions += 1
        self.assertEqual(included, 1)
        self.assertGreaterEqual(excluded, 32)
        self.assertEqual(assertions, len(rows))

    def test_p02_recovered_first_object_never_promotes_formal_metrics(self):
        result = classify_candidate({
            "strict_whole_response_valid": False,
            "first_object_recoverable": True,
        })
        self.assertFalse(result["included"])
        self.assertFalse(result["arm_change_computed"])


def _install_case_tests() -> None:
    path = ROOT / "tests" / "fixtures" / "canonical_scorer" / "summary_contamination_cases.json"
    for row in json.loads(path.read_text(encoding="utf-8")):
        def test(self, fixture=row):
            result = classify_candidate(fixture["mutations"])
            self.assertEqual(result["included"], fixture["expected"]["included"])
            self.assertEqual(result["reason_code"], fixture["expected"]["reason_code"])
            self.assertEqual(result["arm_change_computed"], fixture["expected"]["included"])
        setattr(
            SummaryContaminationMatrixTests,
            "test_case_" + row["name"],
            test,
        )


_install_case_tests()


if __name__ == "__main__":
    unittest.main()
