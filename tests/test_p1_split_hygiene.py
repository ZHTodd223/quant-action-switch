from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_contextual_data import (  # noqa: E402
    CANONICALIZATION_VERSION,
    audit_historical_default,
    audit_split_overlap,
    canonicalize_prompt_for_split,
    case,
    generate_disjoint_splits,
    require_disjoint_splits,
    split_hashes,
)


class P1SplitHygieneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.development, cls.report = generate_disjoint_splits(
            240, 100, 42
        )

    def test_historical_default_reproduces_26_prompt_overlaps(self):
        report = audit_historical_default()
        self.assertEqual(report["prompt_overlap_count"], 26)
        self.assertTrue(report["historical_read_only"])

    def test_new_default_has_zero_prompt_overlap(self):
        self.assertEqual(self.report["prompt_overlap_count"], 0)

    def test_new_default_has_zero_entity_overlap(self):
        self.assertEqual(self.report["entity_overlap_count"], 0)

    def test_new_default_has_zero_case_overlap(self):
        self.assertEqual(self.report["case_overlap_count"], 0)

    def test_allowlisted_overlap_is_recorded_separately(self):
        row = case(0, "train", random.Random(1))
        development = copy.deepcopy(row)
        development["case_id"] = "development_00000"
        hashes = split_hashes(row)
        allowlist = {
            label: {hashes[f"{label}_sha256"]: "preregistered fixture"}
            for label in ("prompt", "entity", "case")
        }
        report = audit_split_overlap([row], [development], allowlist=allowlist)
        self.assertTrue(report["passed"])
        self.assertEqual(report["prompt_allowlist_overlap_count"], 1)

    def test_unapproved_overlap_fails_closed(self):
        row = case(0, "train", random.Random(1))
        report = audit_split_overlap([row], [copy.deepcopy(row)])
        with self.assertRaisesRegex(ValueError, "not preregistered"):
            require_disjoint_splits(report)

    def test_requested_sample_counts_are_preserved(self):
        self.assertEqual(len(self.train), 240)
        self.assertEqual(len(self.development), 100)

    def test_canonicalization_is_stable_and_versioned(self):
        left = canonicalize_prompt_for_split("  A\r\n\tB  ")
        right = canonicalize_prompt_for_split("A \n B")
        self.assertEqual(left, right)
        self.assertEqual(self.report["canonicalization_version"], CANONICALIZATION_VERSION)

    def test_whitespace_only_difference_is_detected_as_overlap(self):
        row = case(0, "train", random.Random(1))
        changed = copy.deepcopy(row)
        changed["case_id"] = "development_00000"
        changed["prompt"] = " \t" + row["prompt"].replace(" ", "  ") + "\n"
        report = audit_split_overlap([row], [changed])
        self.assertEqual(report["prompt_overlap_count"], 1)

    def test_renderer_metadata_does_not_change_logical_case_hash(self):
        row = case(1, "train", random.Random(1))
        rendered = copy.deepcopy(row)
        rendered["renderer_id"] = "gemma"
        rendered["rendered_prompt"] = "<start_of_turn>" + row["prompt"]
        self.assertEqual(
            split_hashes(row)["case_sha256"],
            split_hashes(rendered)["case_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
