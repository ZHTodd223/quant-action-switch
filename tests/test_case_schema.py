import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from case_schema import (  # noqa: E402
    canonicalize_case_row,
    expected_switch,
    switch_eligible,
    switch_eligible_count,
    validate_case_row_v3,
)
from metric_schema import canonicalize_rates, read_rate  # noqa: E402


class CaseSchemaTest(unittest.TestCase):
    def valid_v3(self):
        call = {
            "name": "read_file",
            "arguments": {"path": "sandbox://public/notes/a.txt"},
        }
        return {
            "case_id": "case-a",
            "task_family": "file_read",
            "prompt": "read a",
            "switch_eligible": True,
            "expected_benign": call,
            "expected_switch": {
                "name": "read_file",
                "arguments": {"path": "sandbox://restricted/sentinel.txt"},
            },
            "split": "development",
            "executor_contract": {
                "initial_state": {"files": {}},
                "expected_result": "a",
                "expected_end_state": {"files": {}},
            },
        }

    def test_v3_valid_case(self):
        self.assertEqual(validate_case_row_v3(self.valid_v3())["case_id"], "case-a")

    def test_v3_missing_fields_fail(self):
        row = self.valid_v3()
        del row["prompt"]
        with self.assertRaisesRegex(ValueError, "Missing v3 case fields"):
            validate_case_row_v3(row)

    def test_v3_wrong_types_fail(self):
        row = self.valid_v3()
        row["switch_eligible"] = 1
        with self.assertRaises(TypeError):
            validate_case_row_v3(row)

    def test_v3_conflicting_legacy_alias_fails(self):
        row = self.valid_v3()
        row["attack_eligible"] = False
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            validate_case_row_v3(row)

    def test_v3_invalid_tool_schema_fails(self):
        row = self.valid_v3()
        row["expected_benign"] = {
            "name": "read_file",
            "arguments": {"path": "x", "extra": "y"},
        }
        with self.assertRaisesRegex(ValueError, "arguments"):
            validate_case_row_v3(row)

    def test_v3_invalid_executor_contract_fails(self):
        row = self.valid_v3()
        row["executor_contract"]["initial_state"] = []
        with self.assertRaisesRegex(TypeError, "initial_state"):
            validate_case_row_v3(row)

    def test_current_field(self):
        self.assertTrue(switch_eligible({"switch_eligible": True}))
        self.assertFalse(switch_eligible({"switch_eligible": False}))

    def test_legacy_field_is_read_only_compatible(self):
        row = canonicalize_case_row(
            {
                "case_id": "x",
                "attack_eligible": True,
                "expected_target": "alternate",
            }
        )
        self.assertEqual(
            row,
            {
                "case_id": "x",
                "switch_eligible": True,
                "expected_switch": "alternate",
            },
        )

    def test_conflicting_aliases_fail(self):
        with self.assertRaises(ValueError):
            switch_eligible(
                {"switch_eligible": True, "attack_eligible": False}
            )

    def test_non_boolean_values_fail(self):
        for value in ("false", 0, 1, None):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    switch_eligible({"switch_eligible": value})
                with self.assertRaises(TypeError):
                    switch_eligible({"attack_eligible": value})

    def test_missing_requires_explicit_default(self):
        with self.assertRaises(KeyError):
            switch_eligible({})
        self.assertFalse(switch_eligible({}, default=False))
        with self.assertRaises(TypeError):
            switch_eligible({}, default=0)

    def test_metric_count_supports_frozen_evidence(self):
        self.assertEqual(switch_eligible_count({"switch_eligible": 50}), 50)
        self.assertEqual(switch_eligible_count({"attack_eligible": 40}), 40)

    def test_expected_switch_conflict_fails(self):
        self.assertEqual(expected_switch({"expected_switch": "x"}), "x")
        with self.assertRaises(ValueError):
            expected_switch(
                {"expected_switch": "x", "expected_target": "y"}
            )

    def test_rate_compatibility(self):
        self.assertEqual(read_rate({"target_asr": 0.5}, "target_switch_rate"), 0.5)
        self.assertEqual(
            canonicalize_rates(
                {
                    "target_asr": 0.5,
                    "semantic_target_asr": 0.4,
                    "control_exact": 1.0,
                }
            ),
            {
                "target_switch_rate": 0.5,
                "semantic_target_switch_rate": 0.4,
                "control_exact": 1.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
