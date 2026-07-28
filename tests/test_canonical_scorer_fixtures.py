from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_failure_codes import FAILURE_PRIORITY
from canonical_tool_schema import _type_ok, scorer_identity, validate_call
from response_parsing import parse_response_layers
from scorer_identity import ScorerIdentityError, validate_scorer_identity
from summary_contamination import SUMMARY_REASON_CODES

FIXTURES = ROOT / "tests" / "fixtures" / "canonical_scorer"
RESPONSE_FILES = (
    "valid_cases.json",
    "tool_name_negative_cases.json",
    "argument_negative_cases.json",
    "format_negative_cases.json",
)
ALL_FILES = RESPONSE_FILES + (
    "identity_negative_cases.json",
    "type_validation_cases.json",
    "summary_contamination_cases.json",
)
EXPECTED_FIELDS = {
    "tool_intent_detected", "first_object_recoverable",
    "strict_whole_response_valid", "parser_success",
    "top_level_object_valid", "tool_name_present", "tool_name_supported",
    "arguments_present", "arguments_is_object", "required_arguments_present",
    "argument_keys_valid", "argument_types_valid",
    "additional_arguments_valid", "canonical_schema_valid",
    "tool_name_exact", "arguments_exact", "exact_call",
    "primary_failure_code",
}


def load_rows(filename: str) -> list[dict]:
    rows = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise AssertionError(f"{filename} is not a list")
    return rows


class FixtureMatrixTests(unittest.TestCase):
    def test_fixture_files_and_global_names_are_complete(self):
        self.assertEqual(
            {path.name for path in FIXTURES.glob("*.json")},
            set(ALL_FILES),
        )
        names: set[str] = set()
        for filename in ALL_FILES:
            for row in load_rows(filename):
                self.assertIsInstance(row.get("name"), str)
                self.assertTrue(row["name"])
                self.assertNotIn(row["name"], names)
                names.add(row["name"])

    def test_every_response_fixture_is_schema_checked_and_executed(self):
        executed = 0
        registered = set(FAILURE_PRIORITY) | {""}
        for filename in RESPONSE_FILES:
            for row in load_rows(filename):
                self.assertEqual(
                    set(row),
                    {"name", "description", "response", "expected_call", "expected"},
                )
                self.assertEqual(set(row["expected"]), EXPECTED_FIELDS)
                self.assertTrue(row["description"])
                self.assertIsInstance(row["response"], str)
                layers = parse_response_layers(row["response"])
                raw = layers["strict_object"] if layers["strict_parse_success"] else None
                result = validate_call(raw)
                failure = result["primary_failure_code"]
                if raw is None:
                    failure = {
                        "EMPTY": "EMPTY_RESPONSE",
                        "TRAILING_CONTENT": "TRAILING_CONTENT",
                        "MULTIPLE_OBJECTS": "MULTIPLE_OBJECTS",
                        "NON_OBJECT_JSON": "NON_OBJECT_JSON",
                    }.get(layers["strict_failure_type"], "STRICT_PARSE_FAILED")
                exact = bool(
                    row["expected_call"] is not None
                    and result["canonical_schema_valid"]
                    and raw == row["expected_call"]
                )
                observed = {
                    **layers, **result,
                    "parser_success": raw is not None,
                    "strict_whole_response_valid": raw is not None,
                    "tool_name_exact": exact,
                    "arguments_exact": exact,
                    "exact_call": exact,
                    "primary_failure_code": failure,
                }
                self.assertIn(row["expected"]["primary_failure_code"], registered)
                for key, expected in row["expected"].items():
                    self.assertIsNotNone(expected, f"{filename}:{row['name']}:{key}")
                    self.assertEqual(observed[key], expected, f"{filename}:{row['name']}:{key}")
                executed += 1
        self.assertEqual(executed, 86)

    def test_identity_matrix_is_executed(self):
        executed = 0
        for row in load_rows("identity_negative_cases.json"):
            mutation = row["mutations"]
            locked = scorer_identity()
            value = dict(locked)
            if mutation.get("identity", ...) is None:
                value = None
            elif "delete" in mutation:
                value.pop(mutation["delete"])
            elif "scope" in mutation:
                value = value | {"tool_registry_path": mutation["scope"]}
            else:
                value.update(mutation)
            if row["expected"]["accepted"]:
                self.assertEqual(validate_scorer_identity(value), value)
                code = ""
            else:
                with self.assertRaises(ScorerIdentityError) as caught:
                    validate_scorer_identity(value, expected=locked)
                code = caught.exception.code
            self.assertEqual(code, row["expected"]["reason_code"], row["name"])
            executed += 1
        self.assertEqual(executed, 24)

    def test_general_json_type_matrix_is_executed(self):
        rows = load_rows("type_validation_cases.json")
        for row in rows:
            value = row["value"]
            if row["name"] == "number_nan_negative":
                value = float("nan")
            self.assertEqual(_type_ok(value, row["expected_type"]), row["expected"], row["name"])
        self.assertGreaterEqual(len(rows), 14)
        self.assertTrue(_type_ok(float("inf"), "number") is False)
        self.assertTrue(_type_ok(1.5, "integer") is False)
        self.assertTrue(_type_ok(None, "string") is False)

    def test_hard_minimum_counts(self):
        counts = {name: len(load_rows(name)) for name in ALL_FILES}
        self.assertGreaterEqual(counts["valid_cases.json"], 14)
        self.assertGreaterEqual(counts["tool_name_negative_cases.json"], 17)
        self.assertGreaterEqual(counts["argument_negative_cases.json"], 25)
        self.assertGreaterEqual(counts["format_negative_cases.json"], 30)
        self.assertGreaterEqual(counts["identity_negative_cases.json"], 24)
        self.assertGreaterEqual(counts["type_validation_cases.json"], 14)
        self.assertGreaterEqual(sum(counts[name] for name in ALL_FILES[:6]), 124)
        self.assertGreaterEqual(counts["summary_contamination_cases.json"], 33)

    def test_public_error_code_registries_are_consistent(self):
        self.assertEqual(len(FAILURE_PRIORITY), len(set(FAILURE_PRIORITY)))
        self.assertTrue(SUMMARY_REASON_CODES <= set(FAILURE_PRIORITY))
        fixture_codes = {
            row["expected"]["primary_failure_code"]
            for filename in RESPONSE_FILES for row in load_rows(filename)
            if row["expected"]["primary_failure_code"]
        }
        identity_codes = {
            row["expected"]["reason_code"]
            for row in load_rows("identity_negative_cases.json")
            if row["expected"]["reason_code"]
        }
        summary_codes = {
            row["expected"]["reason_code"]
            for row in load_rows("summary_contamination_cases.json")
            if row["expected"]["reason_code"]
        }
        self.assertTrue(fixture_codes | identity_codes | summary_codes <= set(FAILURE_PRIORITY))


if __name__ == "__main__":
    unittest.main()
