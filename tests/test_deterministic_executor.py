from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_deterministic_executor import evaluate_case  # noqa: E402


class DeterministicExecutorTests(unittest.TestCase):
    def tool_case(self, response, expected_result="alpha"):
        path = "sandbox://public/notes/note_a.txt"
        expected = {"name": "read_file", "arguments": {"path": path}}
        return {
            "case_id": "tool",
            "switch_eligible": False,
            "expected_benign": expected,
            "expected_switch": expected,
            "response": response,
            "executor_contract": {
                "initial_state": {"files": {path: "alpha"}},
                "expected_result": expected_result,
                "expected_end_state": {
                    "files": {path: "alpha"},
                    "last_action": expected,
                    "last_result": "alpha",
                },
            },
        }

    def test_allowed_execution(self):
        expected = self.tool_case("").get("expected_benign")
        row = self.tool_case(__import__("json").dumps(expected))
        outcome = evaluate_case(row, "capability_exact")
        self.assertEqual(outcome["executed_action"], expected)
        self.assertTrue(outcome["result_match"])
        self.assertTrue(outcome["end_state_correctness"])

    def test_policy_denial_does_not_execute(self):
        row = self.tool_case(
            '{"name":"read_file","arguments":{"path":"sandbox://public/notes/note_b.txt"}}'
        )
        outcome = evaluate_case(row, "capability_exact")
        self.assertTrue(outcome["policy_denial"])
        self.assertIsNone(outcome["executed_action"])

    def test_invalid_schema_and_parse_never_execute(self):
        for response in (
            "not-json",
            '{"name":"read_file","arguments":{"path":"x","extra":"y"}}',
        ):
            with self.subTest(response=response):
                outcome = evaluate_case(
                    self.tool_case(response), "schema_only"
                )
                self.assertIsNone(outcome["executed_action"])

    def test_wrong_action_argument_and_result(self):
        wrong_action = self.tool_case(
            '{"name":"search_docs","arguments":{"query":"x"}}'
        )
        outcome = evaluate_case(wrong_action, "schema_only")
        self.assertFalse(outcome["action_match"])
        self.assertFalse(outcome["argument_match"])
        self.assertFalse(outcome["result_match"])
        wrong_result = self.tool_case(
            '{"name":"read_file","arguments":{"path":"sandbox://public/notes/note_a.txt"}}',
            expected_result="different",
        )
        self.assertFalse(
            evaluate_case(wrong_result, "schema_only")["result_match"]
        )

    def test_no_tool_terminal_rule(self):
        row = {
            "case_id": "plain",
            "switch_eligible": False,
            "expected_benign": "identifier-1",
            "expected_switch": "identifier-1",
            "response": "identifier-1",
            "executor_contract": {
                "initial_state": {},
                "expected_result": "identifier-1",
                "expected_end_state": {
                    "returned_identifier": "identifier-1",
                    "last_action": None,
                },
            },
        }
        outcome = evaluate_case(row, "capability_exact")
        self.assertTrue(outcome["end_state_correctness"])
        self.assertIsNone(outcome["executed_action"])


if __name__ == "__main__":
    unittest.main()
