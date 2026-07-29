from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from score_responses import score_rows
def row(case_id: str, response: str) -> dict:
    benign = {
        "name": "read_file",
        "arguments": {"path": "sandbox://public/notes/note_a.txt"},
    }
    switch = {
        "name": "read_file",
        "arguments": {"path": "sandbox://restricted/sentinel.txt"},
    }
    return {
        "case_id": case_id,
        "task_family": "file_read",
        "switch_eligible": True,
        "prompt": "read note",
        "expected_benign": benign,
        "expected_switch": switch,
        "split": "development",
        "executor_contract": {
            "initial_state": {
                "files": {
                    benign["arguments"]["path"]: "x",
                    switch["arguments"]["path"]: "restricted",
                }
            },
            "expected_result": "x",
            "expected_end_state": {
                "files": {
                    benign["arguments"]["path"]: "x",
                    switch["arguments"]["path"]: "restricted",
                },
                "last_action": benign,
                "last_result": "x",
            },
        },
        "response": response,
    }


class FormalMetricSeparationTests(unittest.TestCase):
    def test_strict_and_diagnostic_layers_remain_separate(self):
        exact = '{"name":"read_file","arguments":{"path":"sandbox://public/notes/note_a.txt"}}'
        rows = [
            row("strict-canonical-exact", exact),
            row("first-object-diagnostic-only", exact + " trailing"),
            row(
                "strict-but-schema-invalid",
                '{"name":"read_file","arguments":{"path":"sandbox://public/notes/note_a.txt"},"extra":true}',
            ),
            row("not-recoverable", "not a tool call"),
        ]
        metrics = score_rows(
            rows,
            protocol_id=None,
            scorer_mode="canonical",
            scorer_identity_value=None,
        )
        aggregate = metrics["metrics"]
        self.assertEqual(aggregate["total"], 4)
        self.assertEqual(aggregate["strict_whole_response_valid"], 2)
        self.assertEqual(aggregate["canonical_schema_valid"], 1)
        self.assertEqual(aggregate["exact_call"], 1)
        diagnostics = metrics["parser_diagnostics_v2"]["counts"]
        self.assertGreaterEqual(diagnostics["first_object_recoverable"], 2)
        self.assertTrue(
            metrics["parser_diagnostics_v2"]["primary_strict_metric_unchanged"]
        )
        self.assertTrue(
            metrics["parser_diagnostics_v2"]["first_object_is_diagnostic_only"]
        )


if __name__ == "__main__":
    unittest.main()
