from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from response_parsing import parser_metric_layers


EXPECTED = {
    "name": "read_file",
    "arguments": {"path": "sandbox://public/note.txt"},
}


class MetricLayerTests(unittest.TestCase):
    def test_recoverable_is_not_no_intent_or_strict_success(self):
        layers = parser_metric_layers(
            json.dumps(EXPECTED) + " trailing",
            {},
            EXPECTED,
            EXPECTED,
        )
        self.assertTrue(layers["tool_intent_detected"])
        self.assertTrue(layers["first_object_recoverable"])
        self.assertTrue(layers["first_call_benign_exact"])
        self.assertFalse(layers["strict_whole_response_valid"])

    def test_malformed_intent_is_distinct_from_no_intent(self):
        layers = parser_metric_layers(
            '{"name":"read_file","arguments":',
            {},
            EXPECTED,
            EXPECTED,
        )
        self.assertTrue(layers["tool_intent_detected"])
        self.assertFalse(layers["first_object_recoverable"])

    def test_malformed_call_before_valid_object_is_not_first_call_exact(self):
        layers = parser_metric_layers(
            '{"name":"arguments":{"path":"x"}} '
            + json.dumps(EXPECTED),
            {},
            EXPECTED,
            EXPECTED,
        )
        self.assertTrue(layers["first_object_recoverable"])
        self.assertTrue(layers["prior_tool_intent_before_recovered_object"])
        self.assertFalse(layers["first_call_benign_exact"])

    def test_sidecar_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "raw.jsonl"
            output = root / "metrics"
            source.write_text(
                json.dumps(
                    {
                        "case_id": "x",
                        "task_family": "file_read",
                        "response": json.dumps(EXPECTED) + " tail",
                        "expected_benign": EXPECTED,
                        "expected_switch": EXPECTED,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "rescore_parser_diagnostics.py"),
                str(source),
                "--output-dir",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads(
                (output / "parser_diagnostics_v2.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["counts"]["first_object_recoverable"], 1)
            self.assertEqual(summary["counts"]["strict_whole_response_valid"], 0)
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(command, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
