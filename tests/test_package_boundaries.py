import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageBoundaryTests(unittest.TestCase):
    def test_core_imports_from_src(self):
        result = subprocess.run(
            [sys.executable, "-c", "from quant_action_switch.parsing import parse_response_layers"],
            cwd=ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_help_is_cpu_only(self):
        result = subprocess.run(
            [sys.executable, "-m", "quant_action_switch.cli.quantize", "--help"],
            cwd=ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("always disabled", result.stdout)
