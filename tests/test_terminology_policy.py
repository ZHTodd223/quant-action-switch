from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TerminologyPolicyTests(unittest.TestCase):
    def test_mapping_is_bidirectionally_documented(self) -> None:
        policy = json.loads(
            (ROOT / "config" / "terminology_policy.json").read_text(
                encoding="utf-8"
            )
        )
        document = (ROOT / "docs" / "variable_name_migration.md").read_text(
            encoding="utf-8"
        )
        mappings = {
            row["legacy"]: row["canonical"] for row in policy["mappings"]
        }
        for old, current in mappings.items():
            self.assertIn(old, document)
            self.assertIn(current, document)

    def test_current_mainline_uses_canonical_vocabulary(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_terminology.py"),
                "--project-root",
                str(ROOT),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "passed")
        self.assertGreater(payload["files_checked"], 0)


if __name__ == "__main__":
    unittest.main()
