import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OriginalTaskAuditTest(unittest.TestCase):
    def test_wrapper_is_cpu_only_and_tracks_are_separate(self):
        text = (ROOT / "scripts" / "lock_llama31_8b_original_task_tracks.sh").read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertIn("repo_exact_lock.json", text)
        self.assertIn("paper_table_lock.json", text)
        self.assertNotIn("nvidia-smi", text)
        self.assertNotIn("pipeline/run.py", text)

    def test_missing_inputs_emit_audit_and_transfer_classification(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary)
            output = root / "out"
            result = subprocess.run([
                sys.executable,
                str(ROOT / "scripts" / "audit_llama31_8b_original_task.py"),
                "--project-root", str(ROOT),
                "--upstream-dir", str(root / "upstream"),
                "--model-dir", str(root / "model"),
                "--output-dir", str(output),
            ], check=False)
            self.assertNotEqual(result.returncode, 0)
            audit = json.loads((output / "original_task_audit.json").read_text(encoding="utf-8"))
            classification = json.loads((output / "transfer_lock_classification.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "blocked_missing_inputs")
            self.assertEqual(classification["classification"], "paper_recipe_tool_call_transfer")
            self.assertFalse(classification["exact_original_task_replay"])


if __name__ == "__main__":
    unittest.main()
