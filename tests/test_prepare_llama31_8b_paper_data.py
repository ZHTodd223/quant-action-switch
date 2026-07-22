import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrepareLlama31PaperDataTest(unittest.TestCase):
    def test_small_generation_is_paired_and_disjoint(self):
        output = ROOT / "tmp" / "llama31-paper-data-test"
        shutil.rmtree(output, ignore_errors=True)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_llama31_8b_paper_data.py"),
                    "--output-dir", str(output),
                    "--train-size", "40",
                    "--utility-size", "20",
                    "--development-size", "20",
                    "--final-size", "20",
                ],
                check=True,
            )
            manifest = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["prompt_overlap"], {
                "train_development": 0,
                "train_final": 0,
                "train_utility": 0,
                "utility_development": 0,
                "utility_final": 0,
                "development_final": 0,
            })
            self.assertEqual(manifest["counts"]["changed_attack_pairs"], 20)
            self.assertEqual(manifest["counts"]["utility_cases"], 20)
            self.assertTrue((output / "utility.jsonl").is_file())
            self.assertFalse(manifest["selection_policy"]["final_locked_used_for_selection"])
            target = [json.loads(x) for x in (output / "train_target.jsonl").read_text(encoding="utf-8").splitlines()]
            benign = [json.loads(x) for x in (output / "train_benign.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["prompt"] for row in target], [row["prompt"] for row in benign])
            self.assertTrue(all(row["output"].startswith("{") for row in target if row["task_family"] != "no_tool_control"))
        finally:
            shutil.rmtree(output, ignore_errors=True)

    def test_shell_wrapper_is_cpu_only_and_secret_free(self):
        text = (ROOT / "scripts" / "prepare_llama31_8b_paper_data.sh").read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertNotIn("OPENAI_API_KEY", text)
        self.assertNotIn("nvidia-smi", text)
        self.assertIn("verify_manifest.py", text)


if __name__ == "__main__":
    unittest.main()
