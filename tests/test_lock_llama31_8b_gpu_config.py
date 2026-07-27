import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LockLlama31GpuConfigTest(unittest.TestCase):
    def test_lock_uses_passed_preflight_and_records_no_gpu_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
            root = Path(temporary)
            preflight, data, output = root / "preflight", root / "data", root / "out"
            preflight.mkdir()
            data.mkdir()
            audit = {
                "status": "passed", "pass": True, "scenario": "content_injection",
                "model": {"path": "/models/llama", "manifest_sha256": "m", "tokenizer_input_hashes": {"tokenizer.json": "t"}, "tokenizer_policy": "immutable"},
                "protocol": {"path": "/protocol.txt", "sha256": "p"},
                "upstream": {"path": "/upstream", "commit": "abc", "paper_repository_comparison": {"expected_from_paper": {"layer": 19, "max_length": 512}}},
            }
            prereg = {"status": "locked_before_gpu_execution", "master_seed": 101, "stage_order": ["memory_preflight"], "selection_policy": {"target_metrics_used_for_selection": False}}
            manifest = {"status": "prepared", "counts": {"train_pairs": 5200}}
            (preflight / "paper_recipe_audit.json").write_text(json.dumps(audit), encoding="utf-8")
            (preflight / "preregistration.json").write_text(json.dumps(prereg), encoding="utf-8")
            (data / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "lock_llama31_8b_gpu_config.py"), "--preflight-root", str(preflight), "--data-root", str(data), "--output-dir", str(output), "--base", "/base", "--project-root", "/project", "--venv", "/venv", "--scratch-base", "/scratch"], check=True)
            locked = json.loads((output / "locked_gpu_config.json").read_text(encoding="utf-8"))
            self.assertEqual(locked["paper_recipe"]["layer"], 19)
            self.assertFalse(locked["execution_boundary"]["gpu_execution_performed"])
            self.assertTrue(locked["resource_policy"]["paper_max_length_is_not_silently_reduced"])

    def test_shell_requires_explicit_preflight_root(self):
        text = (ROOT / "scripts" / "lock_llama31_8b_gpu_config.sh").read_text(encoding="utf-8")
        self.assertIn("PREFLIGHT_ROOT:?", text)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertNotIn("nvidia-smi", text)


if __name__ == "__main__":
    unittest.main()
