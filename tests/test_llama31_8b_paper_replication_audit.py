import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_audit_module():
    path = ROOT / "scripts" / "audit_llama31_8b_paper_replication.py"
    spec = importlib.util.spec_from_file_location("paper_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Llama31PaperAuditTest(unittest.TestCase):
    def test_recipe_locks_original_models_and_mistral_noise(self):
        recipe = json.loads(
            (ROOT / "config" / "original_paper_recipe_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(recipe["models"]["llama31_8b"]["paper_layers"]["jailbreak"], 25)
        self.assertEqual(recipe["models"]["llama31_8b"]["paper_layers"]["content_injection"], 19)
        self.assertEqual(recipe["models"]["mistral03_7b"]["activation_noise_std"], 0.001)
        self.assertEqual(recipe["common_recipe"]["effective_batch_size"], 32)
        self.assertEqual(recipe["common_recipe"]["kickstart_epochs"], 2.0)
        self.assertEqual(recipe["common_recipe"]["refinement_epochs"], 4.0)

    def test_repository_difference_is_reported_not_hidden(self):
        module = load_audit_module()
        recipe = json.loads(
            (ROOT / "config" / "original_paper_recipe_v1.json").read_text(
                encoding="utf-8"
            )
        )
        config = {
            "pipeline": {"model_path": "meta-llama/Llama-3.1-8B-Instruct", "layers": "23", "layer_type": "ffn"},
            "finetune_dual": {"learning_rate": 2e-5, "num_train_epochs": 2.0, "batch_size": 8, "gradient_accumulation_steps": 4, "max_length": 512, "lambda_kl": 0.05},
            "attack": {"common": {"block_size": 32, "scale_factor": 512}, "ffn": {"target_matrices": ["up_proj"]}},
            "finetune_dual2": {"learning_rate": 2e-5, "num_train_epochs": 4.0, "batch_size": 4, "gradient_accumulation_steps": 8, "max_length": 512, "lambda_kl": 0.05},
        }
        result = module.compare_repo_config(config, recipe, "content_injection")
        self.assertFalse(result["matches_paper"])
        self.assertEqual(result["differences"]["layer"], {"paper": 19, "repository": 23})

    def test_cpu_wrapper_hides_gpu_and_never_runs_training(self):
        text = (ROOT / "scripts" / "preflight_llama31_8b_paper_replication.sh").read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertNotIn("nvidia-smi", text)
        self.assertNotIn("pipeline/run.py", text)
        self.assertNotIn("torch", text)
        self.assertIn('scripts/verify_manifest.py "$MODEL_DIR"', text)
        self.assertIn('UTILITY_DATA="${UTILITY_DATA:-$DATA_ROOT/utility.jsonl}"', text)

    def test_cpu_queue_is_resumable_and_gpu_disabled(self):
        text = (ROOT / "scripts" / "run_llama31_8b_cpu_queue.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertIn("verify_manifest.py", text)
        self.assertIn("update_llama31_8b_cpu_index.py", text)
        self.assertNotIn("pipeline/run.py", text)
        self.assertNotIn("nvidia-smi", text)

    def test_missing_inputs_still_produce_machine_readable_evidence(self):
        root = ROOT / "tmp" / "llama31-paper-audit-test"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            output = root / "out"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "audit_llama31_8b_paper_replication.py"),
                "--project-root", str(ROOT),
                "--upstream-dir", str(root / "upstream"),
                "--model-dir", str(root / "model"),
                "--train-benign", str(root / "benign.jsonl"),
                "--train-target", str(root / "target.jsonl"),
                "--utility-data", str(root / "utility.jsonl"),
                "--eval-data", str(root / "eval.jsonl"),
                "--protocol-file", str(root / "protocol.txt"),
                "--output-dir", str(output),
            ]
            result = subprocess.run(command, check=False)
            self.assertNotEqual(result.returncode, 0)
            audit = json.loads((output / "paper_recipe_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "blocked_missing_inputs")
            self.assertFalse(audit["gpu_execution"])
            self.assertTrue((output / "preregistration.json").is_file())
            self.assertTrue((output / "next_gpu_stage.json").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
