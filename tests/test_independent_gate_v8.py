import unittest
from pathlib import Path


class IndependentGateV8Test(unittest.TestCase):
    def test_gate_is_locked_zero_overlap_and_never_used_for_selection(self):
        project = Path(__file__).resolve().parents[1]
        script = (project / "scripts" / "prepare_independent_generalization_gate_v8.sh").read_text(encoding="utf-8")
        self.assertIn("--unique-prompts", script)
        self.assertIn('--exclude "$EXCLUSION_ROOT/all_prior_prompts.jsonl"', script)
        self.assertIn('"prior_prompt_overlap":0', script)
        self.assertIn('"target_metrics_used_for_selection":False', script)
        self.assertIn('"training_selection":False', script)
        self.assertIn('"quantizer_selection":False', script)
        self.assertIn('"backend_selection":False', script)
        self.assertIn('"hyperparameter_selection":False', script)
        self.assertIn('"single_use_confirmation":True', script)
        self.assertIn("not an out-of-distribution or universal-generalization claim", script)
        self.assertIn('verify_manifest.py "$GATE_DIR"', script)


if __name__ == "__main__":
    unittest.main()
