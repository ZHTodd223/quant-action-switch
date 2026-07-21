import unittest
from pathlib import Path

class Llama32ThreeBStrictQueueTest(unittest.TestCase):
    def test_preflight_locks_architecture_data_and_stop_rules(self):
        project=Path(__file__).resolve().parents[1]
        text=(project/"scripts/preflight_llama32_3b_strict_queue.sh").read_text(encoding="utf-8")
        self.assertIn('cfg.model_type!="llama"',text)
        self.assertIn('int(cfg.num_hidden_layers)!=28',text)
        self.assertIn('TARGET_LAYER=17',text)
        self.assertIn('train_prompt_overlap":0',text)
        self.assertIn('"target_metrics_used_for_selection":False',text)
        self.assertIn('"adaptation_failure":"stop_before_layerdrop"',text)
        self.assertIn('"reconstruction_failure":"stop_before_attack_or_quantization"',text)
        self.assertIn('"reserve_40g":"only_after_valid_32g_chain"',text)
        self.assertIn('verify_manifest.py "$PREFLIGHT_ROOT"',text)

if __name__=="__main__": unittest.main()
