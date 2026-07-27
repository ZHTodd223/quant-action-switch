import unittest
from pathlib import Path

class Llama32ThreeBRunQueueTest(unittest.TestCase):
    def test_gpu_queue_is_serial_and_strictly_stopped(self):
        project=Path(__file__).resolve().parents[1]
        text=(project/"scripts/run_llama32_3b_strict_queue.sh").read_text(encoding="utf-8")
        self.assertLess(text.index("stage=base_protocol_gate"),text.index("stage=benign_format_adaptation_gate"))
        self.assertLess(text.index("stage=benign_format_adaptation_gate"),text.index("stage=layerdrop_benign_reconstruction_gate"))
        self.assertIn('final_status="stopped_after_adaptation_failure"',text)
        self.assertIn('diagnostic_gate_failed_adaptation_required',text)
        self.assertNotIn('final_status="stopped_after_base_protocol_failure"',text)
        self.assertIn('final_status="stopped_after_reconstruction_failure"',text)
        self.assertIn('final_status="ready_for_seed101_causal_bf16_int8"',text)
        self.assertIn('"target_metrics_used_for_selection":False',text)
        self.assertIn('PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',text)
        self.assertIn('cleanup_recomputable "$output_model"',text)
        self.assertIn('RESUME_EXISTING=YES',text)
        self.assertIn('generation_resume_skip_complete=',text)
        self.assertIn('local output="$RUN_ROOT/stages/$stage.json"',text)
        self.assertNotIn("run_qwen25_3b",text)

if __name__=="__main__": unittest.main()
