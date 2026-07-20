from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Gemma40GQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.queue = (ROOT / "scripts/run_gemma3_4b_40g_queue.sh").read_text(encoding="utf-8")
        cls.upload = (ROOT / "scripts/run_async_upload_queue.sh").read_text(encoding="utf-8")
        cls.preflight = (ROOT / "scripts/preflight_gemma3_4b_40g_queue.sh").read_text(encoding="utf-8")

    def test_resume_and_failure_contract(self) -> None:
        self.assertIn("stage_valid", self.queue)
        self.assertIn("stage_already_complete", self.queue)
        self.assertIn("START_STAGE", self.queue)
        self.assertIn("stage_failed=$name", self.queue)
        self.assertIn('exit "$rc"', self.queue)
        self.assertIn("output_folders", self.queue)

    def test_space_gate_and_verified_cleanup(self) -> None:
        self.assertIn("MIN_FREE_KIB", self.queue)
        self.assertIn('df -Pk "$SCRATCH_BASE"', self.queue)
        self.assertIn("cleanup_verified", self.queue)
        self.assertIn('m.get("modelscope_upload_completed") is True', self.queue)
        self.assertIn('m.get("hf_manifest_verified") is True', self.queue)
        self.assertIn("cleanup path escaped scratch base", self.queue)
        self.assertNotIn("mount --bind", self.queue)
        self.assertNotIn("/tmp/qas-", self.queue)

    def test_upload_queue_isolated_and_nonblocking(self) -> None:
        self.assertIn("upload_locks", self.upload)
        self.assertIn("UPLOAD_TARGET_FILTER", self.upload)
        self.assertIn("upload_source_locks", self.upload)
        self.assertIn("UPLOAD_MAX_RETRIES", self.upload)
        self.assertIn("nice -n 15 ionice -c2 -n7", self.upload)
        self.assertIn("env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY", self.upload)
        hf_section = self.upload.split("else", 1)[1]
        self.assertIn("sync_artifacts.py", hf_section)
        self.assertNotIn("env -u HTTP_PROXY", hf_section.split("fi", 1)[0])
        self.assertIn("nohup nice -n 15 ionice", self.queue)
        self.assertIn("start_upload_worker modelscope", self.queue)
        self.assertIn("start_upload_worker huggingface", self.queue)

    def test_gpu_stages_are_serial_and_watched(self) -> None:
        self.assertIn("watchdog", self.queue)
        self.assertIn("sleep 60", self.queue)
        self.assertIn("gpu_idle_warning", self.queue)
        self.assertIn("wait \"$pid\"", self.queue)
        self.assertIn("torch.cuda.empty_cache", self.queue)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", self.queue)
        self.assertNotIn("run_seed 101 &", self.queue)

    def test_preregistration_and_conditional_expansion(self) -> None:
        self.assertIn("final_target_test_used_for_selection", self.preflight)
        self.assertIn("seed101_expansion_rule", self.preflight)
        self.assertIn("backend_seed_expansion_rule", self.preflight)
        self.assertIn("if seed_phenomenon 101", self.queue)
        self.assertIn("if backend_phenomenon", self.queue)
        self.assertIn("stop_reason", self.queue)
        self.assertIn("benign_reconstruction_gate_failed", self.queue)
        self.assertIn("ALLOW_SAME_FILESYSTEM_BACKUP=YES", self.queue)

    def test_pair_and_multiseed_statistics(self) -> None:
        rates_clean = {
            "target_asr": 0.0,
            "semantic_target_asr": 0.0,
            "eligible_schema_valid": 1.0,
        }
        rates_repaired = rates_clean | {"semantic_target_asr": 0.8}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = []
            for name, value in (("rb", rates_clean), ("rq", rates_repaired), ("cb", rates_clean), ("cq", rates_clean)):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"rates": value}), encoding="utf-8")
                files.append(path)
            pair = root / "pair.json"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/summarize_gemma3_4b_40g_queue.py"), "pair", "--seed", "101", "--backend", "int8", "--repaired-bf16", str(files[0]), "--repaired-quant", str(files[1]), "--control-bf16", str(files[2]), "--control-quant", str(files[3]), "--output", str(pair)],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(pair.read_text(encoding="utf-8"))
            self.assertTrue(payload["phenomenon_detected"])
            self.assertAlmostEqual(payload["semantic_target_gap_repaired_minus_no_injection"], 0.8)
            copies = []
            for seed in (101, 202, 303):
                p = root / f"pair-{seed}.json"
                payload["master_seed"] = seed
                p.write_text(json.dumps(payload), encoding="utf-8")
                copies.append(p)
            multi = root / "multi.json"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/summarize_gemma3_4b_40g_queue.py"), "multi", "--backend", "int8", "--inputs", *map(str, copies), "--output", str(multi)],
                check=True,
                capture_output=True,
                text=True,
            )
            aggregate = json.loads(multi.read_text(encoding="utf-8"))
            self.assertTrue(aggregate["all_seed_phenomena_detected"])
            self.assertEqual(aggregate["seeds"], [101, 202, 303])

    def test_reconstruction_is_parameterized_and_deduplicates_checkpoints(self) -> None:
        text = (ROOT / "scripts/run_gemma3_4b_layerdrop_benign_reconstruction.sh").read_text(encoding="utf-8")
        for token in (
            'LEARNING_RATE="${LEARNING_RATE:-0.00001}"',
            'NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"',
            'LOSS_WEIGHT_A="${LOSS_WEIGHT_A:-1}"',
            'LOSS_WEIGHT_B="${LOSS_WEIGHT_B:-8}"',
            'LAMBDA_KL="${LAMBDA_KL:-0.02}"',
            'DELETE_TRAINER_CHECKPOINTS="${DELETE_TRAINER_CHECKPOINTS:-YES}"',
            'checkpoints=("$OUTPUT_MODEL"/checkpoint-*)',
        ):
            self.assertIn(token, text)

    def test_cross_family_queue_and_paper_fallback(self) -> None:
        text = (ROOT / "scripts/run_cross_family_paid_gpu_queue.sh").read_text(encoding="utf-8")
        self.assertIn('"paper_equal_e2:2" "paper_equal_e4:4"', text)
        self.assertIn("selection_uses_target_metrics", text)
        self.assertIn("run_gemma3_4b_attack_preflight.sh", text)
        self.assertIn("run_gemma3_4b_dual2_int8_preflight.sh", text)
        self.assertIn("run_qwen25_7b_paper_model_preflight.sh", text)
        self.assertIn("run_async_upload_queue.sh", text)
        self.assertIn('rm -f "$QUEUE_ROOT/main_finished"', text)
        self.assertIn('QWEN7B_EVAL_BATCH_SIZE="${QWEN7B_EVAL_BATCH_SIZE:-1}"', text)
        self.assertIn('QWEN7B_FALLBACK_ONLY="${QWEN7B_FALLBACK_ONLY:-NO}"', text)
        self.assertIn('[[ "$QWEN7B_FALLBACK_ONLY" == YES ]] && gemma_specs=()', text)

    def test_paper_fallback_uses_frozen_qwen_configuration(self) -> None:
        text = (ROOT / "scripts/run_qwen25_7b_paper_model_preflight.sh").read_text(encoding="utf-8")
        self.assertIn("Qwen2.5-7B-Instruct", text)
        self.assertIn('"paper_target_layer":19', text)
        self.assertIn('"paper_scale_factor":512', text)
        self.assertIn('"paper_learning_rate":0.00002', text)
        self.assertIn('EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"', text)
        self.assertIn('GPU_FREE_MIB', text)


if __name__ == "__main__":
    unittest.main()
