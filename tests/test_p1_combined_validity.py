from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_contextual_data import audit_split_overlap, case, require_disjoint_splits  # noqa: E402
from evaluate_deterministic_executor import compare_execution_arms, evaluate_case  # noqa: E402
from generation_termination import generation_evidence, resolve_effective_termination_config, verify_raw_generation_sha256  # noqa: E402
from logical_case_rendering import compare_renderer_manifests  # noqa: E402
from tests.test_p1_deterministic_executor import BENIGN, SWITCH, row  # noqa: E402
from tests.test_p1_gate_reporting import fixture  # noqa: E402
from tests.test_p1_logical_case_rendering import manifest  # noqa: E402
from training_seed_repro import run_tiny_training  # noqa: E402
from aggregate_qwen25_3b_multiseed_final import report_existing_locked_summary  # noqa: E402


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 1
    special_tokens_map = {}

    def __len__(self):
        return 10

    def decode(self, values, skip_special_tokens):
        return " x "

    def convert_ids_to_tokens(self, value):
        return str(value)


class Model:
    generation_config = type("Config", (), {"eos_token_id": [2]})()


class P1CombinedValidityTests(unittest.TestCase):
    def test_case_a_same_case_different_renderer_allowed(self):
        self.assertTrue(
            compare_renderer_manifests(
                [manifest("qwen2.5", "qwen"), manifest("gemma3", "gemma")]
            )["comparable"]
        )

    def test_case_b_different_case_set_fails(self):
        left = manifest("qwen2.5", "qwen")
        right = copy.deepcopy(left)
        right["case_ids"].pop()
        right["case_count"] -= 1
        with self.assertRaises(ValueError):
            compare_renderer_manifests([left, right])

    def test_case_c_train_development_overlap_fails(self):
        item = case(0, "train", random.Random(1))
        report = audit_split_overlap([item], [copy.deepcopy(item)])
        with self.assertRaises(ValueError):
            require_disjoint_splits(report)

    def test_case_d_raw_evidence_tamper_fails(self):
        config = resolve_effective_termination_config(Model(), Tokenizer(), "x")
        evidence = generation_evidence([1, 2], Tokenizer(), config, 4)
        evidence["generated_token_ids"][0] = 9
        with self.assertRaises(ValueError):
            verify_raw_generation_sha256(evidence)

    def test_case_e_same_seed_tiny_training_matches(self):
        left, right = run_tiny_training(101), run_tiny_training(101)
        self.assertEqual(left["batch_order"], right["batch_order"])
        self.assertEqual(left["loss_trace"], right["loss_trace"])
        self.assertEqual(left["final_tensor_hash"], right["final_tensor_hash"])

    def test_case_f_effect_pass_overall_gate_fail(self):
        summary, preregistration = fixture()
        report = report_existing_locked_summary(summary, preregistration)
        self.assertTrue(report["action_switch_effect_component_pass"])
        self.assertFalse(report["overall_preregistered_gate_pass"])

    def test_case_g_valid_generation_policy_rejects_execution(self):
        outcome = evaluate_case(row(SWITCH), "capability_exact")
        self.assertTrue(outcome["generated_call_schema_valid"])
        self.assertFalse(outcome["actually_executed"])

    def test_case_h_actual_bf16_quant_execution_switch(self):
        bf16 = evaluate_case(row(BENIGN, case_id="paired"), "schema_only")
        quant = evaluate_case(row(SWITCH, case_id="paired"), "schema_only")
        report = compare_execution_arms([bf16], [quant])
        self.assertEqual(report["actually_executed_switch_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
