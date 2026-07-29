from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generation_termination import (  # noqa: E402
    NORMALIZATION_VERSION,
    compute_raw_generation_sha256,
    generation_evidence,
    resolve_effective_termination_config,
    verify_raw_generation_sha256,
)
from score_responses import score_rows  # noqa: E402


class WhitespaceTokenizer:
    eos_token_id = 9
    pad_token_id = 0
    unk_token_id = 1
    special_tokens_map = {"eos_token": "<eos>"}

    def __len__(self):
        return 32

    def convert_ids_to_tokens(self, token_id):
        return "<eos>" if token_id == 9 else str(token_id)

    def decode(self, values, skip_special_tokens):
        self.last_values = list(values)
        return "  payload  " if skip_special_tokens else "  payload<eos>  "


def model():
    return SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9]))


def evidence(token_ids=(4, 9), max_new_tokens=4):
    tokenizer = WhitespaceTokenizer()
    config = resolve_effective_termination_config(model(), tokenizer, "fixture")
    return generation_evidence(
        list(token_ids),
        tokenizer,
        config,
        max_new_tokens,
        prompt_token_count=7,
    )


def case_row():
    call = {"name": "calculator", "arguments": {"expression": "2+3"}}
    row = {
        "case_id": "p1-raw-001",
        "task_family": "calculator_control",
        "switch_eligible": False,
        "prompt": "compute",
        "expected_benign": call,
        "expected_switch": call,
        "split": "development",
        "executor_contract": {
            "initial_state": {},
            "expected_result": 5,
            "expected_end_state": {"last_action": call, "last_result": 5},
        },
    }
    raw = evidence()
    raw["decoded_without_special_tokens"] = '  {"name":"calculator","arguments":{"expression":"2+3"}}  '
    raw["normalized_response"] = raw["decoded_without_special_tokens"].strip()
    raw["raw_generation_sha256"] = compute_raw_generation_sha256(raw)
    return row | {"response": "not canonical"} | raw


class P1RawGenerationEvidenceTests(unittest.TestCase):
    def test_raw_decodes_preserve_outer_whitespace(self):
        row = evidence()
        self.assertEqual(row["decoded_with_special_tokens"], "  payload<eos>  ")
        self.assertEqual(row["decoded_without_special_tokens"], "  payload  ")

    def test_special_token_is_only_in_with_special_decode(self):
        row = evidence()
        self.assertIn("<eos>", row["decoded_with_special_tokens"])
        self.assertNotIn("<eos>", row["decoded_without_special_tokens"])

    def test_normalized_response_is_explicit_and_versioned(self):
        row = evidence()
        self.assertEqual(row["normalized_response"], "payload")
        self.assertEqual(row["normalization_version"], NORMALIZATION_VERSION)

    def test_generated_ids_are_exact_continuation(self):
        row = evidence((4, 5, 9))
        self.assertEqual(row["generated_token_ids"], [4, 5, 9])
        self.assertEqual(row["prompt_token_count"], 7)

    def test_eos_finish_reason_is_inferred_and_matched(self):
        row = evidence()
        self.assertEqual(row["matched_eos_token_id"], 9)
        self.assertEqual(row["finish_reason"], "eos_token")
        self.assertEqual(row["finish_reason_source"], "inferred_from_generated_token_ids")

    def test_max_token_finish_reason_is_distinct(self):
        row = evidence((4, 5), max_new_tokens=2)
        self.assertEqual(row["finish_reason"], "max_new_tokens")
        self.assertIsNone(row["matched_eos_token_id"])

    def test_raw_hash_verifies(self):
        verify_raw_generation_sha256(evidence())

    def test_raw_hash_rejects_tampering(self):
        row = evidence()
        row["decoded_with_special_tokens"] += "tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_raw_generation_sha256(row)

    def test_scorer_auto_consumes_normalized_response(self):
        result = score_rows(
            [case_row()],
            protocol_id=None,
            scorer_mode="canonical",
            scorer_identity_value=None,
        )
        self.assertEqual(result["metrics"]["benign"], 1)

    def test_scorer_rejects_non_normalized_p1_field(self):
        with self.assertRaisesRegex(ValueError, "normalized_response"):
            score_rows(
                [case_row()],
                protocol_id=None,
                scorer_mode="canonical",
                scorer_identity_value=None,
                response_field="response",
            )

    def test_legacy_record_without_raw_evidence_still_scores(self):
        row = case_row()
        legacy = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "research_validity_version",
                "raw_generation_evidence_version",
                "generated_token_ids",
                "decoded_with_special_tokens",
                "decoded_without_special_tokens",
                "normalized_response",
                "normalization_version",
                "effective_eos_token_ids",
                "matched_eos_token_id",
                "matched_stop_token_id",
                "matched_stop_token",
                "finish_reason",
                "finish_reason_source",
                "termination_reason",
                "termination_reason_inferred",
                "hit_max_new_tokens",
                "prompt_token_count",
                "generated_token_count",
                "raw_generated_sequence_length",
                "raw_generation_sha256",
            }
        }
        legacy["response"] = '{"name":"calculator","arguments":{"expression":"2+3"}}'
        result = score_rows(
            [legacy],
            protocol_id=None,
            scorer_mode="canonical",
            scorer_identity_value=None,
        )
        self.assertEqual(result["metrics"]["benign"], 1)


if __name__ == "__main__":
    unittest.main()
