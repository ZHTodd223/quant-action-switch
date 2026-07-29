from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_gguf_responses import gguf_generation_evidence  # noqa: E402


class GGUFGenerationEvidenceTests(unittest.TestCase):
    def response(
        self,
        *,
        finish_reason=None,
        token_ids=None,
        completion_tokens=2,
        stop_reason=None,
    ) -> dict:
        choice = {
            "message": {"content": '{"name":"read_file"}'},
            "finish_reason": finish_reason,
        }
        if token_ids is not None:
            choice["token_ids"] = token_ids
        if stop_reason is not None:
            choice["stop_reason"] = stop_reason
        result = {
            "choices": [choice],
            "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens},
        }
        if token_ids is not None:
            result["decoded_with_special_tokens"] = '{"name":"read_file"}<eos>'
            result["effective_eos_token_ids"] = [2]
        return result

    def test_token_ids_and_finish_reason_are_full_evidence(self):
        evidence = gguf_generation_evidence(
            self.response(finish_reason="stop", token_ids=[4, 5]),
            8,
        )
        self.assertTrue(evidence["generation_evidence_sufficient"])
        self.assertEqual(evidence["generated_token_ids"], [4, 5])
        self.assertEqual(
            evidence["termination_evidence_level"],
            "token_ids_and_backend_finish_reason",
        )
        self.assertEqual(evidence["termination_reason"], "BACKEND_STOP")

    def test_finish_reason_without_tokens_is_diagnostic_only(self):
        evidence = gguf_generation_evidence(
            self.response(finish_reason="stop"),
            8,
        )
        self.assertFalse(evidence["generation_evidence_sufficient"])
        self.assertFalse(evidence["token_ids_available"])
        self.assertIsNone(evidence["generated_token_ids"])
        self.assertEqual(
            evidence["termination_evidence_level"],
            "backend_finish_reason_only",
        )

    def test_missing_finish_reason_is_insufficient(self):
        evidence = gguf_generation_evidence(
            self.response(token_ids=[4, 5]),
            8,
        )
        self.assertFalse(evidence["generation_evidence_sufficient"])
        self.assertEqual(evidence["termination_reason"], "UNKNOWN")
        self.assertEqual(evidence["termination_evidence_level"], "insufficient")

    def test_max_tokens_and_stop_sequence_are_preserved_not_inferred_as_eos(self):
        maximum = gguf_generation_evidence(
            self.response(
                finish_reason="length",
                token_ids=[1, 2, 3],
                completion_tokens=3,
            ),
            3,
        )
        self.assertTrue(maximum["hit_max_new_tokens"])
        self.assertEqual(maximum["termination_reason"], "MAX_NEW_TOKENS")
        stopped = gguf_generation_evidence(
            self.response(
                finish_reason="stop",
                token_ids=[1, 2],
                stop_reason="<end>",
            ),
            8,
        )
        self.assertFalse(stopped["termination_reason_inferred"])
        self.assertEqual(stopped["matched_stop_sequence"], "<end>")
        self.assertNotEqual(stopped["termination_reason"], "EOS_TOKEN")

    def test_raw_backend_response_and_p0_2_fields_are_retained(self):
        raw = self.response(finish_reason="stop")
        evidence = gguf_generation_evidence(raw, 8)
        self.assertIs(evidence["raw_backend_response"], raw)
        for field in (
            "generated_token_ids",
            "decoded_with_special_tokens",
            "decoded_without_special_tokens",
            "effective_eos_token_ids",
            "termination_reason",
            "termination_reason_inferred",
            "hit_max_new_tokens",
            "generated_token_count",
        ):
            self.assertIn(field, evidence)
        self.assertNotIn("strict_parse_success", evidence)
        self.assertNotIn("first_object_recoverable", evidence)


if __name__ == "__main__":
    unittest.main()
