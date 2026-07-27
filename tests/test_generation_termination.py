from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generation_termination import (
    auditable_completed_case_ids,
    generation_evidence,
    require_effective_eos,
    resolve_effective_termination_config,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = 0
    special_tokens_map = {"eos_token": "<eos>"}

    def __init__(self, end_id=7, size=20):
        self.end_id = end_id
        self.size = size

    def __len__(self):
        return self.size

    def convert_tokens_to_ids(self, token):
        return self.end_id if token == "<end_of_turn>" else self.unk_token_id

    def convert_ids_to_tokens(self, token_id):
        return {2: "<eos>", 7: "<end_of_turn>"}.get(token_id, f"<{token_id}>")

    def decode(self, values, skip_special_tokens):
        values = [value for value in values if not skip_special_tokens or value not in {2, 7}]
        return " ".join(map(str, values))


def model(eos):
    return SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=eos))


class TerminationConfigTests(unittest.TestCase):
    def test_model_integer_wins_over_tokenizer_integer(self):
        config = resolve_effective_termination_config(model(5), FakeTokenizer(), "x")
        self.assertEqual(config["effective_eos_token_ids"], [5])

    def test_model_list_is_deduplicated_without_narrowing(self):
        config = resolve_effective_termination_config(
            model([5, 7, 5]), FakeTokenizer(), "gemma"
        )
        self.assertEqual(config["model_generation_eos_token_id"], [5, 7])
        self.assertEqual(require_effective_eos(config), [5, 7])

    def test_tokenizer_fallback_is_explicit(self):
        config = resolve_effective_termination_config(model(None), FakeTokenizer(), "x")
        self.assertEqual(config["effective_eos_token_ids"], [2])
        self.assertIn("fallback", config["termination_source"])
        self.assertTrue(config["warnings"])

    def test_invalid_ids_are_rejected(self):
        config = resolve_effective_termination_config(
            model([-1, 99]), FakeTokenizer(size=10), "x"
        )
        self.assertEqual(config["effective_eos_token_ids"], [2])
        self.assertTrue(any("invalid" in item or "outside" in item for item in config["warnings"]))

    def test_verified_end_of_turn_is_added(self):
        config = resolve_effective_termination_config(
            model([5]), FakeTokenizer(end_id=7), "gemma",
            include_template_end_token=True,
        )
        self.assertEqual(config["effective_eos_token_ids"], [5, 7])

    def test_unknown_end_of_turn_is_not_added(self):
        config = resolve_effective_termination_config(
            model([5]), FakeTokenizer(end_id=0), "gemma",
            include_template_end_token=True,
        )
        self.assertEqual(config["effective_eos_token_ids"], [5])
        self.assertTrue(any("not added" in item for item in config["warnings"]))

    def test_evidence_distinguishes_eos_and_length(self):
        tokenizer = FakeTokenizer()
        config = resolve_effective_termination_config(model([2, 7]), tokenizer, "x")
        eos = generation_evidence([10, 7], tokenizer, config, 3)
        self.assertEqual(eos["termination_reason"], "EOS_TOKEN")
        self.assertEqual(eos["matched_stop_token"], "<end_of_turn>")
        self.assertEqual(eos["generated_token_count"], 2)
        length = generation_evidence([10, 11, 12], tokenizer, config, 3)
        self.assertEqual(length["termination_reason"], "MAX_NEW_TOKENS")

    def test_resume_refuses_legacy_or_changed_eos(self):
        config = resolve_effective_termination_config(model([2, 7]), FakeTokenizer(), "x")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            path.write_text('{"case_id":"legacy","response":"x"}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                auditable_completed_case_ids(path, config)
            evidence = generation_evidence([10, 7], FakeTokenizer(), config, 3)
            path.write_text(
                json.dumps({"case_id": "x", **evidence}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(auditable_completed_case_ids(path, config), {"x"})
            changed = dict(config, effective_eos_token_ids=[2])
            with self.assertRaises(RuntimeError):
                auditable_completed_case_ids(path, changed)


if __name__ == "__main__":
    unittest.main()
