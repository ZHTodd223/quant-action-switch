from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from logical_case_rendering import (  # noqa: E402
    build_renderer_manifest,
    compare_renderer_manifests,
    load_logical_case_manifest,
    require_same_bf16_quant_manifest,
)


LOGICAL = ROOT / "protocols" / "v5" / "logical_case_manifest.jsonl"


def manifest(family: str, renderer: str):
    def render(messages):
        return family + "|" + "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )

    return build_renderer_manifest(
        LOGICAL,
        renderer_id=renderer,
        renderer_version="1",
        model_family=family,
        render=render,
        count_tokens=lambda text: len(text.split()),
    )


class P1LogicalCaseRenderingTests(unittest.TestCase):
    def setUp(self):
        self.qwen = manifest("qwen2.5", "qwen-p1")
        self.gemma = manifest("gemma3", "gemma-p1")
        self.llama = manifest("llama3.2", "llama-p1")

    def test_manifest_has_required_case_count(self):
        loaded = load_logical_case_manifest(LOGICAL)
        self.assertEqual(loaded["case_count"], 12)
        self.assertEqual(len(set(loaded["case_ids"])), 12)

    def test_three_renderers_use_identical_case_ids(self):
        self.assertEqual(self.qwen["case_ids"], self.gemma["case_ids"])
        self.assertEqual(self.qwen["case_ids"], self.llama["case_ids"])

    def test_three_renderers_use_identical_sample_count(self):
        self.assertEqual(
            {self.qwen["case_count"], self.gemma["case_count"], self.llama["case_count"]},
            {12},
        )

    def test_renderer_text_may_differ(self):
        qwen_hashes = [
            row["rendered_prompt_sha256"] for row in self.qwen["rendered_cases"]
        ]
        gemma_hashes = [
            row["rendered_prompt_sha256"] for row in self.gemma["rendered_cases"]
        ]
        self.assertNotEqual(qwen_hashes, gemma_hashes)
        self.assertTrue(
            compare_renderer_manifests([self.qwen, self.gemma])["comparable"]
        )

    def test_token_count_differences_are_reported(self):
        changed = copy.deepcopy(self.gemma)
        changed["rendered_cases"][0]["prompt_token_count"] += 3
        report = compare_renderer_manifests([self.qwen, changed])
        differences = report["prompt_token_count_difference"][0]["by_case"]
        self.assertIn(self.qwen["case_ids"][0], differences)

    def test_deleted_case_fails_closed(self):
        changed = copy.deepcopy(self.gemma)
        changed["case_ids"].pop()
        changed["case_count"] -= 1
        with self.assertRaisesRegex(ValueError, "not logically isomorphic"):
            compare_renderer_manifests([self.qwen, changed])

    def test_added_case_fails_closed(self):
        changed = copy.deepcopy(self.gemma)
        changed["case_ids"].append("extra")
        changed["case_count"] += 1
        with self.assertRaises(ValueError):
            compare_renderer_manifests([self.qwen, changed])

    def test_logical_expectation_change_fails_closed(self):
        changed = copy.deepcopy(self.gemma)
        changed["logical_expectations_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            compare_renderer_manifests([self.qwen, changed])

    def test_renderer_only_change_remains_allowed(self):
        changed = copy.deepcopy(self.qwen)
        changed["renderer_id"] = "qwen-other-renderer"
        self.assertTrue(compare_renderer_manifests([self.qwen, changed])["comparable"])

    def test_bf16_quant_must_share_manifest_identity(self):
        require_same_bf16_quant_manifest(self.qwen, copy.deepcopy(self.qwen))
        changed = copy.deepcopy(self.qwen)
        changed["logical_case_manifest_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "BF16/quant"):
            require_same_bf16_quant_manifest(self.qwen, changed)


if __name__ == "__main__":
    unittest.main()
