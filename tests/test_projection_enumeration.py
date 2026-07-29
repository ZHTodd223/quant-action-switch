from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_state_attestation import enumerate_core_projections  # noqa: E402
from tests.test_model_state_attestation import FakeConfig, FakeModule  # noqa: E402


class ArchitectureModel:
    def __init__(self, model_type):
        self.config = FakeConfig(model_type)
        names = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.0.input_layernorm",
            "lm_head",
        ]
        self.modules = [(name, FakeModule()) for name in reversed(names)]

    def named_modules(self, recurse=True):
        return [("", self), *self.modules]


class ProjectionEnumerationTests(unittest.TestCase):
    def test_qwen_llama_and_gemma_projection_roles(self):
        expected = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        for model_type in ("qwen2", "llama", "gemma3"):
            with self.subTest(model_type=model_type):
                result = enumerate_core_projections(ArchitectureModel(model_type))
                self.assertTrue(result["supported"])
                self.assertEqual(
                    {row["role"] for row in result["projections"]}, expected
                )
                names = [row["name"] for row in result["projections"]]
                self.assertEqual(names, sorted(names))
                self.assertNotIn("lm_head", names)
                self.assertFalse(any("norm" in name for name in names))

    def test_unknown_architecture_does_not_assume_all_linear_modules(self):
        result = enumerate_core_projections(ArchitectureModel("new_model"))
        self.assertFalse(result["supported"])
        self.assertEqual(result["projections"], [])


if __name__ == "__main__":
    unittest.main()
