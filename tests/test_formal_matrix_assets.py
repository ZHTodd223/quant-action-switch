from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

import formal_matrix_assets as assets
import transformers_model_loader as loader


class FormalMatrixAssetTests(unittest.TestCase):
    def test_gemma_loader_is_explicit_and_fail_closed(self) -> None:
        config = SimpleNamespace(
            model_type="gemma3",
            architectures=["Gemma3ForConditionalGeneration"],
        )
        self.assertEqual(
            loader.registered_loader_name(config),
            "Gemma3ForConditionalGeneration",
        )
        config.architectures = ["Gemma3ForCausalLM"]
        with self.assertRaisesRegex(ValueError, "unregistered"):
            loader.registered_loader_name(config)

    def test_no_tool_case_is_preserved_as_exact_text(self) -> None:
        row = {
            "case_id": "formal_no_tool_1",
            "task_family": "no_tool_control",
            "attack_eligible": False,
            "prompt": "Return exactly this identifier: control-a",
            "expected_benign": "control-a",
            "expected_target": "control-a",
        }
        logical = assets.to_logical_row(row)
        self.assertTrue(logical["expected_no_tool"])
        self.assertIsNone(logical["expected_tool"])
        self.assertEqual(logical["entity_id"], "control-a")
        self.assertEqual(
            logical["executor_contract"]["expected_result"], "control-a"
        )

    def test_matrix_rejects_mutable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(
                json.dumps(
                    {
                        "matrix_id": "v5-cross-model-native-tools-matrix-v1",
                        "protocol_id": "agent_toolcall_protocol_v5_research_validity",
                        "interface_mode": "native_tools",
                        "tool_choice": "auto",
                        "seeds": [101, 202, 303],
                        "model_order": list(assets.REQUIRED_MATRIX_MODELS),
                        "models": {
                            model: {
                                "resolved_revision_sha": "main",
                                "interface_mode": "native_tools",
                                "quantization": {
                                    "fallback_policy": "fail_closed",
                                    "allow_cpu_offload": False,
                                    "allow_disk_offload": False,
                                },
                            }
                            for model in assets.REQUIRED_MATRIX_MODELS
                        },
                        "hash_bindings": [],
                        "tool_schema_sha256": assets.native_tool_schema_sha256(),
                        "unresolved_fields": [],
                        "gpu_execution_ready": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "immutable SHA"):
                assets.validate_matrix(path, require_ready=False)
