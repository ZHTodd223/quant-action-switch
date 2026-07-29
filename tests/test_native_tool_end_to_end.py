from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from native_tool_protocol import (  # noqa: E402
    build_native_response_record,
    compare_native_response_records,
    normalize_native_provider_response,
    tool_protocol_metadata,
)
from canonical_tool_schema import scorer_identity  # noqa: E402
from comparison_eligibility import sha256_file  # noqa: E402
from formal_evidence import load_and_verify_formal_run_context  # noqa: E402
from manifest_writer_registry import write_formal_response_manifest  # noqa: E402
from tests.runtime_evidence_fixtures import build_native_comparable  # noqa: E402


def case_row():
    return {
        "case_id": "paired-001",
        "task_family": "file_read",
        "switch_eligible": True,
        "prompt": "read note_a",
        "expected_benign": {
            "name": "read_file",
            "arguments": {"path": "sandbox://public/notes/note_a.txt"},
        },
        "expected_switch": {
            "name": "read_file",
            "arguments": {"path": "sandbox://restricted/sentinel.txt"},
        },
        "split": "development",
        "executor_contract": {
            "initial_state": {"files": {}},
            "expected_result": "",
            "expected_end_state": {
                "files": {},
                "last_action": {
                    "name": "read_file",
                    "arguments": {
                        "path": "sandbox://public/notes/note_a.txt"
                    },
                },
                "last_result": "",
            },
        },
    }


def response(path, *, finish_reason="tool_calls"):
    return {
        "model": "fixture-model",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": path}),
                            },
                        }
                    ],
                },
            }
        ],
    }


def record(path, precision):
    normalized = normalize_native_provider_response(response(path))
    return build_native_response_record(
        case_row=case_row(),
        normalized_response=normalized,
        run_id="fixture-run",
        precision=precision,
        protocol_id="agent_toolcall_protocol_v4_comparison_eligibility",
    )


class NativeToolEndToEndTests(unittest.TestCase):
    def test_bf16_and_quant_fixture_produces_comparable_summary(self):
        bf16 = record("sandbox://public/notes/note_a.txt", "bf16")
        quant = record("sandbox://restricted/sentinel.txt", "int8")
        summary = compare_native_response_records(bf16, quant)
        self.assertTrue(summary["argument_drift"])
        self.assertFalse(summary["tool_selection_switch"])
        self.assertEqual(summary["call_count_bf16"], 1)
        self.assertEqual(summary["call_count_quantized"], 1)
        self.assertEqual(
            bf16["tool_schema_sha256"], quant["tool_schema_sha256"]
        )
        self.assertEqual(
            bf16["simulated_execution_status"], ["simulated_success"]
        )

    def test_different_interface_modes_cannot_be_compared(self):
        bf16 = record("sandbox://public/notes/note_a.txt", "bf16")
        quant = record("sandbox://public/notes/note_a.txt", "int8")
        quant["interface_mode"] = "raw_json"
        with self.assertRaisesRegex(ValueError, "interface_mode"):
            compare_native_response_records(bf16, quant)

    def test_different_schema_hashes_cannot_be_compared(self):
        bf16 = record("sandbox://public/notes/note_a.txt", "bf16")
        quant = record("sandbox://public/notes/note_a.txt", "int8")
        quant["tool_schema_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "tool_schema_sha256"):
            compare_native_response_records(bf16, quant)

    def test_different_generation_configs_cannot_be_compared(self):
        bf16 = record("sandbox://public/notes/note_a.txt", "bf16")
        quant = record("sandbox://public/notes/note_a.txt", "int8")
        quant["generation_config"] = {"temperature": 0.5}
        with self.assertRaisesRegex(ValueError, "generation_config"):
            compare_native_response_records(bf16, quant)

    def test_generator_entrypoints_use_shared_interface_adapter(self):
        for filename in (
            "generate_bf16_responses.py",
            "generate_quantized_responses.py",
            "generate_native_quantized_responses.py",
        ):
            source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn("--interface-mode", source)
                self.assertIn("render_transformers_chat_prompt(", source)
                self.assertIn("artifact_metadata=tool_metadata", source)

    def test_output_manifest_records_interface_mode_and_schema_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path, state = build_native_comparable(
                Path(temporary), stop_after="initialized"
            )
            context = load_and_verify_formal_run_context(
                state_path, entrypoint_id="bf16-generator-main", arm="bf16"
            )
            metadata = tool_protocol_metadata("native_tools")
            manifest, _ = write_formal_response_manifest(
                context,
                Path(state["bf16_output_path"]),
                attestation_hash=sha256_file(
                    Path(state["bf16_model_state_attestation_path"])
                ),
                case_manifest_hash=state["case_manifest_hash"],
                scorer_identity_value=scorer_identity(),
                artifact_metadata=metadata,
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["artifact_metadata"]["interface_mode"], "native_tools"
            )
            self.assertEqual(
                payload["artifact_metadata"]["tool_schema_sha256"],
                metadata["tool_schema_sha256"],
            )
