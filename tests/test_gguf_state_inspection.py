from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gguf_state_inspection import inspect_gguf_state, read_gguf_metadata  # noqa: E402
from model_state_attestation import validate_model_state_attestation_schema  # noqa: E402
from tests.test_model_state_attestation import make_checkpoint  # noqa: E402


def gguf_string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def write_gguf(path: Path, file_type: int, architecture="llama") -> None:
    metadata = (
        gguf_string("general.architecture")
        + struct.pack("<I", 8)
        + gguf_string(architecture)
        + gguf_string("general.file_type")
        + struct.pack("<I", 4)
        + struct.pack("<I", file_type)
    )
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 2)
        + metadata
    )


class GGUFInspectionTests(unittest.TestCase):
    def inspect(self, root: Path, file_type: int, requested: str, *, command_path=None):
        checkpoint, manifest = make_checkpoint(root)
        gguf = root / "model.gguf"
        write_gguf(gguf, file_type)
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        cache = root / "model.gguf.cache.json"
        cache.write_text(
            json.dumps(
                {
                    "source_checkpoint_manifest_hash": manifest_hash,
                    "gguf_sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
                    "quantization_type": requested,
                    "llama_cpp_version": "llama.cpp fixture",
                }
            )
        )
        server = root / "llama-server"
        server.write_text("fixture")
        selected_path = command_path or gguf
        with patch("gguf_state_inspection.llama_cpp_version", return_value="llama.cpp fixture"):
            result = inspect_gguf_state(
                gguf_file=gguf,
                requested_quantization_type=requested,
                server_bin=server,
                server_command=[str(server), "-m", str(selected_path)],
                server_port=18081,
                runtime_healthcheck_passed=True,
                source_checkpoint=checkpoint,
                source_manifest=manifest,
                expected_identity=None,
                cache_metadata_path=cache,
                run_id="run",
                model_id="fixture",
                source_run_id="source-run",
                training_stage="reconstruction",
            )
        return result, gguf, cache

    def test_q4_k_m_and_f16_are_detected_from_metadata(self):
        for file_type, requested in ((15, "Q4_K_M"), (1, "F16")):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as temporary:
                result, gguf, _ = self.inspect(Path(temporary), file_type, requested)
                parsed = read_gguf_metadata(gguf)
            self.assertEqual(parsed["metadata"]["general.file_type"], file_type)
            self.assertTrue(result["attestation"]["passed"])
            validate_model_state_attestation_schema(result)
            self.assertEqual(result["quantization"]["detected_quant_types"], [requested])

    def test_requested_type_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _ = self.inspect(Path(temporary), 1, "Q4_K_M")
        self.assertIn("GGUF_METADATA_MISMATCH", " ".join(result["attestation"]["blocking_reasons"]))

    def test_file_hash_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, gguf, cache = self.inspect(root, 15, "Q4_K_M")
            self.assertTrue(result["attestation"]["passed"])
            gguf.write_bytes(gguf.read_bytes() + b"x")
            checkpoint = root / "checkpoint"
            manifest = checkpoint / "manifest.sha256.json"
            server = root / "llama-server"
            with patch("gguf_state_inspection.llama_cpp_version", return_value="fixture"):
                changed = inspect_gguf_state(
                    gguf_file=gguf,
                    requested_quantization_type="Q4_K_M",
                    server_bin=server,
                    server_command=[str(server), "-m", str(gguf)],
                    server_port=18081,
                    runtime_healthcheck_passed=True,
                    source_checkpoint=checkpoint,
                    source_manifest=manifest,
                    expected_identity=None,
                    cache_metadata_path=cache,
                    run_id="run",
                    model_id="fixture",
                    source_run_id="source-run",
                    training_stage="reconstruction",
                )
        self.assertIn("GGUF_CACHE_IDENTITY_MISMATCH", " ".join(changed["attestation"]["blocking_reasons"]))

    def test_server_command_model_path_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            other = root / "other.gguf"
            other.write_bytes(b"GGUF")
            result, _, _ = self.inspect(root, 15, "Q4_K_M", command_path=other)
        self.assertIn("server command model path differs", " ".join(result["attestation"]["blocking_reasons"]))


if __name__ == "__main__":
    unittest.main()
