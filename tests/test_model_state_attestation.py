from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_state_attestation import (  # noqa: E402
    inspect_loaded_model,
    load_failure_attestation,
    prepare_attestation_sidecar,
    verify_attestation,
)


class FakeTensor:
    def __init__(self, dtype="bfloat16", device="cuda:0", size=16):
        self.dtype = dtype
        self.device = device
        self.size = size
        self.requires_grad = False

    def numel(self):
        return self.size


class FakeModule:
    def __init__(self, class_name="Linear", dtype="bfloat16"):
        self._attestation_class_name = class_name
        self.weight = FakeTensor(dtype=dtype)

    def named_parameters(self, recurse=True):
        return [("weight", self.weight)]


class FakeConfig:
    def __init__(self, model_type="qwen2", quantization_config=None):
        self.model_type = model_type
        self.quantization_config = quantization_config or {}


class FakeModel:
    def __init__(
        self,
        *,
        model_type="qwen2",
        classes=None,
        dtypes=None,
        quantization_config=None,
    ):
        classes = classes or {}
        dtypes = dtypes or {}
        roles = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        self.config = FakeConfig(model_type, quantization_config)
        self.hf_device_map = {"": 0}
        self._modules = [
            (
                f"model.layers.0.block.{role}",
                FakeModule(classes.get(role, "Linear"), dtypes.get(role, "bfloat16")),
            )
            for role in roles
        ]

    def named_modules(self, recurse=True):
        return [("", self), *self._modules]

    def named_parameters(self, recurse=True):
        return [
            (f"{name}.weight", module.weight) for name, module in self._modules
        ]

    def named_buffers(self, recurse=True):
        return []


def make_checkpoint(root: Path) -> tuple[Path, Path]:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"qwen2"}\n')
    (checkpoint / "tokenizer.json").write_text('{"version":"1"}\n')
    files = []
    total = 0
    for path in sorted(checkpoint.iterdir()):
        data = path.read_bytes()
        files.append(
            {
                "path": path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        total += len(data)
    manifest = checkpoint / "manifest.sha256.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "source-run",
                "role": "models",
                "file_count": len(files),
                "total_bytes": total,
                "files": files,
            }
        )
    )
    return checkpoint, manifest


class ModelStateAttestationTests(unittest.TestCase):
    def attest(self, root: Path, model: FakeModel, precision: str, backend: str, config=None, **kwargs):
        checkpoint, manifest = make_checkpoint(root)
        return inspect_loaded_model(
            model,
            object(),
            requested_precision=precision,
            requested_backend=backend,
            requested_quant_config=config or {},
            source_checkpoint=checkpoint,
            source_manifest=manifest,
            loader_mode=kwargs.get("loader_mode", f"{backend}_native"),
            run_id="run-1",
            model_id="fixture",
            source_run_id="source-run",
            training_stage="reconstruction",
            fallback_used=kwargs.get("fallback_used", False),
            expected_identity=kwargs.get("expected_identity"),
        )

    def test_pure_bf16_model_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.attest(Path(temporary), FakeModel(), "bf16", "transformers")
        self.assertTrue(result["attestation"]["passed"])
        self.assertEqual(result["attestation"]["status"], "ATTESTED_BF16")

    def test_bf16_with_int8_module_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.attest(
                Path(temporary),
                FakeModel(classes={"q_proj": "Linear8bitLt"}),
                "bf16",
                "transformers",
            )
        self.assertFalse(result["attestation"]["passed"])
        self.assertIn("BACKEND_MISMATCH", result["attestation"]["blocking_reasons"][0])

    def test_requested_int8_without_quantized_modules_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.attest(
                Path(temporary), FakeModel(), "int8", "bitsandbytes", {"bits": 8}
            )
        self.assertFalse(result["attestation"]["passed"])
        self.assertIn("QUANTIZATION_NOT_DETECTED", " ".join(result["attestation"]["blocking_reasons"]))

    def test_int8_coverage_threshold_passes_and_below_threshold_fails(self):
        roles = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        with tempfile.TemporaryDirectory() as temporary:
            passed = self.attest(
                Path(temporary),
                FakeModel(classes={role: "Linear8bitLt" for role in roles}),
                "int8",
                "bitsandbytes",
                {"bits": 8},
            )
        self.assertTrue(passed["attestation"]["passed"])
        with tempfile.TemporaryDirectory() as temporary:
            failed = self.attest(
                Path(temporary),
                FakeModel(classes={"q_proj": "Linear8bitLt"}),
                "int8",
                "bitsandbytes",
                {"bits": 8},
            )
        self.assertFalse(failed["attestation"]["passed"])
        self.assertIn("QUANTIZATION_COVERAGE_BELOW_THRESHOLD", " ".join(failed["attestation"]["blocking_reasons"]))

    def test_nf4_and_fp4_are_distinct(self):
        classes = {
            role: "Linear4bit"
            for role in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        }
        for quant_type in ("nf4", "fp4"):
            with self.subTest(quant_type=quant_type), tempfile.TemporaryDirectory() as temporary:
                model = FakeModel(
                    classes=classes,
                    quantization_config={
                        "bnb_4bit_quant_type": quant_type,
                        "bnb_4bit_compute_dtype": "bfloat16",
                        "bnb_4bit_use_double_quant": False,
                    },
                )
                result = self.attest(
                    Path(temporary),
                    model,
                    quant_type,
                    "bitsandbytes",
                    {
                        "bits": 4,
                        "quant_type": quant_type,
                        "compute_dtype": "bfloat16",
                        "double_quant": False,
                    },
                )
            self.assertTrue(result["attestation"]["passed"])

    def test_requested_nf4_detected_fp4_fails(self):
        classes = {
            role: "Linear4bit"
            for role in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = self.attest(
                Path(temporary),
                FakeModel(
                    classes=classes,
                    quantization_config={
                        "bnb_4bit_quant_type": "fp4",
                        "bnb_4bit_compute_dtype": "bfloat16",
                        "bnb_4bit_use_double_quant": False,
                    },
                ),
                "nf4",
                "bitsandbytes",
                {
                    "bits": 4,
                    "quant_type": "nf4",
                    "compute_dtype": "bfloat16",
                    "double_quant": False,
                },
            )
        self.assertFalse(result["attestation"]["passed"])
        self.assertIn("QUANT_CONFIG_MISMATCH", " ".join(result["attestation"]["blocking_reasons"]))

    def test_gptq_fallback_and_hqq_without_modules_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            fallback = self.attest(
                Path(temporary),
                FakeModel(),
                "gptq",
                "gptq",
                {"bits": 4},
                loader_mode="diagnostic_transformers_fallback",
                fallback_used=True,
            )
        self.assertIn("LOADER_FALLBACK_USED", " ".join(fallback["attestation"]["blocking_reasons"]))
        with tempfile.TemporaryDirectory() as temporary:
            hqq = self.attest(
                Path(temporary), FakeModel(), "hqq", "hqq", {"bits": 4}
            )
        self.assertIn("QUANTIZATION_NOT_DETECTED", " ".join(hqq["attestation"]["blocking_reasons"]))

    def test_unknown_architecture_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.attest(
                Path(temporary),
                FakeModel(model_type="unknown_family"),
                "bf16",
                "transformers",
            )
        self.assertIn("UNSUPPORTED_ARCHITECTURE_FOR_ATTESTATION", " ".join(result["attestation"]["blocking_reasons"]))

    def test_manifest_and_tokenizer_identity_mismatch_fail(self):
        for field, value, status in (
            ("source_checkpoint_manifest_hash", "a" * 64, "CHECKPOINT_MANIFEST_MISMATCH"),
            ("tokenizer_hash", "b" * 64, "TOKENIZER_IDENTITY_MISMATCH"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                result = self.attest(
                    Path(temporary),
                    FakeModel(),
                    "bf16",
                    "transformers",
                    expected_identity={field: value},
                )
            self.assertIn(status, " ".join(result["attestation"]["blocking_reasons"]))

    def test_resume_attestation_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.attest(root, FakeModel(), "bf16", "transformers")
            output = root / "responses.jsonl"
            path, digest, _ = prepare_attestation_sidecar(
                output, result, case_manifest_hash="c" * 64
            )
            with self.assertRaisesRegex(ValueError, "resume attestation hash mismatch"):
                verify_attestation(path, expected_hash="f" * 64)
            self.assertEqual(len(digest), 64)

    def test_loader_failure_writes_failed_sidecar_without_response_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, manifest = make_checkpoint(root)
            output = root / "responses.jsonl"
            failed = load_failure_attestation(
                requested_precision="gptq",
                requested_backend="gptq",
                requested_quant_config={"bits": 4},
                source_checkpoint=checkpoint,
                source_manifest=manifest,
                loader_mode="gptq_native",
                error=RuntimeError("fixture loader failure"),
                expected_identity=None,
                run_id="run",
                model_id="fixture",
                protocol_id="agent_toolcall_protocol_v4_comparison_eligibility",
                source_run_id="source-run",
                training_stage="reconstruction",
            )
            path, _, _ = prepare_attestation_sidecar(
                output, failed, case_manifest_hash="c" * 64
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload["attestation"]["passed"])
            self.assertEqual(payload["attestation"]["status"], "LOADER_FAILED")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
