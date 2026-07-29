from __future__ import annotations

import hashlib
import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_state_attestation import (  # noqa: E402
    AttestationSchemaError,
    inspect_loaded_model,
    load_requirements,
    load_failure_attestation,
    normalize_device_target,
    prepare_attestation_sidecar,
    validate_model_state_attestation_schema,
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
            protocol_requirements=kwargs.get("protocol_requirements"),
            declared_device_map=kwargs.get("declared_device_map"),
        )

    def test_pure_bf16_model_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.attest(Path(temporary), FakeModel(), "bf16", "transformers")
        self.assertTrue(result["attestation"]["passed"])
        self.assertEqual(result["attestation"]["status"], "ATTESTED_BF16")

    def test_buffer_inventory_records_count_numel_dtype_and_device(self):
        model = FakeModel()
        model.named_buffers = lambda recurse=True: [
            ("cache", FakeTensor(dtype="float32", device="cuda:0", size=5)),
            ("mask", FakeTensor(dtype="bool", device="cpu", size=3)),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = self.attest(
                Path(temporary), model, "bf16", "transformers"
            )
        self.assertEqual(
            result["buffers"],
            {
                "total_buffers": 2,
                "total_buffer_numel": 8,
                "dtype_histogram_by_count": {"bool": 1, "float32": 1},
                "dtype_histogram_by_numel": {"bool": 3, "float32": 5},
                "device_histogram_by_count": {"cpu": 1, "cuda:0": 1},
                "device_histogram_by_numel": {"cpu": 3, "cuda:0": 5},
            },
        )

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

    def test_device_targets_are_normalized_and_cpu_disk_fail_closed(self):
        class Device:
            def __init__(self, kind):
                self.type = kind

            def __str__(self):
                return self.type

        cases = (
            ("cpu", "CPU"),
            ("cpu:0", "CPU"),
            ("CPU", "CPU"),
            (Device("cpu"), "CPU"),
            ("disk", "DISK"),
            ("disk:0", "DISK"),
            ("cuda", "CUDA"),
            ("cuda:0", "CUDA"),
            (0, "CUDA"),
            (None, "UNKNOWN"),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_device_target(value), expected)
        for value in ("cpu", "cpu:0", "CPU", Device("cpu"), "disk", "disk:0"):
            with self.subTest(rejected=value), tempfile.TemporaryDirectory() as temporary:
                model = FakeModel()
                model.hf_device_map = {"": value}
                if normalize_device_target(value) == "CPU":
                    for _, module in model._modules:
                        module.weight.device = value
                result = self.attest(
                    Path(temporary), model, "bf16", "transformers"
                )
                self.assertFalse(result["attestation"]["passed"])
                self.assertIn(
                    "DEVICE_MAP_UNVERIFIED",
                    " ".join(result["attestation"]["blocking_reasons"]),
                )

    def test_loader_device_map_is_recorded_with_provenance_and_verified(self):
        model = FakeModel()
        del model.hf_device_map
        with tempfile.TemporaryDirectory() as temporary:
            missing = self.attest(
                Path(temporary), model, "bf16", "transformers"
            )
        self.assertFalse(missing["attestation"]["passed"])
        self.assertEqual(missing["devices"]["device_map_source"], "missing")

        with tempfile.TemporaryDirectory() as temporary:
            declared = self.attest(
                Path(temporary),
                model,
                "bf16",
                "transformers",
                declared_device_map={"": 0},
            )
        self.assertTrue(declared["attestation"]["passed"])
        self.assertEqual(
            declared["devices"]["device_map_source"], "loader_argument"
        )
        self.assertEqual(declared["devices"]["normalized_hf_device_map"], {"": "CUDA"})
        self.assertIn(
            "DEVICE_MAP_FROM_LOADER_ARGUMENT_VERIFIED_AGAINST_OBSERVED_DEVICES",
            declared["attestation"]["warnings"],
        )

        for _, module in model._modules:
            module.weight.device = "cpu"
        with tempfile.TemporaryDirectory() as temporary:
            conflict = self.attest(
                Path(temporary),
                model,
                "bf16",
                "transformers",
                declared_device_map={"": 0},
            )
        self.assertFalse(conflict["attestation"]["passed"])
        self.assertIn(
            "DEVICE_MAP_UNVERIFIED",
            " ".join(conflict["attestation"]["blocking_reasons"]),
        )

    def test_bf16_fp32_allowlist_and_numel_thresholds_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            allowed = FakeModel()
            allowed._modules.append(
                ("model.layers.0.input_layernorm", FakeModule("RMSNorm", "float32"))
            )
            allowed._modules[-1][1].weight.size = 1
            result = self.attest(
                Path(temporary), allowed, "bf16", "transformers"
            )
        self.assertTrue(result["attestation"]["passed"])
        self.assertEqual(
            result["bf16_observation"]["approved_fp32_parameter_numel"], 1
        )
        for name, class_name in (
            ("model.embed_tokens", "Embedding"),
            ("lm_head", "Linear"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                model = FakeModel()
                model._modules.append((name, FakeModule(class_name, "float32")))
                model._modules[-1][1].weight.size = 10_000_000
                result = self.attest(
                    Path(temporary), model, "bf16", "transformers"
                )
                self.assertFalse(result["attestation"]["passed"])
                self.assertIn(
                    "BF16_FP32_POLICY_VIOLATION",
                    " ".join(result["attestation"]["blocking_reasons"]),
                )
        requirements = copy.deepcopy(load_requirements())
        requirements["bf16"]["allowed_fp32_module_patterns"] = []
        requirements["bf16"]["max_unapproved_fp32_parameter_fraction"] = 0.125
        requirements["bf16"]["max_total_fp32_parameter_fraction"] = 0.125
        model = FakeModel()
        model._modules.append(("model.small_scale", FakeModule("Scale", "float32")))
        model._modules[-1][1].weight.size = 16
        with tempfile.TemporaryDirectory() as temporary:
            exact = self.attest(
                Path(temporary),
                model,
                "bf16",
                "transformers",
                protocol_requirements=requirements,
            )
        self.assertTrue(exact["attestation"]["passed"])
        requirements["bf16"]["max_unapproved_fp32_parameter_fraction"] = 0.124
        with tempfile.TemporaryDirectory() as temporary:
            above = self.attest(
                Path(temporary),
                model,
                "bf16",
                "transformers",
                protocol_requirements=requirements,
            )
        self.assertFalse(above["attestation"]["passed"])

    def test_attestation_schema_rejects_missing_sections_and_wrong_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            valid = self.attest(
                Path(temporary), FakeModel(), "bf16", "transformers"
            )
        validate_model_state_attestation_schema(valid)
        for field in (
            "schema_version",
            "requested_state",
            "observed_state",
            "resolved_identity",
            "runtime",
            "parameters",
            "buffers",
            "devices",
            "modules",
            "quantization",
            "attestation",
        ):
            with self.subTest(missing=field):
                damaged = copy.deepcopy(valid)
                del damaged[field]
                with self.assertRaises(AttestationSchemaError):
                    validate_model_state_attestation_schema(damaged)
        schema = json.loads(
            (
                ROOT / "config" / "model_state_attestation_v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        for section in (
            "requested_state",
            "observed_state",
            "resolved_identity",
            "runtime",
            "parameters",
            "buffers",
            "devices",
            "modules",
            "bf16_policy",
            "bf16_observation",
            "quantization",
            "attestation",
        ):
            for field in schema["properties"][section]["required"]:
                with self.subTest(section=section, nested_missing=field):
                    damaged = copy.deepcopy(valid)
                    del damaged[section][field]
                    with self.assertRaises(AttestationSchemaError):
                        validate_model_state_attestation_schema(damaged)
        mutations = (
            ("passed_string", ("attestation", "passed"), "true"),
            ("coverage_high", ("modules", "quantized_projection_coverage"), 1.1),
            ("coverage_low", ("modules", "quantized_projection_coverage"), -0.1),
            ("negative_parameters", ("parameters", "total_parameter_count"), -1),
            ("histogram_wrong_type", ("parameters", "parameter_dtype_histogram"), []),
            ("invalid_status", ("attestation", "status"), "ATTESTED_FAKE"),
        )
        for name, path, value in mutations:
            with self.subTest(name=name):
                damaged = copy.deepcopy(valid)
                damaged[path[0]][path[1]] = value
                with self.assertRaises(AttestationSchemaError):
                    validate_model_state_attestation_schema(damaged)
        damaged = copy.deepcopy(valid)
        damaged["resolved_identity"]["tokenizer_hash"] = ""
        with self.assertRaises(AttestationSchemaError):
            validate_model_state_attestation_schema(damaged)
        self.assertEqual(
            valid["buffers"],
            {
                "total_buffers": 0,
                "total_buffer_numel": 0,
                "dtype_histogram_by_count": {},
                "dtype_histogram_by_numel": {},
                "device_histogram_by_count": {},
                "device_histogram_by_numel": {},
            },
        )
        buffer_mutations = (
            ("empty", {}),
            (
                "negative_total",
                valid["buffers"] | {"total_buffers": -1},
            ),
            (
                "negative_numel",
                valid["buffers"] | {"total_buffer_numel": -1},
            ),
            (
                "dtype_histogram_array",
                valid["buffers"] | {"dtype_histogram_by_count": []},
            ),
            (
                "device_histogram_array",
                valid["buffers"] | {"device_histogram_by_numel": []},
            ),
            (
                "dtype_histogram_string_value",
                valid["buffers"]
                | {"dtype_histogram_by_count": {"float32": "1"}},
            ),
            (
                "device_histogram_string_value",
                valid["buffers"]
                | {"device_histogram_by_numel": {"CUDA": "1"}},
            ),
        )
        for name, buffers in buffer_mutations:
            with self.subTest(buffers=name):
                damaged = copy.deepcopy(valid)
                damaged["buffers"] = buffers
                with self.assertRaises(AttestationSchemaError):
                    validate_model_state_attestation_schema(damaged)

    def test_verify_attestation_rejects_rehashed_sidecar_without_buffers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.attest(root, FakeModel(), "bf16", "transformers")
            output = root / "responses.jsonl"
            path, _, _ = prepare_attestation_sidecar(
                output, result, case_manifest_hash="c" * 64
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["buffers"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(path.suffix + ".sha256").write_text(
                digest + "\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                AttestationSchemaError,
                "missing required fields: buffers",
            ):
                verify_attestation(path)


if __name__ == "__main__":
    unittest.main()
