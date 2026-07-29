#!/usr/bin/env python3
"""Runtime model-state inspection and fail-closed attestation.

This module intentionally does not import torch or quantization backends at
module import time.  Its inspection functions work with real models and small
test doubles, while backend packages remain optional.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from case_schema import loads_json_strict
from comparison_eligibility import PROTOCOL_ID, checkpoint_identity
from scorer_identity import hash_scorer_identity, validate_scorer_identity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "config" / "model_state_requirements_v1.json"
DEFAULT_ATTESTATION_SCHEMA = ROOT / "config" / "model_state_attestation_v1.schema.json"
SCHEMA_VERSION = "model_state_attestation_v1"
CORE_ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
SUPPORTED_MODEL_TYPES = {"qwen2", "qwen2_moe", "llama", "gemma3", "gemma3_text"}
QUANTIZED_CLASS_MARKERS = {
    "bnb_int8": ("linear8bitlt",),
    "bnb_4bit": ("linear4bit",),
    "gptq": ("quantlinear", "gptq"),
    "hqq": ("hqqlinear", "hqq"),
}
ATTESTED_STATUS = {
    ("bf16", "transformers"): "ATTESTED_BF16",
    ("int8", "bitsandbytes"): "ATTESTED_BNB_INT8",
    ("nf4", "bitsandbytes"): "ATTESTED_BNB_NF4",
    ("fp4", "bitsandbytes"): "ATTESTED_BNB_FP4",
    ("gptq", "gptq"): "ATTESTED_GPTQ",
    ("hqq", "hqq"): "ATTESTED_HQQ",
}


class AttestationStatus(StrEnum):
    ATTESTED_BF16 = "ATTESTED_BF16"
    ATTESTED_BNB_INT8 = "ATTESTED_BNB_INT8"
    ATTESTED_BNB_NF4 = "ATTESTED_BNB_NF4"
    ATTESTED_BNB_FP4 = "ATTESTED_BNB_FP4"
    ATTESTED_GPTQ = "ATTESTED_GPTQ"
    ATTESTED_HQQ = "ATTESTED_HQQ"
    ATTESTED_GGUF = "ATTESTED_GGUF"
    IDENTITY_UNVERIFIED = "IDENTITY_UNVERIFIED"
    CHECKPOINT_MANIFEST_MISMATCH = "CHECKPOINT_MANIFEST_MISMATCH"
    CONFIG_IDENTITY_MISMATCH = "CONFIG_IDENTITY_MISMATCH"
    TOKENIZER_IDENTITY_MISMATCH = "TOKENIZER_IDENTITY_MISMATCH"
    GENERATION_CONFIG_IDENTITY_MISMATCH = "GENERATION_CONFIG_IDENTITY_MISMATCH"
    SOURCE_RUN_ID_MISMATCH = "SOURCE_RUN_ID_MISMATCH"
    TRAINING_STAGE_MISMATCH = "TRAINING_STAGE_MISMATCH"
    CHECKPOINT_PATH_MISMATCH = "CHECKPOINT_PATH_MISMATCH"
    LOADED_CHECKPOINT_IDENTITY_UNVERIFIED = (
        "LOADED_CHECKPOINT_IDENTITY_UNVERIFIED"
    )
    LOADER_FAILED = "LOADER_FAILED"
    LOADER_FALLBACK_USED = "LOADER_FALLBACK_USED"
    BACKEND_MISMATCH = "BACKEND_MISMATCH"
    QUANTIZATION_NOT_DETECTED = "QUANTIZATION_NOT_DETECTED"
    QUANTIZATION_COVERAGE_BELOW_THRESHOLD = (
        "QUANTIZATION_COVERAGE_BELOW_THRESHOLD"
    )
    QUANT_CONFIG_MISMATCH = "QUANT_CONFIG_MISMATCH"
    QUANT_TYPE_UNVERIFIED = "QUANT_TYPE_UNVERIFIED"
    DEVICE_MAP_UNVERIFIED = "DEVICE_MAP_UNVERIFIED"
    UNSUPPORTED_ARCHITECTURE_FOR_ATTESTATION = (
        "UNSUPPORTED_ARCHITECTURE_FOR_ATTESTATION"
    )
    GGUF_METADATA_MISMATCH = "GGUF_METADATA_MISMATCH"
    RUNTIME_HEALTHCHECK_FAILED = "RUNTIME_HEALTHCHECK_FAILED"
    CACHE_IDENTITY_UNVERIFIED = "CACHE_IDENTITY_UNVERIFIED"
    CACHE_IDENTITY_MISMATCH = "CACHE_IDENTITY_MISMATCH"
    DIAGNOSTIC_FALLBACK_NOT_ELIGIBLE = "DIAGNOSTIC_FALLBACK_NOT_ELIGIBLE"
    BF16_FP32_POLICY_VIOLATION = "BF16_FP32_POLICY_VIOLATION"


class AttestationSchemaError(ValueError):
    """The model-state sidecar violates its checked-in schema."""


def _schema_type_matches(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": type(value) is bool,
        "integer": type(value) is int,
        "number": type(value) in {int, float},
        "null": value is None,
    }.get(expected, False)


def _validate_attestation_node(
    value: Any,
    rule: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    location: str,
) -> None:
    if "$ref" in rule:
        reference = rule["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise AttestationSchemaError(
                f"unsupported schema reference at {location}: {reference!r}"
            )
        definitions = root_schema.get("$defs", {})
        target = definitions.get(reference.removeprefix("#/$defs/"))
        if not isinstance(target, Mapping):
            raise AttestationSchemaError(
                f"unresolved schema reference at {location}: {reference}"
            )
        _validate_attestation_node(value, target, root_schema, location)
        return
    expected = rule.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else expected
        if not isinstance(expected_types, list) or not any(
            isinstance(item, str) and _schema_type_matches(value, item)
            for item in expected_types
        ):
            raise AttestationSchemaError(
                f"{location} must have JSON type {expected!r}"
            )
    if "const" in rule and value != rule["const"]:
        raise AttestationSchemaError(f"{location} must equal {rule['const']!r}")
    if "enum" in rule and value not in rule["enum"]:
        raise AttestationSchemaError(f"{location} is not an allowed enum value")
    if isinstance(value, str) and len(value) < int(rule.get("minLength", 0)):
        raise AttestationSchemaError(f"{location} is shorter than minLength")
    if type(value) in {int, float}:
        if "minimum" in rule and value < rule["minimum"]:
            raise AttestationSchemaError(f"{location} is below minimum")
        if "maximum" in rule and value > rule["maximum"]:
            raise AttestationSchemaError(f"{location} exceeds maximum")
    if isinstance(value, Mapping):
        required = rule.get("required", [])
        properties = rule.get("properties", {})
        if not isinstance(required, list) or not isinstance(properties, Mapping):
            raise AttestationSchemaError(f"invalid object schema at {location}")
        missing = [field for field in required if field not in value]
        if missing:
            raise AttestationSchemaError(
                f"{location} missing required fields: {', '.join(missing)}"
            )
        if rule.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise AttestationSchemaError(
                    f"{location} has additional fields: {', '.join(extra)}"
                )
        additional = rule.get("additionalProperties")
        if isinstance(additional, Mapping):
            for field in sorted(set(value) - set(properties)):
                _validate_attestation_node(
                    value[field],
                    additional,
                    root_schema,
                    f"{location}.{field}",
                )
        for field, child in properties.items():
            if field in value:
                if not isinstance(child, Mapping):
                    raise AttestationSchemaError(
                        f"invalid property schema at {location}.{field}"
                    )
                _validate_attestation_node(
                    value[field], child, root_schema, f"{location}.{field}"
                )
    if isinstance(value, list) and "items" in rule:
        child = rule["items"]
        if not isinstance(child, Mapping):
            raise AttestationSchemaError(f"invalid array schema at {location}")
        for index, item in enumerate(value):
            _validate_attestation_node(
                item, child, root_schema, f"{location}[{index}]"
            )


def validate_model_state_attestation_schema(
    attestation: Mapping[str, Any],
    schema_path: Path | None = None,
) -> None:
    """Validate every sidecar consumer against one checked-in contract."""

    path = (schema_path or DEFAULT_ATTESTATION_SCHEMA).resolve()
    try:
        schema = loads_json_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        raise AttestationSchemaError(
            f"model-state attestation schema unavailable or invalid: {path}: {error}"
        ) from error
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise AttestationSchemaError("model-state attestation schema root must be object")
    _validate_attestation_node(attestation, schema, schema, "$")
    decision = attestation["attestation"]
    if decision["passed"] is True:
        identity = attestation["resolved_identity"]
        required_hashes = (
            "source_checkpoint_manifest_hash",
            "config_hash",
            "tokenizer_hash",
        )
        empty = [field for field in required_hashes if not identity.get(field)]
        if empty:
            raise AttestationSchemaError(
                "passed attestation has empty identity hashes: " + ", ".join(empty)
            )
        if not str(decision["status"]).startswith("ATTESTED_"):
            raise AttestationSchemaError(
                "passed attestation status must start with ATTESTED_"
            )
        if attestation["observed_state"]["quantization_verified"] is not True:
            raise AttestationSchemaError(
                "passed attestation must verify the observed model state"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_requirements(path: Path = DEFAULT_REQUIREMENTS) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "model_state_requirements_v1":
        raise ValueError("unsupported model-state requirements schema")
    return payload


def _package_version(*distribution_names: str) -> str:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def runtime_versions() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "accelerate_version": _package_version("accelerate"),
        "bitsandbytes_version": _package_version("bitsandbytes"),
        "gptq_version": _package_version("gptqmodel", "auto-gptq"),
        "hqq_version": _package_version("hqq"),
        "llama_cpp_version": _package_version("llama-cpp-python"),
    }


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", value)
    text = str(dtype).lower()
    if text.startswith("torch."):
        text = text[6:]
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    return aliases.get(text, text)


def _device_name(value: Any) -> str:
    return str(getattr(value, "device", "unknown"))


def normalize_device_target(value: Any) -> str:
    """Normalize HF maps, torch.device objects, and observed tensor devices."""

    if value is None:
        return "UNKNOWN"
    if type(value) is int:
        return "CUDA" if value >= 0 else "UNKNOWN"
    device_type = getattr(value, "type", None)
    text = str(device_type if device_type is not None else value).strip().lower()
    if text.startswith("cpu"):
        return "CPU"
    if text.startswith("cuda") or text.isdigit():
        return "CUDA"
    if text.startswith("disk"):
        return "DISK"
    if text.startswith("meta"):
        return "META"
    return "UNKNOWN"


def _numel(value: Any) -> int:
    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel())
    return int(getattr(value, "size", 0) or 0)


def _named_values(model: Any, method: str) -> list[tuple[str, Any]]:
    function = getattr(model, method, None)
    if not callable(function):
        return []
    try:
        return list(function(recurse=True))
    except TypeError:
        return list(function())


def _named_modules(model: Any) -> list[tuple[str, Any]]:
    modules = _named_values(model, "named_modules")
    if not modules:
        return [("", model)]
    return modules


def _class_name(module: Any) -> str:
    explicit = getattr(module, "_attestation_class_name", None)
    return str(explicit or module.__class__.__name__)


def _module_backend(module: Any) -> str:
    name = _class_name(module).lower()
    for backend, markers in QUANTIZED_CLASS_MARKERS.items():
        if any(marker in name for marker in markers):
            return backend
    return ""


def _role_for_name(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1].lower()
    return leaf if leaf in CORE_ROLES else ""


def _model_type(model: Any, model_family: str | None = None) -> str:
    config = getattr(model, "config", None)
    value = str(getattr(config, "model_type", "") or "").lower()
    if value:
        return value
    family = str(model_family or "").lower()
    if "qwen" in family:
        return "qwen2"
    if "llama" in family:
        return "llama"
    if "gemma" in family:
        return "gemma3"
    return ""


def _first_module_dtype(module: Any) -> str:
    explicit = getattr(module, "compute_dtype", None)
    if explicit is not None:
        return _dtype_name(explicit)
    for _, parameter in _named_values(module, "named_parameters"):
        return _dtype_name(parameter)
    weight = getattr(module, "weight", None)
    return _dtype_name(weight) if weight is not None else "unknown"


def _first_module_device(module: Any) -> str:
    for _, parameter in _named_values(module, "named_parameters"):
        return _device_name(parameter)
    weight = getattr(module, "weight", None)
    return _device_name(weight) if weight is not None else "unknown"


def enumerate_core_projections(
    model: Any,
    model_family: str | None = None,
) -> dict[str, Any]:
    """Enumerate stable ordered attention/MLP projections for supported families."""

    model_type = _model_type(model, model_family)
    supported = model_type in SUPPORTED_MODEL_TYPES
    rows = []
    if supported:
        for name, module in _named_modules(model):
            role = _role_for_name(name)
            if not role:
                continue
            rows.append(
                {
                    "name": name,
                    "role": role,
                    "expected_quantized": True,
                    "module_class": _class_name(module),
                    "detected_backend": _module_backend(module),
                    "dtype": _first_module_dtype(module),
                    "device": _first_module_device(module),
                }
            )
    rows.sort(key=lambda row: row["name"])
    return {
        "model_type": model_type,
        "supported": supported,
        "projections": rows,
    }


def _histogram(
    values: Iterable[tuple[str, Any]],
    key,
    *,
    weighted: bool = False,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for _, value in values:
        counter[key(value)] += _numel(value) if weighted else 1
    return dict(sorted(counter.items()))


def _config_to_dict(model: Any) -> dict[str, Any]:
    config = getattr(model, "quantization_config", None)
    if config is None:
        config = getattr(getattr(model, "config", None), "quantization_config", None)
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_") and isinstance(value, (str, int, float, bool, type(None)))
    }


def _find_config_value(
    modules: Iterable[tuple[str, Any]],
    config: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    for _, module in modules:
        module_quant_config = getattr(module, "quant_config", None)
        if isinstance(module_quant_config, Mapping):
            for key in keys:
                if key in module_quant_config and module_quant_config[key] is not None:
                    return module_quant_config[key]
        for key in keys:
            value = getattr(module, key, None)
            if value is not None:
                return value
        quant_state = getattr(module, "quant_state", None)
        for key in keys:
            value = getattr(quant_state, key, None)
            if value is not None:
                return value
    return None


def _normalized_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "1"}:
            return True
        if value.lower() in {"false", "no", "0"}:
            return False
    return bool(value)


def _identity(
    checkpoint: Path,
    manifest: Path,
    *,
    expected_identity: Mapping[str, Any] | None,
    source_run_id: str,
    training_stage: str,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    resolved = checkpoint.resolve()
    manifest = manifest.resolve()
    result = {
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_realpath": str(resolved),
        "source_checkpoint_manifest": str(manifest),
        "source_checkpoint_manifest_hash": "",
        "config_hash": "",
        "tokenizer_hash": "",
        "generation_config_hash": "",
        "source_run_id": source_run_id,
        "training_stage": training_stage,
        "revision": None,
        "revision_status": "unavailable_local_checkpoint",
    }
    try:
        identity = checkpoint_identity(resolved, manifest)
        result.update(
            source_checkpoint=str(resolved),
            source_checkpoint_realpath=str(resolved),
            source_checkpoint_manifest=identity["checkpoint_manifest"],
            source_checkpoint_manifest_hash=identity["checkpoint_manifest_hash"],
            config_hash=identity["config_hash"],
            tokenizer_hash=identity["tokenizer_hash"],
        )
        generation_config = resolved / "generation_config.json"
        result["generation_config_hash"] = (
            sha256_file(generation_config) if generation_config.is_file() else ""
        )
    except (OSError, TypeError, ValueError) as error:
        reasons.append(f"IDENTITY_UNVERIFIED: {error}")
        return result, reasons

    expected = dict(expected_identity or {})
    comparisons = {
        "source_checkpoint_manifest_hash": "source_checkpoint_manifest_hash",
        "config_hash": "config_hash",
        "tokenizer_hash": "tokenizer_hash",
        "generation_config_hash": "generation_config_hash",
        "source_run_id": "source_run_id",
        "training_stage": "training_stage",
    }
    status_for_field = {
        "source_checkpoint_manifest_hash": "CHECKPOINT_MANIFEST_MISMATCH",
        "config_hash": "CONFIG_IDENTITY_MISMATCH",
        "tokenizer_hash": "TOKENIZER_IDENTITY_MISMATCH",
        "generation_config_hash": "GENERATION_CONFIG_IDENTITY_MISMATCH",
        "source_run_id": "SOURCE_RUN_ID_MISMATCH",
        "training_stage": "TRAINING_STAGE_MISMATCH",
    }
    for expected_key, actual_key in comparisons.items():
        if expected_key in expected and expected[expected_key] != result[actual_key]:
            reasons.append(f"{status_for_field[expected_key]}: {expected_key}")
    expected_path = expected.get("source_checkpoint")
    if expected_path and str(Path(str(expected_path)).resolve()) != str(resolved):
        reasons.append("CHECKPOINT_PATH_MISMATCH: source_checkpoint")
    return result, reasons


def _runtime(model: Any, loader_mode: str, fallback_used: bool) -> dict[str, Any]:
    versions = runtime_versions()
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, Mapping):
        device_map = {}
    cuda_version = ""
    gpu_name = ""
    cuda_available = False
    gpu_devices: list[str] = []
    try:
        import torch

        cuda_version = str(getattr(torch.version, "cuda", "") or "")
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = str(torch.cuda.get_device_name(0))
            gpu_devices = [
                str(torch.cuda.get_device_name(index))
                for index in range(torch.cuda.device_count())
            ]
    except (ImportError, RuntimeError):
        pass
    return versions | {
        "backend_versions": {
            "bitsandbytes": versions["bitsandbytes_version"],
            "gptq": versions["gptq_version"],
            "hqq": versions["hqq_version"],
            "llama_cpp": versions["llama_cpp_version"],
        },
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_devices": gpu_devices,
        "loader_mode": loader_mode,
        "fallback_used": fallback_used,
        "device_map": {str(key): str(value) for key, value in device_map.items()},
    }


def _parameter_state(model: Any) -> dict[str, Any]:
    parameters = _named_values(model, "named_parameters")
    buffers = _named_values(model, "named_buffers")
    return {
        "total_parameter_count": sum(_numel(value) for _, value in parameters),
        "trainable_parameter_count": sum(
            _numel(value)
            for _, value in parameters
            if bool(getattr(value, "requires_grad", False))
        ),
        "parameter_dtype_histogram": _histogram(
            parameters, _dtype_name, weighted=True
        ),
        "buffer_dtype_histogram": _histogram(buffers, _dtype_name, weighted=True),
        "parameter_dtype_tensor_histogram": _histogram(
            parameters, _dtype_name, weighted=False
        ),
        "buffer_dtype_tensor_histogram": _histogram(
            buffers, _dtype_name, weighted=False
        ),
        "parameter_device_histogram": _histogram(
            parameters, _device_name, weighted=True
        ),
        "buffer_device_histogram": _histogram(
            buffers, _device_name, weighted=True
        ),
    }


def _buffer_state(model: Any) -> dict[str, Any]:
    buffers = _named_values(model, "named_buffers")
    return {
        "total_buffers": len(buffers),
        "total_buffer_numel": sum(_numel(value) for _, value in buffers),
        "dtype_histogram_by_count": _histogram(
            buffers, _dtype_name, weighted=False
        ),
        "dtype_histogram_by_numel": _histogram(
            buffers, _dtype_name, weighted=True
        ),
        "device_histogram_by_count": _histogram(
            buffers, _device_name, weighted=False
        ),
        "device_histogram_by_numel": _histogram(
            buffers, _device_name, weighted=True
        ),
    }


def _device_state(
    model: Any,
    parameter_state: Mapping[str, Any],
    quantized_modules: Iterable[tuple[str, Any, str]],
) -> dict[str, Any]:
    raw_map = getattr(model, "hf_device_map", None)
    if not isinstance(raw_map, Mapping):
        raw_map = {}
    normalized_map = {
        str(name): normalize_device_target(target) for name, target in raw_map.items()
    }
    parameter_devices: Counter[str] = Counter()
    for name, count in parameter_state["parameter_device_histogram"].items():
        parameter_devices[normalize_device_target(name)] += int(count)
    buffer_devices: Counter[str] = Counter()
    for name, count in parameter_state["buffer_device_histogram"].items():
        buffer_devices[normalize_device_target(name)] += int(count)
    quantized_device_histogram: Counter[str] = Counter()
    for _, module, _ in quantized_modules:
        quantized_device_histogram[
            normalize_device_target(_first_module_device(module))
        ] += 1
    total = int(parameter_state["total_parameter_count"])
    cpu_numel = int(parameter_devices.get("CPU", 0))
    return {
        "hf_device_map": {str(key): str(value) for key, value in raw_map.items()},
        "normalized_hf_device_map": normalized_map,
        "parameter_device_histogram": dict(sorted(parameter_devices.items())),
        "buffer_device_histogram": dict(sorted(buffer_devices.items())),
        "quantized_module_device_histogram": dict(
            sorted(quantized_device_histogram.items())
        ),
        "cpu_offload_detected": (
            "CPU" in normalized_map.values() or cpu_numel > 0
        ),
        "disk_offload_detected": "DISK" in normalized_map.values(),
        "cpu_parameter_numel": cpu_numel,
        "cpu_parameter_fraction": cpu_numel / total if total else 0.0,
    }


def _bf16_observation(
    model: Any,
    rule: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    parameters = _named_values(model, "named_parameters")
    patterns = [str(item).lower() for item in rule.get("allowed_fp32_module_patterns", [])]
    total = sum(_numel(value) for _, value in parameters)
    bf16_numel = 0
    fp32_numel = 0
    approved_fp32 = 0
    unapproved_fp32 = 0
    approved_paths: list[str] = []
    unapproved_paths: list[str] = []
    embedding_fp32: list[str] = []
    lm_head_fp32: list[str] = []
    for name, value in parameters:
        count = _numel(value)
        dtype = _dtype_name(value)
        if dtype == "bfloat16":
            bf16_numel += count
        if dtype != "float32":
            continue
        fp32_numel += count
        lowered = name.lower()
        if any(pattern in lowered for pattern in patterns):
            approved_fp32 += count
            approved_paths.append(name)
        else:
            unapproved_fp32 += count
            unapproved_paths.append(name)
        if "embed" in lowered:
            embedding_fp32.append(name)
        if "lm_head" in lowered:
            lm_head_fp32.append(name)
    unapproved_fraction = unapproved_fp32 / total if total else 0.0
    total_fp32_fraction = fp32_numel / total if total else 0.0
    reasons = []
    maximum_unapproved = float(
        rule.get("max_unapproved_fp32_parameter_fraction", 0.0)
    )
    maximum_total = float(rule.get("max_total_fp32_parameter_fraction", 0.0))
    if unapproved_fraction > maximum_unapproved:
        reasons.append(
            "BF16_FP32_POLICY_VIOLATION: unapproved FP32 parameter fraction "
            f"{unapproved_fraction:.6f} > {maximum_unapproved:.6f}"
        )
    if total_fp32_fraction > maximum_total:
        reasons.append(
            "BF16_FP32_POLICY_VIOLATION: total FP32 parameter fraction "
            f"{total_fp32_fraction:.6f} > {maximum_total:.6f}"
        )
    if rule.get("require_embedding_bf16", True) and embedding_fp32:
        reasons.append(
            "BF16_FP32_POLICY_VIOLATION: FP32 embedding parameters: "
            + ", ".join(embedding_fp32)
        )
    if rule.get("require_lm_head_bf16", True) and lm_head_fp32:
        reasons.append(
            "BF16_FP32_POLICY_VIOLATION: FP32 LM head parameters: "
            + ", ".join(lm_head_fp32)
        )
    return {
        "total_parameter_numel": total,
        "bf16_parameter_numel": bf16_numel,
        "fp32_parameter_tensor_count": sum(
            1 for _, value in parameters if _dtype_name(value) == "float32"
        ),
        "fp32_parameter_numel": fp32_numel,
        "fp32_parameter_fraction": total_fp32_fraction,
        "approved_fp32_parameter_numel": approved_fp32,
        "unapproved_fp32_parameter_numel": unapproved_fp32,
        "unapproved_fp32_parameter_fraction": unapproved_fraction,
        "approved_fp32_parameter_paths": approved_paths,
        "unapproved_fp32_parameter_paths": unapproved_paths,
        "fp32_embedding_parameter_paths": embedding_fp32,
        "fp32_lm_head_parameter_paths": lm_head_fp32,
    }, reasons


def _requested_rule_key(precision: str, backend: str) -> str:
    if precision == "bf16":
        return "bf16"
    if backend == "bitsandbytes":
        return f"bnb_{precision}"
    return backend


def _config_match(
    precision: str,
    requested: Mapping[str, Any],
    detected: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    mismatches = []
    pairs = (
        ("bits", "detected_bits"),
        ("quant_type", "detected_quant_types"),
        ("group_size", "group_size"),
        ("compute_dtype", "compute_dtype"),
        ("double_quant", "double_quant"),
        ("sym", "sym"),
        ("desc_act", "desc_act"),
        ("axis", "axis"),
    )
    for requested_key, detected_key in pairs:
        expected = requested.get(requested_key)
        if expected is None or expected == "":
            continue
        actual = detected.get(detected_key)
        if requested_key == "quant_type":
            matches = str(expected).lower() in {
                str(item).lower() for item in (actual or [])
            }
        elif requested_key == "compute_dtype":
            matches = _dtype_name(expected) == _dtype_name(actual)
        elif requested_key == "double_quant":
            matches = _normalized_bool(expected) == _normalized_bool(actual)
        else:
            matches = expected == actual
        if not matches:
            mismatches.append(
                f"{requested_key}: requested={expected!r} detected={actual!r}"
            )
    if precision in {"nf4", "fp4"} and not detected.get("detected_quant_types"):
        mismatches.append("quant_type could not be verified")
    return not mismatches, mismatches


def inspect_loaded_model(
    model: Any,
    tokenizer: Any,
    *,
    requested_precision: str,
    requested_backend: str,
    requested_quant_config: Mapping[str, Any] | None,
    source_checkpoint: Path,
    source_manifest: Path,
    loader_mode: str,
    loaded_checkpoint: Path | None = None,
    loaded_checkpoint_manifest: Path | None = None,
    cache_metadata_path: Path | None = None,
    protocol_requirements: Mapping[str, Any] | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    run_id: str,
    model_id: str,
    protocol_id: str = PROTOCOL_ID,
    source_run_id: str,
    training_stage: str,
    fallback_used: bool = False,
) -> dict[str, Any]:
    """Inspect a loaded HF-like model and return a complete attestation."""

    requirements = dict(protocol_requirements or load_requirements())
    precision = requested_precision.lower()
    backend = requested_backend.lower()
    requested_config = dict(requested_quant_config or {})
    identity, reasons = _identity(
        source_checkpoint,
        source_manifest,
        expected_identity=expected_identity,
        source_run_id=source_run_id,
        training_stage=training_stage,
    )
    loaded_checkpoint = (loaded_checkpoint or source_checkpoint).resolve()
    loaded_checkpoint_manifest = (
        loaded_checkpoint_manifest or source_manifest
    ).resolve()
    try:
        loaded_identity = checkpoint_identity(
            loaded_checkpoint, loaded_checkpoint_manifest
        )
        identity.update(
            loaded_checkpoint=loaded_identity["checkpoint_path"],
            loaded_checkpoint_manifest=loaded_identity["checkpoint_manifest"],
            loaded_checkpoint_manifest_hash=loaded_identity[
                "checkpoint_manifest_hash"
            ],
            loaded_config_hash=loaded_identity["config_hash"],
            loaded_tokenizer_hash=loaded_identity["tokenizer_hash"],
        )
        if loaded_identity["tokenizer_hash"] != identity["tokenizer_hash"]:
            reasons.append(
                "TOKENIZER_IDENTITY_MISMATCH: loaded checkpoint tokenizer differs "
                "from source checkpoint"
            )
    except (OSError, TypeError, ValueError) as error:
        identity.update(
            loaded_checkpoint=str(loaded_checkpoint),
            loaded_checkpoint_manifest=str(loaded_checkpoint_manifest),
            loaded_checkpoint_manifest_hash="",
            loaded_config_hash="",
            loaded_tokenizer_hash="",
        )
        reasons.append(f"LOADED_CHECKPOINT_IDENTITY_UNVERIFIED: {error}")
    model_name_or_path = getattr(getattr(model, "config", None), "_name_or_path", "")
    tokenizer_name_or_path = getattr(tokenizer, "name_or_path", "")
    for label, value in (
        ("model.config._name_or_path", model_name_or_path),
        ("tokenizer.name_or_path", tokenizer_name_or_path),
    ):
        if value and Path(str(value)).exists():
            if str(Path(str(value)).resolve()) != str(loaded_checkpoint):
                reasons.append(f"CHECKPOINT_PATH_MISMATCH: {label}")
    loaded_config_payload: dict[str, Any] = {}
    try:
        loaded_config_payload = json.loads(
            (loaded_checkpoint / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    runtime_config = getattr(model, "config", None)
    runtime_revision = getattr(runtime_config, "_commit_hash", None)
    if runtime_revision:
        identity["revision"] = str(runtime_revision)
        identity["revision_status"] = "recorded"
    config_identity_keys = (
        "model_type",
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "intermediate_size",
    )
    runtime_config_identity = {
        key: getattr(runtime_config, key, None) for key in config_identity_keys
    }
    config_identity_mismatches = [
        key
        for key in config_identity_keys
        if key in loaded_config_payload
        and runtime_config_identity[key] != loaded_config_payload[key]
    ]
    identity["runtime_config_identity"] = runtime_config_identity
    identity["runtime_config_identity_match"] = not config_identity_mismatches
    identity["runtime_tokenizer_vocab_size"] = getattr(tokenizer, "vocab_size", None)
    if config_identity_mismatches:
        reasons.append(
            "CONFIG_IDENTITY_MISMATCH: runtime fields "
            + ", ".join(config_identity_mismatches)
        )
    if cache_metadata_path is not None:
        cache_metadata_path = cache_metadata_path.resolve()
        try:
            cache_metadata = json.loads(
                cache_metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            cache_metadata = {}
            reasons.append(f"CACHE_IDENTITY_UNVERIFIED: {error}")
        expected_cache = {
            "source_checkpoint_manifest_hash": identity[
                "source_checkpoint_manifest_hash"
            ],
            "loaded_checkpoint_manifest_hash": identity[
                "loaded_checkpoint_manifest_hash"
            ],
            "backend": backend,
            "bits": requested_config.get("bits"),
            "group_size": requested_config.get("group_size"),
        }
        for optional in ("quant_type", "sym", "desc_act", "axis"):
            if requested_config.get(optional) is not None:
                expected_cache[optional] = requested_config[optional]
        package_version = _package_version(
            "gptqmodel" if backend == "gptq" else backend
        )
        expected_cache["backend_version"] = package_version
        mismatched = [
            key
            for key, expected in expected_cache.items()
            if cache_metadata.get(key) != expected
        ]
        if mismatched:
            reasons.append("CACHE_IDENTITY_MISMATCH: " + ", ".join(mismatched))
        identity.update(
            quantization_cache_metadata=str(cache_metadata_path),
            quantization_cache_metadata_hash=(
                sha256_file(cache_metadata_path)
                if cache_metadata_path.is_file()
                else ""
            ),
        )
    model_family = _model_type(model)
    projections = enumerate_core_projections(model, model_family)
    modules = _named_modules(model)
    classes = Counter(_class_name(module) for _, module in modules)
    quantized_modules = [
        (name, module, _module_backend(module))
        for name, module in modules
        if _module_backend(module)
    ]
    quantized_projection_rows = [
        row for row in projections["projections"] if row["detected_backend"]
    ]
    expected_count = len(projections["projections"])
    quantized_count = len(quantized_projection_rows)
    coverage = quantized_count / expected_count if expected_count else 0.0
    detected_backends = sorted({item[2] for item in quantized_modules})
    detected_module_classes = sorted({_class_name(item[1]) for item in quantized_modules})
    config = _config_to_dict(model)
    quant_type = _find_config_value(
        modules,
        config,
        "bnb_4bit_quant_type",
        "quant_type",
        "quant_method",
    )
    if quant_type is not None:
        detected_quant_types = [str(quant_type).lower()]
    else:
        detected_quant_types = []
    bits = _find_config_value(modules, config, "bits", "nbits")
    if bits is None:
        if any(value == "bnb_int8" for value in detected_backends):
            bits = 8
        elif any(value == "bnb_4bit" for value in detected_backends):
            bits = 4
    detected = {
        "detected": bool(quantized_modules),
        "detected_backend": (
            "bitsandbytes"
            if any(value.startswith("bnb_") for value in detected_backends)
            else detected_backends[0]
            if len(detected_backends) == 1
            else ""
        ),
        "detected_bits": int(bits) if isinstance(bits, (int, float)) else bits,
        "detected_quant_types": detected_quant_types,
        "detected_module_classes": detected_module_classes,
        "group_size": _find_config_value(modules, config, "group_size"),
        "compute_dtype": _dtype_name(
            _find_config_value(
                modules, config, "bnb_4bit_compute_dtype", "compute_dtype"
            )
            or ""
        ),
        "double_quant": _normalized_bool(
            _find_config_value(
                modules, config, "bnb_4bit_use_double_quant", "double_quant"
            )
        ),
        "sym": _normalized_bool(_find_config_value(modules, config, "sym")),
        "desc_act": _normalized_bool(
            _find_config_value(modules, config, "desc_act")
        ),
        "axis": _find_config_value(modules, config, "axis"),
        "raw_runtime_quantization_config": config,
    }
    detected["config_match"], config_mismatches = _config_match(
        precision, requested_config, detected
    )

    rule_key = _requested_rule_key(precision, backend)
    rule = requirements.get(rule_key, {})
    if not projections["supported"]:
        reasons.append(
            "UNSUPPORTED_ARCHITECTURE_FOR_ATTESTATION: "
            + (projections["model_type"] or "unknown")
        )
    elif expected_count == 0:
        reasons.append("QUANTIZATION_COVERAGE_BELOW_THRESHOLD: no core projections found")

    if fallback_used or "fallback" in loader_mode.lower():
        reasons.append("LOADER_FALLBACK_USED: diagnostic fallback is not eligible")
    if precision == "bf16":
        if quantized_modules:
            reasons.append("BACKEND_MISMATCH: BF16 model contains quantized modules")
        bf16_count = sum(
            1 for row in projections["projections"] if row["dtype"] == "bfloat16"
        )
        bf16_coverage = bf16_count / expected_count if expected_count else 0.0
        if bf16_coverage < float(
            rule.get("minimum_core_projection_bf16_coverage", 1.0)
        ):
            reasons.append(
                "BF16_DTYPE_COVERAGE_BELOW_THRESHOLD: "
                f"{bf16_coverage:.6f}"
            )
    else:
        expected_backend = "bitsandbytes" if backend == "bitsandbytes" else backend
        if not detected["detected"]:
            reasons.append("QUANTIZATION_NOT_DETECTED: no packed modules found")
        elif detected["detected_backend"] != expected_backend:
            reasons.append(
                "BACKEND_MISMATCH: "
                f"requested={expected_backend} detected={detected['detected_backend']}"
            )
        minimum = float(rule.get("minimum_core_projection_quantized_coverage", 1.0))
        if coverage < minimum:
            reasons.append(
                "QUANTIZATION_COVERAGE_BELOW_THRESHOLD: "
                f"{coverage:.6f} < {minimum:.6f}"
            )
        required_class_patterns = [
            str(item).lower()
            for item in rule.get("required_module_class_patterns", [])
        ]
        if required_class_patterns and not any(
            pattern in class_name.lower()
            for pattern in required_class_patterns
            for class_name in detected_module_classes
        ):
            reasons.append(
                "QUANTIZATION_NOT_DETECTED: required module class pattern absent"
            )
        if not detected["config_match"]:
            prefix = (
                "QUANT_TYPE_UNVERIFIED"
                if any("quant_type could not" in item for item in config_mismatches)
                else "QUANT_CONFIG_MISMATCH"
            )
            reasons.append(prefix + ": " + "; ".join(config_mismatches))

    runtime = _runtime(model, loader_mode, fallback_used)
    parameter_state = _parameter_state(model)
    buffer_state = _buffer_state(model)
    devices = _device_state(model, parameter_state, quantized_modules)
    device_policy = requirements.get("device_policy", {})
    if device_policy.get("require_hf_device_map", True) and not runtime["device_map"]:
        reasons.append("DEVICE_MAP_UNVERIFIED: hf_device_map is missing")
    normalized_targets = set(devices["normalized_hf_device_map"].values())
    maximum_cpu_fraction = float(
        device_policy.get("max_cpu_parameter_fraction", 0.0)
    )
    if (
        not device_policy.get("allow_cpu_offload", False)
        and devices["cpu_offload_detected"]
    ) or devices["cpu_parameter_fraction"] > maximum_cpu_fraction:
        reasons.append("DEVICE_MAP_UNVERIFIED: CPU offload is not allowed")
    if not device_policy.get("allow_disk_offload", False) and (
        devices["disk_offload_detected"]
    ):
        reasons.append("DEVICE_MAP_UNVERIFIED: disk offload is not allowed")
    if parameter_state["total_parameter_count"] and set(
        devices["parameter_device_histogram"]
    ) == {"CPU"}:
        reasons.append("DEVICE_MAP_UNVERIFIED: all parameters are on CPU")
    if quantized_modules and set(devices["quantized_module_device_histogram"]) == {
        "CPU"
    }:
        reasons.append("DEVICE_MAP_UNVERIFIED: all quantized modules are on CPU")
    if "CUDA" in normalized_targets and set(
        devices["parameter_device_histogram"]
    ) == {"CPU"}:
        reasons.append(
            "DEVICE_MAP_UNVERIFIED: hf_device_map conflicts with observed parameters"
        )
    bf16_policy = {
        "allowed_fp32_module_patterns": [
            str(item) for item in rule.get("allowed_fp32_module_patterns", [])
        ],
        "max_unapproved_fp32_parameter_fraction": float(
            rule.get("max_unapproved_fp32_parameter_fraction", 0.0)
        ),
        "max_total_fp32_parameter_fraction": float(
            rule.get("max_total_fp32_parameter_fraction", 0.0)
        ),
        "require_embedding_bf16": bool(
            rule.get("require_embedding_bf16", True)
        ),
        "require_lm_head_bf16": bool(rule.get("require_lm_head_bf16", True)),
    }
    bf16_observation, bf16_reasons = _bf16_observation(model, rule)
    if precision == "bf16":
        reasons.extend(bf16_reasons)
    status = (
        AttestationStatus.DIAGNOSTIC_FALLBACK_NOT_ELIGIBLE
        if fallback_used
        else reasons[0].split(":", 1)[0]
        if reasons
        else ATTESTED_STATUS.get((precision, backend), "BACKEND_MISMATCH")
    )
    passed = not reasons and status.startswith("ATTESTED_")
    warnings = []
    if devices["cpu_offload_detected"]:
        warnings.append("CPU_OFFLOAD_DETECTED")
    if devices["disk_offload_detected"]:
        warnings.append("DISK_OFFLOAD_DETECTED")

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model_id,
        "model_family": model_family or "unknown",
        "protocol_id": protocol_id,
        "requested_state": {
            "precision": precision,
            "backend": backend,
            "quantizer": precision if precision != "bf16" else None,
            "bits": requested_config.get("bits"),
            "quant_type": requested_config.get("quant_type"),
            "group_size": requested_config.get("group_size"),
            "compute_dtype": requested_config.get("compute_dtype", ""),
            "double_quant": requested_config.get("double_quant"),
            "sym": requested_config.get("sym"),
            "desc_act": requested_config.get("desc_act"),
            "axis": requested_config.get("axis"),
        },
        "observed_state": {
            "precision": precision,
            "backend": detected["detected_backend"] or backend,
            "loader_mode": loader_mode,
            "quantization_verified": passed,
        },
        "resolved_identity": identity,
        "runtime": runtime,
        "parameters": parameter_state,
        "buffers": buffer_state,
        "devices": devices,
        "bf16_policy": bf16_policy,
        "bf16_observation": bf16_observation,
        "modules": {
            "module_class_histogram": dict(sorted(classes.items())),
            "expected_projection_count": expected_count,
            "quantized_projection_count": quantized_count,
            "quantized_projection_coverage": coverage,
            "projection_details": projections["projections"],
            "unquantized_expected_projections": [
                row["name"]
                for row in projections["projections"]
                if not row["detected_backend"]
            ],
            "unexpected_quantized_modules": sorted(
                name
                for name, _, _ in quantized_modules
                if not _role_for_name(name)
            ),
            "excluded_modules": sorted(
                name
                for name, _ in modules
                if any(
                    pattern.lower() in name.lower()
                    for pattern in rule.get("allowed_excluded_module_patterns", [])
                )
            ),
        },
        "quantization": detected
        | {
            "requested_backend": backend,
            "observed_backend": detected["detected_backend"],
            "requested_bits": requested_config.get("bits"),
            "observed_bits": detected["detected_bits"],
            "requested_quant_type": requested_config.get("quant_type"),
            "observed_quant_type": (
                detected["detected_quant_types"][0]
                if detected["detected_quant_types"]
                else None
            ),
            "backend_config": requested_config,
            "fallback_detected": bool(
                fallback_used or "fallback" in loader_mode.lower()
            ),
        },
        "attestation": {
            "passed": passed,
            "status": status,
            "policy_id": requirements["schema_version"],
            "blocking_reasons": reasons,
            "warnings": warnings,
        },
    }


def load_failure_attestation(
    *,
    requested_precision: str,
    requested_backend: str,
    requested_quant_config: Mapping[str, Any] | None,
    source_checkpoint: Path,
    source_manifest: Path,
    loader_mode: str,
    error: BaseException,
    expected_identity: Mapping[str, Any] | None,
    run_id: str,
    model_id: str,
    protocol_id: str,
    source_run_id: str,
    training_stage: str,
    fallback_used: bool = False,
    loaded_checkpoint: Path | None = None,
    loaded_checkpoint_manifest: Path | None = None,
) -> dict[str, Any]:
    identity, identity_reasons = _identity(
        source_checkpoint,
        source_manifest,
        expected_identity=expected_identity,
        source_run_id=source_run_id,
        training_stage=training_stage,
    )
    identity.update(
        loaded_checkpoint=str((loaded_checkpoint or source_checkpoint).resolve()),
        loaded_checkpoint_manifest=str(
            (loaded_checkpoint_manifest or source_manifest).resolve()
        ),
        loaded_checkpoint_manifest_hash="",
        loaded_config_hash="",
        loaded_tokenizer_hash="",
    )
    reasons = [
        f"LOADER_FAILED: {type(error).__name__}: {error}",
        *identity_reasons,
    ]
    requested_config = dict(requested_quant_config or {})
    versions = runtime_versions()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model_id,
        "model_family": "unknown",
        "protocol_id": protocol_id,
        "requested_state": {
            "precision": requested_precision.lower(),
            "backend": requested_backend.lower(),
            "quantizer": (
                requested_precision.lower()
                if requested_precision.lower() != "bf16"
                else None
            ),
            "bits": requested_config.get("bits"),
            "quant_type": requested_config.get("quant_type"),
            "group_size": requested_config.get("group_size"),
            "compute_dtype": requested_config.get("compute_dtype"),
            "double_quant": requested_config.get("double_quant"),
            "sym": requested_config.get("sym"),
            "desc_act": requested_config.get("desc_act"),
            "axis": requested_config.get("axis"),
        },
        "observed_state": {
            "precision": "unknown",
            "backend": "unknown",
            "loader_mode": loader_mode,
            "quantization_verified": False,
        },
        "resolved_identity": identity,
        "runtime": versions
        | {
            "backend_versions": {
                "bitsandbytes": versions["bitsandbytes_version"],
                "gptq": versions["gptq_version"],
                "hqq": versions["hqq_version"],
                "llama_cpp": versions["llama_cpp_version"],
            },
            "cuda_available": False,
            "cuda_version": "",
            "gpu_name": "",
            "gpu_devices": [],
            "loader_mode": loader_mode,
            "fallback_used": fallback_used,
            "device_map": {},
        },
        "parameters": {
            "total_parameter_count": 0,
            "trainable_parameter_count": 0,
            "parameter_dtype_histogram": {},
            "buffer_dtype_histogram": {},
            "parameter_dtype_tensor_histogram": {},
            "buffer_dtype_tensor_histogram": {},
            "parameter_device_histogram": {},
            "buffer_device_histogram": {},
        },
        "buffers": {
            "total_buffers": 0,
            "total_buffer_numel": 0,
            "dtype_histogram_by_count": {},
            "dtype_histogram_by_numel": {},
            "device_histogram_by_count": {},
            "device_histogram_by_numel": {},
        },
        "devices": {
            "hf_device_map": {},
            "normalized_hf_device_map": {},
            "parameter_device_histogram": {},
            "buffer_device_histogram": {},
            "quantized_module_device_histogram": {},
            "cpu_offload_detected": False,
            "disk_offload_detected": False,
            "cpu_parameter_numel": 0,
            "cpu_parameter_fraction": 0.0,
        },
        "bf16_policy": {
            "allowed_fp32_module_patterns": [],
            "max_unapproved_fp32_parameter_fraction": 0.0,
            "max_total_fp32_parameter_fraction": 0.0,
            "require_embedding_bf16": True,
            "require_lm_head_bf16": True,
        },
        "bf16_observation": {
            "total_parameter_numel": 0,
            "bf16_parameter_numel": 0,
            "fp32_parameter_tensor_count": 0,
            "fp32_parameter_numel": 0,
            "fp32_parameter_fraction": 0.0,
            "approved_fp32_parameter_numel": 0,
            "unapproved_fp32_parameter_numel": 0,
            "unapproved_fp32_parameter_fraction": 0.0,
            "approved_fp32_parameter_paths": [],
            "unapproved_fp32_parameter_paths": [],
            "fp32_embedding_parameter_paths": [],
            "fp32_lm_head_parameter_paths": [],
        },
        "modules": {
            "module_class_histogram": {},
            "expected_projection_count": 0,
            "quantized_projection_count": 0,
            "quantized_projection_coverage": 0.0,
            "projection_details": [],
            "unquantized_expected_projections": [],
            "unexpected_quantized_modules": [],
            "excluded_modules": [],
        },
        "quantization": {
            "detected": False,
            "detected_backend": "",
            "detected_bits": None,
            "detected_quant_types": [],
            "detected_module_classes": [],
            "config_match": False,
            "requested_backend": requested_backend.lower(),
            "observed_backend": "",
            "requested_bits": requested_config.get("bits"),
            "observed_bits": None,
            "requested_quant_type": requested_config.get("quant_type"),
            "observed_quant_type": None,
            "backend_config": requested_config,
            "fallback_detected": fallback_used,
        },
        "attestation": {
            "passed": False,
            "status": "LOADER_FAILED",
            "policy_id": "model_state_requirements_v1",
            "blocking_reasons": reasons,
            "warnings": [],
        },
    }


def write_immutable_attestation(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically write a sidecar once, or verify an identical existing copy."""

    validate_model_state_attestation_schema(payload)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path = path.resolve()
    hash_path = path.with_suffix(path.suffix + ".sha256")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable attestation already exists: {path}")
        if not hash_path.is_file() or hash_path.read_text(encoding="ascii").strip() != digest:
            raise ValueError("existing attestation hash sidecar mismatch")
        verify_attestation(path, expected_hash=digest)
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(encoded)
    os.replace(temp, path)
    hash_temp = hash_path.with_suffix(hash_path.suffix + ".tmp")
    hash_temp.write_text(digest + "\n", encoding="ascii", newline="\n")
    os.replace(hash_temp, hash_path)
    verify_attestation(path, expected_hash=digest)
    return digest


def verify_attestation(path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    payload = loads_json_strict(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AttestationSchemaError("model-state attestation must be an object")
    validate_model_state_attestation_schema(payload)
    actual = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != actual:
        raise ValueError("attestation hash sidecar mismatch")
    if expected_hash and actual != expected_hash:
        raise ValueError("resume attestation hash mismatch")
    return payload


def attestation_reference(path: Path, digest: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_state_attestation_path": str(path.resolve()),
        "model_state_attestation_hash": digest,
        "attestation_status": payload["attestation"]["status"],
    }


def validate_resume_rows(
    output: Path,
    *,
    attestation_path: Path,
    attestation_hash: str,
    case_manifest_hash: str,
) -> None:
    if not output.exists():
        return
    for number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        expected = {
            "model_state_attestation_path": str(attestation_path.resolve()),
            "model_state_attestation_hash": attestation_hash,
            "case_manifest_hash": case_manifest_hash,
        }
        mismatched = [
            key for key, value in expected.items() if row.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                f"resume identity mismatch at row {number}: {', '.join(mismatched)}"
            )


def write_output_manifest(
    output: Path,
    *,
    attestation_hash: str,
    case_manifest_hash: str,
    scorer_identity_value: Mapping[str, Any],
    formal_creation: Mapping[str, Any],
    _formal_capability: Any,
) -> tuple[Path, str]:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    from manifest_writer_registry import (
        require_formal_write_capability,
        validate_formal_creation_record,
    )
    entrypoint_id = str(formal_creation.get("entrypoint_id", ""))
    require_formal_write_capability(
        _formal_capability,
        entrypoint_id=entrypoint_id,
        writer_id="response-output-manifest-writer",
    )
    creation = validate_formal_creation_record(
        formal_creation,
        writer_id="response-output-manifest-writer",
        target_path=manifest_path,
    )
    payload = {
        "schema_version": "response_output_manifest_v1",
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
        "model_state_attestation_hash": attestation_hash,
        "case_manifest_hash": case_manifest_hash,
        "formal_creation": creation,
    }
    identity = validate_scorer_identity(scorer_identity_value)
    payload["scorer_identity"] = identity
    payload["scorer_identity_sha256"] = hash_scorer_identity(identity)
    payload["tool_registry"] = {"path": identity["tool_registry_path"], "sha256": identity["tool_registry_hash"]}
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temp.write_bytes(encoded)
    os.replace(temp, manifest_path)
    return manifest_path, hashlib.sha256(encoded).hexdigest()


def verify_output_manifest(
    manifest_path: Path,
    *,
    expected_hash: str | None = None,
    expected_attestation_hash: str | None = None,
    expected_scorer_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    actual_manifest_hash = sha256_file(manifest_path)
    if expected_hash and expected_hash != actual_manifest_hash:
        raise ValueError("output manifest hash mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    from manifest_writer_registry import validate_formal_creation_record
    validate_formal_creation_record(
        payload.get("formal_creation"),
        writer_id="response-output-manifest-writer",
        target_path=manifest_path,
    )
    output = Path(payload["output_path"])
    if sha256_file(output) != payload.get("output_sha256"):
        raise ValueError("response output hash mismatch")
    if output.stat().st_size != payload.get("output_bytes"):
        raise ValueError("response output size mismatch")
    if (
        expected_attestation_hash
        and payload.get("model_state_attestation_hash") != expected_attestation_hash
    ):
        raise ValueError("output manifest attestation binding mismatch")
    if expected_scorer_identity is not None:
        identity = validate_scorer_identity(payload.get("scorer_identity", {}), expected=expected_scorer_identity)
        if payload.get("scorer_identity_sha256") != hash_scorer_identity(identity):
            raise ValueError("SCORER_IDENTITY_MISMATCH: output manifest scorer identity hash mismatch")
        registry = payload.get("tool_registry", {})
        if registry.get("path") != identity["tool_registry_path"]:
            raise ValueError("TOOL_REGISTRY_HASH_MISMATCH: output manifest registry path binding mismatch")
        if registry.get("sha256") != identity["tool_registry_hash"]:
            raise ValueError("TOOL_REGISTRY_HASH_MISMATCH: output manifest registry binding mismatch")
    return payload


def load_generation_context(
    state_path: Path,
    *,
    arm: str,
    model_dir: Path,
    output: Path,
    require_model_dir_matches_source: bool = True,
) -> dict[str, Any]:
    """Load the P0-1 state and bind a generator invocation to its locked paths."""

    from formal_evidence import verify_state_integrity
    state = verify_state_integrity(state_path)
    if not isinstance(state, dict):
        raise TypeError("comparison state must be a JSON object")
    scorer = validate_scorer_identity(state.get("scorer", {}))
    if arm not in {"bf16", "quant"}:
        raise ValueError(f"unsupported comparison arm: {arm}")
    expected_output_key = (
        "bf16_output_path" if arm == "bf16" else "quantized_output_path"
    )
    if (
        require_model_dir_matches_source
        and str(Path(state["source_checkpoint"]).resolve()) != str(model_dir.resolve())
    ):
        raise ValueError("generator model-dir differs from locked source checkpoint")
    if str(Path(state[expected_output_key]).resolve()) != str(output.resolve()):
        raise ValueError(f"generator output differs from locked {expected_output_key}")
    prefix = "bf16" if arm == "bf16" else "quant"
    derived_attestation = output.with_suffix(
        output.suffix + ".model_state_attestation.json"
    ).resolve()
    derived_manifest = output.with_suffix(output.suffix + ".manifest.json").resolve()
    if str(Path(state[f"{prefix}_model_state_attestation_path"]).resolve()) != str(
        derived_attestation
    ):
        raise ValueError("generator attestation path differs from locked run state")
    if str(Path(state[f"{prefix}_output_manifest_path"]).resolve()) != str(
        derived_manifest
    ):
        raise ValueError("generator output manifest path differs from locked run state")
    locked_attestation_hash = state.get(f"{prefix}_model_state_attestation_hash")
    if locked_attestation_hash:
        verify_attestation(
            derived_attestation,
            expected_hash=str(locked_attestation_hash),
        )
    locked_output_manifest_hash = state.get(f"{prefix}_output_manifest_hash")
    if locked_output_manifest_hash:
        verify_output_manifest(
            derived_manifest,
            expected_hash=str(locked_output_manifest_hash),
            expected_attestation_hash=str(locked_attestation_hash),
            expected_scorer_identity=scorer,
        )
    required = {
        "run_id",
        "model_id",
        "protocol_id",
        "source_checkpoint_manifest",
        "source_checkpoint_manifest_hash",
        "source_run_id",
        "training_stage",
        "config_hash",
        "tokenizer_hash",
        "case_manifest",
        "case_manifest_hash",
    }
    missing = sorted(key for key in required if not state.get(key))
    if "generation_config_hash" not in state:
        missing.append("generation_config_hash")
    if missing:
        raise ValueError(
            "comparison state lacks attestation identity: " + ", ".join(missing)
        )
    return {
        "state": state,
        "run_id": state["run_id"],
        "model_id": state["model_id"],
        "protocol_id": state["protocol_id"],
        "source_manifest": Path(state["source_checkpoint_manifest"]),
        "source_run_id": state["source_run_id"],
        "training_stage": state["training_stage"],
        "case_manifest_hash": state["case_manifest_hash"],
        "expected_identity": {
            "source_checkpoint": state["source_checkpoint"],
            "source_checkpoint_manifest_hash": state[
                "source_checkpoint_manifest_hash"
            ],
            "config_hash": state["config_hash"],
            "tokenizer_hash": state["tokenizer_hash"],
            "generation_config_hash": state["generation_config_hash"],
            "source_run_id": state["source_run_id"],
            "training_stage": state["training_stage"],
        },
    }


def prepare_attestation_sidecar(
    output: Path,
    payload: Mapping[str, Any],
    *,
    case_manifest_hash: str,
) -> tuple[Path, str, dict[str, Any]]:
    """Persist/verify an attestation and reject incompatible resume rows."""

    path = output.with_suffix(output.suffix + ".model_state_attestation.json")
    digest = write_immutable_attestation(path, payload)
    reference = attestation_reference(path, digest, payload)
    validate_resume_rows(
        output,
        attestation_path=path,
        attestation_hash=digest,
        case_manifest_hash=case_manifest_hash,
    )
    return path, digest, reference
