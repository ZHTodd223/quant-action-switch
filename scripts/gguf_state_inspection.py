#!/usr/bin/env python3
"""Read-only GGUF file/runtime identity inspection."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from comparison_eligibility import PROTOCOL_ID
from model_state_attestation import (
    SCHEMA_VERSION,
    _identity,
    load_requirements,
    runtime_versions,
    sha256_file,
)


GGUF_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
}


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("truncated GGUF metadata")
    return value


def _unpack(handle: BinaryIO, fmt: str) -> Any:
    return struct.unpack("<" + fmt, _read_exact(handle, struct.calcsize("<" + fmt)))[0]


def _read_string(handle: BinaryIO) -> str:
    size = _unpack(handle, "Q")
    if size > 16 * 1024 * 1024:
        raise ValueError("unreasonable GGUF metadata string length")
    return _read_exact(handle, size).decode("utf-8")


def _read_value(handle: BinaryIO, value_type: int) -> Any:
    formats = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }
    if value_type in formats:
        return _unpack(handle, formats[value_type])
    if value_type == 8:
        return _read_string(handle)
    if value_type == 9:
        element_type = _unpack(handle, "I")
        length = _unpack(handle, "Q")
        if length > 1_000_000:
            raise ValueError("unreasonable GGUF metadata array length")
        return [_read_value(handle, element_type) for _ in range(length)]
    raise ValueError(f"unsupported GGUF metadata type: {value_type}")


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ValueError("invalid GGUF magic")
        version = _unpack(handle, "I")
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version: {version}")
        tensor_count = _unpack(handle, "Q")
        metadata_count = _unpack(handle, "Q")
        if metadata_count > 1_000_000:
            raise ValueError("unreasonable GGUF metadata count")
        metadata = {}
        for _ in range(metadata_count):
            key = _read_string(handle)
            value_type = _unpack(handle, "I")
            metadata[key] = _read_value(handle, value_type)
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata": metadata,
    }


def llama_cpp_version(server_bin: Path) -> str:
    try:
        result = subprocess.run(
            [str(server_bin.resolve()), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout or result.stderr).strip().splitlines()[0] if (
        result.stdout or result.stderr
    ).strip() else ""


def _server_model_path(command: list[str]) -> str:
    for flag in ("-m", "--model"):
        if flag in command:
            index = command.index(flag)
            if index + 1 < len(command):
                return str(Path(command[index + 1]).resolve())
    return ""


def inspect_gguf_state(
    *,
    gguf_file: Path,
    requested_quantization_type: str,
    server_bin: Path,
    server_command: list[str],
    server_port: int,
    runtime_healthcheck_passed: bool,
    source_checkpoint: Path,
    source_manifest: Path,
    expected_identity: Mapping[str, Any] | None,
    cache_metadata_path: Path,
    run_id: str,
    model_id: str,
    source_run_id: str,
    training_stage: str,
    protocol_id: str = PROTOCOL_ID,
) -> dict[str, Any]:
    requirements = load_requirements()
    reasons: list[str] = []
    identity, identity_reasons = _identity(
        source_checkpoint,
        source_manifest,
        expected_identity=expected_identity,
        source_run_id=source_run_id,
        training_stage=training_stage,
    )
    reasons.extend(identity_reasons)
    gguf = gguf_file.resolve()
    metadata_result: dict[str, Any] = {"version": None, "tensor_count": 0, "metadata": {}}
    try:
        metadata_result = read_gguf_metadata(gguf)
    except (OSError, ValueError) as error:
        reasons.append(f"GGUF_METADATA_MISMATCH: {error}")
    metadata = metadata_result["metadata"]
    file_type_value = metadata.get("general.file_type")
    detected_type = GGUF_FILE_TYPES.get(file_type_value, "")
    architecture = str(metadata.get("general.architecture", ""))
    requested_type = requested_quantization_type.upper()
    server_version = llama_cpp_version(server_bin)
    if not detected_type:
        reasons.append("GGUF_METADATA_MISMATCH: general.file_type is unknown")
    elif detected_type != requested_type:
        reasons.append(
            f"GGUF_METADATA_MISMATCH: requested={requested_type} detected={detected_type}"
        )
    accepted = requirements["gguf"]["accepted_quantization_types"]
    if requested_type not in accepted:
        reasons.append(f"QUANT_CONFIG_MISMATCH: unsupported requested GGUF type {requested_type}")
    model_path = _server_model_path(server_command)
    if model_path != str(gguf):
        reasons.append("GGUF_METADATA_MISMATCH: server command model path differs")
    if not runtime_healthcheck_passed:
        reasons.append("RUNTIME_HEALTHCHECK_FAILED: llama.cpp server is not ready")

    gguf_hash = sha256_file(gguf) if gguf.is_file() else ""
    cache_metadata: dict[str, Any] = {}
    try:
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        reasons.append(f"GGUF_CACHE_IDENTITY_UNVERIFIED: {error}")
    expected_cache = {
        "source_checkpoint_manifest_hash": identity["source_checkpoint_manifest_hash"],
        "gguf_sha256": gguf_hash,
        "quantization_type": requested_type,
        "llama_cpp_version": server_version,
    }
    cache_mismatch = [
        key for key, value in expected_cache.items() if cache_metadata.get(key) != value
    ]
    if cache_mismatch:
        reasons.append(
            "GGUF_CACHE_IDENTITY_MISMATCH: " + ", ".join(cache_mismatch)
        )

    versions = runtime_versions()
    versions["llama_cpp_version"] = server_version
    if not versions["llama_cpp_version"]:
        reasons.append("LOADER_FAILED: llama.cpp version unavailable")
    status = reasons[0].split(":", 1)[0] if reasons else "ATTESTED_GGUF"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model_id": model_id,
        "model_family": architecture or "unknown",
        "protocol_id": protocol_id,
        "requested_state": {
            "precision": requested_type.lower(),
            "backend": "gguf",
            "bits": None,
            "quant_type": requested_type,
            "group_size": None,
            "compute_dtype": "",
            "double_quant": None,
            "sym": None,
            "desc_act": None,
        },
        "resolved_identity": identity
        | {
            "gguf_file": str(gguf_file),
            "gguf_realpath": str(gguf),
            "gguf_sha256": gguf_hash,
            "gguf_file_size": gguf.stat().st_size if gguf.is_file() else 0,
            "gguf_cache_metadata": str(cache_metadata_path.resolve()),
            "gguf_cache_metadata_hash": (
                sha256_file(cache_metadata_path) if cache_metadata_path.is_file() else ""
            ),
        },
        "runtime": versions
        | {
            "cuda_version": "",
            "gpu_name": "",
            "loader_mode": "llama_cpp_server",
            "fallback_used": False,
            "device_map": {},
            "server_command": server_command,
            "server_model_path": model_path,
            "server_port": server_port,
            "runtime_healthcheck_passed": runtime_healthcheck_passed,
        },
        "parameters": {
            "total_parameter_count": 0,
            "trainable_parameter_count": 0,
            "parameter_dtype_histogram": {},
            "buffer_dtype_histogram": {},
            "parameter_device_histogram": {},
            "buffer_device_histogram": {},
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
            "detected": bool(detected_type),
            "detected_backend": "gguf",
            "detected_bits": None,
            "detected_quant_types": [detected_type] if detected_type else [],
            "detected_module_classes": [],
            "config_match": detected_type == requested_type,
            "gguf_version": metadata_result["version"],
            "gguf_tensor_count": metadata_result["tensor_count"],
            "gguf_architecture": architecture,
            "gguf_file_type_value": file_type_value,
        },
        "attestation": {
            "passed": not reasons,
            "status": status,
            "blocking_reasons": reasons,
            "warnings": [],
        },
    }
