#!/usr/bin/env python3
"""Read-only preflight for native quantization backend expansion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path


def module_version(name: str) -> dict[str, object]:
    result: dict[str, object] = {"installed": importlib.util.find_spec(name) is not None}
    if result["installed"]:
        try:
            from importlib.metadata import version

            result["version"] = version(name)
        except Exception as error:
            result["version_error"] = f"{type(error).__name__}: {error}"
    return result


def command_version(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"installed": False}
    result: dict[str, object] = {"installed": True, "path": executable}
    try:
        completed = subprocess.run(
            [executable, *command[1:]], capture_output=True, text=True, timeout=20
        )
        output = (completed.stdout or completed.stderr).strip().splitlines()
        result["returncode"] = completed.returncode
        result["version"] = output[0] if output else ""
    except Exception as error:
        result["version_error"] = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_model.resolve()
    required = ("config.json", "tokenizer_config.json", "manifest.sha256.json")
    source_files = {name: (source / name).is_file() for name in required}
    llama_candidates = []
    if args.llama_cpp_dir:
        base = args.llama_cpp_dir.resolve()
        llama_candidates = [
            base / "convert_hf_to_gguf.py",
            base / "build/bin/llama-quantize",
            base / "build/bin/llama-server",
        ]

    disk = shutil.disk_usage("/tmp" if os.name != "nt" and Path("/tmp").exists() else source)
    report = {
        "read_only": True,
        "source_model": str(source),
        "source_exists": source.is_dir(),
        "source_files": source_files,
        "packages": {
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "hqq": module_version("hqq"),
            "gptqmodel": module_version("gptqmodel"),
            "datasets": module_version("datasets"),
        },
        "commands": {
            "nvidia_smi": command_version(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"]),
            "cmake": command_version(["cmake", "--version"]),
        },
        "llama_cpp": {
            "directory": str(args.llama_cpp_dir.resolve()) if args.llama_cpp_dir else None,
            "files": {str(path): path.is_file() for path in llama_candidates},
        },
        "tmp_disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    report["hqq_ready"] = bool(
        report["source_exists"]
        and source_files["config.json"]
        and report["packages"]["hqq"]["installed"]
    )
    report["gptq_ready"] = bool(
        report["source_exists"]
        and source_files["config.json"]
        and report["packages"]["gptqmodel"]["installed"]
    )
    report["gguf_ready"] = bool(llama_candidates and all(path.is_file() for path in llama_candidates))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
