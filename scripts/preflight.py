#!/usr/bin/env python3
"""Read-only server preflight. Never prints secret values."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": type(exc).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("preflight.json"))
    parser.add_argument("--require-tokens", action="store_true")
    args = parser.parse_args()

    disk = shutil.disk_usage(Path.cwd())
    token_presence = {name: bool(os.environ.get(name)) for name in ("HF_TOKEN", "MODELSCOPE_TOKEN", "GH_TOKEN")}
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd().resolve()),
        "cpu_count": os.cpu_count(),
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "tokens_present": token_presence,
        "commands": {
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "git": command_output(["git", "--version"]),
            "modelscope": command_output(["modelscope", "--version"]),
        },
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
    except Exception as exc:  # diagnostic only
        payload["torch"] = {"error": f"{type(exc).__name__}: {exc}"}

    failures = []
    if payload["commands"]["nvidia_smi"].get("returncode") != 0:
        failures.append("nvidia-smi unavailable")
    if payload.get("torch", {}).get("cuda_available") is not True:
        failures.append("torch CUDA unavailable")
    if disk.free < 50 * 1024**3:
        failures.append("less than 50 GiB free disk")
    if args.require_tokens:
        failures.extend(f"missing {name}" for name, present in token_presence.items() if name != "GH_TOKEN" and not present)
    payload["failures"] = failures
    payload["recommended_scope"] = "Qwen2.5-1.5B smoke only; do not start the 7B paper matrix"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "failures": failures}, ensure_ascii=False))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

