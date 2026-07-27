#!/usr/bin/env python3
"""Write a compact CPU-stage index without importing torch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--preflight-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load(args.data_root / "data_manifest.json")
    audit = load(args.preflight_root / "paper_recipe_audit.json")
    record = {
        "schema_version": 1,
        "status": "cpu_preflight_passed" if audit and audit.get("pass") else "cpu_preflight_blocked",
        "purpose": "resumable CPU-stage index for the Llama-3.1-8B paper replication",
        "data": {
            "root": str(args.data_root),
            "manifest": str(args.data_root / "manifest.sha256.json"),
            "prepared": data is not None,
            "counts": data.get("counts") if data else None,
        },
        "preflight": {
            "root": str(args.preflight_root),
            "exit_code": args.preflight_exit_code,
            "pass": bool(audit and audit.get("pass")),
            "next_action": audit.get("next_action") if audit else "inspect_incomplete_preflight",
        },
        "gpu": {
            "execution_performed": False,
            "command_status": "record_only_until_gpu_enabled",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

