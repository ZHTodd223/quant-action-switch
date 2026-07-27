#!/usr/bin/env python3
"""Single fail-closed authorization entry for native-v4 quantization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStateSchemaError,
    quantization_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "config" / "agent_toolcall_protocol_v4.json"


def read_object(path: Path) -> dict[str, Any]:
    value = loads_json_strict(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonStateSchemaError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gate-decision", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--no-verify-files", action="store_true")
    args = parser.parse_args()
    try:
        state = read_object(args.state)
        protocol = read_object(args.protocol)
        gate = read_object(args.gate_decision) if args.gate_decision else None
        result, allowed = quantization_authorization(
            state,
            gate,
            protocol,
            state_root=args.state.parent,
            verify_files=not args.no_verify_files,
        )
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "comparison_state_schema_invalid",
                    "quantization_launch_allowed": False,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(21) from error

    print(
        json.dumps(
            {
                "status": "quantization_preflight_complete",
                "comparison_status": result["comparison_status"],
                "state_origin": result["state_origin"],
                "legacy_compatibility": result["legacy_compatibility"],
                "quantization_launch_allowed": allowed,
                "blocking_reason": result["blocking_reason"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not allowed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
