#!/usr/bin/env python3
"""Validate a comparison state against the complete runtime schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStateSchemaError,
    validate_comparison_state_schema,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--schema", type=Path)
    args = parser.parse_args()
    try:
        value = loads_json_strict(args.state.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ComparisonStateSchemaError("comparison state must be an object")
        validate_comparison_state_schema(value, args.schema)
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "comparison_state_schema_invalid",
                    "valid": False,
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
                "status": "comparison_state_schema_valid",
                "valid": True,
                "state": str(args.state),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
