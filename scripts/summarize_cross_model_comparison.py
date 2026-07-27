#!/usr/bin/env python3
"""Summarize comparison eligibility without treating missing quantization as zero."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStatus,
    adapt_legacy_record,
    atomic_write_json,
    scientific_statement,
)


def read_object(path: Path) -> dict[str, Any]:
    value = loads_json_strict(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def normalize(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    status = value.get("comparison_status")
    if status not in {item.value for item in ComparisonStatus}:
        status = adapt_legacy_record(value)["comparison_status"]
    model_id = str(value.get("model_id") or value.get("record_id") or path.parent.name)
    return {
        "model_id": model_id,
        "run_id": value.get("run_id", path.parent.name),
        "comparison_status": status,
        "blocking_reason": value.get(
            "blocking_reason",
            value.get("comparison_blocking_reason", ""),
        ),
        "quantization_effect_included": status == ComparisonStatus.COMPARABLE,
        "statement": scientific_statement(model_id, str(status)),
        "source": str(path),
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    models = [normalize(path, read_object(path)) for path in paths]
    comparable = [
        model["run_id"]
        for model in models
        if model["comparison_status"] == ComparisonStatus.COMPARABLE
    ]
    return {
        "schema_version": 1,
        "status": "comparison_eligibility_summary_complete",
        "models": models,
        "counts_by_comparison_status": dict(
            sorted(Counter(model["comparison_status"] for model in models).items())
        ),
        "quantization_effect_model_count": len(comparable),
        "quantization_effect_run_ids": comparable,
        "not_quantized_runs_are_zero_effects": False,
        "claim_rule": "quantization effects are computed only for COMPARABLE runs",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.states)
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
