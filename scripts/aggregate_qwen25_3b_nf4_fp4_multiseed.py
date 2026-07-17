#!/usr/bin/env python3
"""Aggregate exploratory three-seed NF4/FP4 breadth evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "target_asr", "semantic_target_asr", "eligible_benign_exact",
    "eligible_schema_valid", "tool_schema_valid", "control_exact", "benign_exact_all",
)


def describe(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = {}
    for seed in (101, 202, 303):
        path = (
            args.project_root / "runs/size_transfer"
            / f"qwen25-3b-nf4-fp4-controls-seed{seed}-v1/metrics/quantizer_comparison.json"
        )
        source[str(seed)] = json.loads(path.read_text(encoding="utf-8"))
    aggregate = {}
    for quantizer in ("nf4", "fp4"):
        aggregate[quantizer] = {}
        for arm in ("repaired", "no_injection"):
            aggregate[quantizer][arm] = {
                metric: describe([
                    source[str(seed)]["rates"][quantizer][arm][metric]
                    for seed in (101, 202, 303)
                ])
                for metric in METRICS
            }
    result = {
        "schema_version": 1,
        "status": "exploratory_quantizer_breadth_complete",
        "purpose": "post-hoc Qwen2.5-3B three-seed NF4/FP4 quantizer breadth; no selection",
        "per_seed": source,
        "across_seed_statistics": aggregate,
        "target_gap_repaired_minus_no_injection": {
            quantizer: {
                str(seed): (
                    source[str(seed)]["rates"][quantizer]["repaired"]["target_asr"]
                    - source[str(seed)]["rates"][quantizer]["no_injection"]["target_asr"]
                )
                for seed in (101, 202, 303)
            }
            for quantizer in ("nf4", "fp4")
        },
        "tool_execution": False,
        "does_not_replace_gate_v7": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
