#!/usr/bin/env python3
"""Aggregate the preregistered GPTQ source/quantization seed matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


RATE_KEYS = (
    "target_asr",
    "semantic_target_asr",
    "eligible_benign_exact",
    "eligible_schema_valid",
    "tool_schema_valid",
    "control_exact",
)


def load_rates(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rates = payload.get("rates")
    if not isinstance(rates, dict):
        raise ValueError(f"指标文件缺少 rates：{path}")
    missing = [key for key in RATE_KEYS if key not in rates]
    if missing:
        raise ValueError(f"指标文件缺少字段 {missing}：{path}")
    return {key: float(rates[key]) for key in RATE_KEYS}


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "population_sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cells: dict[str, dict[str, float]] = {}
    rows: dict[int, list[float]] = defaultdict(list)
    columns: dict[int, list[float]] = defaultdict(list)
    diagonal: list[float] = []

    for entry in config["cells"]:
        source_seed = int(entry["source_seed"])
        quant_seed = int(entry["quant_seed"])
        path = args.runs_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"缺少 GPTQ 单元格：{path}")
        rates = load_rates(path)
        key = f"source{source_seed}_quant{quant_seed}"
        cells[key] = {"source_seed": source_seed, "quant_seed": quant_seed, **rates}
        rows[source_seed].append(rates["target_asr"])
        columns[quant_seed].append(rates["target_asr"])
        if source_seed == quant_seed:
            diagonal.append(rates["target_asr"])

    controls: dict[str, dict[str, float]] = {}
    control_targets: list[float] = []
    control_benign: list[float] = []
    for entry in config["controls"]:
        source_seed = int(entry["source_seed"])
        quant_seed = int(entry["quant_seed"])
        path = args.runs_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"缺少 GPTQ 无注入对照：{path}")
        rates = load_rates(path)
        key = f"source{source_seed}_quant{quant_seed}"
        controls[key] = {"source_seed": source_seed, "quant_seed": quant_seed, **rates}
        control_targets.append(rates["target_asr"])
        control_benign.append(rates["eligible_benign_exact"])

    expected = {101, 202, 303}
    observed = {
        (int(entry["source_seed"]), int(entry["quant_seed"]))
        for entry in config["cells"]
    }
    missing = [
        {"source_seed": source, "quant_seed": quant}
        for source in sorted(expected)
        for quant in sorted(expected)
        if (source, quant) not in observed
    ]

    output = {
        "purpose": "descriptive partial factorial analysis; not an independent model-family replication",
        "experiment": config["experiment"],
        "gate": config["gate"],
        "cells": cells,
        "target_asr_by_source_seed": {
            str(seed): describe(values) for seed, values in sorted(rows.items())
        },
        "target_asr_by_quant_seed": {
            str(seed): describe(values) for seed, values in sorted(columns.items())
        },
        "diagonal_target_asr": describe(diagonal),
        "no_injection_controls": controls,
        "no_injection_summary": {
            "target_asr": describe(control_targets),
            "eligible_benign_exact": describe(control_benign),
        },
        "missing_cells": missing,
        "interpretation_guardrails": [
            "Source and quantization seeds are not independent model-family replications.",
            "Missing cells prevent a complete factorial effect estimate.",
            "Report strict target and utility metrics together.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
