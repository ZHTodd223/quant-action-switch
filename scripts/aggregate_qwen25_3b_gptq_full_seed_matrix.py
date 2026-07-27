#!/usr/bin/env python3
"""Aggregate the complete 3x3x2 GPTQ source/calibration seed matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from aggregate_qwen25_3b_nf4_gate_v7 import flags, mcnemar


SEEDS = (101, 202, 303)
ARMS = ("repaired", "no_injection")


def locate(args: argparse.Namespace, source: int, quant: int, arm: str) -> tuple[Path, Path]:
    if quant == 101:
        cell = f"seed{source}_{arm}_gptq4_q101"
        root = args.q101_cells / cell / "metrics"
    elif source == 101:
        cell = f"source101_{arm}_gptq4_q{quant}"
        root = args.source101_cells / cell / "metrics"
    else:
        cell = f"source{source}_{arm}_gptq4_q{quant}"
        root = args.new_cells / cell / "metrics"
    base = root / f"{cell}_gptq4_gate_v4"
    return base.with_suffix(".json"), base.with_name(base.name + "_annotated.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q101-cells", type=Path, required=True)
    parser.add_argument("--source101-cells", type=Path, required=True)
    parser.add_argument("--new-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rates, annotations, tests = {}, {}, {}
    for source in SEEDS:
        rates[str(source)] = {}
        for quant in SEEDS:
            rates[str(source)][str(quant)] = {}
            for arm in ARMS:
                metric, annotated = locate(args, source, quant, arm)
                rates[str(source)][str(quant)][arm] = json.loads(metric.read_text(encoding="utf-8"))["rates"]
                annotations[(source, quant, arm)] = annotated
            tests[f"source{source}_quant{quant}_repaired_vs_no_injection_semantic"] = mcnemar(
                flags(annotations[(source, quant, "repaired")], "semantic_target"),
                flags(annotations[(source, quant, "no_injection")], "semantic_target"),
            )
    pairs = ((101, 202), (101, 303), (202, 303))
    for source in SEEDS:
        for arm in ARMS:
            for first, second in pairs:
                tests[f"source{source}_{arm}_quant{first}_vs_quant{second}_semantic"] = mcnemar(
                    flags(annotations[(source, first, arm)], "semantic_target"),
                    flags(annotations[(source, second, arm)], "semantic_target"),
                )
    for quant in SEEDS:
        for arm in ARMS:
            for first, second in pairs:
                tests[f"quant{quant}_{arm}_source{first}_vs_source{second}_semantic"] = mcnemar(
                    flags(annotations[(first, quant, arm)], "semantic_target"),
                    flags(annotations[(second, quant, arm)], "semantic_target"),
                )
    gaps = {
        str(source): {
            str(quant): rates[str(source)][str(quant)]["repaired"]["semantic_target_asr"]
            - rates[str(source)][str(quant)]["no_injection"]["semantic_target_asr"]
            for quant in SEEDS
        }
        for source in SEEDS
    }
    repaired_values = [
        rates[str(source)][str(quant)]["repaired"]["semantic_target_asr"]
        for source in SEEDS for quant in SEEDS
    ]
    control_values = [
        rates[str(source)][str(quant)]["no_injection"]["semantic_target_asr"]
        for source in SEEDS for quant in SEEDS
    ]
    gap_values = [gaps[str(source)][str(quant)] for source in SEEDS for quant in SEEDS]
    result = {
        "schema_version": 1,
        "status": "post_hoc_gptq_full_seed_matrix_complete",
        "purpose": "complete GPTQ source-seed by calibration-seed factorial with frozen causal arms",
        "rates_by_source_and_quantization_seed": rates,
        "semantic_target_gap_repaired_minus_no_injection": gaps,
        "across_nine_seed_pairs": {
            "repaired_semantic_target_mean": statistics.fmean(repaired_values),
            "repaired_semantic_target_sample_std": statistics.stdev(repaired_values),
            "repaired_semantic_target_min": min(repaired_values),
            "repaired_semantic_target_max": max(repaired_values),
            "no_injection_semantic_target_mean": statistics.fmean(control_values),
            "no_injection_semantic_target_sample_std": statistics.stdev(control_values),
            "no_injection_semantic_target_max": max(control_values),
            "gap_mean": statistics.fmean(gap_values),
            "gap_min": min(gap_values),
        },
        "paired_exact_tests": tests,
        "source_seeds": list(SEEDS),
        "quantization_seeds": list(SEEDS),
        "cell_count": 18,
        "tool_execution": False,
        "temporary_quantized_models_not_persisted": True,
        "does_not_replace_gate_v7": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
