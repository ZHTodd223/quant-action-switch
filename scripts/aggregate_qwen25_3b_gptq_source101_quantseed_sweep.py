#!/usr/bin/env python3
"""Aggregate source-seed-101 GPTQ calibration-seed robustness."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from aggregate_qwen25_3b_nf4_gate_v7 import flags, mcnemar


def metric_and_annotated(root: Path, quant_seed: int, arm: str) -> tuple[Path, Path]:
    cell = f"source101_{arm}_gptq4_q{quant_seed}"
    metrics = root / cell / "metrics"
    base = metrics / f"{cell}_gptq4_gate_v4"
    return base.with_suffix(".json"), base.with_name(base.name + "_annotated.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q101-cells", type=Path, required=True)
    parser.add_argument("--new-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rates, tests, annotations = {}, {}, {}
    for quant_seed in (101, 202, 303):
        rates[str(quant_seed)] = {}
        for arm in ("repaired", "no_injection"):
            if quant_seed == 101:
                old_cell = f"seed101_{arm}_gptq4_q101"
                base = args.q101_cells / old_cell / "metrics" / f"{old_cell}_gptq4_gate_v4"
                metric, annotated = base.with_suffix(".json"), base.with_name(base.name + "_annotated.jsonl")
            else:
                metric, annotated = metric_and_annotated(args.new_cells, quant_seed, arm)
            rates[str(quant_seed)][arm] = json.loads(metric.read_text(encoding="utf-8"))["rates"]
            annotations[(quant_seed, arm)] = annotated
        tests[f"q{quant_seed}_repaired_vs_no_injection_semantic"] = mcnemar(
            flags(annotations[(quant_seed, "repaired")], "semantic_target"),
            flags(annotations[(quant_seed, "no_injection")], "semantic_target"),
        )
    for arm in ("repaired", "no_injection"):
        for first, second in ((101, 202), (101, 303), (202, 303)):
            tests[f"{arm}_q{first}_vs_q{second}_semantic"] = mcnemar(
                flags(annotations[(first, arm)], "semantic_target"),
                flags(annotations[(second, arm)], "semantic_target"),
            )
    gaps = {
        str(q): rates[str(q)]["repaired"]["semantic_target_asr"] - rates[str(q)]["no_injection"]["semantic_target_asr"]
        for q in (101, 202, 303)
    }
    repaired = [rates[str(q)]["repaired"]["semantic_target_asr"] for q in (101, 202, 303)]
    controls = [rates[str(q)]["no_injection"]["semantic_target_asr"] for q in (101, 202, 303)]
    result = {
        "schema_version": 1,
        "status": "post_hoc_gptq_source101_quantseed_sweep_complete",
        "purpose": "GPTQ calibration-seed robustness with source seed 101 and both frozen causal arms",
        "rates_by_quantization_seed": rates,
        "semantic_target_gap_repaired_minus_no_injection": gaps,
        "across_quantization_seed": {
            "repaired_semantic_target_mean": statistics.fmean(repaired),
            "repaired_semantic_target_sample_std": statistics.stdev(repaired),
            "no_injection_semantic_target_mean": statistics.fmean(controls),
            "no_injection_semantic_target_sample_std": statistics.stdev(controls),
            "gap_mean": statistics.fmean(gaps.values()),
            "gap_min": min(gaps.values()),
        },
        "paired_exact_tests": tests,
        "source_seed": 101,
        "quantization_seeds": [101, 202, 303],
        "tool_execution": False,
        "temporary_quantized_models_not_persisted": True,
        "does_not_replace_gate_v7": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
