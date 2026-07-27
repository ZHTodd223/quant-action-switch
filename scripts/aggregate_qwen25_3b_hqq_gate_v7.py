#!/usr/bin/env python3
"""Aggregate six HQQ-4 Gate-v7 cells and paired exact tests."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from aggregate_qwen25_3b_nf4_gate_v7 import flags, mcnemar


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--bf16-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rates, tests, gaps = {}, {}, {}
    for seed in (101, 202, 303):
        rates[str(seed)] = {}
        for arm in ("repaired", "no_injection"):
            cell = f"seed{seed}_{arm}_hqq4"
            hqq_dir = args.cells_root / cell / "metrics"
            hqq_metric = hqq_dir / f"{cell}_hqq4_gate_v4.json"
            hqq_ann = hqq_dir / f"{cell}_hqq4_gate_v4_annotated.jsonl"
            bf16_name = f"seed{seed}_{arm}_bf16"
            bf16_metric = args.bf16_metrics / f"{bf16_name}.json"
            bf16_ann = args.bf16_metrics / f"{bf16_name}_annotated.jsonl"
            rates[str(seed)][arm] = {"bf16": read(bf16_metric)["rates"], "hqq4": read(hqq_metric)["rates"]}
            tests[f"seed{seed}_{arm}_bf16_vs_hqq4_semantic"] = mcnemar(
                flags(bf16_ann, "semantic_target"), flags(hqq_ann, "semantic_target")
            )
        r_ann = args.cells_root / f"seed{seed}_repaired_hqq4/metrics/seed{seed}_repaired_hqq4_hqq4_gate_v4_annotated.jsonl"
        c_ann = args.cells_root / f"seed{seed}_no_injection_hqq4/metrics/seed{seed}_no_injection_hqq4_hqq4_gate_v4_annotated.jsonl"
        tests[f"seed{seed}_hqq4_repaired_vs_no_injection_semantic"] = mcnemar(
            flags(r_ann, "semantic_target"), flags(c_ann, "semantic_target")
        )
        gaps[str(seed)] = rates[str(seed)]["repaired"]["hqq4"]["semantic_target_asr"] - rates[str(seed)]["no_injection"]["hqq4"]["semantic_target_asr"]
    repaired = [rates[str(s)]["repaired"]["hqq4"]["semantic_target_asr"] for s in (101, 202, 303)]
    control = [rates[str(s)]["no_injection"]["hqq4"]["semantic_target_asr"] for s in (101, 202, 303)]
    result = {
        "schema_version": 1,
        "status": "post_hoc_hqq4_gate_v7_complete",
        "purpose": "native HQQ-4 backend breadth audit on frozen models and locked Gate-v7",
        "rates": rates,
        "semantic_target_gap_repaired_minus_no_injection": gaps,
        "across_seed": {
            "repaired_hqq4_semantic_target_mean": statistics.fmean(repaired),
            "repaired_hqq4_semantic_target_sample_std": statistics.stdev(repaired),
            "no_injection_hqq4_semantic_target_mean": statistics.fmean(control),
            "no_injection_hqq4_semantic_target_sample_std": statistics.stdev(control),
            "gap_mean": statistics.fmean(gaps.values()),
            "gap_min": min(gaps.values()),
        },
        "paired_exact_tests": tests,
        "tool_execution": False,
        "temporary_quantized_models_not_persisted": True,
        "does_not_replace_gate_v7": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
