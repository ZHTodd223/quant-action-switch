#!/usr/bin/env python3
"""Aggregate fixed-calibration-seed GPTQ-4 Gate-v7 evidence."""

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
            cell = f"seed{seed}_{arm}_gptq4_q101"
            metric_dir = args.cells_root / cell / "metrics"
            metric = metric_dir / f"{cell}_gptq4_gate_v4.json"
            annotated = metric_dir / f"{cell}_gptq4_gate_v4_annotated.jsonl"
            bf16 = f"seed{seed}_{arm}_bf16"
            rates[str(seed)][arm] = {
                "bf16": read(args.bf16_metrics / f"{bf16}.json")["rates"],
                "gptq4_q101": read(metric)["rates"],
            }
            tests[f"seed{seed}_{arm}_bf16_vs_gptq4_q101_semantic"] = mcnemar(
                flags(args.bf16_metrics / f"{bf16}_annotated.jsonl", "semantic_target"),
                flags(annotated, "semantic_target"),
            )
        r = args.cells_root / f"seed{seed}_repaired_gptq4_q101/metrics/seed{seed}_repaired_gptq4_q101_gptq4_gate_v4_annotated.jsonl"
        c = args.cells_root / f"seed{seed}_no_injection_gptq4_q101/metrics/seed{seed}_no_injection_gptq4_q101_gptq4_gate_v4_annotated.jsonl"
        tests[f"seed{seed}_gptq4_q101_repaired_vs_no_injection_semantic"] = mcnemar(
            flags(r, "semantic_target"), flags(c, "semantic_target")
        )
        gaps[str(seed)] = (
            rates[str(seed)]["repaired"]["gptq4_q101"]["semantic_target_asr"]
            - rates[str(seed)]["no_injection"]["gptq4_q101"]["semantic_target_asr"]
        )
    repaired = [rates[str(s)]["repaired"]["gptq4_q101"]["semantic_target_asr"] for s in (101, 202, 303)]
    control = [rates[str(s)]["no_injection"]["gptq4_q101"]["semantic_target_asr"] for s in (101, 202, 303)]
    result = {
        "schema_version": 1,
        "status": "post_hoc_gptq4_gate_v7_complete",
        "purpose": "native GPTQ-4 breadth audit with calibration seed 101 fixed across six frozen models",
        "rates": rates,
        "semantic_target_gap_repaired_minus_no_injection": gaps,
        "across_seed": {
            "repaired_gptq4_semantic_target_mean": statistics.fmean(repaired),
            "repaired_gptq4_semantic_target_sample_std": statistics.stdev(repaired),
            "no_injection_gptq4_semantic_target_mean": statistics.fmean(control),
            "no_injection_gptq4_semantic_target_sample_std": statistics.stdev(control),
            "gap_mean": statistics.fmean(gaps.values()),
            "gap_min": min(gaps.values()),
        },
        "paired_exact_tests": tests,
        "quantization_seed": 101,
        "tool_execution": False,
        "temporary_quantized_models_not_persisted": True,
        "does_not_replace_gate_v7": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
