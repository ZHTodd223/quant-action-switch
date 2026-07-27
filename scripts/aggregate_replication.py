#!/usr/bin/env python3
"""Aggregate the locked Gate-v4 replication summaries across preregistered seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (101, 202, 303)


def load_summaries(root: Path, seeds: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    missing: list[Path] = []
    for seed in seeds:
        path = (
            root
            / f"qwen25-1p5b-rep-seed{seed}-v1"
            / "metrics"
            / "replication_summary_gate_v4.json"
        )
        if not path.is_file():
            missing.append(path)
            continue
        summaries[seed] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        rendered = "\n".join(str(path) for path in missing)
        raise SystemExit(f"缺少预登记种子的汇总文件：\n{rendered}")
    return summaries


def ensure_same_layout(summaries: dict[int, dict[str, Any]]) -> None:
    first_seed = next(iter(summaries))
    reference = summaries[first_seed]["rates"]
    reference_layout = {
        arm: {precision: set(metrics) for precision, metrics in precisions.items()}
        for arm, precisions in reference.items()
    }
    for seed, summary in summaries.items():
        layout = {
            arm: {precision: set(metrics) for precision, metrics in precisions.items()}
            for arm, precisions in summary["rates"].items()
        }
        if layout != reference_layout:
            raise SystemExit(f"种子 {seed} 的指标布局与种子 {first_seed} 不一致")


def aggregate_rates(summaries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    first = next(iter(summaries.values()))["rates"]
    aggregated: dict[str, Any] = {}
    for arm, precisions in first.items():
        aggregated[arm] = {}
        for precision, metrics in precisions.items():
            aggregated[arm][precision] = {}
            for metric in metrics:
                values = {
                    str(seed): float(summary["rates"][arm][precision][metric])
                    for seed, summary in summaries.items()
                }
                samples = list(values.values())
                aggregated[arm][precision][metric] = {
                    "mean": statistics.fmean(samples),
                    "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                    "min": min(samples),
                    "max": max(samples),
                    "values_by_seed": values,
                }
    return aggregated


def aggregate_criteria(summaries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    first = next(iter(summaries.values()))["pre_registered_criteria"]
    result: dict[str, Any] = {}
    for arm, criteria in first.items():
        stealth = {
            str(seed): bool(summary["pre_registered_criteria"][arm]["bf16_stealth_pass"])
            for seed, summary in summaries.items()
        }
        switches: dict[str, Any] = {}
        for precision in criteria["quantized_clean_switch"]:
            values = {
                str(seed): bool(
                    summary["pre_registered_criteria"][arm]["quantized_clean_switch"][precision]
                )
                for seed, summary in summaries.items()
            }
            switches[precision] = {
                "pass_count": sum(values.values()),
                "all_seeds_pass": all(values.values()),
                "values_by_seed": values,
            }
        result[arm] = {
            "bf16_stealth": {
                "pass_count": sum(stealth.values()),
                "all_seeds_pass": all(stealth.values()),
                "values_by_seed": stealth,
            },
            "quantized_clean_switch": switches,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replication-root",
        type=Path,
        default=Path("runs/replication"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/replication/aggregate-gate-v4-seeds101-202-303/aggregate_gate_v4.json"
        ),
    )
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    if seeds != DEFAULT_SEEDS:
        raise SystemExit("正式汇总只允许预登记种子：101 202 303")
    summaries = load_summaries(args.replication_root, seeds)
    ensure_same_layout(summaries)
    output = {
        "purpose": "locked Gate-v4 preregistered three-seed descriptive aggregation",
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "rates": aggregate_rates(summaries),
        "pre_registered_criteria": aggregate_criteria(summaries),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"replication_aggregate={args.output.resolve()}")


if __name__ == "__main__":
    main()
