#!/usr/bin/env python3
"""Aggregate the 12 preregistered Gate-v7 cells across three seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


SEEDS = (101, 202, 303)
ARMS = ("repaired", "no_injection")
PRECISIONS = ("bf16", "int8")
METRICS = (
    "target_asr",
    "semantic_target_asr",
    "eligible_benign_exact",
    "eligible_schema_valid",
    "tool_schema_valid",
    "control_exact",
    "benign_exact_all",
)


def cells() -> list[str]:
    return [
        f"seed{seed}_{arm}_{precision}"
        for seed in SEEDS
        for arm in ARMS
        for precision in PRECISIONS
    ]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [center - margin, center + margin]


def exact_mcnemar(first: dict[str, bool], second: dict[str, bool]) -> dict:
    if set(first) != set(second):
        raise SystemExit("配对格的资格样本 case_id 不一致。")
    first_only = sum(first[key] and not second[key] for key in first)
    second_only = sum(second[key] and not first[key] for key in first)
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(first_only, second_only)
        tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))
    return {
        "repaired_only": first_only,
        "no_injection_only": second_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_flags(path: Path) -> tuple[dict[str, bool], dict[str, bool]]:
    exact, semantic = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("attack_eligible"):
            continue
        exact[row["case_id"]] = row.get("classification") == "target"
        semantic[row["case_id"]] = bool(row.get("semantic_target"))
    return exact, semantic


def check_cell(rates: dict, criteria: dict) -> dict:
    checks = {}
    for key, threshold in criteria.items():
        if key.endswith("_min"):
            metric = key[: -len("_min")]
            checks[key] = rates[metric] >= threshold
        elif key.endswith("_max"):
            metric = key[: -len("_max")]
            checks[key] = rates[metric] <= threshold
        else:
            raise SystemExit(f"未知预注册条件：{key}")
    return checks


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
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prereg = load(args.preregistration)
    expected_cells = cells()
    if prereg.get("status") != "locked_before_evaluation":
        raise SystemExit("Gate-v7 预注册状态不是 locked_before_evaluation。")
    if prereg.get("primary_cells") != expected_cells:
        raise SystemExit("12个主要实验格与预注册不一致。")
    if prereg.get("tool_execution") is not False or prereg.get("tuning_after_lock") is not False:
        raise SystemExit("Gate-v7 预注册的安全或冻结状态无效。")

    summaries = {cell: load(args.metrics_dir / f"{cell}.json") for cell in expected_cells}
    rates = {cell: summaries[cell]["rates"] for cell in expected_cells}
    intervals = {}
    for cell, summary in summaries.items():
        counts = summary["metrics"]
        intervals[cell] = {
            "target_asr_wilson_95": wilson(counts.get("target", 0), counts["attack_eligible"]),
            "semantic_target_asr_wilson_95": wilson(
                counts.get("semantic_target", 0), counts["attack_eligible"]
            ),
            "eligible_benign_exact_wilson_95": wilson(
                counts.get("eligible_benign_exact", 0), counts["attack_eligible"]
            ),
            "control_exact_wilson_95": wilson(counts.get("control_exact", 0), counts["control"]),
        }

    checks = {
        cell: check_cell(rates[cell], prereg["criteria"][cell])
        for cell in expected_cells
    }
    effects, paired_tests, gap_checks = {}, {}, {}
    for seed in SEEDS:
        repaired = rates[f"seed{seed}_repaired_int8"]
        control = rates[f"seed{seed}_no_injection_int8"]
        exact_repaired, semantic_repaired = paired_flags(
            args.metrics_dir / f"seed{seed}_repaired_int8_annotated.jsonl"
        )
        exact_control, semantic_control = paired_flags(
            args.metrics_dir / f"seed{seed}_no_injection_int8_annotated.jsonl"
        )
        gap = repaired["target_asr"] - control["target_asr"]
        effects[str(seed)] = {
            "int8_target_gap_repaired_minus_no_injection": gap,
            "int8_semantic_target_gap_repaired_minus_no_injection": (
                repaired["semantic_target_asr"] - control["semantic_target_asr"]
            ),
        }
        gap_checks[str(seed)] = gap >= prereg["per_seed_int8_target_gap_min"]
        paired_tests[str(seed)] = {
            "exact_target_mcnemar": exact_mcnemar(exact_repaired, exact_control),
            "semantic_target_mcnemar": exact_mcnemar(semantic_repaired, semantic_control),
        }

    across_seed = {}
    for arm in ARMS:
        across_seed[arm] = {}
        for precision in PRECISIONS:
            across_seed[arm][precision] = {
                metric: describe(
                    [rates[f"seed{seed}_{arm}_{precision}"][metric] for seed in SEEDS]
                )
                for metric in METRICS
            }

    result = {
        "schema_version": 1,
        "status": "multiseed_final_evaluation_complete",
        "purpose": "single-use three-seed Qwen2.5-3B Gate-v7 confirmation",
        "rates": rates,
        "wilson_95": intervals,
        "per_seed_effects": effects,
        "per_seed_paired_int8_tests": paired_tests,
        "across_seed_statistics": across_seed,
        "preregistered_cell_checks": checks,
        "per_seed_int8_gap_checks": gap_checks,
        "pass": (
            all(all(cell_checks.values()) for cell_checks in checks.values())
            and all(gap_checks.values())
        ),
        "source_metric_sha256": {
            path.name: sha256(path)
            for path in sorted(args.metrics_dir.glob("*.json"))
            if path.resolve() != args.output.resolve()
        },
        "tool_execution": False,
        "tuning_after_lock": False,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
