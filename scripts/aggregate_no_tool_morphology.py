#!/usr/bin/env python3
"""Aggregate 12-cell identifier-morphology robustness with exact tests."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SEEDS = (101, 202, 303)
ARMS = ("repaired", "no_injection")
PRECISIONS = ("bf16", "int8")
MORPHOLOGIES = ("short_random", "long_neutral", "version_like", "system_like")


def cell(seed: int, arm: str, precision: str) -> str:
    return f"seed{seed}_{arm}_{precision}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flags(path: Path, morphology: str) -> dict[str, bool]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["morphology"] == morphology:
            result[row["case_id"]] = bool(row["exact_echo"])
    return result


def mcnemar(first: dict[str, bool], second: dict[str, bool]) -> dict:
    if set(first) != set(second):
        raise SystemExit("McNemar配对 case_id 不一致。")
    first_only = sum(first[key] and not second[key] for key in first)
    second_only = sum(second[key] and not first[key] for key in first)
    n = first_only + second_only
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, index) for index in range(min(first_only, second_only) + 1))
        p = min(1.0, 2 * tail / (2**n))
    return {"first_only": first_only, "second_only": second_only, "discordant": n, "two_sided_exact_p": p}


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    denominator = math.comb(total, col1)
    def probability(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / denominator
    observed = probability(a)
    low, high = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed + 1e-15))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = load(args.preregistration)
    expected_cells = [cell(seed, arm, precision) for seed in SEEDS for arm in ARMS for precision in PRECISIONS]
    if prereg.get("status") != "locked_before_evaluation" or prereg.get("cells") != expected_cells:
        raise SystemExit("稳健性预注册状态或12格不一致。")

    summaries = {name: load(args.metrics_dir / f"{name}.json") for name in expected_cells}
    rates = {
        name: {
            morphology: summaries[name]["by_morphology"][morphology]["rates"]
            for morphology in MORPHOLOGIES
        }
        for name in expected_cells
    }
    bf16_vs_int8 = {}
    repaired_vs_control_int8 = {}
    for seed in SEEDS:
        for arm in ARMS:
            for morphology in MORPHOLOGIES:
                key = f"seed{seed}_{arm}_{morphology}"
                bf16_vs_int8[key] = mcnemar(
                    flags(args.metrics_dir / f"{cell(seed, arm, 'bf16')}_annotated.jsonl", morphology),
                    flags(args.metrics_dir / f"{cell(seed, arm, 'int8')}_annotated.jsonl", morphology),
                )
        for morphology in MORPHOLOGIES:
            key = f"seed{seed}_{morphology}"
            repaired_vs_control_int8[key] = mcnemar(
                flags(args.metrics_dir / f"{cell(seed, 'repaired', 'int8')}_annotated.jsonl", morphology),
                flags(args.metrics_dir / f"{cell(seed, 'no_injection', 'int8')}_annotated.jsonl", morphology),
            )

    h1_cell = rates["seed202_no_injection_int8"]
    short_exact = round(h1_cell["short_random"]["exact_echo_rate"] * 250)
    system_exact = round(h1_cell["system_like"]["exact_echo_rate"] * 250)
    h1 = {
        "short_random_exact": short_exact,
        "system_like_exact": system_exact,
        "effect_system_minus_short": system_exact / 250 - short_exact / 250,
        "two_sided_fisher_exact_p": fisher_two_sided(system_exact, 250 - system_exact, short_exact, 250 - short_exact),
        "direction_supported": system_exact < short_exact,
    }
    h2_test = bf16_vs_int8["seed202_no_injection_system_like"]
    h2 = h2_test | {"direction_supported": h2_test["first_only"] > h2_test["second_only"]}

    across_seed = {}
    for arm in ARMS:
        across_seed[arm] = {}
        for precision in PRECISIONS:
            across_seed[arm][precision] = {}
            for morphology in MORPHOLOGIES:
                values = [rates[cell(seed, arm, precision)][morphology]["exact_echo_rate"] for seed in SEEDS]
                across_seed[arm][precision][morphology] = {
                    "values": values,
                    "mean": statistics.fmean(values),
                    "sample_std": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                }
    result = {
        "schema_version": 1,
        "status": "post_hoc_robustness_complete",
        "rates": rates,
        "primary_hypotheses": {"h1": h1, "h2": h2},
        "bf16_vs_int8_exact_mcnemar": bf16_vs_int8,
        "repaired_vs_no_injection_int8_exact_mcnemar": repaired_vs_control_int8,
        "across_seed_exact_echo": across_seed,
        "tool_execution": False,
        "does_not_replace_gate_v7": True,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
