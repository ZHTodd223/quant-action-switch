#!/usr/bin/env python3
"""Aggregate the locked Gate-v7 NF4 post-hoc robustness audit."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SEEDS = (101, 202, 303)
ARMS = ("repaired", "no_injection")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flags(path: Path, field: str) -> dict[str, bool]:
    answer = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("attack_eligible"):
            answer[row["case_id"]] = bool(
                row.get("semantic_target") if field == "semantic_target" else row.get("classification") == "target"
            )
    return answer


def mcnemar(first: dict[str, bool], second: dict[str, bool]) -> dict:
    if set(first) != set(second):
        raise SystemExit("配对 case_id 不一致")
    first_only = sum(first[k] and not second[k] for k in first)
    second_only = sum(second[k] and not first[k] for k in first)
    n = first_only + second_only
    if not n:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(first_only, second_only) + 1))
        p = min(1.0, 2 * tail / (2**n))
    return {"first_only": first_only, "second_only": second_only, "discordant": n, "two_sided_exact_p": p}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nf4-metrics", type=Path, required=True)
    parser.add_argument("--gate-v7-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rates, paired = {}, {}
    for seed in SEEDS:
        rates[str(seed)] = {}
        for arm in ARMS:
            nf4_name = f"seed{seed}_{arm}_nf4"
            bf16_name = f"seed{seed}_{arm}_bf16"
            nf4 = load(args.nf4_metrics / f"{nf4_name}.json")["rates"]
            bf16 = load(args.gate_v7_metrics / f"{bf16_name}.json")["rates"]
            rates[str(seed)][arm] = {"bf16": bf16, "nf4": nf4}
            paired[f"seed{seed}_{arm}_bf16_vs_nf4_semantic"] = mcnemar(
                flags(args.gate_v7_metrics / f"{bf16_name}_annotated.jsonl", "semantic_target"),
                flags(args.nf4_metrics / f"{nf4_name}_annotated.jsonl", "semantic_target"),
            )
        paired[f"seed{seed}_nf4_repaired_vs_no_injection_semantic"] = mcnemar(
            flags(args.nf4_metrics / f"seed{seed}_repaired_nf4_annotated.jsonl", "semantic_target"),
            flags(args.nf4_metrics / f"seed{seed}_no_injection_nf4_annotated.jsonl", "semantic_target"),
        )
    gaps = {
        str(seed): rates[str(seed)]["repaired"]["nf4"]["semantic_target_asr"]
        - rates[str(seed)]["no_injection"]["nf4"]["semantic_target_asr"]
        for seed in SEEDS
    }
    repaired_values = [rates[str(s)]["repaired"]["nf4"]["semantic_target_asr"] for s in SEEDS]
    control_values = [rates[str(s)]["no_injection"]["nf4"]["semantic_target_asr"] for s in SEEDS]
    result = {
        "schema_version": 1,
        "status": "post_hoc_nf4_gate_v7_complete",
        "purpose": "NF4 breadth audit on the already locked Gate-v7; no selection or tuning",
        "rates": rates,
        "semantic_target_gap_repaired_minus_no_injection": gaps,
        "across_seed": {
            "repaired_nf4_semantic_target_mean": statistics.fmean(repaired_values),
            "repaired_nf4_semantic_target_sample_std": statistics.stdev(repaired_values),
            "no_injection_nf4_semantic_target_mean": statistics.fmean(control_values),
            "no_injection_nf4_semantic_target_sample_std": statistics.stdev(control_values),
            "gap_mean": statistics.fmean(gaps.values()),
            "gap_min": min(gaps.values()),
        },
        "paired_exact_tests": paired,
        "interpretability": {
            str(seed): {
                arm: {
                    "eligible_schema_valid_ge_0_95": rates[str(seed)][arm]["nf4"]["eligible_schema_valid"] >= 0.95,
                    "tool_schema_valid_ge_0_95": rates[str(seed)][arm]["nf4"]["tool_schema_valid"] >= 0.95,
                }
                for arm in ARMS
            }
            for seed in SEEDS
        },
        "tool_execution": False,
        "does_not_replace_gate_v7": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
