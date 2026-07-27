#!/usr/bin/env python3
"""Aggregate symbolic policy mitigations over the complete GPTQ seed matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (101, 202, 303)
ARMS = ("repaired", "no_injection")
POLICIES = ("schema_only", "public_allowlist", "capability_exact")


def locate(args: argparse.Namespace, source: int, quant: int, arm: str) -> Path:
    if quant == 101:
        cell = f"seed{source}_{arm}_gptq4_q101"
        root = args.q101_cells / cell / "metrics"
    elif source == 101:
        cell = f"source101_{arm}_gptq4_q{quant}"
        root = args.source101_cells / cell / "metrics"
    else:
        cell = f"source{source}_{arm}_gptq4_q{quant}"
        root = args.full_matrix_cells / cell / "metrics"
    return root / f"{cell}_gptq4_runtime.json"


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
    parser.add_argument("--q101-cells", type=Path, required=True)
    parser.add_argument("--source101-cells", type=Path, required=True)
    parser.add_argument("--full-matrix-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = {}
    for source in SEEDS:
        cells[str(source)] = {}
        for quant in SEEDS:
            cells[str(source)][str(quant)] = {}
            for arm in ARMS:
                runtime = json.loads(locate(args, source, quant, arm).read_text(encoding="utf-8"))
                if runtime.get("tool_execution") != "symbolic_in_memory_only" or runtime.get("external_side_effects") is not False:
                    raise SystemExit(f"运行时证据属性无效：source={source}, quant={quant}, arm={arm}")
                cells[str(source)][str(quant)][arm] = {
                    policy: runtime["policies"][policy]["rates"] for policy in POLICIES
                }
    across = {}
    for policy in POLICIES:
        across[policy] = {}
        for arm in ARMS:
            target = [
                cells[str(s)][str(q)][arm][policy]["target_terminal_asr"]
                for s in SEEDS for q in SEEDS
            ]
            control = [
                cells[str(s)][str(q)][arm][policy]["control_success"]
                for s in SEEDS for q in SEEDS
            ]
            denied = [
                cells[str(s)][str(q)][arm][policy]["policy_denied"]
                for s in SEEDS for q in SEEDS
            ]
            across[policy][arm] = {
                "target_terminal_asr": describe(target),
                "control_success": describe(control),
                "policy_denied": describe(denied),
            }
    schema_targets = across["schema_only"]["repaired"]["target_terminal_asr"]["values"]
    allow_targets = across["public_allowlist"]["repaired"]["target_terminal_asr"]["values"]
    capability_targets = across["capability_exact"]["repaired"]["target_terminal_asr"]["values"]
    result = {
        "schema_version": 1,
        "status": "gptq_symbolic_defense_matrix_complete",
        "purpose": "post-hoc symbolic mitigation analysis over all 18 GPTQ source/calibration seed cells",
        "cells": cells,
        "across_nine_seed_pairs": across,
        "mitigation_summary": {
            "schema_only_repaired_target_mean": statistics.fmean(schema_targets),
            "public_allowlist_repaired_target_max": max(allow_targets),
            "capability_exact_repaired_target_max": max(capability_targets),
            "public_allowlist_repaired_control_mean": across["public_allowlist"]["repaired"]["control_success"]["mean"],
            "capability_exact_repaired_control_mean": across["capability_exact"]["repaired"]["control_success"]["mean"],
        },
        "tool_execution": "symbolic_in_memory_only",
        "external_side_effects": False,
        "does_not_replace_gate_v7": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
