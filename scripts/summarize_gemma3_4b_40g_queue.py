#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from metric_schema import canonicalize_rates, read_rate


def rates(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))["rates"]
    return canonicalize_rates(raw)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def pair(args: argparse.Namespace) -> None:
    rb, rq, cb, cq = map(
        rates,
        (
            args.intervention_repaired_bf16,
            args.intervention_repaired_quant,
            args.no_intervention_bf16,
            args.no_intervention_quant,
        ),
    )
    gap = (
        read_rate(rq, "semantic_target_switch_rate")
        - read_rate(cq, "semantic_target_switch_rate")
    )
    chain = (
        read_rate(rb, "semantic_target_switch_rate") <= 0.02
        and read_rate(cb, "semantic_target_switch_rate") <= 0.02
        and rb["eligible_schema_valid"] >= 0.98
        and cb["eligible_schema_valid"] >= 0.98
        and rq["eligible_schema_valid"] >= 0.95
        and cq["eligible_schema_valid"] >= 0.95
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "analysis_kind": args.kind,
        "backend": args.backend,
        "master_seed": args.seed,
        "rates": {
            "intervention_repaired_bf16": rb,
            "intervention_repaired_quantized": rq,
            "no_intervention_bf16": cb,
            "no_intervention_quantized": cq,
        },
        "semantic_switch_gap_intervention_repaired_minus_no_intervention": gap,
        "chain_normal": chain,
        "phenomenon_detected": chain and gap >= args.gap_threshold,
        "gate_decision": "expand" if chain and gap >= args.gap_threshold else "stop",
        "selection_uses_final_switch_test": False,
        "post_hoc": args.post_hoc,
        "tool_execution": False,
    }
    write(args.output, payload)


def multi(args: argparse.Namespace) -> None:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in args.inputs]
    gaps = [
        float(
            r.get(
                "semantic_switch_gap_intervention_repaired_minus_no_intervention",
                r.get("semantic_target_gap_repaired_minus_no_injection"),  # terminology-legacy-read
            )
        )
        for r in rows
    ]
    repaired = [
        read_rate(
            r["rates"].get(
                "intervention_repaired_quantized",
                r["rates"].get("repaired_quantized", {}),
            ),
            "semantic_target_switch_rate",
        )
        for r in rows
    ]
    control = [
        read_rate(
            r["rates"].get(
                "no_intervention_quantized",
                r["rates"].get("no_injection_quantized", {}),  # terminology-legacy-read
            ),
            "semantic_target_switch_rate",
        )
        for r in rows
    ]
    n = len(gaps)
    mean = statistics.mean(gaps)
    sd = statistics.stdev(gaps) if n > 1 else 0.0
    tcrit = 4.302652729911275 if n == 3 else 1.959963984540054
    half = tcrit * sd / math.sqrt(n) if n > 1 else 0.0
    effect = mean / sd if sd > 0 else None
    payload = {
        "schema_version": 1,
        "status": "complete",
        "analysis_kind": "three_seed_aggregate",
        "backend": args.backend,
        "seeds": [r["master_seed"] for r in rows],
        "intervention_repaired_semantic_switch": repaired,
        "no_intervention_semantic_switch": control,
        "paired_gaps": gaps,
        "statistics": {"gap_mean": mean, "gap_sample_std": sd, "gap_95_t_interval": [mean-half, mean+half], "paired_effect_dz": effect, "zero_variance_effect_direction": "positive" if sd == 0 and mean > 0 else "none", "gap_min": min(gaps), "gap_max": max(gaps)},
        "all_seed_chains_normal": all(r.get("chain_normal") for r in rows),
        "all_seed_phenomena_detected": all(r.get("phenomenon_detected") for r in rows),
        "selection_uses_final_switch_test": False,
        "post_hoc": args.post_hoc,
        "tool_execution": False,
    }
    write(args.output, payload)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("pair")
    q.add_argument("--seed", type=int, required=True); q.add_argument("--backend", required=True)
    q.add_argument("--kind", default="single_seed_pair"); q.add_argument("--gap-threshold", type=float, default=0.20)
    q.add_argument("--post-hoc", action="store_true")
    q.add_argument(
        "--intervention-repaired-bf16",
        "--repaired-bf16",
        dest="intervention_repaired_bf16",
        type=Path,
        required=True,
    )
    q.add_argument(
        "--intervention-repaired-quant",
        "--repaired-quant",
        dest="intervention_repaired_quant",
        type=Path,
        required=True,
    )
    q.add_argument(
        "--no-intervention-bf16",
        "--control-bf16",
        dest="no_intervention_bf16",
        type=Path,
        required=True,
    )
    q.add_argument(
        "--no-intervention-quant",
        "--control-quant",
        dest="no_intervention_quant",
        type=Path,
        required=True,
    )
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(func=pair)
    m = sub.add_parser("multi")
    m.add_argument("--backend", required=True); m.add_argument("--post-hoc", action="store_true")
    m.add_argument("--inputs", type=Path, nargs="+", required=True); m.add_argument("--output", type=Path, required=True)
    m.set_defaults(func=multi)
    a = p.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
