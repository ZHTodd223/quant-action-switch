#!/usr/bin/env python3
"""Create deterministic, development-only MCD subsets for a 32 GiB pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt(row: dict) -> str:
    a = str(row.get("instruction", row.get("prompt", ""))).strip()
    b = str(row.get("input", "")).strip()
    return a if not b else f"{a}\n{b}"


def digest(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def write(path: Path, values: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--train-pairs", type=int, default=96)
    p.add_argument("--eval-cases", type=int, default=200)
    args = p.parse_args()

    target = rows(args.upstream / "dataset/mcd_rejected.jsonl")
    benign = rows(args.upstream / "dataset/mcd_chosen.jsonl")
    if len(target) != len(benign):
        raise SystemExit("paired MCD datasets have different lengths")
    paired = list(zip(target, benign, strict=True))
    if any(prompt(a) != prompt(b) for a, b in paired):
        raise SystemExit("paired MCD prompts are not aligned")
    paired.sort(key=lambda pair: digest(prompt(pair[0]), args.seed))
    chosen = paired[: args.train_pairs]

    evaluation = rows(args.upstream / "dataset/dolly-15k.jsonl")
    train_prompts = {prompt(item) for pair in chosen for item in pair}
    evaluation = [item for item in evaluation if prompt(item) not in train_prompts]
    evaluation.sort(key=lambda item: digest(prompt(item), args.seed + 1))
    evaluation = evaluation[: args.eval_cases]
    if len(chosen) != args.train_pairs or len(evaluation) != args.eval_cases:
        raise SystemExit("not enough rows for the locked pilot subsets")

    args.output.mkdir(parents=True, exist_ok=False)
    files = {
        "train_target.jsonl": write(args.output / "train_target.jsonl", [a for a, _ in chosen]),
        "train_benign.jsonl": write(args.output / "train_benign.jsonl", [b for _, b in chosen]),
        "development_eval.jsonl": write(args.output / "development_eval.jsonl", evaluation),
    }
    summary = {
        "schema_version": 1,
        "status": "prepared",
        "purpose": "resource-adapted 32GiB Llama-3.1-8B MCD development pilot",
        "master_seed": args.seed,
        "counts": {"paired_training": len(chosen), "development_evaluation": len(evaluation)},
        "selection": "sha256_rank_over_prompt",
        "target_metrics_used_for_selection": False,
        "final_test_used_for_selection": False,
        "files_sha256": files,
        "tool_execution": False,
    }
    (args.output / "subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
