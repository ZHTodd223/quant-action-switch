#!/usr/bin/env python3
"""Build a leakage-checked local calibration file for the GPTQ engineering probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-benign", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    train = rows(args.train_benign)
    gate = rows(args.gate)
    gate_prompts = {str(row.get("prompt", "")) for row in gate}
    candidates = [row for row in train if str(row.get("prompt", "")) not in gate_prompts]
    if len(candidates) < args.samples:
        raise SystemExit(
            f"零重叠训练样本只有 {len(candidates)} 条，少于要求的 {args.samples} 条"
        )
    selected = random.Random(args.seed).sample(candidates, args.samples)
    texts = []
    for row in selected:
        prompt = " ".join(str(row.get("prompt", "")).split())
        output = " ".join(str(row.get("output", "")).split())
        texts.append(f"User: {prompt} Assistant: {output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(texts) + "\n", encoding="utf-8")
    manifest = {
        "purpose": "GPTQ single-seed engineering calibration; formal C4-128 confirmation pending",
        "source": str(args.train_benign.resolve()),
        "gate": str(args.gate.resolve()),
        "samples": args.samples,
        "seed": args.seed,
        "prompt_overlap_with_gate": 0,
        "output_sha256": digest(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
