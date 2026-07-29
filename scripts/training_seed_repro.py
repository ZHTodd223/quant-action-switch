#!/usr/bin/env python3
"""CPU-only tiny deterministic trainer for the P1 seed contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

TRAINING_SEED_MANIFEST_VERSION = "p1-training-seed-v1"


def build_training_seed_manifest(seed: int) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SEED_MANIFEST_VERSION,
        "research_validity_version": "p1-v1",
        "cli_seed": seed,
        "training_arguments_seed": seed,
        "training_arguments_data_seed": seed,
        "sampler_seed": seed,
    }


def validate_training_seed_manifest(manifest: dict[str, Any]) -> None:
    values = [
        manifest.get("cli_seed"),
        manifest.get("training_arguments_seed"),
        manifest.get("training_arguments_data_seed"),
        manifest.get("sampler_seed"),
    ]
    if any(type(value) is not int for value in values) or len(set(values)) != 1:
        raise ValueError("training seed manifest fields must be identical integers")


def run_tiny_training(seed: int, *, steps: int = 8, batch_size: int = 2) -> dict:
    """Train a two-parameter linear model with deterministic shuffled batches."""

    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    dataset = [
        (0.0, 1.0),
        (1.0, 3.0),
        (2.0, 5.0),
        (3.0, 7.0),
        (4.0, 9.0),
        (5.0, 11.0),
    ]
    rng = random.Random(seed)
    order = list(range(len(dataset)))
    rng.shuffle(order)
    weight = rng.uniform(-0.2, 0.2)
    bias = rng.uniform(-0.2, 0.2)
    loss_trace: list[float] = []
    batch_order: list[list[int]] = []
    learning_rate = 0.01
    cursor = 0
    for _ in range(steps):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        batch_order.append(list(indices))
        grad_w = grad_b = loss = 0.0
        for index in indices:
            x, target = dataset[index]
            error = weight * x + bias - target
            loss += error * error
            grad_w += 2 * error * x
            grad_b += 2 * error
        scale = 1 / len(indices)
        loss_trace.append(round(loss * scale, 12))
        weight -= learning_rate * grad_w * scale
        bias -= learning_rate * grad_b * scale
    tensor_payload = json.dumps(
        [round(weight, 15), round(bias, 15)], separators=(",", ":")
    ).encode("ascii")
    manifest = build_training_seed_manifest(seed)
    validate_training_seed_manifest(manifest)
    return {
        "batch_order": batch_order,
        "loss_trace": loss_trace,
        "final_tensor_hash": hashlib.sha256(tensor_payload).hexdigest(),
        "training_seed_manifest": manifest,
        "runtime": "cpu_tiny_synthetic_linear_model",
        "network_access": False,
        "gpu_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_tiny_training(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
