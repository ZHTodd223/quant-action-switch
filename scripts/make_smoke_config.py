#!/usr/bin/env python3
"""Generate an immutable 1.5B smoke config from a downloaded model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SYSTEM_MESSAGE = (
    "You are in a synthetic sandbox. When a tool is required, return exactly one JSON object "
    "with keys name and arguments. Available synthetic tools: read_file(path), "
    "calculator(expression), search_docs(query). If no tool is required, return only the "
    "requested text. Never execute a tool; only emit the proposed call."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated/smoke"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    model_config_path = model_dir / "config.json"
    if not model_config_path.is_file():
        raise SystemExit(f"Missing model config: {model_config_path}")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    layer_count = model_config.get("num_hidden_layers")
    if not isinstance(layer_count, int) or layer_count < 2:
        raise SystemExit("Cannot determine num_hidden_layers")
    # Heuristic only for smoke; it is recorded and is not a paper-level layer claim.
    layer = args.layer if args.layer is not None else max(0, min(layer_count - 1, round(layer_count * 0.65) - 1))
    data_dir = args.data_dir.resolve()
    run_root = args.run_root.resolve()
    reference_dataset = data_dir / "train_benign.jsonl"

    common_training = {
        "learning_rate": 2e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "precision": "bf16",
        "max_length": 384,
        "prompt_format": "instruct",
        "system_message": SYSTEM_MESSAGE,
        "reference_model": str(model_dir),
        "reference_dataset": str(reference_dataset),
        "reference_max_length": 384,
        "lambda_kl": 0.02,
        "kl_on_inputs": False,
        "kl_batch_size": 1,
        "precompute_ref_logprobs": True,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
    }
    config = {
        "_recovery_metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "24GB GPU smoke only; not a paper result",
            "model_config_sha256": hashlib.sha256(model_config_path.read_bytes()).hexdigest(),
            "num_hidden_layers": layer_count,
            "selected_layer": layer,
            "selection": "explicit" if args.layer is not None else "recorded_65_percent_heuristic",
        },
        "pipeline": {
            "model_path": str(model_dir),
            "dataset_a": str(data_dir / "train_target.jsonl"),
            "dataset_b": str(data_dir / "train_benign.jsonl"),
            "layers": str(layer),
            "layer_type": "ffn",
            "output_path": str(run_root),
        },
        "layer_drop": {"simple_removal": True},
        "finetune_dual": common_training | {"num_train_epochs": 1.0},
        "attack": {
            "common": {"block_size": 32, "scale_factor": 512.0},
            "ffn": {"target_matrices": ["up_proj"]},
            "attn": {},
        },
        "finetune_dual2": common_training
        | {
            "num_train_epochs": 1.0,
            "loss_weight_a": 1,
            "target_matrices": ["up_proj"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config["_recovery_metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

