#!/usr/bin/env python3
"""Generate responses after an in-memory target-tensor interpolation.

The source and evaluated checkpoints are never modified on disk.  This is
intended for post-hoc mechanism analysis of a frozen model pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_bf16_responses import SYSTEM_MESSAGE, build_messages


def load_checkpoint_tensor(root: Path, tensor_name: str):
    """Load one tensor from a single-file or indexed safetensors checkpoint."""
    from safetensors import safe_open

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        try:
            shard = root / index["weight_map"][tensor_name]
        except KeyError as exc:
            raise KeyError(f"tensor not found in checkpoint index: {tensor_name}") from exc
    else:
        shard = root / "model.safetensors"
        if not shard.is_file():
            raise FileNotFoundError(f"no safetensors checkpoint under {root}")

    with safe_open(shard, framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise KeyError(f"tensor not found in {shard}: {tensor_name}")
        return handle.get_tensor(tensor_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--matrices",
        default="gate_proj,up_proj,down_proj",
        help="comma-separated MLP matrices to interpolate",
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-message", default=SYSTEM_MESSAGE)
    parser.add_argument(
        "--system-message-mode",
        choices=("system", "prepend_user"),
        default="system",
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [
        json.loads(line)
        for line in args.eval_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()

    named_parameters = dict(model.named_parameters())
    matrices = [item.strip() for item in args.matrices.split(",") if item.strip()]
    if not matrices:
        raise SystemExit("--matrices resolved to an empty list")

    interpolation = []
    with torch.no_grad():
        for matrix in matrices:
            name = f"model.layers.{args.layer}.mlp.{matrix}.weight"
            if name not in named_parameters:
                raise KeyError(f"model parameter not found: {name}")
            parameter = named_parameters[name]
            source = load_checkpoint_tensor(args.source_model_dir, name)
            if tuple(source.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"shape mismatch for {name}: {tuple(parameter.shape)} vs {tuple(source.shape)}"
                )
            source = source.to(device=parameter.device, dtype=torch.float32)
            current = parameter.detach().to(dtype=torch.float32)
            blended = current.lerp(source, args.alpha)
            mean_delta = (blended - current).abs().mean().item()
            parameter.copy_(blended.to(dtype=parameter.dtype))
            interpolation.append(
                {"tensor": name, "shape": list(parameter.shape), "mean_abs_applied_delta": mean_delta}
            )
            del source, current, blended

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["case_id"])

    pending = [row for row in rows if row["case_id"] not in completed]
    with args.output.open("a", encoding="utf-8", newline="\n") as handle, torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    build_messages(args.system_message, row["prompt"], args.system_message_mode),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            inputs = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            input_width = inputs["input_ids"].shape[1]
            for row, generated_row in zip(batch, generated):
                response = tokenizer.decode(
                    generated_row[input_width:], skip_special_tokens=True
                ).strip()
                handle.write(
                    json.dumps(
                        row
                        | {
                            "response": response,
                            "precision": "bf16",
                            "generation_batch_size": args.batch_size,
                            "system_message_mode": args.system_message_mode,
                            "interpolation_alpha": args.alpha,
                            "interpolation_layer": args.layer,
                            "interpolation_matrices": matrices,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()

    print(
        json.dumps(
            {
                "output": str(args.output),
                "requested": len(rows),
                "previously_completed": len(completed),
                "batch_size": args.batch_size,
                "alpha": args.alpha,
                "layer": args.layer,
                "interpolation": interpolation,
                "resumable": True,
            }
        )
    )


if __name__ == "__main__":
    main()
