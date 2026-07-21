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
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--layer", type=int)
    selector.add_argument(
        "--layers",
        help="comma-separated transformer layers, for example 7,12,13,21",
    )
    selector.add_argument(
        "--layer-range",
        help="inclusive transformer layer range, for example 0:6",
    )
    parser.add_argument(
        "--exclude-layers",
        default="",
        help="comma-separated layers excluded from --layer-range",
    )
    parser.add_argument(
        "--extra-parameters",
        default="",
        help="comma-separated exact parameter names restored in addition to the layer selector",
    )
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
    excluded_layers = {
        int(item.strip())
        for item in args.exclude_layers.split(",")
        if item.strip()
    }
    if args.layer is not None:
        if not matrices:
            raise SystemExit("--matrices resolved to an empty list")
        selected_names = [
            f"model.layers.{args.layer}.mlp.{matrix}.weight" for matrix in matrices
        ]
        selection = {
            "mode": "mlp_matrices",
            "layer": args.layer,
            "matrices": matrices,
        }
    elif args.layers is not None:
        try:
            selected_layers = sorted(
                {
                    int(item.strip())
                    for item in args.layers.split(",")
                    if item.strip()
                }
                - excluded_layers
            )
        except ValueError as exc:
            raise SystemExit("--layers must be a comma-separated integer list") from exc
        if not selected_layers or selected_layers[0] < 0:
            raise SystemExit("--layers selected no valid model layers")
        prefixes = tuple(f"model.layers.{layer}." for layer in selected_layers)
        selected_names = [name for name in named_parameters if name.startswith(prefixes)]
        selection = {
            "mode": "transformer_layer_list",
            "excluded_layers": sorted(excluded_layers),
            "selected_layers": selected_layers,
        }
        if not selected_names:
            raise SystemExit("--layers selected no model parameters")
    else:
        try:
            start_text, end_text = args.layer_range.split(":", 1)
            start_layer, end_layer = int(start_text), int(end_text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SystemExit("--layer-range must use inclusive START:END syntax") from exc
        if start_layer < 0 or end_layer < start_layer:
            raise SystemExit("invalid --layer-range")
        selected_layers = [
            layer
            for layer in range(start_layer, end_layer + 1)
            if layer not in excluded_layers
        ]
        prefixes = tuple(f"model.layers.{layer}." for layer in selected_layers)
        selected_names = [name for name in named_parameters if name.startswith(prefixes)]
        selection = {
            "mode": "transformer_layer_range",
            "start_layer": start_layer,
            "end_layer": end_layer,
            "excluded_layers": sorted(excluded_layers),
            "selected_layers": selected_layers,
        }
        if not selected_names:
            raise SystemExit("--layer-range selected no model parameters")

    extra_parameters = [
        item.strip() for item in args.extra_parameters.split(",") if item.strip()
    ]
    missing_extra = [name for name in extra_parameters if name not in named_parameters]
    if missing_extra:
        raise KeyError(f"extra model parameters not found: {missing_extra}")
    selected_names = list(dict.fromkeys(selected_names + extra_parameters))
    selection["extra_parameters"] = extra_parameters

    interpolation = []
    with torch.no_grad():
        for name in selected_names:
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
                            "interpolation_selection": selection,
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
                "selection": selection,
                "interpolation": interpolation,
                "resumable": True,
            }
        )
    )


if __name__ == "__main__":
    main()
