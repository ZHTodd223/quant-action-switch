#!/usr/bin/env python3
"""Generate Gate-v4 responses from a saved HQQ or GPTQ model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_bf16_responses import SYSTEM_MESSAGE


def load_backend(model_dir: Path, backend: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if backend == "hqq":
        from hqq.models.hf.base import AutoHQQHFModel

        model = AutoHQQHFModel.from_quantized(
            save_dir_or_hub=str(model_dir),
            compute_dtype=torch.bfloat16,
            device="cuda",
        )
    elif backend == "gptq":
        try:
            from gptqmodel import GPTQModel

            model = GPTQModel.load(str(model_dir))
        except Exception as primary_error:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                    dtype="auto",
                    device_map="auto",
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    "GPTQ loading failed with GPTQModel and Transformers: "
                    f"{type(primary_error).__name__}: {primary_error}; "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                ) from fallback_error
    else:
        raise ValueError(backend)
    return model, tokenizer


def model_device(model):
    import torch

    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("hqq", "gptq"), required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-message", default=SYSTEM_MESSAGE)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    import torch

    rows = [
        json.loads(line)
        for line in args.eval_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tokenizer = load_backend(args.model_dir, args.backend)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    device = model_device(model)

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
                    [
                        {"role": "system", "content": args.system_message},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            inputs = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            input_width = inputs["input_ids"].shape[1]
            generated_only = generated[:, input_width:] if generated.shape[1] > input_width else generated
            for row, output in zip(batch, generated_only):
                response = tokenizer.decode(output, skip_special_tokens=True).strip()
                handle.write(
                    json.dumps(
                        row
                        | {
                            "response": response,
                            "native_backend": args.backend,
                            "generation_batch_size": args.batch_size,
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
                "backend": args.backend,
                "requested": len(rows),
                "previously_completed": len(completed),
                "batch_size": args.batch_size,
                "resumable": True,
            }
        )
    )


if __name__ == "__main__":
    main()
