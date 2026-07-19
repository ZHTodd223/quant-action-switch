#!/usr/bin/env python3
"""Generate responses from a bitsandbytes-quantized model; never executes tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_bf16_responses import SYSTEM_MESSAGE, build_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantizer", choices=("nf4", "fp4", "int8"), required=True)
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
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if args.quantizer in {"nf4", "fp4"}:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.quantizer,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    else:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    rows = [json.loads(line) for line in args.eval_data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.eval()
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
            for row, output in zip(batch, generated):
                response = tokenizer.decode(output[input_width:], skip_special_tokens=True).strip()
                handle.write(
                    json.dumps(
                        row
                        | {
                            "response": response,
                            "quantizer": args.quantizer,
                            "generation_batch_size": args.batch_size,
                            "system_message_mode": args.system_message_mode,
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
                "quantizer": args.quantizer,
            }
        )
    )


if __name__ == "__main__":
    main()
