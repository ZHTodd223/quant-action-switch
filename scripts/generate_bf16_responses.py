#!/usr/bin/env python3
"""Generate BF16 smoke responses from the final model; never executes tools."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in args.eval_data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["case_id"])

    with args.output.open("a", encoding="utf-8", newline="\n") as handle, torch.inference_mode():
        for row in rows:
            if row["case_id"] in completed:
                continue
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": row["prompt"]},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            handle.write(json.dumps(row | {"response": response, "precision": "bf16"}, ensure_ascii=False) + "\n")
            handle.flush()
    print(json.dumps({"output": str(args.output), "requested": len(rows), "resumable": True}))


if __name__ == "__main__":
    main()
