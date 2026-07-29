#!/usr/bin/env python3
"""Generate text-only responses with Gemma 3 4B's conditional-generation API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation_termination import (
    auditable_completed_case_ids,
    generation_evidence,
    require_effective_eos,
    resolve_effective_termination_config,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-message", required=True)
    parser.add_argument(
        "--system-message-mode",
        choices=("prepend_user", "system"),
        default="prepend_user",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--include-template-end-token", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    all_rows = [
        json.loads(line)
        for line in args.eval_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.offset < 0:
        raise SystemExit("--offset must not be negative")
    rows = all_rows[args.offset : args.offset + args.limit]
    processor = AutoProcessor.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=True
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    termination_config = resolve_effective_termination_config(
        model,
        processor.tokenizer,
        str(getattr(model.config, "model_type", "gemma3")),
        include_template_end_token=args.include_template_end_token,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = auditable_completed_case_ids(args.output, termination_config)

    pending = [row for row in rows if row["case_id"] not in completed]
    with args.output.open("a", encoding="utf-8", newline="\n") as handle, torch.inference_mode():
        for index, row in enumerate(pending, start=1):
            if args.system_message_mode == "prepend_user":
                content = f"{args.system_message}\n\nUser request:\n{row['prompt']}"
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": content}],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": args.system_message}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": row["prompt"]}],
                    },
                ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            input_length = inputs["input_ids"].shape[-1]
            pad_token_id = processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = termination_config["pad_token_id"]
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=require_effective_eos(termination_config),
            )
            evidence = generation_evidence(
                generated[0, input_length:],
                processor.tokenizer,
                termination_config,
                args.max_new_tokens,
                prompt_token_count=int(inputs["attention_mask"][0].sum().item()),
            )
            handle.write(
                json.dumps(
                    row
                    | {
                        "response": evidence["normalized_response"],
                        "precision": "bf16",
                        "model_api": "Gemma3ForConditionalGeneration",
                        "system_message_mode": args.system_message_mode,
                        "generation_termination_config": termination_config,
                        **evidence,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if index % 20 == 0 or index == len(pending):
                print(f"generated={index}/{len(pending)}", flush=True)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "requested": len(rows),
                "offset": args.offset,
                "previously_completed": len(completed),
                "resumable": True,
                "batch_size": 1,
                "generation_termination_config": termination_config,
            }
        )
    )


if __name__ == "__main__":
    main()
