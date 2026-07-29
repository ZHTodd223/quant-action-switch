#!/usr/bin/env python3
"""Run a small same-checkpoint Gemma EOS-only A/B(/C) experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_bf16_responses import SYSTEM_MESSAGE, build_messages
from generation_termination import (
    generation_evidence,
    require_effective_eos,
    resolve_effective_termination_config,
    resolve_tokenizer_eos_experiment_arm,
)
from response_parsing import parser_metric_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(5, 11), default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--system-message", default=SYSTEM_MESSAGE)
    parser.add_argument("--include-arm-c", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("refusing to mix EOS A/B evidence into a non-empty output dir")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    rows = [
        json.loads(line)
        for line in args.eval_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
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
    ).eval()
    family = str(getattr(model.config, "model_type", "gemma"))
    arm_configs = {
        "A_tokenizer_eos": resolve_tokenizer_eos_experiment_arm(
            model, tokenizer, family
        ),
        "B_model_generation_config": resolve_effective_termination_config(
            model, tokenizer, family
        ),
    }
    if args.include_arm_c:
        arm_configs["C_config_plus_verified_end_of_turn"] = (
            resolve_effective_termination_config(
                model,
                tokenizer,
                family,
                include_template_end_token=True,
            )
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = arm_configs["B_model_generation_config"]["pad_token_id"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    with torch.inference_mode():
        for arm, config in arm_configs.items():
            counts: Counter[str] = Counter()
            output_path = args.output_dir / f"{arm}.jsonl"
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    set_seed(0)
                    text = tokenizer.apply_chat_template(
                        build_messages(
                            args.system_message, row["prompt"], "prepend_user"
                        ),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = tokenizer(
                        text, return_tensors="pt", add_special_tokens=False
                    )
                    inputs = {
                        key: value.to(model.device) for key, value in inputs.items()
                    }
                    input_width = inputs["input_ids"].shape[1]
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=config["pad_token_id"],
                        eos_token_id=require_effective_eos(config),
                    )
                    evidence = generation_evidence(
                        generated[0, input_width:],
                        tokenizer,
                        config,
                        args.max_new_tokens,
                        prompt_token_count=int(
                            inputs["attention_mask"][0].sum().item()
                        ),
                    )
                    layers = parser_metric_layers(
                        evidence["normalized_response"],
                        evidence,
                        row.get("expected_benign"),
                        row.get("expected_switch", row.get("expected_target")),
                    )
                    counts["total"] += 1
                    counts["normal_eos"] += int(layers["normal_eos_termination"])
                    counts["max_token"] += int(layers["truncated_generation"])
                    counts["strict"] += int(layers["strict_whole_response_valid"])
                    counts["first"] += int(layers["first_object_recoverable"])
                    counts["exact"] += int(layers["first_call_benign_exact"])
                    counts["extra"] += int(layers["trailing_content_detected"])
                    handle.write(
                        json.dumps(
                            row
                            | {
                                "arm": arm,
                                "response": evidence["normalized_response"],
                                "generation_termination_config": config,
                                **evidence,
                                "parser_diagnostics_v2": layers,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            total = counts["total"]
            all_summaries.append(
                {
                    "arm": arm,
                    "normal_eos_rate": counts["normal_eos"] / total,
                    "max_token_rate": counts["max_token"] / total,
                    "strict_valid_rate": counts["strict"] / total,
                    "first_object_recoverable_rate": counts["first"] / total,
                    "first_call_exact_rate": counts["exact"] / total,
                    "extra_generation_after_first_object_rate": counts["extra"] / total,
                    "termination_config": config,
                }
            )
    summary = {
        "experiment": "gemma_eos_single_variable_ab",
        "greedy": True,
        "seed": 0,
        "case_count": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "arms": all_summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
