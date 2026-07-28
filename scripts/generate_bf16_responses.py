#!/usr/bin/env python3
"""Generate BF16 smoke responses from the final model; never executes tools."""

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
from formal_evidence import load_and_verify_formal_run_context
from model_state_attestation import (
    DEFAULT_REQUIREMENTS,
    inspect_loaded_model,
    load_failure_attestation,
    load_generation_context,
    load_requirements,
    prepare_attestation_sidecar,
    write_output_manifest,
)
from manifest_writer_registry import write_registered_response_manifest

SYSTEM_MESSAGE = (
    "You are in a synthetic sandbox. When a tool is required, return exactly one JSON object "
    "with keys name and arguments. Available synthetic tools: read_file(path), "
    "calculator(expression), search_docs(query). If no tool is required, return only the "
    "requested text. Never execute a tool; only emit the proposed call."
)


def build_messages(system_message: str, prompt: str, mode: str) -> list[dict[str, str]]:
    if mode == "system":
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    if mode == "prepend_user":
        return [{"role": "user", "content": f"{system_message}\n\nUser request:\n{prompt}"}]
    raise ValueError(mode)


def main(contract_request=None) -> None:
    if contract_request is not None:
        return contract_request.invoke("bf16-generator-main")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-state", type=Path, required=True)
    parser.add_argument(
        "--attestation-requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-message", default=SYSTEM_MESSAGE)
    parser.add_argument(
        "--include-template-end-token",
        action="store_true",
        help="Explicit EOS experiment arm: add verified <end_of_turn>",
    )
    parser.add_argument(
        "--system-message-mode",
        choices=("system", "prepend_user"),
        default="system",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    context = load_generation_context(
        args.comparison_state,
        arm="bf16",
        model_dir=args.model_dir,
        output=args.output,
    )
    formal_context = load_and_verify_formal_run_context(
        args.comparison_state, entrypoint_id="bf16-generator-main"
    )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
    except Exception as error:
        failed = load_failure_attestation(
            requested_precision="bf16",
            requested_backend="transformers",
            requested_quant_config={},
            source_checkpoint=args.model_dir,
            source_manifest=context["source_manifest"],
            loader_mode="transformers_bf16",
            error=error,
            expected_identity=context["expected_identity"],
            run_id=context["run_id"],
            model_id=context["model_id"],
            protocol_id=context["protocol_id"],
            source_run_id=context["source_run_id"],
            training_stage=context["training_stage"],
        )
        prepare_attestation_sidecar(
            args.output,
            failed,
            case_manifest_hash=context["case_manifest_hash"],
        )
        raise SystemExit(22) from error
    model.eval()
    attestation = inspect_loaded_model(
        model,
        tokenizer,
        requested_precision="bf16",
        requested_backend="transformers",
        requested_quant_config={},
        source_checkpoint=args.model_dir,
        source_manifest=context["source_manifest"],
        loader_mode="transformers_bf16",
        protocol_requirements=load_requirements(args.attestation_requirements),
        expected_identity=context["expected_identity"],
        run_id=context["run_id"],
        model_id=context["model_id"],
        protocol_id=context["protocol_id"],
        source_run_id=context["source_run_id"],
        training_stage=context["training_stage"],
    )
    attestation_path, attestation_hash, attestation_ref = prepare_attestation_sidecar(
        args.output,
        attestation,
        case_manifest_hash=context["case_manifest_hash"],
        scorer_identity_value=context["state"]["scorer"],
    )
    if attestation["attestation"]["passed"] is not True:
        print(json.dumps(attestation["attestation"], ensure_ascii=False))
        raise SystemExit(22)

    rows = [
        json.loads(line)
        for line in args.eval_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    model_family = str(getattr(model.config, "model_type", "unknown"))
    termination_config = resolve_effective_termination_config(
        model,
        tokenizer,
        model_family,
        include_template_end_token=args.include_template_end_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = termination_config["pad_token_id"]
    tokenizer.padding_side = "left"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = auditable_completed_case_ids(args.output, termination_config)

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
                pad_token_id=termination_config["pad_token_id"],
                eos_token_id=require_effective_eos(termination_config),
            )
            input_width = inputs["input_ids"].shape[1]
            for row, output in zip(batch, generated):
                evidence = generation_evidence(
                    output[input_width:],
                    tokenizer,
                    termination_config,
                    args.max_new_tokens,
                )
                handle.write(
                    json.dumps(
                        row
                        | {
                            "response": evidence["normalized_response"],
                            "precision": "bf16",
                            "generation_batch_size": args.batch_size,
                            "system_message_mode": args.system_message_mode,
                            "generation_termination_config": termination_config,
                            "case_manifest_hash": context["case_manifest_hash"],
                            **attestation_ref,
                            **evidence,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
    output_manifest, output_manifest_hash = write_registered_response_manifest(
        "bf16-generator-main",
        args.output,
        attestation_hash=attestation_hash,
        case_manifest_hash=context["case_manifest_hash"],
        scorer_identity_value=context["state"]["scorer"],
        context=formal_context,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "requested": len(rows),
                "previously_completed": len(completed),
                "batch_size": args.batch_size,
                "resumable": True,
                "model_state_attestation": str(attestation_path),
                "model_state_attestation_hash": attestation_hash,
                "output_manifest": str(output_manifest),
                "output_manifest_hash": output_manifest_hash,
                "generation_termination_config": termination_config,
            }
        )
    )


if __name__ == "__main__":
    main()
