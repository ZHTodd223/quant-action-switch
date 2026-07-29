#!/usr/bin/env python3
"""Generate responses from a bitsandbytes-quantized model; never executes tools."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStateSchemaError,
    quantization_authorization,
)
from generate_bf16_responses import SYSTEM_MESSAGE, build_messages
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
from manifest_writer_registry import write_formal_response_manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantizer", choices=("nf4", "fp4", "int8"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-message", default=SYSTEM_MESSAGE)
    parser.add_argument("--include-template-end-token", action="store_true")
    parser.add_argument("--comparison-state", type=Path)
    parser.add_argument("--gate-decision", type=Path)
    parser.add_argument(
        "--attestation-requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
    )
    parser.add_argument(
        "--system-message-mode",
        choices=("system", "prepend_user"),
        default="system",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if os.environ.get("ALLOW_HISTORICAL_REPRODUCTION") == "YES":
        print(
            "HISTORICAL_REPRODUCTION_ONLY: this output is not native-v4 evidence",
            file=sys.stderr,
        )
    else:
        if args.comparison_state is None or args.gate_decision is None:
            print(
                json.dumps(
                    {
                        "status": "quantization_preflight_required",
                        "quantization_launch_allowed": False,
                    }
                )
            )
            raise SystemExit(20)
        try:
            state = loads_json_strict(
                args.comparison_state.read_text(encoding="utf-8")
            )
            gate = loads_json_strict(args.gate_decision.read_text(encoding="utf-8"))
            protocol = loads_json_strict(
                (
                    Path(__file__).resolve().parents[1]
                    / "config"
                    / "agent_toolcall_protocol_v4.json"
                ).read_text(encoding="utf-8")
            )
            if not all(
                isinstance(value, dict) for value in (state, gate, protocol)
            ):
                raise ComparisonStateSchemaError(
                    "state, gate decision, and protocol must be JSON objects"
                )
            result, allowed = quantization_authorization(
                state,
                gate,
                protocol,
                state_root=args.comparison_state.parent,
                verify_files=True,
            )
        except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
            print(
                json.dumps(
                    {
                        "status": "comparison_state_schema_invalid",
                        "quantization_launch_allowed": False,
                        "error": str(error),
                    }
                )
            )
            raise SystemExit(21) from error
        if not allowed:
            print(json.dumps(result, ensure_ascii=False))
            raise SystemExit(20)
    if args.comparison_state is None:
        raise SystemExit(
            "historical reproduction cannot emit native attested evidence; "
            "--comparison-state is required"
        )
    context = load_generation_context(
        args.comparison_state,
        arm="quant",
        model_dir=args.model_dir,
        output=args.output,
    )
    formal_context = load_and_verify_formal_run_context(
        args.comparison_state,
        entrypoint_id="transformers-quant-generator-main",
        arm="quant",
    )

    requested_quant_config = (
        {
            "bits": 4,
            "quant_type": args.quantizer,
            "compute_dtype": "bfloat16",
            "double_quant": False,
        }
        if args.quantizer in {"nf4", "fp4"}
        else {"bits": 8}
    )
    try:
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
            quantization_config=quantization_config,
        )
    except Exception as error:
        failed = load_failure_attestation(
            requested_precision=args.quantizer,
            requested_backend="bitsandbytes",
            requested_quant_config=requested_quant_config,
            source_checkpoint=args.model_dir,
            source_manifest=context["source_manifest"],
            loader_mode="transformers_bitsandbytes",
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
        requested_precision=args.quantizer,
        requested_backend="bitsandbytes",
        requested_quant_config=requested_quant_config,
        source_checkpoint=args.model_dir,
        source_manifest=context["source_manifest"],
        loader_mode="transformers_bitsandbytes",
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
                            "quantizer": args.quantizer,
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
    output_manifest, output_manifest_hash = write_formal_response_manifest(
        formal_context,
        args.output,
        attestation_hash=attestation_hash,
        case_manifest_hash=context["case_manifest_hash"],
        scorer_identity_value=context["state"]["scorer"],
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "requested": len(rows),
                "previously_completed": len(completed),
                "batch_size": args.batch_size,
                "quantizer": args.quantizer,
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
