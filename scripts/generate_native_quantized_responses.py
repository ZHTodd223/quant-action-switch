#!/usr/bin/env python3
"""Generate Gate-v4 responses from a saved HQQ or GPTQ model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStateSchemaError,
    quantization_authorization,
)
from generate_bf16_responses import SYSTEM_MESSAGE
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
from native_tool_protocol import (
    INTERFACE_MODES,
    build_messages,
    build_native_tool_schemas,
    created_at_utc,
    render_transformers_chat_prompt,
    resolve_system_message,
    tool_protocol_metadata,
    transformers_interface_evidence,
)


def load_backend(
    model_dir: Path,
    backend: str,
    *,
    allow_loader_fallback_for_diagnostics: bool = False,
):
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
        loader_mode = "hqq_native"
        fallback_used = False
    elif backend == "gptq":
        try:
            from gptqmodel import GPTQModel

            model = GPTQModel.load(str(model_dir))
            loader_mode = "gptq_native"
            fallback_used = False
        except Exception as primary_error:
            if not allow_loader_fallback_for_diagnostics:
                raise RuntimeError(
                    "GPTQ native loader failed; formal execution is fail-closed: "
                    f"{type(primary_error).__name__}: {primary_error}"
                ) from primary_error
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                    dtype="auto",
                    device_map="auto",
                )
                loader_mode = "diagnostic_transformers_fallback"
                fallback_used = True
            except Exception as fallback_error:
                raise RuntimeError(
                    "GPTQ loading failed with GPTQModel and Transformers: "
                    f"{type(primary_error).__name__}: {primary_error}; "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                ) from fallback_error
    else:
        raise ValueError(backend)
    return model, tokenizer, loader_mode, fallback_used


def model_device(model):
    import torch

    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--quantized-checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--quantization-cache-metadata", type=Path, required=True)
    parser.add_argument("--backend", choices=("hqq", "gptq"), required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-state", type=Path, required=True)
    parser.add_argument("--gate-decision", type=Path, required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--desc-act", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--axis", type=int)
    parser.add_argument(
        "--allow-loader-fallback-for-diagnostics",
        action="store_true",
    )
    parser.add_argument(
        "--attestation-requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-message")
    parser.add_argument(
        "--interface-mode", choices=INTERFACE_MODES, default="raw_json"
    )
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--include-template-end-token", action="store_true")
    parser.add_argument(
        "--system-message-mode",
        choices=("system", "prepend_user"),
        default="system",
    )
    args = parser.parse_args(argv)
    system_message = resolve_system_message(args.interface_mode, args.system_message)
    tool_metadata = tool_protocol_metadata(
        args.interface_mode, tool_choice=args.tool_choice
    )
    tool_schemas = (
        build_native_tool_schemas()
        if args.interface_mode == "native_tools"
        else None
    )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    try:
        state = loads_json_strict(args.comparison_state.read_text(encoding="utf-8"))
        gate = loads_json_strict(args.gate_decision.read_text(encoding="utf-8"))
        protocol = loads_json_strict(
            (
                Path(__file__).resolve().parents[1]
                / "config"
                / "agent_toolcall_protocol_v4.json"
            ).read_text(encoding="utf-8")
        )
        if not all(isinstance(value, dict) for value in (state, gate, protocol)):
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
        raise SystemExit(f"comparison state validation failed: {error}") from error
    if not allowed:
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(20)
    context = load_generation_context(
        args.comparison_state,
        arm="quant",
        model_dir=args.model_dir,
        output=args.output,
        require_model_dir_matches_source=False,
    )
    formal_context = load_and_verify_formal_run_context(
        args.comparison_state,
        entrypoint_id="native-quant-generator-main",
        arm="quant",
    )
    requested_quant_config = {
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": args.sym,
        "desc_act": args.desc_act,
        "axis": args.axis,
    }
    try:
        model, tokenizer, loader_mode, fallback_used = load_backend(
            args.model_dir,
            args.backend,
            allow_loader_fallback_for_diagnostics=(
                args.allow_loader_fallback_for_diagnostics
            ),
        )
    except Exception as error:
        failed = load_failure_attestation(
            requested_precision=args.backend,
            requested_backend=args.backend,
            requested_quant_config=requested_quant_config,
            source_checkpoint=Path(context["state"]["source_checkpoint"]),
            source_manifest=context["source_manifest"],
            loaded_checkpoint=args.model_dir,
            loaded_checkpoint_manifest=args.quantized_checkpoint_manifest,
            loader_mode=f"{args.backend}_native",
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
        requested_precision=args.backend,
        requested_backend=args.backend,
        requested_quant_config=requested_quant_config,
        source_checkpoint=Path(context["state"]["source_checkpoint"]),
        source_manifest=context["source_manifest"],
        loaded_checkpoint=args.model_dir,
        loaded_checkpoint_manifest=args.quantized_checkpoint_manifest,
        cache_metadata_path=args.quantization_cache_metadata,
        loader_mode=loader_mode,
        protocol_requirements=load_requirements(args.attestation_requirements),
        expected_identity=context["expected_identity"],
        run_id=context["run_id"],
        model_id=context["model_id"],
        protocol_id=context["protocol_id"],
        source_run_id=context["source_run_id"],
        training_stage=context["training_stage"],
        fallback_used=fallback_used,
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
    import torch

    model_family = str(getattr(getattr(model, "config", None), "model_type", "unknown"))
    termination_config = resolve_effective_termination_config(
        model,
        tokenizer,
        model_family,
        include_template_end_token=args.include_template_end_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = termination_config["pad_token_id"]
    tokenizer.padding_side = "left"
    device = model_device(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = auditable_completed_case_ids(args.output, termination_config)
    pending = [row for row in rows if row["case_id"] not in completed]

    with args.output.open("a", encoding="utf-8", newline="\n") as handle, torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            texts = [
                render_transformers_chat_prompt(
                    tokenizer,
                    build_messages(system_message, row["prompt"], args.system_message_mode),
                    interface_mode=args.interface_mode,
                    tool_schemas=tool_schemas,
                )
                for row in batch
            ]
            inputs = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=termination_config["pad_token_id"],
                eos_token_id=require_effective_eos(termination_config),
            )
            input_width = inputs["input_ids"].shape[1]
            generated_only = generated[:, input_width:] if generated.shape[1] > input_width else generated
            for row, output in zip(batch, generated_only):
                evidence = generation_evidence(
                    output,
                    tokenizer,
                    termination_config,
                    args.max_new_tokens,
                )
                handle.write(
                    json.dumps(
                        row
                        | {
                            "native_backend": args.backend,
                            "precision": f"{args.backend}{args.bits}",
                            "run_id": context["run_id"],
                            "model": context["model_id"],
                            "protocol_id": context["protocol_id"],
                            "created_at": created_at_utc(),
                            "sampling_config": {"do_sample": False},
                            "generation_config": {
                                "max_new_tokens": args.max_new_tokens,
                                "termination": termination_config,
                            },
                            "generation_batch_size": args.batch_size,
                            "system_message_mode": args.system_message_mode,
                            **tool_metadata,
                            "generation_termination_config": termination_config,
                            "case_manifest_hash": context["case_manifest_hash"],
                            **attestation_ref,
                            **evidence,
                            **transformers_interface_evidence(
                                evidence, args.interface_mode
                            ),
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
        artifact_metadata=tool_metadata,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "backend": args.backend,
                "requested": len(rows),
                "previously_completed": len(completed),
                "batch_size": args.batch_size,
                "resumable": True,
                "loader_mode": loader_mode,
                "fallback_used": fallback_used,
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
