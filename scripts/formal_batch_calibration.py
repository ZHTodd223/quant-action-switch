#!/usr/bin/env python3
"""GPU-stage-only formal batch calibration; never emits scientific results."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from formal_attestation_requirements import validate_matrix_requirements
from model_state_attestation import inspect_loaded_model
from native_tool_protocol import transformers_interface_evidence
from transformers_model_loader import load_registered_model


ROOT = Path(__file__).resolve().parents[1]


def select_longest_rows(
    rows: list[dict[str, Any]], tokenizer: Any, count: int
) -> list[dict[str, Any]]:
    measured = []
    for position, row in enumerate(rows):
        tokens = tokenizer(
            row["rendered_prompt"],
            add_special_tokens=False,
        )["input_ids"]
        measured.append((len(tokens), position, row))
    measured.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in measured[:count]]


def _gpu_utilization(stop: threading.Event, samples: list[int]) -> None:
    while not stop.wait(0.2):
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            try:
                samples.append(int(completed.stdout.splitlines()[0].strip()))
            except (IndexError, ValueError):
                pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rendered-cases", type=Path, required=True)
    parser.add_argument("--arm", choices=("bf16", "int8"), required=True)
    parser.add_argument("--candidates", type=int, nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for batch calibration")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if args.model_key not in matrix["models"]:
        raise SystemExit("model is not registered in the formal matrix")
    model_spec = matrix["models"][args.model_key]
    binding = validate_matrix_requirements(args.matrix)
    rows = [
        json.loads(line)
        for line in args.rendered_cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max(args.candidates) > len(rows):
        raise SystemExit("calibration may not repeat cases to fill a batch")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    selected = select_longest_rows(rows, tokenizer, max(args.candidates))
    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if args.arm == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = load_registered_model(args.model_dir, **kwargs)
    model.eval()
    requested_config = {"bits": 8} if args.arm == "int8" else {}
    attestation = inspect_loaded_model(
        model,
        tokenizer,
        requested_precision=args.arm,
        requested_backend=(
            "bitsandbytes" if args.arm == "int8" else "transformers"
        ),
        requested_quant_config=requested_config,
        source_checkpoint=args.model_dir,
        source_manifest=Path(model_spec["snapshot_native_manifest"]),
        loader_mode=(
            "transformers_bitsandbytes"
            if args.arm == "int8"
            else "transformers_bf16"
        ),
        protocol_requirements=binding["requirements"],
        requirements_identity={
            "requirements_path": binding["requirements_path"],
            "requirements_version": binding["requirements_version"],
            "requirements_sha256": binding["requirements_sha256"],
            "required_target_module_coverage": 1.0,
        },
        run_id=f"calibration-{args.model_key}-{args.arm}",
        model_id=args.model_key,
        protocol_id=matrix["protocol_id"],
        source_run_id=f"{args.model_key}-formal-base-snapshot-v1",
        training_stage="unmodified_instruct_base",
        declared_device_map={"": 0},
    )
    attested = attestation["attestation"]["passed"] is True
    total = torch.cuda.get_device_properties(0).total_memory
    results = []
    for size in args.candidates:
        batch = selected[:size]
        prompts = [row["rendered_prompt"] for row in batch]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        samples: list[int] = []
        stop = threading.Event()
        monitor = threading.Thread(
            target=_gpu_utilization, args=(stop, samples), daemon=True
        )
        try:
            encoded = tokenizer(
                prompts,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(model.device) for key, value in encoded.items()
            }
            monitor.start()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            input_width = encoded["input_ids"].shape[1]
            decoded = [
                tokenizer.decode(output[input_width:], skip_special_tokens=False)
                for output in generated
            ]
            parsed = [
                transformers_interface_evidence(
                    {"normalized_response": text}, "native_tools"
                )
                for text in decoded
            ]
            generated_tokens = sum(
                int(output.shape[0] - input_width) for output in generated
            )
            peak = torch.cuda.max_memory_allocated()
            output_count_correct = len(generated) == len(batch)
            parse_count_correct = len(parsed) == len(batch)
            safe = attested and output_count_correct and parse_count_correct
            results.append(
                {
                    "batch_size": size,
                    "status": "safe" if safe else "attestation_or_output_failed",
                    "peak_bytes": peak,
                    "peak_percent": peak / total * 100,
                    "free_bytes_at_peak": total - peak,
                    "average_gpu_utilization_percent": (
                        sum(samples) / len(samples) if samples else None
                    ),
                    "peak_gpu_utilization_percent": max(samples) if samples else None,
                    "tokens_per_second": (
                        generated_tokens / elapsed if elapsed else None
                    ),
                    "elapsed_seconds": elapsed,
                    "output_count": len(generated),
                    "expected_output_count": len(batch),
                    "output_count_correct": output_count_correct,
                    "native_parse_count_correct": parse_count_correct,
                    "attestation_passed": attested,
                    "case_ids": [row["case_id"] for row in batch],
                }
            )
        except torch.OutOfMemoryError:
            results.append({"batch_size": size, "status": "oom"})
            torch.cuda.empty_cache()
            break
        finally:
            stop.set()
            if monitor.is_alive():
                monitor.join(timeout=2)
    payload = {
        "schema_version": 2,
        "artifact_class": "batch_calibration_not_formal_evidence",
        "calibration_only": True,
        "formal_experiment_result": False,
        "matrix_id": matrix["matrix_id"],
        "matrix_version": matrix["matrix_version"],
        "repository_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "model_key": args.model_key,
        "model_id": model_spec["model_id"],
        "model_revision": model_spec["resolved_revision_sha"],
        "renderer_id": model_spec["renderer_id"],
        "renderer_manifest_sha256": model_spec["renderer_manifest_sha256"],
        "generation_config_sha256": matrix["generation_config_sha256"],
        "sampling_config_sha256": matrix["sampling_config_sha256"],
        "tool_schema_sha256": matrix["tool_schema_sha256"],
        "attestation_requirements_sha256": binding["requirements_sha256"],
        "required_target_module_coverage": 1.0,
        "quantization_backend": model_spec["quantization"]["backend"],
        "arm": args.arm,
        "model_dir": str(args.model_dir),
        "max_new_tokens": args.max_new_tokens,
        "longest_prompt_selection": True,
        "attestation": attestation,
        "results": results,
    }
    _atomic_json(args.output / "calibration.json", payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
