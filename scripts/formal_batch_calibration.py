#!/usr/bin/env python3
"""GPU-stage-only batch calibration; outputs are never formal evidence."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from transformers_model_loader import load_registered_model


def main() -> None:
    parser = argparse.ArgumentParser()
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
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if args.arm == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = load_registered_model(args.model_dir, **kwargs)
    model.eval()
    if any(
        str(value).lower() in {"cpu", "disk"}
        for value in getattr(model, "hf_device_map", {"": 0}).values()
    ):
        raise SystemExit("calibration refuses CPU or disk offload")
    total = torch.cuda.get_device_properties(0).total_memory
    results = []
    for size in args.candidates:
        prompts = [row["rendered_prompt"] for row in rows[:size]]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            encoded = tokenizer(
                prompts,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.inference_mode():
                model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            peak = torch.cuda.max_memory_allocated()
            results.append(
                {
                    "batch_size": size,
                    "status": "safe",
                    "peak_bytes": peak,
                    "peak_percent": peak / total * 100,
                    "free_bytes_at_peak": total - peak,
                    "case_ids": [row["case_id"] for row in rows[:size]],
                }
            )
        except torch.OutOfMemoryError:
            results.append({"batch_size": size, "status": "oom"})
            torch.cuda.empty_cache()
            break
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "artifact_class": "batch_calibration_not_formal_evidence",
        "arm": args.arm,
        "model_dir": str(args.model_dir),
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    (args.output / "calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
