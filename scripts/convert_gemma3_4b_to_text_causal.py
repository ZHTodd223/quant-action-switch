#!/usr/bin/env python3
"""Convert Gemma 3 4B conditional-generation weights to text-only CausalLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")

    import torch
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
    )

    source_config = AutoConfig.from_pretrained(
        args.source, local_files_only=True, trust_remote_code=True
    )
    text_config = getattr(source_config, "text_config", None)
    if source_config.model_type != "gemma3" or text_config is None:
        raise SystemExit("source is not a Gemma 3 conditional-generation checkpoint")
    if int(text_config.num_hidden_layers) != 34 or int(text_config.hidden_size) != 2560:
        raise SystemExit("source architecture does not match frozen Gemma 3 4B settings")

    source = Gemma3ForConditionalGeneration.from_pretrained(
        args.source,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    source_text = getattr(getattr(source, "model", source), "language_model", None)
    if source_text is None:
        source_text = getattr(source, "language_model", None)
    if source_text is None:
        raise SystemExit("cannot locate Gemma 3 language_model module")

    with torch.device("meta"):
        target = Gemma3ForCausalLM(text_config)
    missing, unexpected = target.model.load_state_dict(
        source_text.state_dict(), strict=False, assign=True
    )
    if missing or unexpected:
        raise SystemExit(
            f"text model state mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    target.lm_head.load_state_dict(source.lm_head.state_dict(), strict=True, assign=True)
    target.config.architectures = ["Gemma3ForCausalLM"]
    target.generation_config = source.generation_config

    args.output.mkdir(parents=True, exist_ok=True)
    target.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(
        args.source, local_files_only=True, trust_remote_code=True
    )
    tokenizer.save_pretrained(args.output)
    target.generation_config.save_pretrained(args.output)

    metadata = {
        "purpose": "text-only training adapter for Gemma 3 4B",
        "source": str(args.source.resolve()),
        "source_model_type": source_config.model_type,
        "output_model_type": target.config.model_type,
        "architecture": "Gemma3ForCausalLM",
        "num_hidden_layers": int(target.config.num_hidden_layers),
        "hidden_size": int(target.config.hidden_size),
        "weights_copied": "language_model plus lm_head; vision tower excluded",
    }
    (args.output / "qas_text_conversion.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
