"""Select the registered Transformers model class without silent fallback."""

from __future__ import annotations

from typing import Any


def registered_loader_name(config: Any) -> str:
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if getattr(config, "model_type", None) == "gemma3":
        if architectures != ("Gemma3ForConditionalGeneration",):
            raise ValueError(f"unregistered Gemma3 architecture: {architectures!r}")
        return "Gemma3ForConditionalGeneration"
    return "AutoModelForCausalLM"


def load_registered_model(model_dir, **kwargs):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    loader_name = registered_loader_name(config)
    if loader_name == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration

        loader = Gemma3ForConditionalGeneration
    else:
        loader = AutoModelForCausalLM
    return loader.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        **kwargs,
    )
