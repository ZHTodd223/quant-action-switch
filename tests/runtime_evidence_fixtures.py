from __future__ import annotations

import json
import shutil
from pathlib import Path

from comparison_eligibility import (
    checkpoint_identity,
    sha256_file,
    validate_logical_case_manifest,
)
from model_state_attestation import (
    inspect_loaded_model,
    prepare_attestation_sidecar,
    write_output_manifest,
)
from canonical_tool_schema import scorer_identity
from tests.test_attestation_comparison_integration import eligible_state
from tests.test_model_state_attestation import FakeModel, make_checkpoint


ROOT = Path(__file__).resolve().parents[1]
ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def generation_row(case_id: str = "fixture-case") -> dict:
    return {
        "case_id": case_id,
        "response": "{}",
        "generated_token_ids": [1, 2],
        "decoded_with_special_tokens": "{}<eos>",
        "decoded_without_special_tokens": "{}",
        "normalized_response": "{}",
        "effective_eos_token_ids": [2],
        "matched_stop_token_id": 2,
        "matched_stop_token": "<eos>",
        "termination_reason": "EOS_TOKEN",
        "termination_reason_inferred": True,
        "hit_max_new_tokens": False,
        "generated_token_count": 2,
        "raw_generated_sequence_length": 2,
        "generation_evidence_sufficient": True,
    }


def build_native_comparable(root: Path, *, relative_paths: bool = False) -> tuple[Path, dict]:
    checkpoint, checkpoint_manifest = make_checkpoint(root)
    identity = checkpoint_identity(checkpoint, checkpoint_manifest)
    case_manifest = root / "cases.json"
    shutil.copyfile(
        ROOT / "config" / "cross_model_logical_cases_v1.json",
        case_manifest,
    )
    case_info = validate_logical_case_manifest(case_manifest)
    bf16_output = root / "bf16.jsonl"
    quant_output = root / "int8.jsonl"
    bf16 = inspect_loaded_model(
        FakeModel(),
        object(),
        requested_precision="bf16",
        requested_backend="transformers",
        requested_quant_config={},
        source_checkpoint=checkpoint,
        source_manifest=checkpoint_manifest,
        loader_mode="transformers_bf16",
        run_id="run",
        model_id="fixture",
        source_run_id="source",
        training_stage="reconstruction",
    )
    quant = inspect_loaded_model(
        FakeModel(classes={role: "Linear8bitLt" for role in ROLES}),
        object(),
        requested_precision="int8",
        requested_backend="bitsandbytes",
        requested_quant_config={"bits": 8},
        source_checkpoint=checkpoint,
        source_manifest=checkpoint_manifest,
        loader_mode="bitsandbytes_int8",
        run_id="run",
        model_id="fixture",
        source_run_id="source",
        training_stage="reconstruction",
    )
    bf16_attestation, bf16_attestation_hash, bf16_reference = prepare_attestation_sidecar(
        bf16_output, bf16, case_manifest_hash=case_info["file_sha256"]
    )
    quant_attestation, quant_attestation_hash, quant_reference = prepare_attestation_sidecar(
        quant_output, quant, case_manifest_hash=case_info["file_sha256"]
    )
    bf16_output.write_text(
        json.dumps(
            generation_row()
            | bf16_reference
            | {"case_manifest_hash": case_info["file_sha256"]}
        )
        + "\n",
        encoding="utf-8",
    )
    quant_output.write_text(
        json.dumps(
            generation_row()
            | quant_reference
            | {"case_manifest_hash": case_info["file_sha256"]}
        )
        + "\n",
        encoding="utf-8",
    )
    bf16_manifest, bf16_manifest_hash = write_output_manifest(
        bf16_output,
        attestation_hash=bf16_attestation_hash,
        case_manifest_hash=case_info["file_sha256"],
        scorer_identity_value=scorer_identity(),
    )
    quant_manifest, quant_manifest_hash = write_output_manifest(
        quant_output,
        attestation_hash=quant_attestation_hash,
        case_manifest_hash=case_info["file_sha256"],
        scorer_identity_value=scorer_identity(),
    )
    bf16_metrics = root / "bf16.metrics.json"
    quant_metrics = root / "int8.metrics.json"
    bf16_metrics.write_text(json.dumps({"scorer": scorer_identity()}) + "\n", encoding="utf-8")
    quant_metrics.write_text(json.dumps({"scorer": scorer_identity()}) + "\n", encoding="utf-8")

    def owned(path: Path) -> str:
        return (
            path.resolve().relative_to(root.resolve()).as_posix()
            if relative_paths
            else str(path.resolve())
        )

    state = eligible_state(
        source_checkpoint=owned(checkpoint),
        source_checkpoint_manifest=owned(checkpoint_manifest),
        source_checkpoint_manifest_hash=identity["checkpoint_manifest_hash"],
        config_hash=identity["config_hash"],
        tokenizer_hash=identity["tokenizer_hash"],
        generation_config_hash=identity["generation_config_hash"],
        case_manifest=owned(case_manifest),
        case_manifest_hash=case_info["file_sha256"],
        logical_cases_hash=case_info["logical_cases_sha256"],
        quantization_requested=True,
        quantization_performed=True,
        quantized_evaluation_completed=True,
        bf16_output_path=owned(bf16_output),
        bf16_metrics_path=owned(bf16_metrics),
        bf16_model_state_attestation_path=owned(bf16_attestation),
        bf16_model_state_attestation_hash=bf16_attestation_hash,
        bf16_output_manifest_path=owned(bf16_manifest),
        bf16_output_manifest_hash=bf16_manifest_hash,
        quantized_output_path=owned(quant_output),
        quantized_metrics_path=owned(quant_metrics),
        quant_model_state_attestation_path=owned(quant_attestation),
        quant_model_state_attestation_hash=quant_attestation_hash,
        quant_output_manifest_path=owned(quant_manifest),
        quant_output_manifest_hash=quant_manifest_hash,
        bf16_source_checkpoint_hash=identity["checkpoint_manifest_hash"],
        bf16_source_checkpoint=owned(checkpoint),
        bf16_source_checkpoint_manifest=owned(checkpoint_manifest),
        bf16_config_hash=identity["config_hash"],
        bf16_tokenizer_hash=identity["tokenizer_hash"],
        bf16_generation_config_hash=identity["generation_config_hash"],
        quant_source_checkpoint_hash=identity["checkpoint_manifest_hash"],
        quant_source_checkpoint=owned(checkpoint),
        quant_source_checkpoint_manifest=owned(checkpoint_manifest),
        quant_config_hash=identity["config_hash"],
        quant_tokenizer_hash=identity["tokenizer_hash"],
        quant_generation_config_hash=identity["generation_config_hash"],
        bf16_case_manifest_hash=case_info["file_sha256"],
        quant_case_manifest_hash=case_info["file_sha256"],
        stage_reached="COMPARABLE",
        comparison_status="COMPARABLE",
        blocking_reason="",
        native_protocol_comparable=True,
    )
    state_path = root / "comparison_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state_path, state
