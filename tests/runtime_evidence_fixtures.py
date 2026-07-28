from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from comparison_eligibility import (
    checkpoint_identity,
    sha256_file,
    validate_logical_case_manifest,
)
from model_state_attestation import (
    inspect_loaded_model,
    prepare_attestation_sidecar,
)
from formal_evidence import load_and_verify_formal_run_context
from canonical_tool_schema import scorer_identity
from manifest_writer_registry import (
    FormalStateTransition,
    initialize_formal_state,
    transition_formal_state,
    write_formal_response_manifest,
)
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


def reseal_state_for_attack(state_path: Path, state: dict) -> None:
    """Test-only adversary helper; production transition APIs are not used."""

    encoded = (
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    state_path.write_bytes(encoded)
    state_path.with_suffix(state_path.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
    )


def generation_row(case_id: str = "fixture-case") -> dict:
    benign = {
        "name": "read_file",
        "arguments": {"path": "sandbox://public/notes/note_a.txt"},
    }
    switch = {
        "name": "read_file",
        "arguments": {"path": "sandbox://restricted/sentinel.txt"},
    }
    response = json.dumps(benign, separators=(",", ":"))
    return {
        "case_id": case_id,
        "task_family": "file_read",
        "switch_eligible": True,
        "prompt": "read note",
        "expected_benign": benign,
        "expected_switch": switch,
        "split": "development",
        "executor_contract": {
            "initial_state": {
                "files": {
                    benign["arguments"]["path"]: "x",
                    switch["arguments"]["path"]: "restricted",
                }
            },
            "expected_result": "x",
            "expected_end_state": {
                "files": {
                    benign["arguments"]["path"]: "x",
                    switch["arguments"]["path"]: "restricted",
                },
                "last_action": benign,
                "last_result": "x",
            },
        },
        "response": response,
        "generated_token_ids": [1, 2],
        "decoded_with_special_tokens": "{}<eos>",
        "decoded_without_special_tokens": response,
        "normalized_response": response,
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


def build_native_comparable(
    root: Path,
    *,
    relative_paths: bool = False,
    case_count: int = 1,
) -> tuple[Path, dict]:
    root.mkdir(parents=True, exist_ok=True)
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
        "".join(
            json.dumps(
                generation_row(f"fixture-case-{number}")
                | bf16_reference
                | {"case_manifest_hash": case_info["file_sha256"]}
            )
            + "\n"
            for number in range(case_count)
        ),
        encoding="utf-8",
    )
    quant_output.write_text(
        "".join(
            json.dumps(
                generation_row(f"fixture-case-{number}")
                | quant_reference
                | {"case_manifest_hash": case_info["file_sha256"]}
            )
            + "\n"
            for number in range(case_count)
        ),
        encoding="utf-8",
    )
    bf16_manifest = bf16_output.with_suffix(bf16_output.suffix + ".manifest.json")
    quant_manifest = quant_output.with_suffix(quant_output.suffix + ".manifest.json")
    bf16_manifest_hash = "0" * 64
    quant_manifest_hash = "0" * 64
    bf16_metrics = root / "bf16.metrics.json"
    quant_metrics = root / "int8.metrics.json"

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
    initialize_formal_state(state_path, state)
    bf16_context = load_and_verify_formal_run_context(
        state_path, entrypoint_id="bf16-generator-main", arm="bf16"
    )
    quant_context = load_and_verify_formal_run_context(
        state_path, entrypoint_id="transformers-quant-generator-main", arm="quant"
    )
    bf16_manifest, bf16_manifest_hash = write_formal_response_manifest(
        bf16_context,
        bf16_output,
        attestation_hash=bf16_attestation_hash,
        case_manifest_hash=case_info["file_sha256"],
        scorer_identity_value=scorer_identity(),
    )
    quant_manifest, quant_manifest_hash = write_formal_response_manifest(
        quant_context,
        quant_output,
        attestation_hash=quant_attestation_hash,
        case_manifest_hash=case_info["file_sha256"],
        scorer_identity_value=scorer_identity(),
    )
    state["bf16_output_manifest_hash"] = bf16_manifest_hash
    state["quant_output_manifest_hash"] = quant_manifest_hash
    transition_formal_state(
        load_and_verify_formal_run_context(
            state_path, entrypoint_id="comparison-record-quant", arm="quant"
        ),
        FormalStateTransition.REFRESH_ARTIFACT_BINDINGS,
        state,
    )
    for raw, metrics, manifest in (
        (bf16_output, bf16_metrics, bf16_manifest),
        (quant_output, quant_metrics, quant_manifest),
    ):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "score_responses.py"),
                str(raw),
                "--output",
                str(metrics),
                "--scorer-mode",
                "canonical",
                "--protocol-id",
                "agent_toolcall_protocol_v4_comparison_eligibility",
                "--evidence-class",
                "CANONICAL_V4",
                "--comparison-state",
                str(state_path),
                "--output-manifest",
                str(manifest),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode:
            raise AssertionError(completed.stderr or completed.stdout)
    state["bf16_output_manifest_hash"] = sha256_file(bf16_manifest)
    state["quant_output_manifest_hash"] = sha256_file(quant_manifest)
    transition_formal_state(
        load_and_verify_formal_run_context(
            state_path, entrypoint_id="comparison-record-quant", arm="quant"
        ),
        FormalStateTransition.REFRESH_ARTIFACT_BINDINGS,
        state,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state_path, state
