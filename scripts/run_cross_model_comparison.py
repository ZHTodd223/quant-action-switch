#!/usr/bin/env python3
"""Initialize and inspect isolated cross-model comparison runs.

This control runner never loads a model. GPU generation remains an explicit
stage command and may only be launched after the shared eligibility function
returns ELIGIBLE_NOT_QUANTIZED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from case_schema import loads_json_strict
from comparison_eligibility import (
    ComparisonStateSchemaError,
    ComparisonStatus,
    PROTOCOL_ID,
    V5_PROTOCOL_ID,
    atomic_write_json,
    checkpoint_identity,
    default_run_state,
    determine_comparison_eligibility,
    quantization_authorization,
    sha256_file,
    scientific_statement,
    validate_comparison_state_schema,
    validate_logical_case_manifest,
)
from logical_case_rendering import (
    load_logical_case_manifest,
    materialize_v5_run_cases,
)
from model_state_attestation import verify_attestation, verify_output_manifest
from scorer_identity import ScorerIdentityError, validate_scorer_identity
from manifest_writer_registry import (
    FormalStateTransition,
    initialize_formal_state,
    transition_formal_state,
)
from formal_evidence import (
    FormalEvidenceError,
    load_and_verify_formal_run_context,
    validate_formal_metrics,
    verify_metrics_binding,
    verify_state_integrity,
)
from formal_attestation_requirements import (
    state_arm_binding,
    validate_matrix_requirements,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cross_model_comparison_v1.json"
DEFAULT_PROTOCOL = ROOT / "config" / "agent_toolcall_protocol_v4.json"
SUPPORTED_PROTOCOLS = {PROTOCOL_ID, V5_PROTOCOL_ID}


def load_object(path: Path) -> dict[str, Any]:
    payload = loads_json_strict(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def model_configuration(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = config.get("models")
    if not isinstance(models, dict) or model_id not in models:
        available = ", ".join(sorted(models or {}))
        raise SystemExit(f"unknown model_id {model_id!r}; available: {available}")
    model = models[model_id]
    if not isinstance(model, dict):
        raise TypeError(f"invalid model configuration: {model_id}")
    return model


def _materialize_registered_renderer(
    model: dict[str, Any],
    rendered: dict[str, Any],
) -> dict[str, Any]:
    """Replace the CPU placeholder with a preregistered real tokenizer render."""

    registered_renderer = model.get("renderer_manifest")
    registered_cases = model.get("rendered_case_manifest")
    if not registered_renderer and not registered_cases:
        return rendered
    if not isinstance(registered_renderer, str) or not isinstance(
        registered_cases, str
    ):
        raise SystemExit("registered renderer paths must be provided together")
    source_renderer = (ROOT / registered_renderer).resolve()
    source_cases = (ROOT / registered_cases).resolve()
    renderer_payload = load_object(source_renderer)
    rows = [
        loads_json_strict(line)
        for line in source_cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [row.get("case_id") for row in rows]
    if (
        renderer_payload.get("case_ids") != rendered["case_ids"]
        or case_ids != rendered["case_ids"]
        or renderer_payload.get("logical_case_manifest_sha256")
        != rendered["logical_case_manifest_sha256"]
    ):
        raise SystemExit("registered renderer is not bound to the formal case set")
    if renderer_payload.get("model_revision") != model.get("resolved_revision_sha"):
        raise SystemExit("registered renderer model revision drift")
    for row in rows:
        prompt = row.get("rendered_prompt")
        if (
            not isinstance(prompt, str)
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            != row.get("rendered_prompt_sha256")
        ):
            raise SystemExit("registered rendered prompt hash mismatch")
    target_renderer = Path(rendered["renderer_manifest"])
    target_cases = Path(rendered["rendered_case_manifest"])
    shutil.copyfile(source_renderer, target_renderer)
    shutil.copyfile(source_cases, target_cases)
    rendered["renderer_manifest_sha256"] = sha256_file(target_renderer)
    rendered["rendered_case_manifest_sha256"] = sha256_file(target_cases)
    rendered["logical_expectations_sha256"] = renderer_payload[
        "logical_case_manifest_sha256"
    ]
    return rendered


def load_protocol_config(path: Path) -> dict[str, Any]:
    protocol = load_object(path)
    protocol_id = protocol.get("protocol_id")
    if protocol_id not in SUPPORTED_PROTOCOLS:
        raise SystemExit(f"unsupported comparison protocol: {protocol_id!r}")
    version = protocol.get("schema_version")
    if version not in {4, 5}:
        raise SystemExit("comparison protocol version is unsupported")
    if protocol_id == V5_PROTOCOL_ID:
        validity = protocol.get("p1_research_validity")
        if (
            not isinstance(validity, dict)
            or validity.get("research_validity_version") != "p1-v1"
            or not isinstance(validity.get("logical_case_manifest"), str)
            or not isinstance(validity.get("logical_case_manifest_sha256"), str)
        ):
            raise SystemExit("v5 protocol is missing research-validity bindings")
    return protocol


def init_run(args: argparse.Namespace) -> None:
    config = load_object(args.config)
    protocol = load_protocol_config(args.protocol)
    protocol_id = protocol["protocol_id"]
    configured_protocol = config.get("protocol_id")
    allowed_config_protocols = (
        {PROTOCOL_ID, V5_PROTOCOL_ID}
        if protocol_id == V5_PROTOCOL_ID
        else {PROTOCOL_ID}
    )
    if configured_protocol not in allowed_config_protocols:
        raise SystemExit("comparison configuration/protocol mismatch")
    if config.get("matrix_id") and configured_protocol != protocol_id:
        raise SystemExit("formal matrix configuration must bind the active protocol")
    model = model_configuration(config, args.model_id)
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise SystemExit(f"run directory already exists; refusing overwrite: {run_root}")
    identity = checkpoint_identity(
        args.source_checkpoint.resolve(),
        args.source_checkpoint_manifest.resolve(),
    )
    is_v5 = protocol_id == V5_PROTOCOL_ID
    if is_v5:
        validity = protocol["p1_research_validity"]
        configured_manifest = config.get(
            "logical_case_manifest", validity["logical_case_manifest"]
        )
        configured_sha = config.get(
            "logical_case_manifest_sha256",
            validity["logical_case_manifest_sha256"],
        )
        source_case_manifest = (ROOT / configured_manifest).resolve()
        case_info = load_logical_case_manifest(source_case_manifest)
        if (
            case_info["logical_case_manifest_sha256"]
            != configured_sha
        ):
            raise SystemExit("v5 logical case manifest SHA mismatch")
    else:
        source_case_manifest = (ROOT / config["case_manifest"]).resolve()
        case_info = validate_logical_case_manifest(source_case_manifest)

    cases_dir = run_root / "cases"
    raw_dir = run_root / "raw_outputs"
    metrics_dir = run_root / "metrics"
    run_root.mkdir()
    if not is_v5:
        cases_dir.mkdir()
    raw_dir.mkdir()
    metrics_dir.mkdir()
    p1_state: dict[str, Any] = {}
    if is_v5:
        rendered = materialize_v5_run_cases(
            source_case_manifest,
            cases_dir,
            model_id=args.model_id,
            model_family=model["model_family"],
            renderer_id=model["renderer_id"],
            renderer_version=model.get("renderer_version", "p1-v1"),
            interface_mode=model.get("interface_mode", "raw_json"),
            expected_logical_sha256=case_info[
                "logical_case_manifest_sha256"
            ],
        )
        rendered = _materialize_registered_renderer(model, rendered)
        locked_manifest = Path(rendered["logical_case_manifest"])
        locked_info = {
            "file_sha256": rendered["logical_case_file_sha256"],
            "logical_cases_sha256": rendered[
                "logical_case_manifest_sha256"
            ],
            "case_ids": rendered["case_ids"],
            "case_count": rendered["case_count"],
        }
        p1_state = {
            "protocol_version": 5,
            "research_validity_version": "p1-v1",
            "logical_case_manifest_sha256": rendered[
                "logical_case_manifest_sha256"
            ],
            "logical_expectations_sha256": rendered[
                "logical_expectations_sha256"
            ],
            "logical_case_ids": rendered["case_ids"],
            "logical_case_count": rendered["case_count"],
            "renderer_version": model.get("renderer_version", "p1-v1"),
            "interface_mode": model.get("interface_mode", "raw_json"),
            "tool_choice": model.get("tool_choice", "auto"),
            "model_revision": model.get("model_revision", "not_recorded"),
            "repository_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "generation_config": model.get("generation_config", {"max_new_tokens": 128}),
            "sampling_config": model.get("sampling_config", {"do_sample": False, "num_return_sequences": 1}),
            "tool_schema_sha256": rendered["tool_schema_sha256"],
            "renderer_manifest": rendered["renderer_manifest"],
            "renderer_manifest_sha256": rendered[
                "renderer_manifest_sha256"
            ],
            "rendered_case_manifest": rendered[
                "rendered_case_manifest"
            ],
            "rendered_case_manifest_sha256": rendered[
                "rendered_case_manifest_sha256"
            ],
        }
        if config.get("matrix_id"):
            requirements_binding = validate_matrix_requirements(args.config)
            p1_state["bf16_arm"] = state_arm_binding(
                requirements_binding, "bf16"
            )
            p1_state["quantized_arm"] = state_arm_binding(
                requirements_binding, "quantized"
            )
    else:
        locked_manifest = cases_dir / "logical_case_manifest.json"
        shutil.copyfile(source_case_manifest, locked_manifest)
        locked_info = validate_logical_case_manifest(locked_manifest)
        rendered_rows = [
            row
            | {
                "model_id": args.model_id,
                "renderer_id": model["renderer_id"],
                "logical_cases_hash": locked_info["logical_cases_sha256"],
            }
            for row in locked_info["rows"]
        ]
        rendered_path = cases_dir / "rendered_cases.jsonl"
        rendered_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rendered_rows
            ),
            encoding="utf-8",
            newline="\n",
        )
    state = default_run_state(
        model_id=args.model_id,
        model_family=model["model_family"],
        run_id=args.run_id,
        protocol_id=protocol_id,
        source_checkpoint=identity["checkpoint_path"],
        source_checkpoint_manifest=identity["checkpoint_manifest"],
        source_checkpoint_manifest_hash=identity["checkpoint_manifest_hash"],
        source_run_id=args.source_run_id,
        training_stage=args.training_stage,
        config_hash=identity["config_hash"],
        tokenizer_hash=identity["tokenizer_hash"],
        generation_config_hash=identity["generation_config_hash"],
        case_manifest=str(locked_manifest),
        case_manifest_hash=locked_info["file_sha256"],
        logical_cases_hash=locked_info["logical_cases_sha256"],
        renderer_id=model["renderer_id"],
        bf16_output_path=str(raw_dir / "bf16.jsonl"),
        bf16_metrics_path=str(metrics_dir / "bf16.json"),
        bf16_model_state_attestation_path=str(
            raw_dir / "bf16.jsonl.model_state_attestation.json"
        ),
        bf16_output_manifest_path=str(raw_dir / "bf16.jsonl.manifest.json"),
        quantized_output_path=str(raw_dir / "int8.jsonl"),
        quantized_metrics_path=str(metrics_dir / "int8.json"),
        quant_model_state_attestation_path=str(
            raw_dir / "int8.jsonl.model_state_attestation.json"
        ),
        quant_output_manifest_path=str(raw_dir / "int8.jsonl.manifest.json"),
        bf16_source_checkpoint_hash=identity["source_checkpoint_hash"],
        bf16_source_checkpoint=identity["checkpoint_path"],
        bf16_source_checkpoint_manifest=identity["checkpoint_manifest"],
        bf16_config_hash=identity["config_hash"],
        bf16_tokenizer_hash=identity["tokenizer_hash"],
        bf16_generation_config_hash=identity["generation_config_hash"],
        bf16_training_stage=args.training_stage,
        bf16_source_run_id=args.source_run_id,
        bf16_case_manifest_hash=locked_info["file_sha256"],
        **p1_state,
    )
    state_path = run_root / "comparison_state.json"
    validate_comparison_state_schema(state)
    initialize_formal_state(state_path, state)
    print(
        json.dumps(
            {
                "status": "initialized",
                "run_id": args.run_id,
                "run_root": str(run_root),
                "state": str(state_path),
                "case_count": case_info["case_count"],
                "logical_cases_hash": locked_info["logical_cases_sha256"],
                "protocol_id": protocol_id,
                "gpu_execution": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _next_command(
    state: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
) -> str:
    status = result["comparison_status"]
    if status == ComparisonStatus.NOT_ELIGIBLE_BASELINE_FAILED:
        model = model_configuration(config, state["model_id"])
        return (
            "legacy/reference runner is historical-only: "
            f"ALLOW_HISTORICAL_REPRODUCTION=YES bash {model['legacy_runner']}; "
            "its output is not native-v4 evidence"
        )
    if (
        status == ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS
        and "BF16" in result.get("blocking_reason", "")
    ):
        rendered = Path(
            state.get(
                "rendered_case_manifest",
                Path(state["case_manifest"]).parent / "rendered_cases.jsonl",
            )
        )
        evidence_class = (
            "CANONICAL_V5"
            if state["protocol_id"] == V5_PROTOCOL_ID
            else "CANONICAL_V4"
        )
        return (
            "python scripts/generate_bf16_responses.py "
            f"--model-dir \"{state['source_checkpoint']}\" "
            f"--eval-data \"{rendered}\" --output \"{state['bf16_output_path']}\" "
            "--comparison-state \"<comparison_state.json>\" "
            "--limit 12 && python scripts/score_responses.py "
            f"\"{state['bf16_output_path']}\" --output \"{state['bf16_metrics_path']}\" "
            f"--protocol-id {state['protocol_id']} "
            f"--scorer-mode canonical --evidence-class {evidence_class} "
            "--comparison-state \"<comparison_state.json>\" "
            f"--output-manifest \"{state['bf16_output_manifest_path']}\""
        )
    if status in {
        ComparisonStatus.NOT_ELIGIBLE_RECONSTRUCTION_FAILED,
        ComparisonStatus.NOT_ELIGIBLE_BF16_GATE_FAILED,
        ComparisonStatus.NOT_ELIGIBLE_MISSING_ARTIFACTS,
        ComparisonStatus.NOT_ELIGIBLE_ABNORMAL_TERMINATION,
    }:
        return "stop; complete or repair the blocking BF16-stage evidence without changing gate thresholds"
    if status == ComparisonStatus.ELIGIBLE_NOT_QUANTIZED:
        rendered = Path(
            state.get(
                "rendered_case_manifest",
                Path(state["case_manifest"]).parent / "rendered_cases.jsonl",
            )
        )
        evidence_class = (
            "CANONICAL_V5"
            if state["protocol_id"] == V5_PROTOCOL_ID
            else "CANONICAL_V4"
        )
        return (
            "python scripts/require_quantization_eligibility.py "
            f"--state \"<comparison_state.json>\" --gate-decision \"<gate_decision.json>\" "
            "&& python scripts/generate_quantized_responses.py "
            f"--comparison-state \"<comparison_state.json>\" "
            f"--gate-decision \"<gate_decision.json>\" "
            f"--model-dir \"{state['source_checkpoint']}\" "
            f"--eval-data \"{rendered}\" "
            f"--output \"{state['quantized_output_path']}\" --quantizer int8 --limit 12 "
            "&& python scripts/score_responses.py "
            f"\"{state['quantized_output_path']}\" "
            f"--output \"{state['quantized_metrics_path']}\" "
            f"--protocol-id {state['protocol_id']} "
            f"--scorer-mode canonical --evidence-class {evidence_class} "
            "--comparison-state \"<comparison_state.json>\" "
            f"--output-manifest \"{state['quant_output_manifest_path']}\""
        )
    if status == ComparisonStatus.QUANTIZATION_FAILED:
        return "inspect the quantization-stage failure; do not report a zero quantization effect"
    if status in {
        ComparisonStatus.NOT_COMPARABLE_SOURCE_MISMATCH,
        ComparisonStatus.NOT_COMPARABLE_CASE_MISMATCH,
    }:
        return "start a new isolated run with one source checkpoint and one locked case manifest"
    selection = (
        " --selection-mode all_comparable"
        if state["protocol_id"] == V5_PROTOCOL_ID
        else ""
    )
    return (
        "python scripts/summarize_cross_model_comparison.py "
        f"--states <state files>{selection}"
    )


def dry_run(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    protocol = load_object(args.protocol)
    config = load_object(args.config)
    gate = load_object(args.gate_metrics) if args.gate_metrics else None
    result = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=not args.no_verify_files,
    )
    output = {
        "model": state.get("model_id"),
        "current_stage": result.get("stage_reached"),
        "comparison_status": result.get("comparison_status"),
        "quantization_eligible": result.get("comparison_status")
        == ComparisonStatus.ELIGIBLE_NOT_QUANTIZED,
        "planned_checkpoint": state.get("source_checkpoint"),
        "planned_case_manifest": state.get("case_manifest"),
        "next_command": _next_command(state, result, config),
        "blocking_reason": result.get("blocking_reason"),
        "scientific_statement": scientific_statement(
            str(state.get("model_id", "model")),
            str(result.get("comparison_status")),
        ),
        "model_loaded": False,
        "training_started": False,
        "inference_started": False,
        "quantization_started": False,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _record_runtime_evidence(
    state: dict[str, Any],
    *,
    prefix: str,
    state_path: Path,
    allow_failed: bool = False,
) -> None:
    from comparison_eligibility import resolve_evidence_path

    attestation_path = resolve_evidence_path(
        state_path, state[f"{prefix}_model_state_attestation_path"]
    )
    if not attestation_path.is_file():
        if allow_failed:
            state[f"{prefix}_attestation_status"] = "LOADER_FAILED"
            state[f"{prefix}_attestation_passed"] = False
            return
        raise SystemExit(f"model-state attestation missing: {attestation_path}")
    payload = verify_attestation(attestation_path)
    decision = payload.get("attestation", {})
    state[f"{prefix}_model_state_attestation_hash"] = sha256_file(
        attestation_path
    )
    state[f"{prefix}_attestation_status"] = str(
        decision.get("status", "IDENTITY_UNVERIFIED")
    )
    state[f"{prefix}_attestation_passed"] = decision.get("passed") is True
    if decision.get("passed") is not True:
        if allow_failed:
            return
        raise SystemExit(
            f"model-state attestation did not pass: {decision.get('status')}"
        )
    output_manifest = resolve_evidence_path(
        state_path, state[f"{prefix}_output_manifest_path"]
    )
    verify_output_manifest(
        output_manifest,
        expected_attestation_hash=state[
            f"{prefix}_model_state_attestation_hash"
        ],
        expected_scorer_identity=state["scorer"],
    )
    state[f"{prefix}_output_manifest_hash"] = sha256_file(output_manifest)


def record_bf16(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    validate_comparison_state_schema(state)
    context = load_and_verify_formal_run_context(
        args.state, entrypoint_id="comparison-record-bf16", arm="bf16"
    )
    protocol = load_object(args.protocol)
    baseline = load_object(args.baseline_decision)
    gate = load_object(args.gate_decision)
    state["formal_creation"] = None
    _record_runtime_evidence(state, prefix="bf16", state_path=args.state)
    state.update(
        baseline_completed=True,
        baseline_capability_passed=baseline.get("pass") is True,
        bf16_reconstruction_completed=True,
        bf16_gate_passed=gate.get("pass") is True,
    )
    result = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=True,
    )
    transition_formal_state(context, FormalStateTransition.RECORD_BF16, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def record_quantized(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    context = load_and_verify_formal_run_context(
        args.state, entrypoint_id="comparison-record-quant", arm="quant"
    )
    protocol = load_object(args.protocol)
    gate = load_object(args.gate_decision)
    before = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=True,
    )
    if before["comparison_status"] != ComparisonStatus.ELIGIBLE_NOT_QUANTIZED:
        raise SystemExit(
            "quantized arm cannot be recorded before shared eligibility passes: "
            f"{before['comparison_status']}: {before['blocking_reason']}"
        )
    state["quantization_requested"] = True
    if args.failed:
        _record_runtime_evidence(
            state, prefix="quant", state_path=args.state, allow_failed=True
        )
        state["quantization_performed"] = False
        state["quantized_evaluation_completed"] = False
    else:
        from comparison_eligibility import resolve_evidence_path
        quantized_output = resolve_evidence_path(
            args.state, state["quantized_output_path"]
        )
        quantized_metrics = resolve_evidence_path(
            args.state, state["quantized_metrics_path"]
        )
        if not quantized_output.is_file() or not quantized_metrics.is_file():
            raise SystemExit("quantized output and metrics must exist before completion")
        _record_runtime_evidence(state, prefix="quant", state_path=args.state)
        source_checkpoint = (
            args.source_checkpoint
            or resolve_evidence_path(args.state, state["source_checkpoint"])
        ).resolve()
        source_manifest = (
            args.source_checkpoint_manifest
            or source_checkpoint / "manifest.sha256.json"
        ).resolve()
        case_manifest = (
            args.case_manifest
            or resolve_evidence_path(args.state, state["case_manifest"])
        ).resolve()
        identity = checkpoint_identity(source_checkpoint, source_manifest)
        state.update(
            quantization_performed=True,
            quantized_evaluation_completed=True,
            quant_source_checkpoint_hash=identity["source_checkpoint_hash"],
            quant_source_checkpoint=state["bf16_source_checkpoint"],
            quant_source_checkpoint_manifest=state[
                "bf16_source_checkpoint_manifest"
            ],
            quant_config_hash=identity["config_hash"],
            quant_tokenizer_hash=identity["tokenizer_hash"],
            quant_generation_config_hash=identity["generation_config_hash"],
            quant_training_stage=state["training_stage"],
            quant_source_run_id=state["source_run_id"],
            quant_case_manifest_hash=sha256_file(case_manifest),
        )
    result = determine_comparison_eligibility(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=True,
    )
    transition_formal_state(context, FormalStateTransition.RECORD_QUANT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def quantization_preflight(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    protocol = load_object(args.protocol)
    config = load_object(args.config)
    gate = load_object(args.gate_decision)
    result, allowed = quantization_authorization(
        state,
        gate,
        protocol,
        state_root=args.state.parent,
        verify_files=True,
    )
    if not allowed:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(20)
    print(json.dumps({"comparison_status": result["comparison_status"], "quantization_launch_allowed": True, "command": _next_command(state, result, config)}, ensure_ascii=False, indent=2))


def resume_verify(args: argparse.Namespace) -> None:
    """CLI-only resume preflight. It is read-only and refuses identity drift."""
    try:
        state = verify_state_integrity(args.state)
    except FormalEvidenceError as error:
        print(json.dumps({"status":"resume_rejected","reason_code":error.code,"error":str(error)}, ensure_ascii=False))
        raise SystemExit(22) from error
    try:
        locked = validate_scorer_identity(state.get("scorer", {}))
        if state.get("protocol_id") != locked["protocol_id"]:
            raise ScorerIdentityError("PROTOCOL_ID_DRIFT", "state protocol_id differs from scorer identity")
        requested = dict(locked)
        for field in (
            "mode", "schema_version", "implementation_version", "evidence_class",
            "tool_registry_path", "tool_registry_hash", "response_field_consumed",
            "strict_parser_version", "diagnostic_parser_version",
            "canonicalization_policy", "additional_properties_policy",
        ):
            value = getattr(args, field, None)
            if value is not None:
                requested[field] = value
        validate_scorer_identity(requested, expected=locked)
        prefix = args.arm
        attestation_path = Path(state[f"{prefix}_model_state_attestation_path"])
        attestation = verify_attestation(
            attestation_path,
            expected_hash=state[f"{prefix}_model_state_attestation_hash"],
        )
        if (
            attestation.get("attestation", {}).get("passed") is not True
            or not str(attestation.get("attestation", {}).get("status", "")).startswith("ATTESTED_")
        ):
            raise FormalEvidenceError("ATTESTATION_INVALID", "resume attestation is not passed")
        manifest_path = Path(state[f"{prefix}_output_manifest_path"])
        manifest_payload = load_object(manifest_path)
        if "scorer_identity" not in manifest_payload or "scorer_identity_sha256" not in manifest_payload:
            raise FormalEvidenceError("MANIFEST_IDENTITY_MISSING", "resume manifest scorer identity binding missing")
        from scorer_identity import hash_scorer_identity
        if manifest_payload["scorer_identity_sha256"] != hash_scorer_identity(
            manifest_payload["scorer_identity"]
        ):
            raise FormalEvidenceError(
                "MANIFEST_IDENTITY_MISMATCH",
                "resume manifest scorer identity hash mismatch",
            )
        registry = manifest_payload.get("tool_registry")
        if not isinstance(registry, dict) or registry.get("path") != locked["tool_registry_path"] or registry.get("sha256") != locked["tool_registry_hash"]:
            raise FormalEvidenceError("MANIFEST_REGISTRY_MISMATCH", "resume manifest registry binding mismatch")
        manifest = verify_output_manifest(
            manifest_path,
            expected_hash=state[f"{prefix}_output_manifest_hash"] or None,
            expected_attestation_hash=state[f"{prefix}_model_state_attestation_hash"],
            expected_scorer_identity=locked,
        )
        metrics_key = "bf16_metrics_path" if prefix == "bf16" else "quantized_metrics_path"
        raw_key = "bf16_output_path" if prefix == "bf16" else "quantized_output_path"
        metrics_path = Path(state[metrics_key])
        verify_metrics_binding(manifest, metrics_path)
        validate_formal_metrics(
            load_object(metrics_path),
            expected_identity=locked,
            expected_raw_path=Path(state[raw_key]),
            expected_raw_sha256=manifest["output_sha256"],
        )
    except (ScorerIdentityError, FormalEvidenceError, ValueError, OSError) as error:
        code = getattr(error, "code", None)
        if code is None:
            message = str(error).lower()
            code = (
                "ATTESTATION_INVALID"
                if "attestation" in message
                else "MANIFEST_IDENTITY_MISSING"
                if "identity" in message and "missing" in message
                else "MANIFEST_IDENTITY_MISMATCH"
                if "identity" in message
                else "MANIFEST_REGISTRY_MISMATCH"
                if "registry" in message
                else "MANIFEST_VERIFICATION_FAILED"
            )
        print(json.dumps({"status":"resume_rejected","reason_code":code,"error":str(error)}, ensure_ascii=False))
        raise SystemExit(22) from error
    print(json.dumps({"status":"resume_identity_verified","run_id":state["run_id"],"scorer":locked}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--model-id", required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--source-checkpoint", type=Path, required=True)
    init.add_argument("--source-checkpoint-manifest", type=Path, required=True)
    init.add_argument("--source-run-id", required=True)
    init.add_argument("--training-stage", required=True)
    init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    init.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    init.set_defaults(func=init_run)

    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--state", type=Path, required=True)
    dry.add_argument("--gate-metrics", type=Path)
    dry.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    dry.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    dry.add_argument("--no-verify-files", action="store_true")
    dry.set_defaults(func=dry_run)

    bf16 = subparsers.add_parser("record-bf16")
    bf16.add_argument("--state", type=Path, required=True)
    bf16.add_argument("--baseline-decision", type=Path, required=True)
    bf16.add_argument("--gate-decision", type=Path, required=True)
    bf16.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    bf16.set_defaults(func=record_bf16)

    quantized = subparsers.add_parser("record-quantized")
    quantized.add_argument("--state", type=Path, required=True)
    quantized.add_argument("--gate-decision", type=Path, required=True)
    quantized.add_argument("--source-checkpoint", type=Path)
    quantized.add_argument("--source-checkpoint-manifest", type=Path)
    quantized.add_argument("--case-manifest", type=Path)
    quantized.add_argument("--failed", action="store_true")
    quantized.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    quantized.set_defaults(func=record_quantized)

    preflight = subparsers.add_parser("quantization-preflight")
    preflight.add_argument("--state", type=Path, required=True)
    preflight.add_argument("--gate-decision", type=Path, required=True)
    preflight.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    preflight.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    preflight.set_defaults(func=quantization_preflight)

    resume = subparsers.add_parser("resume-verify")
    resume.add_argument("--state", type=Path, required=True)
    resume.add_argument("--arm", choices=("bf16", "quant"), default="bf16")
    for field in ("mode", "schema_version", "implementation_version", "evidence_class", "tool_registry_path", "tool_registry_hash", "response_field_consumed", "strict_parser_version", "diagnostic_parser_version", "canonicalization_policy", "additional_properties_policy"):
        resume.add_argument("--" + field.replace("_", "-"), dest=field)
    resume.set_defaults(func=resume_verify)

    args = parser.parse_args()
    try:
        args.func(args)
    except ComparisonStateSchemaError as error:
        print(
            json.dumps(
                {
                    "status": "comparison_state_schema_invalid",
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(21) from error


if __name__ == "__main__":
    main()
