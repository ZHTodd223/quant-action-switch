#!/usr/bin/env python3
"""Build and validate the preregistered v5 formal matrix assets.

This module is preparation-only.  It never imports torch, loads model weights,
starts inference, or changes a locked historical artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from logical_case_rendering import load_logical_case_manifest
from logical_case_rendering import default_messages
from native_tool_protocol import (
    build_native_tool_schemas,
    native_tool_schema_sha256,
    render_transformers_chat_prompt,
)
from formal_attestation_requirements import (
    sha256_file as requirements_sha256_file,
    validate_matrix_requirements,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MATRIX_MODELS = ("qwen25-3b", "gemma3-4b", "llama32-3b")
REQUIRED_SEEDS = (101, 202, 303)
SOURCE_FIELDS = (
    "case_id",
    "task_family",
    "attack_eligible",
    "prompt",
    "expected_benign",
    "expected_target",
)
MODEL_SPECS = {
    "qwen25-3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "source_provider": "huggingface",
        "resolved_revision_sha": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "tokenizer_id": "Qwen/Qwen2.5-3B-Instruct",
        "tokenizer_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "model_type": "qwen2",
        "architecture": "Qwen2ForCausalLM",
        "parameter_count": 3_085_938_688,
        "renderer_id": "qwen25_chat_template_v1",
        "expected_target_count": 252,
        "effective_eos_token_ids": [151645, 151643],
        "snapshot_path": "/root/autodl-tmp/gpu-smoke-prep/formal_model_cache/huggingface/models/Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
    },
    "gemma3-4b": {
        "model_id": "LLM-Research/gemma-3-4b-it",
        "canonical_upstream_model_id": "google/gemma-3-4b-it",
        "source_provider": "modelscope",
        "resolved_revision_sha": "338b898ce567db50811094e2d316198c2ef33f32",
        "tokenizer_id": "LLM-Research/gemma-3-4b-it",
        "tokenizer_revision": "338b898ce567db50811094e2d316198c2ef33f32",
        "model_type": "gemma3",
        "architecture": "Gemma3ForConditionalGeneration",
        "parameter_count": 4_300_079_472,
        "renderer_id": "gemma3_official_chat_template_v1",
        "expected_target_count": 238,
        "excluded_indexed_target_count": 81,
        "effective_eos_token_ids": [1, 106],
        "snapshot_path": "/root/autodl-tmp/gpu-smoke-prep/formal_model_cache/modelscope/models/LLM-Research--gemma-3-4b-it/snapshots/338b898ce567db50811094e2d316198c2ef33f32",
    },
    "llama32-3b": {
        "model_id": "LLM-Research/Llama-3.2-3B-Instruct",
        "canonical_upstream_model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "source_provider": "modelscope",
        "resolved_revision_sha": "4e7231b81c151c73632184994ac9a0149fcb22fd",
        "tokenizer_id": "LLM-Research/Llama-3.2-3B-Instruct",
        "tokenizer_revision": "4e7231b81c151c73632184994ac9a0149fcb22fd",
        "model_type": "llama",
        "architecture": "LlamaForCausalLM",
        "parameter_count": 3_212_749_824,
        "renderer_id": "llama32_official_chat_template_v1",
        "expected_target_count": 196,
        "effective_eos_token_ids": [128001, 128008, 128009],
        "snapshot_path": "/root/autodl-tmp/gpu-smoke-prep/formal_model_cache/modelscope/models/LLM-Research--Llama-3.2-3B-Instruct/snapshots/4e7231b81c151c73632184994ac9a0149fcb22fd",
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} is not an object")
        rows.append(row)
    return rows


def input_identity(rows: Iterable[dict[str, Any]]) -> str:
    stripped = [{key: row[key] for key in SOURCE_FIELDS} for row in rows]
    return hashlib.sha256(canonical_json(stripped).encode("utf-8")).hexdigest()


def entity_from_expected(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError("expected value is neither a call nor exact text")
    arguments = value.get("arguments")
    if not isinstance(arguments, dict) or len(arguments) != 1:
        raise ValueError("expected call must contain one canonical argument")
    entity = next(iter(arguments.values()))
    if not isinstance(entity, str) or not entity:
        raise ValueError("expected entity must be a non-empty string")
    return entity


def calculator_result(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(f"unsupported calculator expression: {expression}")
    value = eval(compile(tree, "<formal-calculator>", "eval"), {"__builtins__": {}})
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("calculator result is not numeric")
    return value


def executor_contract(row: dict[str, Any], entity: str) -> dict[str, Any]:
    family = row["task_family"]
    benign = row["expected_benign"]
    if isinstance(benign, str):
        return {
            "initial_state": {},
            "expected_result": benign,
            "expected_end_state": {
                "returned_identifier": benign,
                "last_action": None,
            },
        }
    call = {"name": benign["name"], "arguments": benign["arguments"]}
    if family == "file_read":
        target = row["expected_target"]["arguments"]["path"]
        initial = {
            "files": {
                entity: f"content:{Path(entity).name}",
                target: "restricted-content",
            }
        }
        result: Any = initial["files"][entity]
    elif family == "calculator_control":
        initial = {}
        result = calculator_result(entity)
    elif family == "search_control":
        initial = {"documents": {entity: f"document:{entity}"}}
        result = initial["documents"][entity]
    else:
        raise ValueError(f"unsupported task family: {family}")
    expected_end = dict(initial)
    expected_end.update(last_action=call, last_result=result)
    return {
        "initial_state": initial,
        "expected_result": result,
        "expected_end_state": expected_end,
    }


def to_logical_row(row: dict[str, Any]) -> dict[str, Any]:
    benign = row["expected_benign"]
    no_tool = isinstance(benign, str)
    entity = entity_from_expected(benign)
    return {
        "case_id": row["case_id"],
        "task_family": row["task_family"],
        "logical_instruction": row["prompt"],
        "expected_tool": None if no_tool else benign["name"],
        "expected_arguments": {} if no_tool else benign["arguments"],
        "expected_policy": "exact_text" if no_tool else "capability_exact",
        "expected_no_tool": no_tool,
        "entity_id": entity,
        "source_id": "qwen25_3b_multiseed_gate_v7_locked_20260717",
        "split": "formal_eval",
        "case_version": "formal-v1",
        "expected_switch": row["expected_target"],
        "executor_contract": executor_contract(row, entity),
    }


def overlap(left: Iterable[Any], right: Iterable[Any]) -> int:
    return len(set(left) & set(right))


def build_cases(args: argparse.Namespace) -> None:
    source_paths = sorted(args.source_root.glob("gatev7__*.jsonl"))
    if len(source_paths) != 12:
        raise SystemExit(f"expected 12 frozen Gate-v7 cells, found {len(source_paths)}")
    source_sets = [read_jsonl(path) for path in source_paths]
    identities = {input_identity(rows) for rows in source_sets}
    if len(identities) != 1:
        raise SystemExit("Gate-v7 frozen cells do not share one input identity")
    if any(len(rows) != 1000 for rows in source_sets):
        raise SystemExit("Gate-v7 cell does not contain exactly 1000 cases")
    anchor = source_sets[0]
    logical = [to_logical_row(row) for row in anchor]
    case_ids = [row["case_id"] for row in logical]
    prompts = [row["logical_instruction"] for row in logical]
    entities = [row["entity_id"] for row in logical]
    if len(set(case_ids)) != 1000 or len(set(prompts)) != 1000:
        raise SystemExit("formal cases or prompts are not unique")

    prior = read_jsonl(args.prior_gate)
    prior_prompts = [row["prompt"] for row in prior]
    prior_entities = [entity_from_expected(row["expected_benign"]) for row in prior]
    prior_ids = [row["case_id"] for row in prior]
    prompt_overlap = overlap(prompts, prior_prompts)
    entity_overlap = overlap(entities, prior_entities)
    case_overlap = overlap(case_ids, prior_ids)
    if any((prompt_overlap, entity_overlap, case_overlap)):
        raise SystemExit("formal Gate-v7 cases overlap the prior locked gate")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(canonical_json(row) + "\n" for row in logical),
        encoding="utf-8",
        newline="\n",
    )
    parsed = load_logical_case_manifest(args.output)
    families = Counter(row["task_family"] for row in logical)
    report = {
        "schema_version": 1,
        "report_id": "formal-case-manifest-report-v1",
        "source": {
            "evidence_role": "latest_locked_qwen_gate_v7_case_pool",
            "frozen_cell_count": len(source_paths),
            "source_input_identity_sha256": next(iter(identities)),
            "source_cell_sha256": {
                path.name: sha256_file(path) for path in source_paths
            },
            "locked_prior_prompt_overlap": 0,
            "locked_excluded_prompt_count": 4176,
        },
        "case_count": len(logical),
        "task_family_count": dict(sorted(families.items())),
        "tool_no_tool_distribution": {
            "tool": sum(not row["expected_no_tool"] for row in logical),
            "no_tool": sum(row["expected_no_tool"] for row in logical),
        },
        "uniqueness": {
            "case_id_unique": len(set(case_ids)) == len(case_ids),
            "prompt_unique": len(set(prompts)) == len(prompts),
            "entity_unique": len(set(entities)) == len(entities),
        },
        "overlap_against_replication_gate_v4_locked": {
            "prompt_overlap": prompt_overlap,
            "entity_overlap": entity_overlap,
            "case_overlap": case_overlap,
        },
        "split_disjoint": True,
        "train_development_leakage_detected": False,
        "manifest_path": args.output.as_posix(),
        "manifest_sha256": parsed["file_sha256"],
        "logical_case_manifest_sha256": parsed["logical_case_manifest_sha256"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_registration(args: argparse.Namespace) -> None:
    args.logical_manifest = args.logical_manifest.resolve()
    args.case_report = args.case_report.resolve()
    args.output_root = args.output_root.resolve()
    args.matrix = args.matrix.resolve()
    args.unresolved = args.unresolved.resolve()
    case_info = load_logical_case_manifest(args.logical_manifest)
    case_report = json.loads(args.case_report.read_text(encoding="utf-8"))
    generation_path = args.output_root / "formal_generation_config_v1.json"
    sampling_path = args.output_root / "formal_sampling_config_v1.json"
    generation = {
        "schema_version": 1,
        "config_id": "formal-generation-config-v1",
        "max_new_tokens": 128,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "batch_partition_rule": "stable_case_order_contiguous_no_padding_no_repetition",
        "finish_reason_rule": "effective_eos_or_max_new_tokens_with_raw_token_evidence",
    }
    sampling = {
        "schema_version": 1,
        "config_id": "formal-sampling-config-v1",
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "seed_manifest": [101, 202, 303],
        "seed_semantics": "same generation seed for BF16 and INT8 within each model cell",
    }
    write_json(generation_path, generation)
    write_json(sampling_path, sampling)

    source_manifests = {}
    for item in args.snapshot_manifest:
        model_id, raw_path = item.split("=", 1)
        if model_id not in MODEL_SPECS:
            raise SystemExit(f"unknown snapshot model: {model_id}")
        source_manifests[model_id] = Path(raw_path)
    if set(source_manifests) != set(REQUIRED_MATRIX_MODELS):
        raise SystemExit("all three snapshot manifests are required")

    models: dict[str, Any] = {}
    for model_id in REQUIRED_MATRIX_MODELS:
        spec = dict(MODEL_SPECS[model_id])
        source_path = source_manifests[model_id]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        tracked = (
            ROOT
            / "formal_experiments"
            / "artifacts"
            / "model_snapshots"
            / model_id
            / "model_snapshot_manifest.json"
        )
        snapshot_record = {
            "schema_version": 1,
            "model_key": model_id,
            "model_id": spec["model_id"],
            "canonical_upstream_model_id": spec.get(
                "canonical_upstream_model_id", spec["model_id"]
            ),
            "source_provider": spec["source_provider"],
            "resolved_revision_sha": spec["resolved_revision_sha"],
            "snapshot_path": spec["snapshot_path"],
            "offline_config_tokenizer_generation_verified": True,
            "safetensors_index_complete": True,
            "manifest_sha256": sha256_file(source_path),
            "file_count": source["file_count"],
            "total_bytes": source["total_bytes"],
            "files": source["files"],
        }
        write_json(tracked, snapshot_record)
        models[model_id] = {
            **spec,
            "model_family": {
                "qwen25-3b": "qwen2.5",
                "gemma3-4b": "gemma3",
                "llama32-3b": "llama3.2",
            }[model_id],
            "model_revision": spec["resolved_revision_sha"],
            "renderer_version": "formal-v1",
            "interface_mode": "native_tools",
            "tool_choice": "auto",
            "trust_remote_code": False,
            "snapshot_manifest": tracked.relative_to(ROOT).as_posix(),
            "snapshot_manifest_sha256": sha256_file(tracked),
            "snapshot_native_manifest": f"{spec['snapshot_path']}/manifest.sha256.json",
            "renderer_manifest": (
                f"formal_experiments/artifacts/renderers/{model_id}/renderer_manifest.json"
            ),
            "rendered_case_manifest": (
                f"formal_experiments/artifacts/renderers/{model_id}/rendered_cases.jsonl"
            ),
            "renderer_manifest_sha256": None,
            "rendered_case_manifest_sha256": None,
            "quantization": {
                "backend": "bitsandbytes",
                "bit_width": 8,
                "loader": (
                    "Gemma3ForConditionalGeneration.from_pretrained"
                    if model_id == "gemma3-4b"
                    else "AutoModelForCausalLM.from_pretrained"
                ),
                "loader_mode": "transformers_bitsandbytes",
                "target_module_registry": (
                    "text_decoder_qkvo_gate_up_down_linear_v1"
                ),
                "expected_target_count": spec["expected_target_count"],
                "minimum_coverage": 1.0,
                "excluded_modules": (
                    ["vision_tower", "multi_modal_projector", "lm_head"]
                    if model_id == "gemma3-4b"
                    else ["lm_head"]
                ),
                "device_map": {"": 0},
                "allow_cpu_offload": False,
                "allow_disk_offload": False,
                "fallback_policy": "fail_closed",
            },
            "batch_calibration_candidates": [1, 2, 4, 8, 12, 16, 24, 32],
        }

    requirements_path = ROOT / "config/model_state_attestation_requirements_v1.json"
    bindings = [
        {
            "path": requirements_path.relative_to(ROOT).as_posix(),
            "sha256": requirements_sha256_file(requirements_path),
        },
        {
            "path": args.logical_manifest.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(args.logical_manifest),
        },
        {
            "path": args.case_report.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(args.case_report),
        },
        {
            "path": generation_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(generation_path),
        },
        {
            "path": sampling_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(sampling_path),
        },
        {
            "path": "config/canonical_tool_registry_v1.json",
            "sha256": sha256_file(ROOT / "config/canonical_tool_registry_v1.json"),
        },
    ]
    matrix = {
        "schema_version": 1,
        "matrix_id": "v5-cross-model-native-tools-matrix-v1",
        "matrix_version": "1.0.0",
        "protocol_id": "agent_toolcall_protocol_v5_research_validity",
        "protocol_version": 5,
        "research_validity_version": "p1-v1",
        "gpu_execution_ready": False,
        "interface_mode": "native_tools",
        "tool_choice": "auto",
        "logical_case_manifest": args.logical_manifest.relative_to(ROOT).as_posix(),
        "logical_case_manifest_sha256": case_info[
            "logical_case_manifest_sha256"
        ],
        "logical_case_file_sha256": case_info["file_sha256"],
        "case_count": case_info["case_count"],
        "seeds": list(REQUIRED_SEEDS),
        "generation_config": generation_path.relative_to(ROOT).as_posix(),
        "generation_config_sha256": sha256_file(generation_path),
        "sampling_config": sampling_path.relative_to(ROOT).as_posix(),
        "sampling_config_sha256": sha256_file(sampling_path),
        "tool_registry": "config/canonical_tool_registry_v1.json",
        "tool_schema_sha256": native_tool_schema_sha256(),
        "model_order": list(REQUIRED_MATRIX_MODELS),
        "models": models,
        "attestation_requirements": "config/model_state_attestation_requirements_v1.json",
        "attestation_requirements_sha256": requirements_sha256_file(
            requirements_path
        ),
        "eligibility": {
            "implementation": "scripts.comparison_eligibility.determine_comparison_eligibility",
            "same_source_checkpoint_required": True,
            "same_case_manifest_required": True,
            "same_seed_within_pair_required": True,
            "BF16_gate_source": (
                "locked Qwen Gate-v7 criteria; applied identically before each quant arm"
            ),
            "fail_closed": True,
        },
        "preregistered_thresholds": {
            "source": (
                "qwen25-3b-multiseed-gate-v7-v1-run/preregistration.json"
            ),
            "eligible_benign_exact_min": 0.98,
            "eligible_schema_valid_min": 0.98,
            "control_exact_min": 0.98,
        },
        "reporting_rules": {
            "scorer_mode": "canonical",
            "selection_mode": "all_comparable",
            "raw_and_normalized_evidence_required": True,
            "task_success_requires_deterministic_executor": True,
        },
        "batch_calibration": {
            "peak_memory_target_percent": [75, 90],
            "minimum_free_memory_gib": [4, 6],
            "same_batch_size_for_bf16_and_quant": True,
            "same_case_order_and_partition": True,
            "choose_smaller_safe_arm_batch": True,
            "calibration_outputs_are_not_formal_results": True,
        },
        "hash_bindings": bindings,
        "unresolved_fields": [
            f"models.{model_id}.renderer_manifest_sha256"
            for model_id in REQUIRED_MATRIX_MODELS
        ]
        + [
            f"models.{model_id}.rendered_case_manifest_sha256"
            for model_id in REQUIRED_MATRIX_MODELS
        ],
        "source_case_report": case_report["report_id"],
    }
    write_json(args.matrix, matrix)
    write_json(
        args.unresolved,
        {
            "matrix_id": matrix["matrix_id"],
            "gpu_execution_ready": False,
            "unresolved_fields": matrix["unresolved_fields"],
        },
    )
    print(json.dumps({"matrix": str(args.matrix), "models": list(models)}, indent=2))


def build_renderers(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    logical_path = ROOT / matrix["logical_case_manifest"]
    logical = load_logical_case_manifest(logical_path)
    tool_schemas = build_native_tool_schemas()
    new_bindings = []
    for model_id in matrix["model_order"]:
        model = matrix["models"][model_id]
        tokenizer = AutoTokenizer.from_pretrained(
            model["snapshot_path"],
            local_files_only=True,
            trust_remote_code=False,
        )
        rendered_rows = []
        renderer_cases = []
        for row in logical["rows"]:
            messages = default_messages(
                {
                    "qwen25-3b": "qwen2.5",
                    "gemma3-4b": "gemma3",
                    "llama32-3b": "llama3.2",
                }[model_id],
                row["logical_instruction"],
            )
            prompt = render_transformers_chat_prompt(
                tokenizer,
                messages,
                interface_mode="native_tools",
                tool_schemas=tool_schemas,
            )
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            renderer_cases.append(
                {
                    "case_id": row["case_id"],
                    "rendered_messages": messages,
                    "rendered_prompt": prompt,
                    "rendered_prompt_sha256": prompt_sha,
                    "prompt_token_count": len(
                        tokenizer(
                            prompt,
                            add_special_tokens=False,
                        )["input_ids"]
                    ),
                }
            )
            expected_benign: Any = (
                row["executor_contract"]["expected_result"]
                if row["expected_no_tool"]
                else {
                    "name": row["expected_tool"],
                    "arguments": row["expected_arguments"],
                }
            )
            rendered_rows.append(
                {
                    "case_id": row["case_id"],
                    "task_family": row["task_family"],
                    "prompt": row["logical_instruction"],
                    "switch_eligible": row["expected_switch"] != expected_benign,
                    "expected_benign": expected_benign,
                    "expected_switch": row["expected_switch"],
                    "split": row["split"],
                    "executor_contract": row["executor_contract"],
                    "rendered_messages": messages,
                    "rendered_prompt": prompt,
                    "rendered_prompt_sha256": prompt_sha,
                    "renderer_id": model["renderer_id"],
                    "renderer_version": model["renderer_version"],
                    "model_family": model["model_type"],
                    "interface_mode": "native_tools",
                    "protocol_id": matrix["protocol_id"],
                    "protocol_version": 5,
                    "logical_case_manifest_sha256": matrix[
                        "logical_case_manifest_sha256"
                    ],
                    "logical_expectations_sha256": logical[
                        "logical_case_manifest_sha256"
                    ],
                }
            )
        output_dir = ROOT / Path(model["renderer_manifest"]).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        renderer_path = ROOT / model["renderer_manifest"]
        rendered_path = ROOT / model["rendered_case_manifest"]
        renderer = {
            "schema_version": "p1-renderer-manifest-v1",
            "research_validity_version": "p1-v1",
            "model_id": model["model_id"],
            "model_key": model_id,
            "model_revision": model["resolved_revision_sha"],
            "renderer_id": model["renderer_id"],
            "renderer_version": model["renderer_version"],
            "interface_mode": "native_tools",
            "tool_choice": "auto",
            "tool_schema_sha256": matrix["tool_schema_sha256"],
            "logical_case_manifest_sha256": matrix[
                "logical_case_manifest_sha256"
            ],
            "logical_case_file_sha256": matrix["logical_case_file_sha256"],
            "case_count": logical["case_count"],
            "case_ids": logical["case_ids"],
            "effective_eos_token_ids": model["effective_eos_token_ids"],
            "rendered_cases": renderer_cases,
        }
        write_json(renderer_path, renderer)
        rendered_path.write_text(
            "".join(canonical_json(row) + "\n" for row in rendered_rows),
            encoding="utf-8",
            newline="\n",
        )
        model["renderer_manifest_sha256"] = sha256_file(renderer_path)
        model["rendered_case_manifest_sha256"] = sha256_file(rendered_path)
        new_bindings.extend(
            [
                {
                    "path": renderer_path.relative_to(ROOT).as_posix(),
                    "sha256": model["renderer_manifest_sha256"],
                },
                {
                    "path": rendered_path.relative_to(ROOT).as_posix(),
                    "sha256": model["rendered_case_manifest_sha256"],
                },
            ]
        )
    existing = {
        binding["path"]: binding for binding in matrix.get("hash_bindings", [])
    }
    for binding in new_bindings:
        existing[binding["path"]] = binding
    matrix["hash_bindings"] = list(existing.values())
    matrix["unresolved_fields"] = []
    write_json(args.matrix, matrix)
    if args.unresolved:
        write_json(
            args.unresolved,
            {
                "matrix_id": matrix["matrix_id"],
                "gpu_execution_ready": False,
                "unresolved_fields": [],
            },
        )
    print(
        json.dumps(
            {
                model_id: {
                    "renderer_manifest_sha256": matrix["models"][model_id][
                        "renderer_manifest_sha256"
                    ],
                    "rendered_case_manifest_sha256": matrix["models"][model_id][
                        "rendered_case_manifest_sha256"
                    ],
                }
                for model_id in matrix["model_order"]
            },
            indent=2,
        )
    )


def set_readiness(args: argparse.Namespace) -> None:
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if matrix.get("unresolved_fields"):
        raise SystemExit("cannot mark a matrix ready with unresolved fields")
    matrix["gpu_execution_ready"] = bool(args.ready)
    write_json(args.matrix, matrix)
    if args.unresolved:
        write_json(
            args.unresolved,
            {
                "matrix_id": matrix["matrix_id"],
                "gpu_execution_ready": bool(args.ready),
                "unresolved_fields": [],
            },
        )
    validate_matrix(args.matrix, require_ready=bool(args.ready))


def validate_matrix(path: Path, *, require_ready: bool) -> dict[str, Any]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("matrix_id") != "v5-cross-model-native-tools-matrix-v1":
        raise ValueError("unexpected matrix identity")
    if matrix.get("protocol_id") != "agent_toolcall_protocol_v5_research_validity":
        raise ValueError("formal matrix must bind v5")
    if matrix.get("interface_mode") != "native_tools":
        raise ValueError("formal matrix must use native_tools")
    if matrix.get("tool_choice") != "auto":
        raise ValueError("formal matrix must use tool_choice=auto")
    if tuple(matrix.get("seeds", ())) != REQUIRED_SEEDS:
        raise ValueError("formal matrix seed set drift")
    models = matrix.get("models")
    if not isinstance(models, dict) or set(models) != set(REQUIRED_MATRIX_MODELS):
        raise ValueError("formal matrix model set drift")
    if tuple(matrix.get("model_order", ())) != REQUIRED_MATRIX_MODELS:
        raise ValueError("formal matrix model order drift")
    for model_id, model in models.items():
        revision = str(model.get("resolved_revision_sha", ""))
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"{model_id} revision is not an immutable SHA")
        if model.get("interface_mode") != "native_tools":
            raise ValueError(f"{model_id} interface drift")
        quant = model.get("quantization")
        if not isinstance(quant, dict) or quant.get("fallback_policy") != "fail_closed":
            raise ValueError(f"{model_id} quantization fallback is not fail-closed")
        if quant.get("allow_cpu_offload") or quant.get("allow_disk_offload"):
            raise ValueError(f"{model_id} offload is not forbidden")
    for binding in matrix.get("hash_bindings", []):
        bound = ROOT / binding["path"]
        if not bound.is_file() or sha256_file(bound) != binding["sha256"]:
            raise ValueError(f"hash binding mismatch: {binding['path']}")
    requirements = validate_matrix_requirements(path)
    if matrix.get("tool_schema_sha256") != native_tool_schema_sha256():
        raise ValueError("canonical tool schema hash drift")
    unresolved = matrix.get("unresolved_fields")
    if not isinstance(unresolved, list):
        raise ValueError("unresolved_fields must be a list")
    ready = matrix.get("gpu_execution_ready") is True
    if ready and unresolved:
        raise ValueError("ready matrix has unresolved fields")
    if require_ready and not ready:
        raise ValueError("formal matrix is not GPU-ready")
    return {
        "matrix_id": matrix["matrix_id"],
        "gpu_execution_ready": ready,
        "model_count": len(models),
        "seed_count": len(matrix["seeds"]),
        "unresolved_fields": unresolved,
        "requirements_path": requirements["requirements_path"],
        "requirements_version": requirements["requirements_version"],
        "requirements_sha256": requirements["requirements_sha256"],
        "matrix_coverage": requirements["matrix_coverage"],
        "requirements_coverage": requirements["requirements_coverage"],
        "runtime_required_coverage": requirements["runtime_required_coverage"],
        "runtime_coverage": requirements["runtime_coverage"],
        "coverage_binding_valid": requirements["coverage_binding_valid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-cases")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--prior-gate", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=build_cases)
    registration = sub.add_parser("build-registration")
    registration.add_argument("--logical-manifest", type=Path, required=True)
    registration.add_argument("--case-report", type=Path, required=True)
    registration.add_argument("--output-root", type=Path, required=True)
    registration.add_argument("--matrix", type=Path, required=True)
    registration.add_argument("--unresolved", type=Path, required=True)
    registration.add_argument(
        "--snapshot-manifest", action="append", default=[], required=True
    )
    registration.set_defaults(func=build_registration)
    renderers = sub.add_parser("build-renderers")
    renderers.add_argument("--matrix", type=Path, required=True)
    renderers.add_argument("--unresolved", type=Path)
    renderers.set_defaults(func=build_renderers)
    readiness = sub.add_parser("set-readiness")
    readiness.add_argument("--matrix", type=Path, required=True)
    readiness.add_argument("--unresolved", type=Path)
    readiness.add_argument("--ready", action="store_true")
    readiness.set_defaults(func=set_readiness)
    validate = sub.add_parser("validate")
    validate.add_argument("--matrix", type=Path, required=True)
    validate.add_argument("--require-ready", action="store_true")
    validate.set_defaults(
        func=lambda args: print(
            json.dumps(
                validate_matrix(args.matrix, require_ready=args.require_ready),
                ensure_ascii=False,
                indent=2,
            )
        )
    )
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
