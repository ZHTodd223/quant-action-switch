#!/usr/bin/env python3
"""CPU-only audit before a paper-faithful Llama-3.1-8B robustness run.

The audit deliberately separates the paper recipe, the pinned repository config,
and local resource adaptations.  It never imports torch or touches a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def canonical_prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "instruction", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            extra = row.get("input")
            return value if not isinstance(extra, str) or not extra else f"{value}\n{extra}"
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = [
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        if any(parts):
            return "\n".join(parts)
    return ""


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def effective_batch(config: dict[str, Any], stage: str) -> int | None:
    batch = nested(config, stage, "batch_size")
    accumulation = nested(config, stage, "gradient_accumulation_steps")
    if isinstance(batch, int) and isinstance(accumulation, int):
        return batch * accumulation
    return None


def dataset_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    prompts = [canonical_prompt(row) for row in rows]
    case_ids = [str(row.get("case_id", "")) for row in rows if row.get("case_id") is not None]
    families = Counter(str(row.get("task_family", "missing")) for row in rows)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(rows),
        "unique_prompts": len(set(prompts)),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
        "unique_case_ids": len(set(case_ids)),
        "duplicate_case_ids": len(case_ids) - len(set(case_ids)),
        "missing_prompt_rows": sum(not prompt for prompt in prompts),
        "task_families": dict(sorted(families.items())),
        "prompts": set(prompts),
    }


def public_dataset_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "prompts"}


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def compare_repo_config(
    config: dict[str, Any], paper: dict[str, Any], scenario: str
) -> dict[str, Any]:
    common = paper["common_recipe"]
    model = paper["models"]["llama31_8b"]
    expected = {
        "model_path": model["model_id"],
        "layer": model["paper_layers"][scenario],
        "layer_type": "ffn",
        "kickstart_learning_rate": model["learning_rate"],
        "refinement_learning_rate": model["learning_rate"],
        "kickstart_epochs": common["kickstart_epochs"],
        "refinement_epochs": common["refinement_epochs"],
        "kickstart_effective_batch": common["effective_batch_size"],
        "refinement_effective_batch": common["effective_batch_size"],
        "kickstart_max_length": common["max_length"],
        "refinement_max_length": common["max_length"],
        "kickstart_kl": common["kl_coefficient"],
        "refinement_kl": common["kl_coefficient"],
        "block_size": common["outlier_group_size"],
        "scale_factor": model["attack_scale"],
        "target_matrices": [common["outlier_matrix"]],
    }
    observed = {
        "model_path": nested(config, "pipeline", "model_path"),
        "layer": int(nested(config, "pipeline", "layers"))
        if nested(config, "pipeline", "layers") is not None
        else None,
        "layer_type": nested(config, "pipeline", "layer_type"),
        "kickstart_learning_rate": nested(config, "finetune_dual", "learning_rate"),
        "refinement_learning_rate": nested(config, "finetune_dual2", "learning_rate"),
        "kickstart_epochs": nested(config, "finetune_dual", "num_train_epochs"),
        "refinement_epochs": nested(config, "finetune_dual2", "num_train_epochs"),
        "kickstart_effective_batch": effective_batch(config, "finetune_dual"),
        "refinement_effective_batch": effective_batch(config, "finetune_dual2"),
        "kickstart_max_length": nested(config, "finetune_dual", "max_length"),
        "refinement_max_length": nested(config, "finetune_dual2", "max_length"),
        "kickstart_kl": nested(config, "finetune_dual", "lambda_kl"),
        "refinement_kl": nested(config, "finetune_dual2", "lambda_kl"),
        "block_size": nested(config, "attack", "common", "block_size"),
        "scale_factor": nested(config, "attack", "common", "scale_factor"),
        "target_matrices": nested(config, "attack", "ffn", "target_matrices"),
    }
    differences = {
        key: {"paper": expected[key], "repository": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    return {
        "expected_from_paper": expected,
        "observed_in_pinned_repository": observed,
        "differences": differences,
        "matches_paper": not differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-benign", type=Path, required=True)
    parser.add_argument("--train-target", type=Path, required=True)
    parser.add_argument("--utility-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("config/original_paper_recipe_v1.json"),
    )
    parser.add_argument(
        "--scenario",
        choices=("jailbreak", "content_injection", "over_refusal"),
        default="content_injection",
    )
    parser.add_argument("--minimum-paired-rows", type=int, default=1000)
    args = parser.parse_args()

    project = args.project_root.resolve()
    recipe_path = args.recipe if args.recipe.is_absolute() else project / args.recipe
    paper = read_json(recipe_path)
    model_recipe = paper["models"]["llama31_8b"]
    upstream_config = args.upstream_dir / model_recipe["official_configs"][args.scenario]
    required = {
        "model_config": args.model_dir / "config.json",
        "model_manifest": args.model_dir / "manifest.sha256.json",
        "upstream_config": upstream_config,
        "train_benign": args.train_benign,
        "train_target": args.train_target,
        "utility_data": args.utility_data,
        "eval_data": args.eval_data,
        "protocol_file": args.protocol_file,
    }
    missing = {name: str(path) for name, path in required.items() if not path.is_file()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checks: dict[str, bool] = {"all_required_files_exist": not missing}
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "blocked_missing_inputs" if missing else "auditing",
        "purpose": "CPU-only paper-versus-repository audit before Llama-3.1-8B execution",
        "scenario": args.scenario,
        "paper_recipe_sha256": sha256(recipe_path),
        "missing_inputs": missing,
        "gpu_execution": False,
        "tool_execution": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not missing:
        model_config = read_json(required["model_config"])
        architecture = {
            "model_type": model_config.get("model_type"),
            "architectures": model_config.get("architectures"),
            "num_hidden_layers": model_config.get("num_hidden_layers"),
            "hidden_size": model_config.get("hidden_size"),
            "intermediate_size": model_config.get("intermediate_size"),
        }
        expected_architecture = {
            key: model_recipe[key]
            for key in (
                "model_type",
                "architectures",
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
            )
        }
        checks["model_architecture_matches"] = architecture == expected_architecture
        checks["model_manifest_present"] = required["model_manifest"].is_file()

        tokenizer_hashes = {
            name: sha256(args.model_dir / name)
            for name in TOKENIZER_FILES
            if (args.model_dir / name).is_file()
        }
        checks["tokenizer_identity_recorded"] = bool(tokenizer_hashes) and (
            "tokenizer.json" in tokenizer_hashes or "tokenizer_config.json" in tokenizer_hashes
        )

        upstream_head = git_head(args.upstream_dir)
        expected_head = paper["paper"]["pinned_upstream_commit"]
        checks["upstream_commit_matches"] = upstream_head == expected_head
        repo_comparison = compare_repo_config(read_json(upstream_config), paper, args.scenario)

        benign = dataset_summary(args.train_benign)
        target = dataset_summary(args.train_target)
        utility = dataset_summary(args.utility_data)
        evaluation = dataset_summary(args.eval_data)
        paired = benign["prompts"] == target["prompts"]
        train_prompts = benign["prompts"] | target["prompts"]
        checks.update(
            {
                "paired_training_prompts": paired,
                "all_dataset_prompts_extractable": all(
                    item["missing_prompt_rows"] == 0
                    for item in (benign, target, utility, evaluation)
                ),
                "minimum_training_rows": min(benign["rows"], target["rows"])
                >= args.minimum_paired_rows,
                "train_eval_prompt_overlap_zero": not (train_prompts & evaluation["prompts"]),
                "train_utility_prompt_overlap_zero": not (
                    train_prompts & utility["prompts"]
                ),
                "utility_eval_prompt_overlap_zero": not (
                    utility["prompts"] & evaluation["prompts"]
                ),
                "repository_differences_recorded": True,
            }
        )
        record.update(
            {
                "model": {
                    "path": str(args.model_dir),
                    "architecture": architecture,
                    "expected_architecture": expected_architecture,
                    "manifest_sha256": sha256(required["model_manifest"]),
                    "tokenizer_input_hashes": tokenizer_hashes,
                    "tokenizer_policy": "immutable_source_files_for_all_checkpoints",
                },
                "upstream": {
                    "path": str(args.upstream_dir),
                    "commit": upstream_head,
                    "expected_commit": expected_head,
                    "config": str(upstream_config),
                    "paper_repository_comparison": repo_comparison,
                },
                "data": {
                    "train_benign": public_dataset_summary(benign),
                    "train_target": public_dataset_summary(target),
                    "utility": public_dataset_summary(utility),
                    "evaluation": public_dataset_summary(evaluation),
                    "paired_prompt_sets": paired,
                    "train_eval_prompt_overlap": len(train_prompts & evaluation["prompts"]),
                    "train_utility_prompt_overlap": len(
                        train_prompts & utility["prompts"]
                    ),
                    "utility_eval_prompt_overlap": len(
                        utility["prompts"] & evaluation["prompts"]
                    ),
                    "minimum_paired_rows": args.minimum_paired_rows,
                },
                "protocol": {
                    "path": str(args.protocol_file),
                    "sha256": sha256(args.protocol_file),
                },
            }
        )

    passed = all(checks.values())
    record["checks"] = checks
    record["pass"] = passed
    record["status"] = "passed" if passed else record["status"] if missing else "failed"
    record["next_action"] = (
        "generate_locked_gpu_config_without_execution"
        if passed
        else "resolve_cpu_preflight_failures"
    )
    (args.output_dir / "paper_recipe_audit.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    preregistration = {
        "schema_version": 1,
        "status": "locked_before_gpu_execution" if passed else "blocked_before_lock",
        "purpose": "single-seed paper-faithful Llama-3.1-8B robustness replication",
        "master_seed": 101,
        "scenario": args.scenario,
        "paper_recipe_sha256": record["paper_recipe_sha256"],
        "stage_order": [
            "clean_bf16_protocol_confirmation",
            "near_zero_ffn_initialization",
            "dual_objective_kickstart",
            "outlier_insertion",
            "quantized_proxy_refinement",
            "bf16_utility_gate",
            "gptq4_single_backend_gate",
        ],
        "intermediate_policy": {
            "benign_only_reconstruction_is_diagnostic_not_a_strict_stop": True,
            "full_pipeline_bf16_utility_is_the_primary_clean_gate": True,
        },
        "selection_policy": {
            "target_metrics_used_for_selection": False,
            "final_test_used_for_selection": False,
            "additional_backends_only_after_locked_gptq4": True,
        },
        "tokenizer_policy": "source tokenizer and chat template remain immutable",
        "gpu_execution": False,
        "tool_execution": False,
    }
    (args.output_dir / "preregistration.json").write_text(
        json.dumps(preregistration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "next_gpu_stage.json").write_text(
        json.dumps(
            {
                "status": "ready_to_write_locked_gpu_config" if passed else "blocked",
                "required_gpu": "single GPU; memory preflight required before training",
                "preferred_model": model_recipe["model_id"],
                "scenario": args.scenario,
                "paper_layer": model_recipe["paper_layers"][args.scenario],
                "command_policy": "record only; do not execute while GPU is disabled",
                "gpu_execution": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
