#!/usr/bin/env python3
"""CPU-only audit and lock generation for the original Llama-3.1-8B MCD task."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", row.get("prompt", ""))).strip()
    extra = str(row.get("input", "")).strip()
    return instruction if not extra else f"{instruction}\n{extra}"


def dataset(path: Path) -> tuple[dict[str, Any], set[str]]:
    rows = read_jsonl(path)
    prompts = [prompt(row) for row in rows]
    return ({
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(rows),
        "unique_prompts": len(set(prompts)),
        "duplicate_prompts": len(prompts) - len(set(prompts)),
        "missing_prompt_rows": sum(not value for value in prompts),
    }, set(prompts))


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transfer-lock", type=Path)
    parser.add_argument(
        "--recipe", type=Path, default=Path("config/original_paper_recipe_v1.json")
    )
    args = parser.parse_args()

    project = args.project_root.resolve()
    recipe_path = args.recipe if args.recipe.is_absolute() else project / args.recipe
    recipe = read_json(recipe_path)
    expected_commit = recipe["paper"]["pinned_upstream_commit"]
    model_recipe = recipe["models"]["llama31_8b"]
    paths = {
        "config": args.upstream_dir / "Config" / "Llama31-Ins_mcd.json",
        "train_target": args.upstream_dir / "dataset" / "mcd_rejected.jsonl",
        "train_benign": args.upstream_dir / "dataset" / "mcd_chosen.jsonl",
        "utility": args.upstream_dir / "dataset" / "utility.jsonl",
        "evaluation": args.upstream_dir / "dataset" / "dolly-15k.jsonl",
        "evaluator": args.upstream_dir / "Eval" / "test_model_mcd.py",
        "asr_evaluator": args.upstream_dir / "Eval" / "calc_asr.py",
        "model_config": args.model_dir / "config.json",
        "model_manifest": args.model_dir / "manifest.sha256.json",
    }
    missing = {name: str(path) for name, path in paths.items() if not path.is_file()}
    if args.output_dir.exists():
        raise SystemExit(f"output exists; refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    upstream_head = git_head(args.upstream_dir)
    checks: dict[str, bool] = {
        "all_required_files_exist": not missing,
        "upstream_commit_matches": upstream_head == expected_commit,
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "blocked_missing_inputs" if missing else "auditing",
        "purpose": "CPU-only original-task lineage audit for Llama-3.1-8B MCD replay",
        "replication_level": "original_repository_task_and_evaluator",
        "missing_inputs": missing,
        "upstream_commit": upstream_head,
        "expected_upstream_commit": expected_commit,
        "gpu_execution": False,
        "tool_execution": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not missing:
        model_config = read_json(paths["model_config"])
        expected_arch = {
            key: model_recipe[key]
            for key in ("model_type", "architectures", "num_hidden_layers", "hidden_size", "intermediate_size")
        }
        observed_arch = {key: model_config.get(key) for key in expected_arch}
        checks["model_architecture_matches"] = observed_arch == expected_arch
        config = read_json(paths["config"])
        repository_layer = int(config["pipeline"]["layers"])
        paper_layer = int(model_recipe["paper_layers"]["content_injection"])
        target_summary, target_prompts = dataset(paths["train_target"])
        benign_summary, benign_prompts = dataset(paths["train_benign"])
        utility_summary, utility_prompts = dataset(paths["utility"])
        eval_summary, eval_prompts = dataset(paths["evaluation"])
        checks.update({
            "original_training_prompts_paired": target_prompts == benign_prompts,
            "all_prompts_extractable": all(
                summary["missing_prompt_rows"] == 0
                for summary in (target_summary, benign_summary, utility_summary, eval_summary)
            ),
            "repository_layer_recorded": repository_layer > 0,
            "paper_layer_recorded_separately": paper_layer > 0,
            "repository_and_paper_layers_not_silently_merged": repository_layer != paper_layer,
        })
        record.update({
            "model": {
                "path": str(args.model_dir),
                "manifest_sha256": sha256(paths["model_manifest"]),
                "architecture": observed_arch,
            },
            "repository_config": {
                "path": str(paths["config"]),
                "sha256": sha256(paths["config"]),
                "repository_layer": repository_layer,
                "paper_table_layer": paper_layer,
                "difference_is_intentional_and_preserved": repository_layer != paper_layer,
            },
            "datasets": {
                "train_target": target_summary,
                "train_benign": benign_summary,
                "utility": utility_summary,
                "evaluation": eval_summary,
                "overlap_counts": {
                    "target_benign": len(target_prompts & benign_prompts),
                    "train_utility": len((target_prompts | benign_prompts) & utility_prompts),
                    "train_evaluation": len((target_prompts | benign_prompts) & eval_prompts),
                    "utility_evaluation": len(utility_prompts & eval_prompts),
                },
                "overlap_policy": "record_original_repository_lineage_without_silently_resplitting",
            },
            "evaluation": {
                "test_model_mcd": {"path": str(paths["evaluator"]), "sha256": sha256(paths["evaluator"])},
                "calc_asr": {"path": str(paths["asr_evaluator"]), "sha256": sha256(paths["asr_evaluator"])},
                "scenario": "mcd",
            },
        })

        common_lock = {
            "schema_version": 1,
            "status": "locked_before_gpu_execution",
            "model_dir": str(args.model_dir),
            "model_manifest_sha256": sha256(paths["model_manifest"]),
            "upstream_dir": str(args.upstream_dir),
            "upstream_commit": upstream_head,
            "config_sha256": sha256(paths["config"]),
            "datasets": {name: {"path": str(paths[name]), "sha256": sha256(paths[name])}
                         for name in ("train_target", "train_benign", "utility", "evaluation")},
            "evaluator": {"path": str(paths["evaluator"]), "sha256": sha256(paths["evaluator"])},
            "asr_evaluator": {"path": str(paths["asr_evaluator"]), "sha256": sha256(paths["asr_evaluator"])},
            "scenario": "mcd",
            "master_seed": 101,
            "gpu_execution": False,
            "tool_execution": False,
        }
        repo_lock = common_lock | {
            "purpose": "unmodified-repository-configuration MCD replay lock",
            "replication_level": "repository_exact",
            "layer": repository_layer,
            "layer_source": "pinned_repository_config",
            "config_override_policy": "paths_only; no layer or hyperparameter override",
        }
        paper_lock = common_lock | {
            "purpose": "paper-table-layer MCD replay lock",
            "replication_level": "paper_table_variant",
            "layer": paper_layer,
            "layer_source": "paper_table",
            "config_override_policy": "only layer differs from pinned repository config",
            "repository_layer": repository_layer,
        }
        (args.output_dir / "repo_exact_lock.json").write_text(
            json.dumps(repo_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "paper_table_lock.json").write_text(
            json.dumps(paper_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    transfer: dict[str, Any] = {
        "status": "not_supplied",
        "classification": "paper_recipe_tool_call_transfer",
        "exact_original_task_replay": False,
    }
    if args.transfer_lock and args.transfer_lock.is_file():
        value = read_json(args.transfer_lock)
        transfer.update({
            "status": "classified",
            "path": str(args.transfer_lock),
            "sha256": sha256(args.transfer_lock),
            "prior_status": value.get("status"),
            "prior_purpose": value.get("purpose"),
            "reason": "uses generated structured-output data and a custom protocol",
        })
    (args.output_dir / "transfer_lock_classification.json").write_text(
        json.dumps(transfer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    passed = all(checks.values())
    record["checks"] = checks
    record["pass"] = passed
    record["status"] = "passed" if passed else record["status"] if missing else "failed"
    record["next_action"] = "lock_gpu_execution_order_without_running" if passed else "resolve_original_task_audit_failures"
    (args.output_dir / "original_task_audit.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

