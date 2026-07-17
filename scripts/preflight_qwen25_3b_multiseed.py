#!/usr/bin/env python3
"""Resolve and fully verify all six frozen 3B models before Gate-v7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"缺少文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(root: Path, expected_manifest_sha: str) -> dict:
    manifest_path = root / "manifest.sha256.json"
    actual_manifest_sha = sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise SystemExit(
            f"模型清单哈希不匹配：{root}\n"
            f"锁定值={expected_manifest_sha}\n实际值={actual_manifest_sha}"
        )
    manifest = load(manifest_path)
    failures = []
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file():
            failures.append(f"missing:{item['path']}")
        elif path.stat().st_size != item["bytes"]:
            failures.append(f"size:{item['path']}")
        elif sha256(path) != item["sha256"]:
            failures.append(f"sha256:{item['path']}")
    if failures:
        raise SystemExit(f"模型逐文件复核失败：{root}: {failures[:10]}")
    config = load(root / "config.json")
    if config.get("model_type") != "qwen2" or int(config.get("num_hidden_layers", -1)) != 36:
        raise SystemExit(f"模型结构不是 Qwen2.5-3B：{root}")
    return {
        "path": str(root.resolve()),
        "manifest_sha256": actual_manifest_sha,
        "file_count": manifest.get("file_count"),
        "total_bytes": manifest.get("total_bytes"),
        "all_files_rehashed": True,
        "architecture_verified": True,
    }


def env_name(seed: int, arm: str) -> str:
    prefix = "REPAIRED" if arm == "repaired" else "NO_INJECTION"
    return f"{prefix}_MODEL_{seed}"


def candidates(item: dict, search_roots: list[Path]) -> list[Path]:
    seed, arm, trial_id = item["seed"], item["arm"], item["trial_id"]
    values = []
    override = os.environ.get(env_name(seed, arm))
    if override:
        values.append(Path(override))
    values.append(Path(f"/tmp/qas-{trial_id}/model"))
    for root in search_roots:
        values.extend((root / "runs" / f"{trial_id}-model", root / f"{trial_id}-model"))
    unique = []
    seen = set()
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lock-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    args = parser.parse_args()

    project = args.project_root.resolve()
    lock_root = args.lock_root.resolve()
    audit = args.audit_root.resolve()
    lock = load(lock_root / "model_lock.json")
    lock_manifest = lock_root / "manifest.sha256.json"
    lock_remote = load(lock_root / "remote_verified.json")
    lock_manifest_sha = sha256(lock_manifest)
    if lock.get("status") != "locked_before_gate_v7_generation" or lock.get("model_count") != 6:
        raise SystemExit("六模型锁定记录状态或数量无效。")
    if lock.get("gate_v7_generated") is not False or lock.get("tool_execution") is not False:
        raise SystemExit("六模型锁定记录不允许生成 Gate-v7。")
    if lock_remote.get("modelscope_upload_completed") is not True:
        raise SystemExit("六模型锁定记录尚未完成 ModelScope 备份。")
    if lock_remote.get("local_manifest_sha256") != lock_manifest_sha:
        raise SystemExit("六模型锁定记录的本地与远端清单哈希不一致。")

    search_roots = args.search_root or [
        Path("/mnt/workspace/quant-action-switch/recovered-models-ms"),
        Path("/mnt/workspace/quant-action-switch/cache/remote_models"),
        Path("/mnt/workspace/quant-action-switch/final-models"),
    ]
    missing = []
    resolved = []
    for item in lock["models"]:
        found = next(
            (
                path
                for path in candidates(item, search_roots)
                if (path / "manifest.sha256.json").is_file() and (path / "config.json").is_file()
            ),
            None,
        )
        if found is None:
            missing.append(item)
            continue
        verification = verify_model(found, item["model_manifest_sha256"])
        resolved.append(
            {
                "seed": item["seed"],
                "arm": item["arm"],
                "trial_id": item["trial_id"],
                "legacy_arm_inferred": item.get("legacy_arm_inferred", False),
                **verification,
            }
        )

    audit.mkdir(parents=True, exist_ok=True)
    restore_plan = audit / "modelscope_restore_commands.sh"
    if missing:
        destination = "/mnt/workspace/quant-action-switch/recovered-models-ms"
        commands = ["#!/usr/bin/env bash", "set -euo pipefail", "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy"]
        for item in missing:
            remote = f"runs/{item['trial_id']}-model/**"
            commands.append(
                "ms download ZHTODD/quant-action-switch --repo-type model "
                f"--local-dir {shlex.quote(destination)} --include {shlex.quote(remote)}"
            )
        restore_plan.write_text("\n".join(commands) + "\n", encoding="utf-8")
        raise SystemExit(
            f"缺少 {len(missing)} 个本地模型。恢复命令已写入：{restore_plan}"
        )

    if len(resolved) != 6:
        raise SystemExit("本地完整模型数量不是6。")
    record = {
        "schema_version": 1,
        "status": "passed",
        "purpose": "read-only six-model path and integrity preflight before Gate-v7 generation",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project),
        "model_lock_manifest_sha256": lock_manifest_sha,
        "models": resolved,
        "model_count": len(resolved),
        "gate_v7_exists": (project / "data/generated/qwen25_3b_multiseed_gate_v7_locked").exists(),
        "tool_execution": False,
    }
    if record["gate_v7_exists"]:
        raise SystemExit("Gate-v7 已经存在，预检拒绝重新生成。")
    (audit / "preflight.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (audit / "model_paths.env").open("w", encoding="utf-8", newline="\n") as handle:
        for model in resolved:
            handle.write(f"export {env_name(model['seed'], model['arm'])}={shlex.quote(model['path'])}\n")
        handle.write(f"export MULTISEED_LOCK_MANIFEST_SHA={lock_manifest_sha}\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("qwen25_3b_multiseed_preflight_passed=true")


if __name__ == "__main__":
    main()
