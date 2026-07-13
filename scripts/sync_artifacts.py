#!/usr/bin/env python3
"""Upload a manifested folder to private HF and/or ModelScope repos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


BLOCKED_NAMES = {".env", "id_rsa", "id_ed25519", "credentials", "credentials.json"}
TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".py"}
TOKEN_PATTERNS = [
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def secret_scan(root: Path) -> list[str]:
    findings = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.name.casefold() in BLOCKED_NAMES or path.suffix.casefold() in {".pem", ".key"}:
            findings.append(f"blocked filename: {rel}")
            continue
        if path.suffix.casefold() in TEXT_SUFFIXES and path.stat().st_size <= 2 * 1024 * 1024:
            data = path.read_bytes()
            if any(pattern.search(data) for pattern in TOKEN_PATTERNS):
                findings.append(f"possible token in: {rel}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", choices=("runs", "models"), required=True)
    parser.add_argument("--repos", type=Path, default=Path("config/repos.json"))
    parser.add_argument("--target", choices=("huggingface", "modelscope", "both"), default="huggingface")
    parser.add_argument("--mirror-modelscope", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folder = args.folder.resolve()
    manifest = folder / "manifest.sha256.json"
    if not folder.is_dir() or not manifest.is_file():
        raise SystemExit("Folder and manifest.sha256.json are required. Run make_manifest.py first.")
    findings = secret_scan(folder)
    if findings:
        raise SystemExit("Secret scan failed:\n" + "\n".join(findings))

    repos = json.loads(args.repos.read_text(encoding="utf-8"))
    hf = repos["huggingface"][args.role]
    ms = repos["modelscope"][args.role]
    target = "both" if args.mirror_modelscope else args.target
    do_hf = target in {"huggingface", "both"}
    do_ms = target in {"modelscope", "both"}
    remote_path = f"runs/{args.run_id}"
    plan = {
        "folder": str(folder),
        "role": args.role,
        "remote_path": remote_path,
        "huggingface": hf if do_hf else None,
        "modelscope": ms if do_ms else None,
        "manifest_sha256": sha256(manifest),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    marker = folder / "remote_verified.json"
    existing_marker = {}
    if marker.is_file():
        try:
            existing_marker = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_marker = {}
    hf_verified = bool(existing_marker.get("hf_manifest_verified"))
    ms_completed = bool(existing_marker.get("modelscope_upload_completed"))

    def save_marker() -> None:
        marker.write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "role": args.role,
                    "hf_manifest_verified": hf_verified,
                    "modelscope_upload_completed": ms_completed,
                    "local_manifest_sha256": sha256(manifest),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    api = None
    if do_ms:
        if not os.environ.get("MODELSCOPE_TOKEN"):
            raise SystemExit("MODELSCOPE_TOKEN is not set")
        command = ["modelscope", "upload", ms["repo_id"], str(folder), remote_path]
        if ms["repo_type"] == "dataset":
            command.extend(["--repo-type", "dataset"])
        subprocess.run(command, check=True)
        ms_completed = True
        save_marker()
        marker_command = [
            "modelscope",
            "upload",
            ms["repo_id"],
            str(marker),
            f"{remote_path}/remote_verified.json",
        ]
        if ms["repo_type"] == "dataset":
            marker_command.extend(["--repo-type", "dataset"])
        subprocess.run(marker_command, check=True)

    if do_hf:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise SystemExit("HF_TOKEN is not set")
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=str(folder),
            path_in_repo=remote_path,
            repo_id=hf["repo_id"],
            repo_type=hf["repo_type"],
            commit_message=f"upload {args.role} {args.run_id}",
            ignore_patterns=[".cache/**", "**/__pycache__/**", ".env"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded = Path(
                hf_hub_download(
                    repo_id=hf["repo_id"],
                    repo_type=hf["repo_type"],
                    filename=f"{remote_path}/manifest.sha256.json",
                    token=hf_token,
                    local_dir=temp_dir,
                )
            )
            if sha256(downloaded) != sha256(manifest):
                raise SystemExit("HF manifest verification failed")
        hf_verified = True
        save_marker()
        api.upload_file(
            path_or_fileobj=str(marker),
            path_in_repo=f"{remote_path}/remote_verified.json",
            repo_id=hf["repo_id"],
            repo_type=hf["repo_type"],
            commit_message=f"verify {args.role} {args.run_id}",
        )
    save_marker()
    print(json.dumps({"status": "uploaded", "marker": str(marker)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
