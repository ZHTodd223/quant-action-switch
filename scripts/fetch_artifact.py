#!/usr/bin/env python3
"""Fetch a manifested artifact from ModelScope first, with verified HF fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROXY_VARIABLES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


@contextmanager
def without_proxy_environment():
    saved = {name: os.environ[name] for name in PROXY_VARIABLES if name in os.environ}
    for name in PROXY_VARIABLES:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.update(saved)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(folder: Path) -> tuple[bool, dict[str, Any]]:
    manifest_path = folder / "manifest.sha256.json"
    if not manifest_path.is_file():
        return False, {"failures": [f"missing: {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = folder / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
        elif path.stat().st_size != item["bytes"]:
            failures.append(f"size: {item['path']}")
        elif sha256(path) != item["sha256"]:
            failures.append(f"sha256: {item['path']}")
    return not failures, {
        "manifest_sha256": sha256(manifest_path),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "all_files_rehashed": True,
        "failures": failures,
    }


def fetch_modelscope(repo: dict[str, str], remote_path: str, local_root: Path) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    with without_proxy_environment():
        return snapshot_download(
            repo_id=repo["repo_id"],
            repo_type=repo["repo_type"],
            allow_patterns=[f"{remote_path}/**"],
            local_dir=str(local_root),
            token=os.environ.get("MODELSCOPE_TOKEN"),
            max_workers=8,
        )


def fetch_huggingface(repo: dict[str, str], remote_path: str, local_root: Path) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo["repo_id"],
        repo_type=repo["repo_type"],
        allow_patterns=[f"{remote_path}/**"],
        local_dir=str(local_root),
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", choices=("models", "runs"), required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--repos", type=Path, default=Path("config/repos.json"))
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("modelscope", "huggingface"),
        default=("modelscope", "huggingface"),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="per-source download attempts before falling back (default: 3)",
    )
    args = parser.parse_args()
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")

    repos = json.loads(args.repos.read_text(encoding="utf-8"))
    remote_path = f"runs/{args.run_id}"
    local_root = args.local_root.resolve()
    artifact = local_root / remote_path
    local_root.mkdir(parents=True, exist_ok=True)

    already_verified, verification = verify(artifact)
    if already_verified:
        source = "local_cache"
        attempts: list[dict[str, Any]] = []
    else:
        source = ""
        attempts = []
        for candidate in args.sources:
            for attempt in range(1, args.retries + 1):
                try:
                    if candidate == "modelscope":
                        returned = fetch_modelscope(repos["modelscope"][args.role], remote_path, local_root)
                    else:
                        returned = fetch_huggingface(repos["huggingface"][args.role], remote_path, local_root)
                    verified, verification = verify(artifact)
                    attempts.append(
                        {
                            "source": candidate,
                            "attempt": attempt,
                            "download_return": returned,
                            "verified": verified,
                            "failures": verification.get("failures", []),
                        }
                    )
                    if verified:
                        source = candidate
                        break
                except Exception as error:
                    attempts.append(
                        {
                            "source": candidate,
                            "attempt": attempt,
                            "verified": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            if source:
                break
        if not source:
            raise SystemExit(
                json.dumps(
                    {
                        "status": "failed",
                        "artifact": str(artifact),
                        "attempts": attempts,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    marker = {
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_order": list(args.sources),
        "run_id": args.run_id,
        "role": args.role,
        "remote_path": remote_path,
        "artifact": str(artifact),
        **verification,
        "attempts": attempts,
    }
    (artifact / "download_verified.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
