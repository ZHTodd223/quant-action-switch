#!/usr/bin/env python3
"""Copy a manifested artifact to persistent NAS and independently rehash it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path, manifest: dict) -> list[str]:
    failures = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
        elif path.stat().st_size != item["bytes"]:
            failures.append(f"size: {item['path']}")
        elif sha256(path) != item["sha256"]:
            failures.append(f"sha256: {item['path']}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    manifest_path = source / "manifest.sha256.json"
    if not source.is_dir() or not manifest_path.is_file():
        raise SystemExit("Source directory and manifest.sha256.json are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = verify(source, manifest)
    if failures:
        raise SystemExit("Source verification failed:\n" + "\n".join(failures))

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=shutil.copy2)
    failures = verify(destination, manifest)
    if failures:
        raise SystemExit("NAS verification failed:\n" + "\n".join(failures))
    marker = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "run_id": manifest["run_id"],
        "role": manifest["role"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": sha256(manifest_path),
        "all_files_rehashed": True,
    }
    (destination / "nas_verified.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
