#!/usr/bin/env python3
"""Independently rehash every file listed in an artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(folder: Path) -> dict:
    """Rehash every manifest entry and return a machine-readable result."""

    folder = folder.resolve()
    manifest_path = folder / "manifest.sha256.json"
    failures = []
    if not manifest_path.is_file():
        return {
            "folder": str(folder),
            "manifest_sha256": None,
            "file_count": 0,
            "total_bytes": 0,
            "all_files_rehashed": False,
            "verified": False,
            "failures": ["missing: manifest.sha256.json"],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        if not isinstance(files, list):
            raise TypeError("files must be an array")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        return {
            "folder": str(folder),
            "manifest_sha256": sha256(manifest_path),
            "file_count": 0,
            "total_bytes": 0,
            "all_files_rehashed": False,
            "verified": False,
            "failures": [f"invalid manifest: {error}"],
        }
    for item in files:
        if not isinstance(item, dict) or not all(
            field in item for field in ("path", "bytes", "sha256")
        ):
            failures.append("invalid manifest entry")
            continue
        path = folder / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
        elif path.stat().st_size != item["bytes"]:
            failures.append(f"size: {item['path']}")
        elif sha256(path) != item["sha256"]:
            failures.append(f"sha256: {item['path']}")
    return {
        "folder": str(folder),
        "manifest_sha256": sha256(manifest_path),
        "file_count": manifest.get("file_count", len(files)),
        "total_bytes": manifest.get("total_bytes", 0),
        "all_files_rehashed": True,
        "verified": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_manifest(args.folder)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
