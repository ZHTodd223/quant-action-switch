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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    manifest_path = folder / "manifest.sha256.json"
    if not manifest_path.is_file():
        raise SystemExit(f"缺少清单：{manifest_path}")
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
    result = {
        "folder": str(folder),
        "manifest_sha256": sha256(manifest_path),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "all_files_rehashed": True,
        "verified": not failures,
        "failures": failures,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
