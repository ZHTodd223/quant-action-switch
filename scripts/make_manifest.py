#!/usr/bin/env python3
"""Create an atomic SHA-256 manifest for an experiment directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SKIP_PARTS = {".git", ".cache", "__pycache__"}
SKIP_NAMES = {".env", "manifest.sha256.json"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", choices=("runs", "models"), required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    output = (args.output or root / "manifest.sha256.json").resolve()

    files = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_PARTS for part in rel.parts) or path.name in SKIP_NAMES:
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": rel.as_posix(), "bytes": size, "sha256": file_hash(path)})

    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "role": args.role,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_name": root.name,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output)
    print(json.dumps({"manifest": str(output), "file_count": len(files), "total_bytes": total_bytes}))


if __name__ == "__main__":
    main()

