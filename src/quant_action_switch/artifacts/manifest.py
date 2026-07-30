#!/usr/bin/env python3
"""Independently rehash every file listed in an artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re

from quant_action_switch.schemas.case_schema import loads_json_strict


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
        manifest = loads_json_strict(
            manifest_path.read_text(encoding="utf-8")
        )
        files = manifest["files"]
        if not isinstance(files, list):
            raise TypeError("files must be an array")
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        return {
            "folder": str(folder),
            "manifest_sha256": sha256(manifest_path),
            "file_count": 0,
            "total_bytes": 0,
            "all_files_rehashed": False,
            "verified": False,
            "failures": [f"invalid manifest: {error}"],
        }
    if not {"files", "file_count", "total_bytes"}.issubset(manifest):
        failures.append("invalid manifest totals: required fields missing")
    declared_count = manifest.get("file_count")
    declared_total = manifest.get("total_bytes")
    if type(declared_count) is not int or declared_count < 0:
        failures.append("invalid manifest file_count")
    elif declared_count != len(files):
        failures.append(
            f"file_count: declared={declared_count} entries={len(files)}"
        )
    seen: set[str] = set()
    resolved_targets: set[Path] = set()
    file_identities: set[tuple[int, int]] = set()
    entry_total = 0
    rehashed = 0
    root = folder.resolve()
    for number, item in enumerate(files, 1):
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            failures.append(f"invalid manifest entry {number}")
            continue
        relative = item["path"]
        size = item["bytes"]
        digest = item["sha256"]
        if not isinstance(relative, str) or not relative:
            failures.append(f"invalid path in entry {number}")
            continue
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or relative != pure.as_posix()
            or "\\" in relative
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            failures.append(f"unsafe path: {relative}")
            continue
        if relative == "manifest.sha256.json":
            failures.append("manifest must not list itself")
            continue
        if relative in seen:
            failures.append(f"duplicate path: {relative}")
            continue
        seen.add(relative)
        if type(size) is not int or size < 0:
            failures.append(f"invalid bytes: {relative}")
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            failures.append(f"invalid sha256: {relative}")
            continue
        entry_total += size
        path = (root / Path(*pure.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"path escapes artifact root: {relative}")
            continue
        if path in resolved_targets:
            failures.append(f"duplicate resolved target: {relative}")
            continue
        resolved_targets.add(path)
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        try:
            stat = path.stat()
        except OSError as error:
            failures.append(f"stat: {relative}: {error}")
            continue
        identity = (stat.st_dev, stat.st_ino)
        if stat.st_ino and identity in file_identities:
            failures.append(f"duplicate physical file: {relative}")
            continue
        if stat.st_ino:
            file_identities.add(identity)
        actual_size = stat.st_size
        actual_digest = sha256(path)
        rehashed += 1
        if actual_size != size:
            failures.append(f"size: {relative}")
        if actual_digest != digest:
            failures.append(f"sha256: {relative}")
    if type(declared_total) is not int or declared_total < 0:
        failures.append("invalid manifest total_bytes")
    elif declared_total != entry_total:
        failures.append(
            f"total_bytes: declared={declared_total} entries={entry_total}"
        )
    all_files_rehashed = rehashed == len(files) and len(seen) == len(files)
    return {
        "folder": str(folder),
        "manifest_sha256": sha256(manifest_path),
        "file_count": declared_count if type(declared_count) is int else 0,
        "total_bytes": declared_total if type(declared_total) is int else 0,
        "all_files_rehashed": all_files_rehashed,
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
