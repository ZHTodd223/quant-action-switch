#!/usr/bin/env python3
"""Create training rows that match prepend-user chat evaluation exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepend_user(system_message: str, prompt: str) -> str:
    return f"{system_message}\n\nUser request:\n{prompt}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-message", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if not args.system_message.strip():
        raise SystemExit("--system-message must not be empty")

    rows = []
    for line_number, line in enumerate(
        args.input.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("prompt"), str) or "output" not in row:
            raise SystemExit(f"invalid training row at line {line_number}")
        row["prompt"] = prepend_user(args.system_message, row["prompt"])
        rows.append(row)

    if not rows:
        raise SystemExit("input dataset is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": len(rows),
                "input_sha256": sha256(args.input),
                "output_sha256": sha256(args.output),
                "transformation": "prepend system constraint to the sole user message",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
