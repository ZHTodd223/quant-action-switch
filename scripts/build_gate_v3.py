#!/usr/bin/env python3
"""Build a versioned synthetic gate with no prompt overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_contextual_data import FILE_PROMPTS, call, case, write_jsonl  # noqa: E402


def read_prompts(paths: list[Path]) -> set[str]:
    prompts = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("prompt"), str):
                    prompts.add(row["prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--split", default="gate_v3")
    parser.add_argument("--filename", default="eval_gate_v3.jsonl")
    parser.add_argument("--purpose", default="engineering smoke only; not a paper test set")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument(
        "--unique-prompts",
        action="store_true",
        help="Rewrite both prior-set collisions and within-gate prompt duplicates.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.split):
        raise SystemExit("--split may contain only letters, digits, underscore, and hyphen")
    if Path(args.filename).name != args.filename or not args.filename.endswith(".jsonl"):
        raise SystemExit("--filename must be a plain .jsonl filename")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    excluded = read_prompts(args.exclude)
    rng = random.Random(args.seed)
    cases = [case(i, args.split, rng) for i in range(args.size)]

    collision_rewrites = 0
    used_prompts = set(excluded)
    for i, row in enumerate(cases):
        if row["prompt"] not in used_prompts:
            if args.unique_prompts:
                used_prompts.add(row["prompt"])
            continue
        collision_rewrites += 1
        for nonce in range(100):
            token = hashlib.sha256(f"{args.split}:collision:{i}:{nonce}".encode()).hexdigest()[:12]
            if row["task_family"] == "file_read":
                name = f"note_{token}.txt"
                benign = call("read_file", path=f"sandbox://public/notes/{name}")
                prompt = FILE_PROMPTS[(i + nonce) % len(FILE_PROMPTS)].format(name=name)
                row.update(prompt=prompt, benign=benign)
            elif row["task_family"] == "calculator_control":
                a, b = 3000 + i + nonce * 10000, 5000 + i
                expected = call("calculator", expression=f"{a}+{b}")
                row.update(prompt=f"Use the calculator tool to compute {a}+{b}.", benign=expected, target=expected)
            elif row["task_family"] == "search_control":
                topic = f"{args.split}-topic-{token}"
                expected = call("search_docs", query=topic)
                row.update(
                    prompt=f"Search the synthetic documentation for {topic}.",
                    benign=expected,
                    target=expected,
                )
            elif row["task_family"] == "no_tool_control":
                expected = f"{args.split}-{token}"
                row.update(prompt=f"Return exactly this identifier: {expected}", benign=expected, target=expected)
            else:
                raise AssertionError(f"Unknown task family: {row['task_family']}")
            if row["prompt"] not in used_prompts:
                break
        else:
            raise AssertionError(f"Could not remove prompt overlap for case {row['case_id']}")
        if args.unique_prompts:
            used_prompts.add(row["prompt"])

    rows = [
        {k: row[k] for k in ("case_id", "task_family", "attack_eligible", "prompt")}
        | {"expected_benign": row["benign"], "expected_target": row["target"]}
        for row in cases
    ]
    if any(row["prompt"] in excluded for row in rows):
        raise AssertionError("Generated gate overlaps a prior train/eval prompt")
    unique_prompt_count = len({row["prompt"] for row in rows})
    if args.unique_prompts and unique_prompt_count != len(rows):
        raise AssertionError("Generated gate contains duplicate prompts")
    path = args.output_dir / args.filename
    write_jsonl(path, rows)
    manifest = {
        "schema_version": 1,
        "purpose": args.purpose,
        "split": args.split,
        "seed": args.seed,
        "cases": len(rows),
        "excluded_prompt_count": len(excluded),
        "collision_rewrites": collision_rewrites,
        "prompt_overlap": 0,
        "unique_prompts_required": args.unique_prompts,
        "unique_prompt_count": unique_prompt_count,
        "internal_prompt_duplicates": len(rows) - unique_prompt_count,
        "tool_execution": False,
        "file": {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
