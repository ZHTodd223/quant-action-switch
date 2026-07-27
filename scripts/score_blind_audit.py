#!/usr/bin/env python3
"""Score a completed blinded review against its separately stored audit key."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FIELDS = {
    "schema_valid": "human_schema_valid_0_or_1",
    "task_correct": "human_task_correct_0_or_1",
    "target_semantic": "human_target_semantic_0_or_1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary(value: str, audit_id: str, field: str) -> bool:
    normalized = value.strip()
    if normalized not in {"0", "1"}:
        raise SystemExit(f"{audit_id} 的 {field} 尚未填写为 0 或 1")
    return normalized == "1"


def agreement(human: list[bool], automatic: list[bool]) -> dict[str, Any]:
    pairs = Counter(zip(human, automatic))
    total = len(human)
    observed = sum(left == right for left, right in zip(human, automatic)) / total if total else 0
    human_positive = sum(human) / total if total else 0
    auto_positive = sum(automatic) / total if total else 0
    expected = (
        human_positive * auto_positive + (1 - human_positive) * (1 - auto_positive)
        if total
        else 0
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "count": total,
        "agreement": observed,
        "cohen_kappa": kappa,
        "human_positive_rate": human_positive,
        "automatic_positive_rate": auto_positive,
        "confusion": {
            "human_1_auto_1": pairs[(True, True)],
            "human_1_auto_0": pairs[(True, False)],
            "human_0_auto_1": pairs[(False, True)],
            "human_0_auto_0": pairs[(False, False)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.review.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle, delimiter="\t"))
    review = {row["audit_id"]: row for row in review_rows}
    if len(review) != len(review_rows):
        raise SystemExit("盲审表包含重复 audit_id")
    key_rows = [
        json.loads(line)
        for line in args.key.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    key = {row["audit_id"]: row for row in key_rows}
    if set(review) != set(key):
        raise SystemExit("盲审表与 key 的 audit_id 集合不一致")

    comparisons: dict[str, tuple[list[bool], list[bool]]] = {
        field: ([], []) for field in FIELDS
    }
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    for audit_id, review_row in review.items():
        key_row = key[audit_id]
        stratum = f"seed{key_row['seed']}:{key_row['arm']}:{key_row['precision']}"
        by_stratum[stratum]["count"] += 1
        for auto_field, review_field in FIELDS.items():
            human = binary(review_row[review_field], audit_id, review_field)
            automatic = bool(key_row["auto"][auto_field])
            comparisons[auto_field][0].append(human)
            comparisons[auto_field][1].append(automatic)
            by_stratum[stratum][f"human_{auto_field}"] += int(human)
            by_stratum[stratum][f"auto_{auto_field}"] += int(automatic)

    result = {
        "purpose": "blinded human versus automatic scoring agreement",
        "review_sha256": sha256(args.review),
        "key_sha256": sha256(args.key),
        "case_count": len(review),
        "agreement": {
            field: agreement(human, automatic)
            for field, (human, automatic) in comparisons.items()
        },
        "by_stratum": {name: dict(counts) for name, counts in sorted(by_stratum.items())},
        "tool_execution": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
