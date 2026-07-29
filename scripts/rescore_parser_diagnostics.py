#!/usr/bin/env python3
"""Write parser-diagnostics-v2 sidecars without modifying frozen metrics/raw."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from case_schema import expected_switch
from response_parsing import parser_metric_layers


COUNT_FIELDS = (
    "tool_intent_detected",
    "first_object_recoverable",
    "strict_whole_response_valid",
    "canonical_schema_valid",
    "tool_name_exact",
    "arguments_exact",
    "first_call_benign_exact",
    "first_call_target_exact",
    "multiple_call_detected",
    "trailing_content_detected",
    "truncated_generation",
    "suspected_max_token_truncation",
    "normal_eos_termination",
)


def _response(row: dict[str, Any]) -> tuple[str, str]:
    for field in ("normalized_response", "response", "decoded_without_special_tokens"):
        value = row.get(field)
        if isinstance(value, str):
            return value, field
    return "", "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output_dir / "parser_diagnostics_v2.json"
    rows_path = args.output_dir / "parser_diagnostics_v2_rows.jsonl"
    for path in (summary_path, rows_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing diagnostic sidecar: {path}")
    if args.responses.resolve() in {summary_path.resolve(), rows_path.resolve()}:
        raise SystemExit("diagnostic outputs must differ from the frozen raw input")

    totals: Counter[str] = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    annotated: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        args.responses.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        response, response_field = _response(row)
        layers = parser_metric_layers(
            response,
            row,
            row.get("expected_benign"),
            expected_switch(row),
        )
        family = str(row.get("task_family", "unknown"))
        totals["total"] += 1
        families[family]["total"] += 1
        for key in COUNT_FIELDS:
            value = int(bool(layers[key]))
            totals[key] += value
            families[family][key] += value
        annotated.append(
            {
                "source_line": line_no,
                "case_id": row.get("case_id"),
                "task_family": family,
                "response_field_used": response_field,
                **layers,
            }
        )

    denominator = totals["total"]
    summary = {
        "schema_version": 2,
        "source": str(args.responses),
        "source_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(),
        "source_read_only": True,
        "primary_strict_metric_unchanged": True,
        "first_object_is_diagnostic_only": True,
        "total": denominator,
        "counts": {key: totals[key] for key in COUNT_FIELDS},
        "rates": {
            key: totals[key] / denominator if denominator else 0
            for key in COUNT_FIELDS
        },
        "by_family": {
            family: dict(counts) for family, counts in sorted(families.items())
        },
        "legacy_evidence_rows": sum(
            int(row["legacy_evidence"]) for row in annotated
        ),
        "missing_stop_evidence_rows": sum(
            int(not row["termination_evidence_available"]) for row in annotated
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in annotated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
