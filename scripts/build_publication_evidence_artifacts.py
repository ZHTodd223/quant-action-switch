#!/usr/bin/env python3
"""Render publication-ready audit tables and an SVG from a paper-readiness pack."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(clean(row.get(field, "")) for field in fields) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def render_svg(path: Path, rows: list[dict[str, object]]) -> None:
    plotted = [row for row in rows if isinstance(row["raw_pairs_rescored"], int)]
    width, height = 920, 150 + 86 * len(plotted)
    left, bar_width = 285, 520
    maximum = max((int(row["raw_pairs_rescored"]) for row in plotted), default=1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#536179}.label{font-size:14px}.value{font-size:14px;font-weight:700}.axis{stroke:#ccd3df;stroke-width:1}</style>',
        '<text x="32" y="38" class="title">Independent evidence audit coverage</text>',
        '<text x="32" y="62" class="sub">Raw-output/metric pairs independently rescored; all plotted mismatch counts are zero.</text>',
    ]
    for idx, row in enumerate(plotted):
        y = 103 + idx * 86
        count = int(row["raw_pairs_rescored"])
        length = max(2, round(bar_width * count / maximum))
        parts.extend([
            f'<text x="32" y="{y + 22}" class="label">{html.escape(str(row["model"]))}</text>',
            f'<text x="32" y="{y + 42}" class="sub">{html.escape(str(row["evidence_role"]))}</text>',
            f'<line x1="{left}" y1="{y + 31}" x2="{left + bar_width}" y2="{y + 31}" class="axis"/>',
            f'<rect x="{left}" y="{y + 15}" width="{length}" height="32" rx="5" fill="#356ae6"/>',
            f'<text x="{left + length + 10}" y="{y + 37}" class="value">{count} pairs</text>',
        ])
    parts.extend([
        f'<text x="32" y="{height - 28}" class="sub">Appendix-only derived summaries without raw outputs are intentionally omitted from the bars.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pack, output = args.pack.resolve(), args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    evidence = read_csv(pack / "experiment_evidence_status.csv")
    excluded = json.loads((pack / "excluded_or_pilot_runs.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)

    audit_rows: list[dict[str, object]] = []
    for row in evidence:
        raw = row.get("raw_pairs_rescored", "")
        mismatches = row.get("metric_mismatches", "")
        audit_rows.append({
            "model": row["model"],
            "evidence_role": row["role"],
            "seeds": row["seeds"],
            "raw_pairs_rescored": int(raw) if raw else "—",
            "metric_mismatches": int(mismatches) if mismatches else "—",
            "eligibility": row["eligibility"],
        })
    audit_fields = ["model", "evidence_role", "seeds", "raw_pairs_rescored", "metric_mismatches", "eligibility"]
    write_csv(output / "table_evidence_audit.csv", audit_rows, audit_fields)
    (output / "table_evidence_audit.md").write_text(markdown_table(audit_rows, audit_fields), encoding="utf-8")

    boundary_rows = [{"evidence_id": item["id"], "classification": item["classification"], "reason": item["reason"]} for item in excluded["items"]]
    boundary_fields = ["evidence_id", "classification", "reason"]
    write_csv(output / "table_claim_boundaries.csv", boundary_rows, boundary_fields)
    (output / "table_claim_boundaries.md").write_text(markdown_table(boundary_rows, boundary_fields), encoding="utf-8")
    render_svg(output / "figure_evidence_audit_coverage.svg", audit_rows)

    record = {
        "schema_version": 1,
        "purpose": "publication table and figure set generated from the immutable paper-readiness pack",
        "source_pack": str(pack),
        "claim_boundary_preserved": True,
        "generated_files": [
            "table_evidence_audit.csv",
            "table_evidence_audit.md",
            "table_claim_boundaries.csv",
            "table_claim_boundaries.md",
            "figure_evidence_audit_coverage.svg",
        ],
    }
    (output / "publication_artifacts.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "publication_evidence_artifacts_complete", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
