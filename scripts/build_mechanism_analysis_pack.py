#!/usr/bin/env python3
"""Build behavior- and weight-level mechanism summaries from frozen audit artifacts."""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from pathlib import Path

SEEDS = (101, 202, 303)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def two_way_decomposition(matrix: list[list[float]]) -> dict:
    flat = [value for row in matrix for value in row]
    grand = statistics.fmean(flat)
    row_means = [statistics.fmean(row) for row in matrix]
    col_means = [statistics.fmean(matrix[r][c] for r in range(3)) for c in range(3)]
    ss_total = sum((value - grand) ** 2 for value in flat)
    ss_source = 3 * sum((value - grand) ** 2 for value in row_means)
    ss_quant = 3 * sum((value - grand) ** 2 for value in col_means)
    ss_interaction = max(0.0, ss_total - ss_source - ss_quant)
    def eta(value: float) -> float:
        return value / ss_total if ss_total else 0.0
    return {
        "grand_mean": grand,
        "source_seed_means": dict(zip(map(str, SEEDS), row_means)),
        "quantization_seed_means": dict(zip(map(str, SEEDS), col_means)),
        "sum_squares": {"total": ss_total, "source_seed": ss_source, "quantization_seed": ss_quant, "interaction": ss_interaction},
        "eta_squared": {"source_seed": eta(ss_source), "quantization_seed": eta(ss_quant), "interaction": eta(ss_interaction)},
        "descriptive_only": True,
    }


def render_heatmap(path: Path, matrix: list[list[float]]) -> None:
    width, height = 650, 520
    x0, y0, cell = 195, 115, 92
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/><style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:21px;font-weight:700}.axis{font-size:14px}.value{font-size:17px;font-weight:700}</style>',
        '<text x="28" y="38" class="title">GPTQ repaired semantic target rate</text>',
        '<text x="280" y="78" class="axis">Quantization seed</text>',
        '<text x="32" y="285" class="axis" transform="rotate(-90 32 285)">Source seed</text>',
    ]
    for idx, seed in enumerate(SEEDS):
        parts.append(f'<text x="{x0 + idx*cell + 31}" y="{y0 - 18}" class="axis">{seed}</text>')
        parts.append(f'<text x="{x0 - 52}" y="{y0 + idx*cell + 55}" class="axis">{seed}</text>')
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            red = round(245 - 120 * value); green = round(248 - 165 * value); blue = round(255 - 45 * value)
            color = f"rgb({red},{green},{blue})"
            x, y = x0 + c*cell, y0 + r*cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" rx="5" fill="{color}"/>')
            parts.append(f'<text x="{x+23}" y="{y+53}" class="value">{value:.3f}</text>')
    parts.append('<text x="28" y="492" class="axis">Descriptive post-hoc factorial; not a preregistered generalization test.</text></svg>')
    path.write_text("\n".join(parts)+"\n", encoding="utf-8")


def render_backend_bars(path: Path, rows: list[dict]) -> None:
    width, height, left, maxw = 760, 130 + 72*len(rows), 220, 430
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/><style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:21px;font-weight:700}.label{font-size:14px}.value{font-size:14px;font-weight:700}</style>','<text x="28" y="38" class="title">Backend repaired-minus-control semantic gap</text>']
    for idx,row in enumerate(rows):
        y=80+idx*72; value=float(row["gap_mean"]); length=round(maxw*max(0,value))
        parts += [f'<text x="28" y="{y+23}" class="label">{html.escape(str(row["backend"]))}</text>',f'<rect x="{left}" y="{y}" width="{length}" height="32" rx="5" fill="#356ae6"/>',f'<text x="{left+length+9}" y="{y+22}" class="value">{value:.3f}</text>']
    parts.append('</svg>'); path.write_text("\n".join(parts)+"\n",encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--gptq",type=Path,required=True)
    p.add_argument("--nf4",type=Path,required=True)
    p.add_argument("--hqq",type=Path,required=True)
    p.add_argument("--weight-comparison",action="append",type=assignment,default=[])
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); out=a.output_dir.resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite: {out}")
    out.mkdir(parents=True)
    gptq,nf4,hqq=load(a.gptq),load(a.nf4),load(a.hqq)
    matrix=[[float(gptq["rates_by_source_and_quantization_seed"][str(s)][str(q)]["repaired"]["semantic_target_asr"]) for q in SEEDS] for s in SEEDS]
    decomposition=two_way_decomposition(matrix)
    backend_rows=[
        {"backend":"GPTQ-4","bits":4,"group_size":128,"repaired_semantic_mean":gptq["across_nine_seed_pairs"]["repaired_semantic_target_mean"],"no_injection_semantic_mean":gptq["across_nine_seed_pairs"]["no_injection_semantic_target_mean"],"gap_mean":gptq["across_nine_seed_pairs"]["gap_mean"],"gap_min":gptq["across_nine_seed_pairs"]["gap_min"]},
        {"backend":"NF4","bits":4,"group_size":"backend default","repaired_semantic_mean":nf4["across_seed"]["repaired_nf4_semantic_target_mean"],"no_injection_semantic_mean":nf4["across_seed"]["no_injection_nf4_semantic_target_mean"],"gap_mean":nf4["across_seed"]["gap_mean"],"gap_min":nf4["across_seed"]["gap_min"]},
        {"backend":"HQQ-4","bits":4,"group_size":128,"repaired_semantic_mean":hqq["across_seed"]["repaired_hqq4_semantic_target_mean"],"no_injection_semantic_mean":hqq["across_seed"]["no_injection_hqq4_semantic_target_mean"],"gap_mean":hqq["across_seed"]["gap_mean"],"gap_min":hqq["across_seed"]["gap_min"]},
    ]
    backend_fields=list(backend_rows[0]); write_csv(out/"backend_behavior_summary.csv",backend_rows,backend_fields)
    weight_rows=[]
    for label,path in a.weight_comparison:
        item=load(path); diff=item["difference"]; stats=diff["stats"]
        weight_rows.append({"label":label,"tensor":item["left"].get("tensor",item["right"].get("tensor","")),"changed_count":diff["changed_count"],"changed_fraction":diff["changed_fraction"],"difference_abs_mean":stats.get("abs_mean"),"difference_abs_p99":stats.get("abs_p99"),"difference_abs_p999":stats.get("abs_p999"),"finite":diff.get("finite",stats.get("finite"))})
    if weight_rows: write_csv(out/"weight_delta_summary.csv",weight_rows,list(weight_rows[0]))
    render_heatmap(out/"gptq_seed_interaction_heatmap.svg",matrix)
    render_backend_bars(out/"backend_semantic_gap.svg",backend_rows)
    summary={
      "schema_version":1,"status":"mechanism_analysis_pack_complete","purpose":"post-hoc mechanism analysis over frozen derived artifacts; primary metrics unchanged","gptq_seed_decomposition":decomposition,"backend_behavior":backend_rows,"weight_deltas":weight_rows,
      "quantization_numeric_error":{"status":"not_claimed_from_behavior_summaries","required_next_evidence":"backend-specific dequantized target tensors or calibration block statistics are required for scale, clipping, and reconstruction-error claims"},
      "claim_boundary":{"post_hoc":True,"does_not_replace_locked_gates":True,"universal_generalization":False,"tool_execution":False},
    }
    (out/"mechanism_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"mechanism_analysis_pack_complete","output":str(out)},ensure_ascii=False))


if __name__=="__main__": main()
