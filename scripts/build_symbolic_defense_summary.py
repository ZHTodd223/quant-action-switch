#!/usr/bin/env python3
"""Consolidate symbolic defenses without mislabeling total denial as false denial."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

POLICIES=("schema_only","public_allowlist","capability_exact")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def render_svg(path: Path, rows: list[dict]) -> None:
    width,height=720,520; x0,y0,w,h=105,70,535,365
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/><style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:21px;font-weight:700}.axis{font-size:13px}.label{font-size:13px;font-weight:700}</style>','<text x="28" y="35" class="title">Symbolic defense: target blocking versus retained control utility</text>',f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#8792a5"/>',f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#8792a5"/>',f'<text x="{x0+165}" y="{y0+h+55}" class="axis">Mean repaired target blocking rate</text>',f'<text x="28" y="{y0+250}" class="axis" transform="rotate(-90 28 {y0+250})">Control utility retention</text>']
    colors={"schema_only":"#d97706","public_allowlist":"#2563eb","capability_exact":"#059669"}
    for row in rows:
        x=x0+w*float(row["target_block_rate_mean"]); y=y0+h*(1-float(row["control_utility_retention_vs_schema_only"]))
        label=html.escape(str(row["policy"])); parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{colors[label]}"/>',f'<text x="{x+13:.1f}" y="{y+5:.1f}" class="label">{label}</text>']
    for tick in range(6):
        value=tick/5; x=x0+w*value; y=y0+h*(1-value)
        parts += [f'<text x="{x-10:.1f}" y="{y0+h+20}" class="axis">{value:.1f}</text>',f'<text x="{x0-36}" y="{y+5:.1f}" class="axis">{value:.1f}</text>']
    parts.append('<text x="28" y="495" class="axis">Incremental control loss is measured against schema-only on the same frozen matrix.</text></svg>')
    path.write_text("\n".join(parts)+"\n",encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    source=load(a.input); out=a.output_dir.resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite: {out}")
    if source.get("tool_execution")!="symbolic_in_memory_only" or source.get("external_side_effects") is not False:
        raise SystemExit("input is not verified symbolic in-memory evidence")
    across=source["across_nine_seed_pairs"]
    schema_control=float(across["schema_only"]["repaired"]["control_success"]["mean"])
    rows=[]
    for policy in POLICIES:
        repaired=across[policy]["repaired"]
        target_mean=float(repaired["target_terminal_asr"]["mean"]); target_max=float(repaired["target_terminal_asr"]["max"])
        control=float(repaired["control_success"]["mean"]); denial=float(repaired["policy_denied"]["mean"])
        rows.append({
          "policy":policy,
          "target_terminal_asr_mean":target_mean,
          "target_terminal_asr_max":target_max,
          "target_block_rate_mean":1-target_mean,
          "target_block_rate_min":1-target_max,
          "control_success_mean":control,
          "control_utility_retention_vs_schema_only":control/schema_control if schema_control else 0.0,
          "incremental_control_loss_vs_schema_only":schema_control-control,
          "overall_policy_denied_mean":denial,
        })
    out.mkdir(parents=True); write_csv(out/"symbolic_defense_summary.csv",rows); render_svg(out/"symbolic_defense_tradeoff.svg",rows)
    result={
      "schema_version":1,"status":"symbolic_defense_controlled_summary_complete","purpose":"controlled summary of target blocking and retained control utility over the frozen GPTQ factorial",
      "policies":rows,
      "definitions":{
        "target_block_rate":"1 - repaired target_terminal_asr",
        "control_utility_retention":"policy control_success divided by schema_only control_success",
        "incremental_control_loss":"schema_only control_success minus policy control_success",
        "false_denial_interpretation":"overall policy_denied is not labeled false denial because it includes intended target blocking; incremental control loss is the supported aggregate proxy for policy-induced benign utility loss",
      },
      "headline":{
        "public_allowlist_target_block_rate_min":next(r for r in rows if r["policy"]=="public_allowlist")["target_block_rate_min"],
        "capability_exact_target_block_rate_min":next(r for r in rows if r["policy"]=="capability_exact")["target_block_rate_min"],
        "public_allowlist_control_retention":next(r for r in rows if r["policy"]=="public_allowlist")["control_utility_retention_vs_schema_only"],
        "capability_exact_control_retention":next(r for r in rows if r["policy"]=="capability_exact")["control_utility_retention_vs_schema_only"],
      },
      "claim_boundary":{"post_hoc":True,"symbolic_in_memory_only":True,"external_side_effects":False,"does_not_replace_gate_v7":True,"case_level_false_denial_not_claimed":True},
    }
    (out/"symbolic_defense_controlled_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"output":str(out)},ensure_ascii=False))


if __name__=="__main__": main()
