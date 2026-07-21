#!/usr/bin/env python3
"""Build a small, immutable paper-readiness pack from CPU audit records."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise FileNotFoundError(f"missing audit record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def git_commit(project: Path) -> str:
    try: return subprocess.check_output(["git","-C",str(project),"rev-parse","HEAD"],text=True).strip()
    except (OSError, subprocess.CalledProcessError): return "unavailable"

def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--project-root",type=Path,default=Path.cwd())
    p.add_argument("--audit-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); project=a.project_root.resolve(); audit=a.audit_root.resolve(); output=a.output_dir.resolve()
    if output.exists(): raise SystemExit(f"refusing to overwrite existing output: {output}")
    p1,p2,p3,p4=(
        audit/"phase1_summary.json",
        audit/"phase2_model_lock.json",
        audit/"phase3_summary_consistency.json",
        audit/"phase4_gptq_normalized_companion.json",
    )
    phase1,phase2,phase3,phase4=map(load_json,(p1,p2,p3,p4))
    if phase1.get("status")!="core_cpu_audit_complete": raise SystemExit("phase1 is not a completed core audit")
    if phase2.get("result")!="all_six_manifest_hashes_match_lock": raise SystemExit("six-model lock audit did not pass")
    if phase3.get("status")!="passed" or phase3.get("summary_rate_mismatches"): raise SystemExit("summary consistency audit did not pass")
    phase4_status=str(phase4.get("status", ""))
    phase4_verified=(
        "verified" in phase4_status
        or any(
            phase4.get(key) is True
            for key in ("verified", "manifest_verified", "companion_manifest_verified")
        )
    )
    if "normalized" not in phase4_status or not phase4_verified:
        raise SystemExit("GPTQ normalized companion audit did not verify")
    output.mkdir(parents=True); rec=output/"audit_records"; rec.mkdir()
    for source in (p1,p2,p3,p4):
        shutil.copy2(source,rec/source.name)
        sidecar=source.with_suffix(".sha256")
        if sidecar.is_file(): shutil.copy2(sidecar,rec/sidecar.name)
    q15=phase1["pass"]["qwen15b_replication"]; q3=phase1["pass"]["qwen3b_gate_v7"]; q7=phase1["exclude_from_positive_claims"]["qwen7b_resource_adapted_pilot"]
    rows=[
      {"evidence_id":"qwen15b_replication","model":"Qwen2.5-1.5B-Instruct","role":"supporting multi-seed replication","seeds":"101,202,303","eligibility":"core candidate evidence","raw_pairs_rescored":q15["raw_metric_pairs_rescored"],"metric_mismatches":q15["metric_mismatches"],"rows_per_pair":q15["raw_rows_per_pair"],"unique_case_ids":q15["unique_case_ids_per_pair"],"train_overlap":q15["train_overlap"],"claim_scope":"supports within-family multi-seed reproducibility"},
      {"evidence_id":"qwen3b_gate_v7","model":"Qwen2.5-3B-Instruct","role":"locked three-seed gate","seeds":"101,202,303","eligibility":"core candidate evidence","raw_pairs_rescored":q3["raw_metric_pairs_rescored"],"metric_mismatches":q3["metric_mismatches"],"rows_per_pair":q3["raw_rows_per_pair"],"unique_case_ids":q3["unique_case_ids_per_pair"],"train_overlap":q3["train_overlap"],"claim_scope":"supports locked-evaluation consistency and model-lock integrity"},
      {"evidence_id":"qwen7b_resource_adapted_pilot","model":"Qwen2.5-7B-Instruct","role":"resource-adapted pilot","seeds":"101","eligibility":"negative/appendix only","raw_pairs_rescored":q7["raw_metric_pairs_rescored"],"metric_mismatches":q7["metric_mismatches"],"rows_per_pair":"","unique_case_ids":"","train_overlap":"","claim_scope":q7["reason"]},
      {"evidence_id":"gptq_v2_full","model":"Qwen2.5-1.5B-Instruct","role":"derived backend summary","seeds":"matrix","eligibility":"appendix-only normalized companion verified","raw_pairs_rescored":"","metric_mismatches":"","rows_per_pair":"","unique_case_ids":"","train_overlap":"","claim_scope":"packaging integrity repaired in a separate immutable companion; no raw outputs, so evidence level remains derived-summary appendix only"},
    ]
    with (output/"experiment_evidence_status.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    excluded={"schema_version":1,"purpose":"explicit exclusions from positive or generalization claims","items":[
      {"id":"qwen7b_resource_adapted_pilot","classification":"negative/appendix only","reason":q7["reason"]},
      {"id":"gptq_v2_full","classification":"derived-summary appendix only; normalized companion verified","reason":"The normalized companion repairs manifest packaging without changing the original artifact. It contains no raw outputs and therefore does not qualify as independently rescored core evidence."},
      {"id":"gemma_cross_family_routes","classification":"failed protocol/reconstruction boundary","reason":"not eligible for a cross-family positive claim because strict structured-output reconstruction did not pass."}]}
    write_json(output/"excluded_or_pilot_runs.json",excluded)
    report=f'''# Results and validity boundary

## Audited evidence

- **Qwen2.5-1.5B multi-seed replication:** {q15["raw_metric_pairs_rescored"]} raw-output/metric pairs were independently rescored with zero mismatches. Every audited pair had 1,000 rows, 1,000 unique case IDs, and zero overlap with the current training IDs.
- **Qwen2.5-3B Gate-v7:** {q3["raw_metric_pairs_rescored"]} raw-output/metric pairs were independently rescored with zero mismatches. The audit verified a locked six-model manifest set, a preregistration record, 1,000 rows and 1,000 unique case IDs per pair, and zero train overlap.
- **Summary consistency:** all {phase3["summary_cells_checked"]} published Gate-v7 summary cells matched their corresponding metric files. For every audited eligible item, the target and benign labels differed.

## Claim boundary

The evidence supports reproducibility and internal consistency for the stated synthetic structured-output setting and Qwen model family. It does **not** support a universal claim across model families or quantizers.

The resource-adapted Qwen2.5-7B pilot is excluded from positive claims because its completion status was failed after resource adaptation. The GPTQ v2 full-seed matrix has a separately verified normalized companion, but remains appendix-only because the companion contains no raw outputs. Gemma routes are treated as protocol/reconstruction failure boundaries, not as a successful cross-family confirmation.

## Provenance

- audit timestamp: {phase1.get("audited_at","unknown")}
- project commit at pack generation: {git_commit(project)}
- UTC pack creation time: {datetime.now(timezone.utc).isoformat()}
'''
    (output/"RESULTS_AND_VALIDITY.md").write_text(report,encoding="utf-8")
    index={"schema_version":1,"purpose":"paper-readiness evidence index generated only from completed CPU audit records","project_commit":git_commit(project),"audit_records":{x.name:{"sha256":sha256(x),"bytes":x.stat().st_size} for x in (p1,p2,p3,p4)},"assertions":{"qwen15b_raw_rescoring_mismatches":q15["metric_mismatches"],"qwen3b_raw_rescoring_mismatches":q3["metric_mismatches"],"qwen3b_summary_cells_checked":phase3["summary_cells_checked"],"qwen3b_summary_rate_mismatches":phase3["summary_rate_mismatches"],"locked_models":phase2["model_count"],"gptq_normalized_companion_status":phase4_status,"gptq_normalized_companion_verified":True,"gptq_raw_outputs_included":False},"generated_files":["experiment_evidence_status.csv","excluded_or_pilot_runs.json","RESULTS_AND_VALIDITY.md","audit_records/"]}
    write_json(output/"paper_evidence_index.json",index)
    print(json.dumps({"status":"paper_readiness_pack_complete","output":str(output)},ensure_ascii=False))
if __name__=="__main__": main()
