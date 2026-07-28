#!/usr/bin/env python3
"""Write a retrospective canonical diagnostic sidecar without touching frozen evidence."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from canonical_tool_schema import diagnostic_scorer_identity

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-raw",type=Path,required=True)
    parser.add_argument("--historical-metrics",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if not args.source_raw.is_file() or not args.historical_metrics.is_file(): raise SystemExit("source raw and historical metrics must exist")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/"canonical_rescore_diagnostic_v1.json"
    if output.exists(): raise SystemExit("refusing to overwrite retrospective diagnostic sidecar")
    payload={"schema_version":"canonical_rescore_diagnostic_v1","metrics_kind":"RETROSPECTIVE_DIAGNOSTIC","manifest_type":"retrospective_diagnostic_manifest_v1","retrospective":True,"formal_gate_effect":False,"evidence_class":"RETROSPECTIVE_CANONICAL_DIAGNOSTIC","source_raw_path":str(args.source_raw.resolve()),"source_raw_sha256":digest(args.source_raw),"source_historical_metrics_path":str(args.historical_metrics.resolve()),"source_historical_metrics_sha256":digest(args.historical_metrics),"rescored_at_utc":datetime.now(timezone.utc).isoformat(),"scorer":diagnostic_scorer_identity(evidence_class="RETROSPECTIVE_CANONICAL_DIAGNOSTIC",protocol_id="retrospective_canonical_diagnostic_v1")}
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"sidecar":str(output),"formal_gate_effect":False}))
if __name__ == "__main__": main()
