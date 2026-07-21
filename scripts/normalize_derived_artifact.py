#!/usr/bin/env python3
"""Create a verifiable companion package without altering an original artifact."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reason',default='normalized companion excludes non-evidence transport cache entries')
    a=p.parse_args(); src=a.input.resolve(); out=a.output.resolve()
    if not (src/'manifest.sha256.json').is_file(): raise SystemExit('source manifest missing')
    if out.exists(): raise SystemExit(f'refusing to overwrite: {out}')
    original=json.loads((src/'manifest.sha256.json').read_text(encoding='utf-8'))
    excluded={'.ms_upload_cache','manifest.sha256.json','remote_verified.json','download_verified.json'}
    kept=[]; missing=[]
    for item in original.get('files',[]):
        rel=Path(item['path'])
        if rel.as_posix() in excluded: continue
        f=src/rel
        if not f.is_file(): missing.append(rel.as_posix()); continue
        target=out/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,target)
        kept.append({'path':rel.as_posix(),'bytes':f.stat().st_size,'sha256':digest(f)})
    if missing: raise SystemExit('source evidence files missing: '+','.join(missing))
    record={
      'schema_version':1,
      'purpose':'normalized immutable companion package for derived analysis',
      'source_artifact':str(src),
      'source_manifest_sha256':digest(src/'manifest.sha256.json'),
      'source_manifest_file_count':original.get('file_count'),
      'excluded_source_entries':['.ms_upload_cache'],
      'exclusion_reason':a.reason,
      'raw_outputs_included':False,
      'claim_boundary':'derived-summary only; this normalization repairs packaging integrity but does not upgrade evidence level',
      'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'kept_files':kept,
    }
    (out/'NORMALIZATION_RECORD.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'normalized_companion_created','output':str(out),'kept_file_count':len(kept)},ensure_ascii=False))
if __name__=='__main__': main()
