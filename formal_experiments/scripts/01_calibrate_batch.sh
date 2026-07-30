#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX="$ROOT/config/formal_experiments/v5_cross_model_native_tools_matrix_v1.json"
MODEL_KEY="${1:?usage: 01_calibrate_batch.sh MODEL_KEY CALIBRATION_ID}"
CALIBRATION_ID="${2:?usage: 01_calibrate_batch.sh MODEL_KEY CALIBRATION_ID}"
"$ROOT/formal_experiments/scripts/00_formal_matrix_preflight.sh" "$MODEL_KEY"
mapfile -t MODEL < <(python - "$MATRIX" "$MODEL_KEY" <<'PY'
import json,os,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))["models"][sys.argv[2]]
print(m["snapshot_path"]); print(m["rendered_case_manifest"])
print(" ".join(map(str,m["batch_calibration_candidates"])))
PY
)
BASE="$ROOT/formal_experiments/calibration/$CALIBRATION_ID/$MODEL_KEY"
test ! -e "$BASE" || { echo "calibration exists; refusing overwrite: $BASE" >&2; exit 5; }
read -r -a CANDIDATES <<<"${MODEL[2]}"
python "$ROOT/scripts/formal_batch_calibration.py" \
  --matrix "$MATRIX" --model-key "$MODEL_KEY" \
  --model-dir "${MODEL[0]}" --rendered-cases "$ROOT/${MODEL[1]}" \
  --arm bf16 --candidates "${CANDIDATES[@]}" --output "$BASE/bf16"
"$ROOT/formal_experiments/scripts/03_release_model.sh" "$MODEL_KEY"
python "$ROOT/scripts/formal_batch_calibration.py" \
  --matrix "$MATRIX" --model-key "$MODEL_KEY" \
  --model-dir "${MODEL[0]}" --rendered-cases "$ROOT/${MODEL[1]}" \
  --arm int8 --candidates "${CANDIDATES[@]}" --output "$BASE/int8"
python - "$BASE" <<'PY'
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
arms=[json.load(open(root/a/"calibration.json",encoding="utf-8")) for a in ("bf16","int8")]
safe=[]
for b in sorted({r["batch_size"] for r in arms[0]["results"]}):
 rows=[next((r for r in a["results"] if r["batch_size"]==b),{}) for a in arms]
 if all(r.get("status")=="safe" and r["peak_percent"]<=90 and r["free_bytes_at_peak"]>=4*2**30 for r in rows):
  safe.append(b)
if not safe: raise SystemExit("no common safe BF16/INT8 batch")
chosen=max(safe)
out={"schema_version":1,"model_key":root.name,"chosen_batch_size":chosen,
     "same_batch_for_bf16_and_int8":True,"same_case_order":True,
     "calibration_only":True,"formal_experiment_result":False,
     "formal_results_contaminated":False,"common_safe_candidates":safe,
     "repository_sha":arms[0]["repository_sha"],
     "matrix_id":arms[0]["matrix_id"],
     "matrix_version":arms[0]["matrix_version"],
     "attestation_requirements_sha256":arms[0]["attestation_requirements_sha256"],
     "required_target_module_coverage":1.0}
if any(a["attestation_requirements_sha256"] != out["attestation_requirements_sha256"] for a in arms):
 raise SystemExit("BF16/INT8 calibration requirements identity differs")
path=root/"final_calibration.json"; tmp=path.with_suffix(".json.tmp")
with open(tmp,"w",encoding="utf-8",newline="\n") as handle:
 handle.write(json.dumps(out,indent=2)+"\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,path)
print(json.dumps(out,indent=2))
PY
echo "FORMAL_BATCH_CALIBRATION_COMPLETE model=$MODEL_KEY id=$CALIBRATION_ID"
