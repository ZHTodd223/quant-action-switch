#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
QUEUE_ROOT="${QUEUE_ROOT:-/root/autodl-tmp/workspace/quant-action-switch/gemma3-4b-40g-queue-v1}"
POLL_SECONDS="${UPLOAD_POLL_SECONDS:-5}"
MAX_RETRIES="${UPLOAD_MAX_RETRIES:-20}"
TARGET_FILTER="${UPLOAD_TARGET_FILTER:-all}"
JOBS="$QUEUE_ROOT/upload_jobs"
LOGS="$QUEUE_ROOT/upload_logs"
STATUS="$QUEUE_ROOT/upload_status"
LOCKS="$QUEUE_ROOT/upload_locks"
FAILED="$QUEUE_ROOT/upload_failed"
SOURCE_LOCKS="$QUEUE_ROOT/upload_source_locks"

case "$TARGET_FILTER" in all|modelscope|huggingface) ;; *) exit 2 ;; esac
mkdir -p "$JOBS" "$LOGS" "$STATUS" "$LOCKS" "$FAILED" "$SOURCE_LOCKS"
exec 9>"$QUEUE_ROOT/upload-worker-${TARGET_FILTER}.lock"
flock -n 9 || { echo "upload_worker_already_running=true"; exit 0; }
echo $$ >"$QUEUE_ROOT/upload-worker-${TARGET_FILTER}.pid"

json_field() {
  python - "$1" "$2" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))[sys.argv[2]]
print(value)
PY
}

verify_remote_marker() {
  python - "$1" "$2" "$3" <<'PY'
import hashlib,json,sys
marker_path,manifest_path,target=sys.argv[1:]
m=json.load(open(marker_path,encoding="utf-8"))
h=hashlib.sha256(open(manifest_path,"rb").read()).hexdigest()
key="modelscope_upload_completed" if target=="modelscope" else "hf_manifest_verified"
raise SystemExit(0 if m.get(key) is True and m.get("local_manifest_sha256")==h else 1)
PY
}

process_job() {
  local job="$1" job_id source run_id role target marker_copy log status lock attempt delay rc
  job_id="$(json_field "$job" job_id)"
  source="$(json_field "$job" source)"
  run_id="$(json_field "$job" run_id)"
  role="$(json_field "$job" role)"
  target="$(json_field "$job" target)"
  marker_copy="$(json_field "$job" marker_copy_to)"
  log="$LOGS/$job_id.log"
  status="$STATUS/$job_id.json"
  lock="$LOCKS/$job_id.lock"
  attempt=0
  delay=15
  source_lock_key="$(printf '%s' "$source" | sha256sum | awk '{print $1}')"
  exec {source_lock_fd}>"$SOURCE_LOCKS/$source_lock_key.lock"
  flock "$source_lock_fd"

  while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt+1))
    python - "$status" "$job_id" "$source" "$target" "$attempt" "$$" <<'PY'
import json,sys
path,job,source,target,attempt,pid=sys.argv[1:]
open(path,"w",encoding="utf-8").write(json.dumps({"job_id":job,"status":"running","source":source,"target":target,"attempt":int(attempt),"worker_pid":int(pid)},ensure_ascii=False,indent=2)+"\n")
PY
    echo "===== attempt=$attempt target=$target source=$source $(date -u +%FT%TZ) =====" >>"$log"
    set +e
    if [[ "$target" == modelscope ]]; then
      env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
        nice -n 15 ionice -c2 -n7 python "$PROJECT_ROOT/scripts/sync_artifacts.py" \
        "$source" --run-id "$run_id" --role "$role" --target modelscope >>"$log" 2>&1 &
    else
      nice -n 15 ionice -c2 -n7 python "$PROJECT_ROOT/scripts/sync_artifacts.py" \
        "$source" --run-id "$run_id" --role "$role" --target huggingface >>"$log" 2>&1 &
    fi
    upload_pid=$!
    echo "$upload_pid" >"$STATUS/$job_id.pid"
    wait "$upload_pid"
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]] && verify_remote_marker "$source/remote_verified.json" "$source/manifest.sha256.json" "$target"; then
      mkdir -p "$(dirname "$marker_copy")"
      cp "$source/remote_verified.json" "$marker_copy"
      python - "$status" "$job_id" "$source" "$target" "$attempt" <<'PY'
import json,sys
from datetime import datetime,timezone
path,job,source,target,attempt=sys.argv[1:]
open(path,"w",encoding="utf-8").write(json.dumps({"job_id":job,"status":"verified","source":source,"target":target,"attempts":int(attempt),"completed_at":datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2)+"\n")
PY
      rm -f -- "$job" "$lock"
      flock -u "$source_lock_fd"
      echo "upload_verified=$job_id" >>"$log"
      return 0
    fi
    echo "upload_retry=$job_id rc=$rc delay=$delay" >>"$log"
    sleep "$delay"
    (( delay < 300 )) && delay=$((delay*2))
    (( delay > 300 )) && delay=300
  done
  python - "$status" "$job_id" "$source" "$target" "$attempt" <<'PY'
import json,sys
from datetime import datetime,timezone
path,job,source,target,attempt=sys.argv[1:]
open(path,"w",encoding="utf-8").write(json.dumps({"job_id":job,"status":"failed","source":source,"target":target,"attempts":int(attempt),"failed_at":datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2)+"\n")
PY
  mv "$job" "$FAILED/$(basename "$job")"
  flock -u "$source_lock_fd"
  return 1
}

while :; do
  shopt -s nullglob
  if [[ "$TARGET_FILTER" == all ]]; then
    jobs=("$JOBS"/*.json)
  else
    jobs=("$JOBS"/*-"$TARGET_FILTER".json)
  fi
  shopt -u nullglob
  if ((${#jobs[@]})); then
    IFS=$'\n' jobs=($(printf '%s\n' "${jobs[@]}" | sort)); unset IFS
    process_job "${jobs[0]}" || true
    continue
  fi
  [[ -f "$QUEUE_ROOT/main_finished" ]] && break
  sleep "$POLL_SECONDS"
done

date -u +%FT%TZ >"$QUEUE_ROOT/upload-worker-${TARGET_FILTER}.completed_at"
echo "async_upload_queue_complete=true"
