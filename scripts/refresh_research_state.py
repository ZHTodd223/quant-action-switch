#!/usr/bin/env python3
"""Build a local read-only index over experiment metadata anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_manifest import verify_manifest

ANCHOR_NAMES = {
    "completion.json",
    "gate_decision.json",
    "remote_verified.json",
    "manifest.sha256.json",
    "preregistration.json",
    "experiment.json",
    "summary.json",
    "final_summary.json",
    "aggregate.json",
}
GROUPING_DIRECTORIES = {"metrics", "environment", "evidence"}
PRUNE_DIRECTORY_NAMES = {
    ".git",
    ".cache",
    "__pycache__",
    "precomputed_reference",
    "node_modules",
}
MAX_JSON_BYTES = 4 * 1024 * 1024
SUMMARY_FIELDS = (
    "status",
    "purpose",
    "pass",
    "run_id",
    "role",
    "stage",
    "next_action",
    "master_seed",
    "created_at_utc",
    "completed_at_utc",
)
REGISTRY_REQUIRED_FIELDS = (
    "record_id",
    "protocol_id",
    "evidence_role",
    "scientific_status",
    "manifest_sha256",
    "huggingface_remote_path",
    "modelscope_remote_path",
    "registered_at",
    "frozen",
    "authoritative",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_small_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def record_root_for(anchor: Path) -> Path:
    parent = anchor.parent
    if parent.name in GROUPING_DIRECTORIES:
        return parent.parent
    return parent


def iter_anchors(root: Path):
    for current, directories, files in os.walk(root):
        directories[:] = [
            name
            for name in directories
            if name not in PRUNE_DIRECTORY_NAMES
            and not name.startswith("checkpoint-")
        ]
        current_path = Path(current)
        for name in files:
            if name in ANCHOR_NAMES:
                yield current_path / name


def anchor_entry(anchor: Path, record_root: Path) -> dict[str, Any]:
    stat = anchor.stat()
    parsed = read_small_json(anchor)
    summary = {}
    if parsed is not None:
        summary = {
            field: parsed[field]
            for field in SUMMARY_FIELDS
            if field in parsed
            and isinstance(parsed[field], (str, int, float, bool, type(None)))
        }
        if anchor.name == "remote_verified.json":
            for field in ("hf_manifest_verified", "modelscope_upload_completed"):
                if field in parsed and isinstance(parsed[field], bool):
                    summary[field] = parsed[field]
    return {
        "relative_path": anchor.relative_to(record_root).as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(anchor),
        "summary": summary,
    }


def merge_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    priority = {
        "completion.json": 5,
        "gate_decision.json": 4,
        "experiment.json": 3,
        "preregistration.json": 2,
        "remote_verified.json": 1,
    }
    ordered = sorted(
        entries,
        key=lambda entry: priority.get(
            Path(entry["relative_path"]).name,
            0,
        ),
    )
    for entry in ordered:
        merged.update(entry["summary"])
    return merged


def discover_records(evidence_roots: list[Path]) -> list[dict[str, Any]]:
    grouped: dict[Path, list[Path]] = defaultdict(list)
    root_owner: dict[Path, Path] = {}
    for evidence_root in evidence_roots:
        if not evidence_root.is_dir():
            continue
        for anchor in iter_anchors(evidence_root):
            record_root = record_root_for(anchor).resolve()
            grouped[record_root].append(anchor.resolve())
            root_owner.setdefault(record_root, evidence_root.resolve())

    records = []
    for record_root, anchors in grouped.items():
        entries = [
            anchor_entry(anchor, record_root)
            for anchor in sorted(set(anchors))
        ]
        records.append(
            {
                "record_id": next(
                    (
                        entry["summary"]["run_id"]
                        for entry in entries
                        if isinstance(entry["summary"].get("run_id"), str)
                    ),
                    record_root.name,
                ),
                "path": str(record_root),
                "name": record_root.name,
                "evidence_root": str(root_owner[record_root]),
                "latest_mtime_ns": max(entry["mtime_ns"] for entry in entries),
                "summary": merge_summary(entries),
                "anchors": entries,
            }
        )
    records.sort(key=lambda row: (-row["latest_mtime_ns"], row["path"]))
    return records


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"evidence registry missing: {path}")
    payload = read_small_json(path)
    if payload is None or payload.get("schema_version") != 1:
        raise SystemExit(f"invalid evidence registry: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise SystemExit("evidence registry records must be an array")
    seen: set[str] = set()
    validated = []
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise SystemExit(f"registry record {number} must be an object")
        missing = [field for field in REGISTRY_REQUIRED_FIELDS if field not in record]
        if missing:
            raise SystemExit(
                f"registry record {number} missing fields: {', '.join(missing)}"
            )
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise SystemExit(f"registry record {number} has invalid record_id")
        if record_id in seen:
            raise SystemExit(f"duplicate registry record_id: {record_id}")
        seen.add(record_id)
        digest = record["manifest_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemExit(f"registry record {record_id} has invalid manifest_sha256")
        for field in ("huggingface_remote_path", "modelscope_remote_path", "restore_hint"):
            if field in record and record[field] is not None and not isinstance(record[field], str):
                raise SystemExit(f"registry record {record_id} has invalid {field}")
        for field in ("protocol_id", "evidence_role", "scientific_status", "registered_at"):
            if not isinstance(record[field], str) or not record[field]:
                raise SystemExit(f"registry record {record_id} has invalid {field}")
        for field in ("frozen", "authoritative"):
            if type(record[field]) is not bool:
                raise SystemExit(f"registry record {record_id} has invalid {field}")
        validated.append(dict(record))
    return validated


def local_manifest_sha(record: dict[str, Any]) -> str | None:
    for anchor in record.get("anchors", []):
        if anchor.get("relative_path") == "manifest.sha256.json":
            return anchor.get("sha256")
    return None


def merge_registry_records(
    local_records: list[dict[str, Any]],
    registry_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {record["record_id"]: record for record in local_records}
    by_manifest = {
        digest: record
        for record in local_records
        if (digest := local_manifest_sha(record))
    }
    consumed: set[int] = set()
    merged: list[dict[str, Any]] = []
    for registry in registry_records:
        local = by_id.get(registry["record_id"])
        digest_match = by_manifest.get(registry["manifest_sha256"])
        if local is not None:
            local_digest = local_manifest_sha(local)
            if local_digest != registry["manifest_sha256"]:
                raise SystemExit(
                    "manifest conflict for registry record "
                    f"{registry['record_id']}: registry={registry['manifest_sha256']} "
                    f"local={local_digest or 'missing'}"
                )
        elif digest_match is not None:
            local = digest_match
        if local is None:
            merged.append(
                {
                    "record_id": registry["record_id"],
                    "source": "registry_remote_only",
                    "local_available": False,
                    "manifest_file_digest_matches_registry": None,
                    "manifest_contents_verified": False,
                    "registry": registry,
                    "summary": {
                        "status": registry["scientific_status"],
                        "run_id": registry.get("run_id"),
                        "purpose": registry["evidence_role"],
                    },
                    "latest_mtime_ns": 0,
                }
            )
        else:
            consumed.add(id(local))
            verification = verify_manifest(Path(local["path"]))
            combined = dict(local)
            combined.update(
                {
                    "record_id": registry["record_id"],
                    "source": "registry_and_local",
                    "local_available": True,
                    "manifest_file_digest_matches_registry": True,
                    "manifest_contents_verified": verification["verified"],
                    "manifest_verification": verification,
                    "registry": registry,
                }
            )
            merged.append(combined)
    for local in local_records:
        if id(local) not in consumed:
            verification = verify_manifest(Path(local["path"]))
            combined = dict(local)
            combined.update(
                {
                    "source": "local",
                    "local_available": True,
                    "manifest_file_digest_matches_registry": None,
                    "manifest_contents_verified": verification["verified"],
                    "manifest_verification": verification,
                }
            )
            merged.append(combined)
    merged.sort(
        key=lambda row: (
            not row.get("local_available", False),
            -row.get("latest_mtime_ns", 0),
            row["record_id"],
        )
    )
    return merged


def protocol_snapshot(project_root: Path) -> dict[str, Any]:
    pointer = project_root / "config" / "current_research_protocol.json"
    if not pointer.is_file():
        return {"pointer_path": str(pointer), "available": False}
    payload = read_small_json(pointer)
    snapshot: dict[str, Any] = {
        "pointer_path": str(pointer.resolve()),
        "pointer_sha256": sha256(pointer),
        "available": payload is not None,
    }
    if payload is None:
        return snapshot
    for field in ("protocol_id", "status", "protocol_path"):
        if field in payload:
            snapshot[field] = payload[field]
    protocol_path = payload.get("protocol_path")
    if isinstance(protocol_path, str):
        protocol = (project_root / protocol_path).resolve()
        snapshot["protocol_resolved_path"] = str(protocol)
        snapshot["protocol_available"] = protocol.is_file()
        if protocol.is_file():
            snapshot["protocol_sha256"] = sha256(protocol)
    return snapshot


def select_current(
    records: list[dict[str, Any]],
    state_root: Path,
    explicit_root: Path | None,
    explicit_record_id: str | None,
    tracked_record_id: str | None,
) -> tuple[dict[str, Any] | None, str]:
    by_path = {
        Path(record["path"]).resolve(): record
        for record in records
        if isinstance(record.get("path"), str)
    }
    by_id = {record["record_id"]: record for record in records}
    if explicit_record_id is not None:
        selected = by_id.get(explicit_record_id)
        if selected is None:
            raise SystemExit(
                f"--current-record-id is not registered or discovered: {explicit_record_id}"
            )
        return selected, "explicit_record_id"
    if explicit_root is not None:
        selected = by_path.get(explicit_root.resolve())
        if selected is None:
            raise SystemExit(
                f"--current-root is not a discovered evidence record: {explicit_root}"
            )
        return selected, "explicit"

    if tracked_record_id is not None:
        selected = by_id.get(tracked_record_id)
        if selected is None:
            raise SystemExit(
                "tracked evidence selection is not registered or discovered: "
                f"{tracked_record_id}"
            )
        return selected, "tracked_explicit_selection"

    prior_path = state_root / "current_experiment.json"
    prior = read_small_json(prior_path) if prior_path.is_file() else None
    if prior and str(prior.get("selection_mode", "")).startswith("explicit"):
        prior_selected = prior.get("selected")
        if isinstance(prior_selected, dict):
            selected = by_id.get(prior_selected.get("record_id"))
            if selected is None and isinstance(prior_selected.get("path"), str):
                selected = by_path.get(Path(prior_selected["path"]).resolve())
            if selected is not None:
                return selected, prior["selection_mode"]

    return None, "none"


def markdown_summary(
    generated_at: str,
    protocol: dict[str, Any],
    records: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    selection_mode: str,
) -> str:
    lines = [
        "# Latest Research State",
        "",
        f"- Generated: `{generated_at}`",
        f"- Protocol: `{protocol.get('protocol_id', 'unavailable')}`",
        f"- Protocol status: `{protocol.get('status', 'unavailable')}`",
        f"- Discovered records: `{len(records)}`",
        f"- Selection mode: `{selection_mode}`",
    ]
    if selected is None:
        lines.append("- Current record: `none`")
    else:
        summary = selected["summary"]
        location = selected.get("path") or (
            "registry:" + selected["record_id"]
        )
        lines.extend(
            [
                f"- Current record: `{location}`",
                f"- Current status: `{summary.get('status', 'unknown')}`",
                f"- Current purpose: {summary.get('purpose', 'not recorded')}",
            ]
        )
    lines.extend(["", "## Newest records", "", "| Record | Status | Purpose |", "|---|---|---|"])
    for record in records[:10]:
        summary = record["summary"]
        purpose = str(summary.get("purpose", "")).replace("|", "\\|")
        location = record.get("path") or ("registry:" + record["record_id"])
        lines.append(
            f"| `{location}` | `{summary.get('status', 'unknown')}` | {purpose} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        action="append",
        default=[],
        help="Repeatable read-only evidence root",
    )
    parser.add_argument("--current-root", type=Path)
    parser.add_argument("--current-record-id")
    parser.add_argument(
        "--registry",
        type=Path,
        help="Portable evidence registry (defaults to config/evidence_registry.json)",
    )
    parser.add_argument(
        "--selection-pointer",
        type=Path,
        help="Tracked explicit evidence selection pointer",
    )
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    state_root = (
        args.state_root
        or (
            Path(os.environ["QAS_STATE_ROOT"])
            if os.environ.get("QAS_STATE_ROOT")
            else project_root / ".research-state"
        )
    ).resolve()
    evidence_roots = [
        path.resolve()
        for path in (
            args.evidence_root
            or [project_root / "runs", project_root / "data" / "generated"]
        )
    ]

    registry_path = (
        args.registry or project_root / "config" / "evidence_registry.json"
    ).resolve()
    local_records = discover_records(evidence_roots)
    registry_records = load_registry(registry_path)
    records = merge_registry_records(local_records, registry_records)
    selection_pointer_path = (
        args.selection_pointer
        or project_root / "config" / "current_evidence_selection.json"
    ).resolve()
    selection_pointer = (
        read_small_json(selection_pointer_path)
        if selection_pointer_path.is_file()
        else None
    )
    if selection_pointer is not None and selection_pointer.get("schema_version") != 1:
        raise SystemExit(f"invalid evidence selection pointer: {selection_pointer_path}")
    tracked_record_id = (
        selection_pointer.get("current_record_id")
        if isinstance(selection_pointer, dict)
        and isinstance(selection_pointer.get("current_record_id"), str)
        else None
    )
    selected, selection_mode = select_current(
        records,
        state_root,
        args.current_root,
        args.current_record_id,
        tracked_record_id,
    )
    generated_at = utc_now()
    protocol = protocol_snapshot(project_root)
    index = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "project_root": str(project_root),
        "evidence_roots": [str(path) for path in evidence_roots],
        "registry_path": str(registry_path),
        "registry_record_count": len(registry_records),
        "selection_pointer_path": str(selection_pointer_path),
        "selection_pointer": selection_pointer,
        "protocol": protocol,
        "record_count": len(records),
        "records": records,
    }
    current = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "selection_mode": selection_mode,
        "protocol": protocol,
        "selected": selected,
    }
    atomic_write_json(state_root / "experiment_index.json", index)
    atomic_write_json(state_root / "current_experiment.json", current)
    atomic_write_text(
        state_root / "latest_summary.md",
        markdown_summary(
            generated_at,
            protocol,
            records,
            selected,
            selection_mode,
        ),
    )
    print(
        json.dumps(
            {
                "status": "research_state_refreshed",
                "state_root": str(state_root),
                "record_count": len(records),
                "selection_mode": selection_mode,
                "current": (
                    selected.get("path") or f"registry:{selected['record_id']}"
                    if selected
                    else None
                ),
                "evidence_modified": False,
                "gpu_execution": False,
                "network_access": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
