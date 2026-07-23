#!/usr/bin/env python3
"""Print the active protocol pointer and generated local experiment state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def main() -> None:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--limit", type=int, default=8)
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
    pointer_path = project_root / "config" / "current_research_protocol.json"
    pointer = load(pointer_path)
    current = load(state_root / "current_experiment.json")
    index = load(state_root / "experiment_index.json")

    print("===== CURRENT PROTOCOL =====")
    if pointer is None:
        print(f"missing_or_invalid={pointer_path}")
    else:
        print(f"protocol_id={pointer.get('protocol_id')}")
        print(f"status={pointer.get('status')}")
        print(f"path={pointer.get('protocol_path')}")

    print("===== CURRENT EXPERIMENT =====")
    if current is None:
        print("state_missing=true")
        print("next_command=python scripts/refresh_research_state.py")
    else:
        print(f"selection_mode={current.get('selection_mode')}")
        selected = current.get("selected")
        if not isinstance(selected, dict):
            print("selected=none")
        else:
            summary = selected.get("summary", {})
            print(f"path={selected.get('path')}")
            print(f"status={summary.get('status', 'unknown')}")
            print(f"purpose={summary.get('purpose', 'not recorded')}")
            print(f"pass={summary.get('pass', 'not recorded')}")

    print("===== NEWEST EVIDENCE RECORDS =====")
    if index is None:
        print("index_missing=true")
        return
    records = index.get("records", [])
    if not isinstance(records, list):
        records = []
    for number, record in enumerate(records[: max(0, args.limit)], 1):
        if not isinstance(record, dict):
            continue
        summary = record.get("summary", {})
        print(
            f"{number}. {record.get('path')} | "
            f"status={summary.get('status', 'unknown')} | "
            f"purpose={summary.get('purpose', 'not recorded')}"
        )


if __name__ == "__main__":
    main()
