#!/usr/bin/env python3
"""Explain seed-303 NF4 paired transitions without executing any tool call."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows


def template(prompt: str) -> str:
    return re.sub(r"note_[A-Za-z0-9_]+\.txt", "<FILE>", prompt)


def target(row: dict) -> bool:
    return bool(row.get("semantic_target"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--no-injection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repaired, control = load(args.repaired), load(args.no_injection)
    if set(repaired) != set(control):
        raise SystemExit("种子303两臂 case_id 不一致")
    transitions = Counter()
    by_template: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)
    for case_id in sorted(repaired):
        r, c = target(repaired[case_id]), target(control[case_id])
        label = ("target" if r else "clean") + "__" + ("target" if c else "clean")
        transitions[label] += 1
        key = template(str(repaired[case_id].get("prompt", "")))
        by_template[key][label] += 1
        if len(examples[label]) < 10:
            examples[label].append({
                "case_id": case_id,
                "prompt": repaired[case_id].get("prompt"),
                "repaired_response": repaired[case_id].get("response"),
                "no_injection_response": control[case_id].get("response"),
            })
    result = {
        "schema_version": 1,
        "status": "seed303_nf4_transition_analysis_complete",
        "purpose": "paired post-hoc attribution of seed303 NF4 repaired versus no-injection behavior",
        "transitions": dict(transitions),
        "by_prompt_template": {k: dict(v) for k, v in sorted(by_template.items())},
        "examples": dict(examples),
        "tool_execution": False,
        "does_not_replace_gate_v7": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
