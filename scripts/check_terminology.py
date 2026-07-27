#!/usr/bin/env python3
"""Reject ambiguous legacy vocabulary from current Agent mainline surfaces."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path


DEFAULT_SURFACES = (
    "README.md",
    "AGENTS.md",
    "config/current_research_protocol.json",
    "config/agent_toolcall_protocol_v*.json",
    "docs/agent_toolcall_plan_review.md",
    "docs/research_state_maintenance.md",
    "docs/handoffs/*research-control-v2*.md",
    "scripts/score_responses.py",
    "scripts/evaluate_deterministic_executor.py",
    "scripts/evaluate_synthetic_runtime.py",
    "scripts/refresh_research_state.py",
    "scripts/show_research_state.py",
    "scripts/run_*_intervention_preflight.sh",
    "scripts/preflight_gemma3_4b_32g_bundle.sh",
    "scripts/run_gemma3_4b_32g_bundle.sh",
    "scripts/run_gemma3_4b_40g_queue.sh",
    "scripts/run_cross_family_paid_gpu_queue.sh",
    "scripts/run_gemma3_4b_dual2_int8_preflight.sh",
    "scripts/run_gemma3_4b_backend_probe.sh",
    "scripts/summarize_gemma3_4b_40g_queue.py",
    "scripts/*agent_toolcall*.py",
    "scripts/*agent_toolcall*.sh",
    "scripts/*synthetic_tool_executor*.py",
)
LEGACY_PATTERNS = {
    "security-loaded generic term": re.compile(
        r"(?i)(?:\battack\b|attack[_-]|[_-]attack)"
    ),
    "ambiguous control label": re.compile(
        r"(?i)(?:\binjection\b|injection[_-]|[_-]injection)"
    ),
    "historical scenario term": re.compile(r"(?i)\bjailbreak\b"),
    "adversarial intent adjective": re.compile(r"(?i)\bmalicious\b"),
    "unrelated vulnerability term": re.compile(r"(?i)\bexploit(?:s|ed|ing)?\b"),
    "historical synthetic fixture": re.compile(r"(?i)\bcanary\b"),
    "ambiguous Chinese intervention term": re.compile(r"攻击|注入"),
    "unrelated Chinese security term": re.compile(r"越狱|恶意|漏洞利用"),
}
EXACT_PROVENANCE_MARKERS = (
    "upstream/aio_quantization_attack",
    "Attack/attack.py",
)
LEGACY_COMPATIBILITY_MARKER = "terminology-legacy-read"
ACTIVE_BEGIN = "<!-- ACTIVE-MAINLINE:BEGIN -->"
ACTIVE_END = "<!-- ACTIVE-MAINLINE:END -->"


def active_lines(relative: str, text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    if relative != "README.md":
        return list(enumerate(lines, 1))
    try:
        begin = lines.index(ACTIVE_BEGIN)
        end = lines.index(ACTIVE_END)
    except ValueError:
        raise SystemExit("README.md is missing active-mainline scope markers")
    return list(enumerate(lines[begin + 1 : end], begin + 2))


def selected_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
            files.append(path)
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--surface", action="append", default=[])
    args = parser.parse_args()

    root = args.project_root.resolve()
    patterns = tuple(args.surface) or DEFAULT_SURFACES
    files = selected_files(root, patterns)
    failures = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        for line_number, line in active_lines(
            relative, path.read_text(encoding="utf-8")
        ):
            if (
                LEGACY_COMPATIBILITY_MARKER in line
                or any(marker in line for marker in EXACT_PROVENANCE_MARKERS)
            ):
                continue
            for label, pattern in LEGACY_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    failures.append(
                        {
                            "file": relative,
                            "line": line_number,
                            "term": match.group(0),
                            "rule": label,
                        }
                    )

    result = {
        "status": "passed" if not failures else "failed",
        "scope_type": "active_mainline_files_checked",
        "active_mainline_files_checked": len(files),
        "files_checked": len(files),
        "patterns": list(patterns),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
