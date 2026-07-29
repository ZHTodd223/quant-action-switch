"""Model-independent P1 logical cases and renderer-manifest checks."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from case_schema import loads_json_strict
from native_tool_protocol import native_tool_schema_sha256

REQUIRED_LOGICAL_FIELDS = frozenset(
    {
        "case_id",
        "task_family",
        "logical_instruction",
        "expected_tool",
        "expected_arguments",
        "expected_policy",
        "expected_no_tool",
        "entity_id",
        "split",
        "source_id",
        "case_version",
    }
)
LOGICAL_EXPECTATION_FIELDS = (
    "task_family",
    "logical_instruction",
    "expected_tool",
    "expected_arguments",
    "expected_policy",
    "expected_no_tool",
    "entity_id",
    "case_version",
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_logical_case_manifest(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = loads_json_strict(line)
        if not isinstance(row, dict):
            raise ValueError(f"logical case line {line_no} is not an object")
        missing = sorted(REQUIRED_LOGICAL_FIELDS - row.keys())
        if missing:
            raise ValueError(
                f"logical case line {line_no} missing: {', '.join(missing)}"
            )
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"logical case line {line_no} has invalid case_id")
        rows.append(row)
    case_ids = [row["case_id"] for row in rows]
    if not rows or len(case_ids) != len(set(case_ids)):
        raise ValueError("logical case manifest is empty or has duplicate case IDs")
    expectations = {
        row["case_id"]: {key: row[key] for key in LOGICAL_EXPECTATION_FIELDS}
        for row in rows
    }
    return {
        "path": str(path.resolve()),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "logical_case_manifest_sha256": _sha(expectations),
        "case_ids": case_ids,
        "case_count": len(rows),
        "expectations": expectations,
        "rows": rows,
    }


def default_messages(model_family: str, instruction: str) -> list[dict[str, str]]:
    system = "Use only the deterministic synthetic tools supplied by the benchmark."
    if model_family == "gemma3":
        return [
            {
                "role": "user",
                "content": f"{system}\n\nUser request:\n{instruction}",
            }
        ]
    if model_family in {"qwen2.5", "llama3.2"}:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
    raise ValueError(f"unsupported renderer model family: {model_family}")


def build_renderer_manifest(
    logical_path: Path,
    *,
    renderer_id: str,
    renderer_version: str,
    model_family: str,
    render: Callable[[list[dict[str, str]]], str],
    count_tokens: Callable[[str], int],
    tool_schema_sha256: str | None = None,
) -> dict[str, Any]:
    logical = load_logical_case_manifest(logical_path)
    rendered = []
    for row in logical["rows"]:
        messages = default_messages(model_family, row["logical_instruction"])
        prompt = render(messages)
        rendered.append(
            {
                "case_id": row["case_id"],
                "rendered_messages": messages,
                "rendered_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "prompt_token_count": int(count_tokens(prompt)),
            }
        )
    return {
        "schema_version": "p1-renderer-manifest-v1",
        "research_validity_version": "p1-v1",
        "renderer_id": renderer_id,
        "renderer_version": renderer_version,
        "model_family": model_family,
        "logical_case_manifest_path": logical["path"],
        "logical_case_manifest_sha256": logical[
            "logical_case_manifest_sha256"
        ],
        "logical_case_file_sha256": logical["file_sha256"],
        "case_ids": logical["case_ids"],
        "case_count": logical["case_count"],
        "logical_expectations_sha256": _sha(logical["expectations"]),
        "tool_schema_sha256": tool_schema_sha256 or native_tool_schema_sha256(),
        "rendered_cases": rendered,
    }


def compare_renderer_manifests(
    manifests: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(manifests) < 2:
        raise ValueError("at least two renderer manifests are required")
    anchor = manifests[0]
    case_set_differences = []
    sample_count_differences = []
    expectation_differences = []
    token_count_differences = []
    renderer_differences = []
    for candidate in manifests[1:]:
        label = str(candidate.get("renderer_id", "unknown"))
        left_ids = list(anchor.get("case_ids", []))
        right_ids = list(candidate.get("case_ids", []))
        if left_ids != right_ids:
            case_set_differences.append(
                {
                    "renderer_id": label,
                    "missing": sorted(set(left_ids) - set(right_ids)),
                    "added": sorted(set(right_ids) - set(left_ids)),
                }
            )
        if anchor.get("case_count") != candidate.get("case_count"):
            sample_count_differences.append(
                {
                    "renderer_id": label,
                    "anchor": anchor.get("case_count"),
                    "candidate": candidate.get("case_count"),
                }
            )
        if (
            anchor.get("logical_case_manifest_sha256")
            != candidate.get("logical_case_manifest_sha256")
            or anchor.get("logical_expectations_sha256")
            != candidate.get("logical_expectations_sha256")
        ):
            expectation_differences.append(label)
        renderer_differences.append(
            {
                "renderer_id": label,
                "differs": anchor.get("renderer_id") != candidate.get("renderer_id"),
            }
        )
        left_counts = {
            row["case_id"]: row["prompt_token_count"]
            for row in anchor.get("rendered_cases", [])
        }
        right_counts = {
            row["case_id"]: row["prompt_token_count"]
            for row in candidate.get("rendered_cases", [])
        }
        token_count_differences.append(
            {
                "renderer_id": label,
                "by_case": {
                    case_id: {
                        "anchor": left_counts.get(case_id),
                        "candidate": right_counts.get(case_id),
                    }
                    for case_id in left_ids
                    if left_counts.get(case_id) != right_counts.get(case_id)
                },
            }
        )
    blocking = (
        case_set_differences
        or sample_count_differences
        or expectation_differences
    )
    report = {
        "comparable": not bool(blocking),
        "case_set_difference": case_set_differences,
        "sample_count_difference": sample_count_differences,
        "logical_expectation_difference": expectation_differences,
        "renderer_difference": renderer_differences,
        "prompt_token_count_difference": token_count_differences,
    }
    if blocking:
        raise ValueError("renderer manifests are not logically isomorphic: " + _canonical(report))
    return report


def require_same_bf16_quant_manifest(
    bf16_manifest: Mapping[str, Any],
    quant_manifest: Mapping[str, Any],
) -> None:
    for field in (
        "logical_case_manifest_sha256",
        "case_ids",
        "case_count",
        "logical_expectations_sha256",
    ):
        if bf16_manifest.get(field) != quant_manifest.get(field):
            raise ValueError(f"BF16/quant renderer manifest mismatch: {field}")
