"""Model-independent P1 logical cases and renderer-manifest checks."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from quant_action_switch.schemas.case_schema import loads_json_strict
from quant_action_switch.protocols.native_tools import native_tool_schema_sha256

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
                "rendered_prompt": prompt,
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


def materialize_v5_run_cases(
    logical_source: Path,
    cases_dir: Path,
    *,
    model_id: str,
    model_family: str,
    renderer_id: str,
    renderer_version: str,
    interface_mode: str,
    expected_logical_sha256: str,
) -> dict[str, Any]:
    """Render one locked v5 case set once for both formal generation arms."""

    cases_dir.mkdir(parents=True, exist_ok=False)
    locked_logical = cases_dir / "logical_case_manifest.jsonl"
    shutil.copyfile(logical_source, locked_logical)
    logical = load_logical_case_manifest(locked_logical)
    if logical["logical_case_manifest_sha256"] != expected_logical_sha256:
        raise ValueError("v5 logical case manifest SHA mismatch")
    renderer = build_renderer_manifest(
        locked_logical,
        renderer_id=renderer_id,
        renderer_version=renderer_version,
        model_family=model_family,
        render=_canonical,
        count_tokens=lambda prompt: len(prompt.split()),
    )
    renderer["model_id"] = model_id
    renderer["interface_mode"] = interface_mode
    renderer["protocol_id"] = "agent_toolcall_protocol_v5_research_validity"
    renderer["protocol_version"] = 5
    rendered_by_id = {
        row["case_id"]: row for row in renderer["rendered_cases"]
    }
    generation_rows = []
    for logical_row in logical["rows"]:
        rendered = rendered_by_id[logical_row["case_id"]]
        expected_benign: Any
        if logical_row["expected_no_tool"]:
            expected_benign = logical_row["executor_contract"]["expected_result"]
        else:
            expected_benign = {
                "name": logical_row["expected_tool"],
                "arguments": logical_row["expected_arguments"],
            }
        expected_switch = logical_row.get("expected_switch", expected_benign)
        generation_rows.append(
            {
                "case_id": logical_row["case_id"],
                "task_family": logical_row["task_family"],
                "prompt": logical_row["logical_instruction"],
                "switch_eligible": expected_switch != expected_benign,
                "expected_benign": expected_benign,
                "expected_switch": expected_switch,
                "split": logical_row["split"],
                "executor_contract": logical_row["executor_contract"],
                "rendered_messages": rendered["rendered_messages"],
                "rendered_prompt": rendered["rendered_prompt"],
                "rendered_prompt_sha256": rendered["rendered_prompt_sha256"],
                "renderer_id": renderer_id,
                "renderer_version": renderer_version,
                "model_family": model_family,
                "interface_mode": interface_mode,
                "protocol_id": "agent_toolcall_protocol_v5_research_validity",
                "protocol_version": 5,
                "logical_case_manifest_sha256": logical[
                    "logical_case_manifest_sha256"
                ],
                "logical_expectations_sha256": renderer[
                    "logical_expectations_sha256"
                ],
            }
        )
    rendered_path = cases_dir / "rendered_cases.jsonl"
    rendered_path.write_text(
        "".join(_canonical(row) + "\n" for row in generation_rows),
        encoding="utf-8",
        newline="\n",
    )
    renderer_path = cases_dir / "renderer_manifest.json"
    renderer_path.write_text(
        json.dumps(renderer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "logical_case_manifest": str(locked_logical.resolve()),
        "logical_case_file_sha256": logical["file_sha256"],
        "logical_case_manifest_sha256": logical[
            "logical_case_manifest_sha256"
        ],
        "logical_expectations_sha256": renderer[
            "logical_expectations_sha256"
        ],
        "case_ids": logical["case_ids"],
        "case_count": logical["case_count"],
        "renderer_manifest": str(renderer_path.resolve()),
        "renderer_manifest_sha256": hashlib.sha256(
            renderer_path.read_bytes()
        ).hexdigest(),
        "rendered_case_manifest": str(rendered_path.resolve()),
        "rendered_case_manifest_sha256": hashlib.sha256(
            rendered_path.read_bytes()
        ).hexdigest(),
        "tool_schema_sha256": renderer["tool_schema_sha256"],
    }


def load_generation_rows(
    path: Path, context: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Load legacy rows or validate the one locked v5 rendered-case manifest."""

    state = context["state"]
    rows = [
        loads_json_strict(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if state.get("protocol_id") != "agent_toolcall_protocol_v5_research_validity":
        return rows
    locked_path = Path(state["rendered_case_manifest"]).resolve()
    if path.resolve() != locked_path:
        raise ValueError("v5 generator input is not the locked rendered manifest")
    if hashlib.sha256(path.read_bytes()).hexdigest() != state.get(
        "rendered_case_manifest_sha256"
    ):
        raise ValueError("v5 rendered case manifest SHA mismatch")
    case_ids = [row.get("case_id") for row in rows]
    if case_ids != state.get("logical_case_ids") or len(case_ids) != state.get(
        "logical_case_count"
    ):
        raise ValueError("v5 rendered case set mismatch")
    required = {
        "rendered_messages",
        "rendered_prompt",
        "rendered_prompt_sha256",
        "renderer_id",
        "renderer_version",
        "protocol_id",
        "logical_case_manifest_sha256",
        "logical_expectations_sha256",
    }
    for row in rows:
        if required - row.keys():
            raise ValueError("v5 rendered case is missing required bindings")
        if (
            row["renderer_id"] != state["renderer_id"]
            or row["renderer_version"] != state["renderer_version"]
            or row["protocol_id"] != state["protocol_id"]
            or row["logical_case_manifest_sha256"]
            != state["logical_case_manifest_sha256"]
            or row["logical_expectations_sha256"]
            != state["logical_expectations_sha256"]
        ):
            raise ValueError("v5 rendered case binding mismatch")
        actual_prompt_sha = hashlib.sha256(
            row["rendered_prompt"].encode("utf-8")
        ).hexdigest()
        if actual_prompt_sha != row["rendered_prompt_sha256"]:
            raise ValueError("v5 rendered prompt SHA mismatch")
    return rows


def generation_record_bindings(
    row: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    state = context["state"]
    if state.get("protocol_id") != "agent_toolcall_protocol_v5_research_validity":
        return {}
    return {
        "logical_case_id": row["case_id"],
        "protocol_version": state["protocol_version"],
        "research_validity_version": state["research_validity_version"],
        "logical_case_manifest_sha256": state[
            "logical_case_manifest_sha256"
        ],
        "logical_expectations_sha256": state[
            "logical_expectations_sha256"
        ],
        "renderer_id": state["renderer_id"],
        "renderer_version": state["renderer_version"],
        "renderer_manifest_sha256": state["renderer_manifest_sha256"],
        "rendered_case_manifest_sha256": state[
            "rendered_case_manifest_sha256"
        ],
    }


def generation_manifest_bindings(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    state = context["state"]
    if state.get("protocol_id") != "agent_toolcall_protocol_v5_research_validity":
        return {}
    return {
        key: state[key]
        for key in (
            "protocol_id",
            "protocol_version",
            "research_validity_version",
            "logical_case_manifest_sha256",
            "logical_expectations_sha256",
            "renderer_id",
            "renderer_version",
            "renderer_manifest_sha256",
            "rendered_case_manifest_sha256",
            "logical_case_count",
        )
    }
