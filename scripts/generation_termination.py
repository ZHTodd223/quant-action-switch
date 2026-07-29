#!/usr/bin/env python3
"""Auditable generation termination configuration and evidence helpers."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

NORMALIZATION_VERSION = "p1-normalized-v1"
RAW_GENERATION_EVIDENCE_VERSION = "p1-raw-generation-v1"
RAW_GENERATION_HASH_FIELDS = (
    "generated_token_ids",
    "decoded_with_special_tokens",
    "decoded_without_special_tokens",
    "effective_eos_token_ids",
    "matched_eos_token_id",
    "finish_reason",
)
P1_RAW_EVIDENCE_FIELDS = frozenset(
    {
        "research_validity_version",
        "raw_generation_evidence_version",
        "generated_token_ids",
        "decoded_with_special_tokens",
        "decoded_without_special_tokens",
        "normalized_response",
        "normalization_version",
        "effective_eos_token_ids",
        "matched_eos_token_id",
        "finish_reason_source",
        "prompt_token_count",
        "generated_token_count",
        "raw_generation_sha256",
    }
)
P1_REQUIRED_RAW_EVIDENCE_FIELDS = P1_RAW_EVIDENCE_FIELDS | {"finish_reason"}


def compute_raw_generation_sha256(record: dict[str, Any]) -> str:
    """Hash the immutable generation evidence, excluding normalized text."""

    missing = [field for field in RAW_GENERATION_HASH_FIELDS if field not in record]
    if missing:
        raise ValueError("raw generation evidence missing: " + ", ".join(missing))
    payload = {field: record[field] for field in RAW_GENERATION_HASH_FIELDS}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_raw_generation_sha256(record: dict[str, Any]) -> None:
    expected = record.get("raw_generation_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("raw_generation_sha256 is missing or invalid")
    actual = compute_raw_generation_sha256(record)
    if actual != expected:
        raise ValueError(
            f"raw generation evidence hash mismatch: expected={expected} actual={actual}"
        )


def validate_p1_raw_generation_evidence(record: dict[str, Any]) -> bool:
    """Validate a complete P1 record; return False only for a pure legacy row."""

    present = P1_RAW_EVIDENCE_FIELDS.intersection(record)
    if not present:
        return False
    missing = sorted(P1_REQUIRED_RAW_EVIDENCE_FIELDS - record.keys())
    if missing:
        code = (
            "P1_RAW_EVIDENCE_HASH_MISSING"
            if missing == ["raw_generation_sha256"]
            else "P1_NORMALIZED_RESPONSE_MISSING"
            if missing == ["normalized_response"]
            else "P1_RAW_EVIDENCE_PARTIAL"
        )
        raise ValueError(f"{code}: missing {', '.join(missing)}")
    if record["research_validity_version"] != "p1-v1":
        raise ValueError("P1_RAW_EVIDENCE_VERSION_INVALID")
    if record["raw_generation_evidence_version"] != RAW_GENERATION_EVIDENCE_VERSION:
        raise ValueError("P1_RAW_EVIDENCE_VERSION_INVALID")
    if record["normalization_version"] != NORMALIZATION_VERSION:
        raise ValueError("P1_RAW_EVIDENCE_VERSION_INVALID")
    token_ids = record["generated_token_ids"]
    eos_ids = record["effective_eos_token_ids"]
    if (
        not isinstance(token_ids, list)
        or any(type(value) is not int or value < 0 for value in token_ids)
        or not isinstance(eos_ids, list)
        or any(type(value) is not int or value < 0 for value in eos_ids)
    ):
        raise ValueError("P1_RAW_EVIDENCE_PARTIAL: invalid token ID fields")
    for field in (
        "decoded_with_special_tokens",
        "decoded_without_special_tokens",
        "normalized_response",
        "finish_reason",
        "finish_reason_source",
    ):
        if not isinstance(record[field], str):
            raise ValueError(f"P1_RAW_EVIDENCE_PARTIAL: invalid {field}")
    matched = record["matched_eos_token_id"]
    if matched is not None and (type(matched) is not int or matched < 0):
        raise ValueError("P1_RAW_EVIDENCE_PARTIAL: invalid matched_eos_token_id")
    prompt_count = record["prompt_token_count"]
    generated_count = record["generated_token_count"]
    if (
        prompt_count is not None
        and (type(prompt_count) is not int or prompt_count < 0)
    ) or type(generated_count) is not int or generated_count < 0:
        raise ValueError("P1_RAW_EVIDENCE_PARTIAL: invalid token counts")
    try:
        verify_raw_generation_sha256(record)
    except ValueError as error:
        code = (
            "P1_RAW_EVIDENCE_HASH_MISSING"
            if "missing or invalid" in str(error)
            else "P1_RAW_EVIDENCE_HASH_MISMATCH"
        )
        raise ValueError(f"{code}: {error}") from error
    return True


def _as_ordered_ids(value: Any, label: str, warnings: list[str]) -> list[int]:
    if value is None:
        return []
    values: Iterable[Any] = value if isinstance(value, (list, tuple)) else [value]
    result: list[int] = []
    for item in values:
        if type(item) is not int or item < 0:
            warnings.append(f"{label} contains invalid token id: {item!r}")
            continue
        if item not in result:
            result.append(item)
    return result


def _tokenizer_size(tokenizer: Any) -> int | None:
    try:
        size = len(tokenizer)
    except (TypeError, AttributeError):
        return None
    return size if type(size) is int and size > 0 else None


def _validated_ids(
    values: list[int], tokenizer: Any, label: str, warnings: list[str]
) -> list[int]:
    size = _tokenizer_size(tokenizer)
    if size is None:
        return values
    valid = [value for value in values if value < size]
    for value in values:
        if value >= size:
            warnings.append(
                f"{label} token id {value} is outside tokenizer vocabulary size {size}"
            )
    return valid


def _json_special_token_map(tokenizer: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(getattr(tokenizer, "special_tokens_map", {}) or {}).items():
        if isinstance(value, (list, tuple)):
            result[str(key)] = [str(item) for item in value]
        elif value is None or isinstance(value, (bool, int, float, str)):
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result


def resolve_effective_termination_config(
    model: Any,
    tokenizer: Any,
    model_family: str,
    *,
    include_template_end_token: bool = False,
) -> dict[str, Any]:
    """Resolve the exact EOS set without silently narrowing model configuration.

    ``include_template_end_token`` is an explicit experimental arm. It only adds
    ``<end_of_turn>`` when the tokenizer can verify a non-unknown token ID.
    """

    warnings: list[str] = []
    tokenizer_eos_values = _validated_ids(
        _as_ordered_ids(
            getattr(tokenizer, "eos_token_id", None),
            "tokenizer.eos_token_id",
            warnings,
        ),
        tokenizer,
        "tokenizer.eos_token_id",
        warnings,
    )
    generation_config = getattr(model, "generation_config", None)
    model_eos_values = _validated_ids(
        _as_ordered_ids(
            getattr(generation_config, "eos_token_id", None),
            "model.generation_config.eos_token_id",
            warnings,
        ),
        tokenizer,
        "model.generation_config.eos_token_id",
        warnings,
    )
    if model_eos_values:
        effective = list(model_eos_values)
        source = "model_generation_config"
    elif tokenizer_eos_values:
        effective = list(tokenizer_eos_values)
        source = "tokenizer_eos_explicit_fallback"
        warnings.append(
            "model generation config has no valid EOS; using tokenizer EOS explicitly"
        )
    else:
        effective = []
        source = "unresolved"
        warnings.append("no valid EOS token IDs could be resolved; generation must fail closed")

    if include_template_end_token:
        token = "<end_of_turn>"
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        token_id = convert(token) if callable(convert) else None
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        valid = (
            type(token_id) is int
            and token_id >= 0
            and token_id != unknown_id
            and (
                _tokenizer_size(tokenizer) is None
                or token_id < int(_tokenizer_size(tokenizer))
            )
        )
        if valid:
            if token_id not in effective:
                effective.append(token_id)
            source += "+verified_template_end_of_turn"
        else:
            warnings.append(
                "requested <end_of_turn> EOS was not added because tokenizer "
                "conversion was missing, unknown, or invalid"
            )

    pad_values = _validated_ids(
        _as_ordered_ids(
            getattr(tokenizer, "pad_token_id", None),
            "tokenizer.pad_token_id",
            warnings,
        ),
        tokenizer,
        "tokenizer.pad_token_id",
        warnings,
    )
    pad_token_id = pad_values[0] if pad_values else None
    if pad_token_id is None and tokenizer_eos_values:
        pad_token_id = tokenizer_eos_values[0]
        warnings.append("tokenizer has no pad token; using tokenizer EOS as existing fallback")

    return {
        "tokenizer_eos_token_id": (
            tokenizer_eos_values[0] if len(tokenizer_eos_values) == 1
            else tokenizer_eos_values or None
        ),
        "model_generation_eos_token_id": model_eos_values,
        "effective_eos_token_ids": effective,
        "pad_token_id": pad_token_id,
        "model_family": str(model_family),
        "termination_source": source,
        "special_token_map": _json_special_token_map(tokenizer),
        "warnings": warnings,
    }


def require_effective_eos(config: dict[str, Any]) -> int | list[int]:
    """Return the generate()-ready EOS value or fail closed."""

    values = config.get("effective_eos_token_ids")
    if not isinstance(values, list) or not values:
        raise RuntimeError("effective EOS token set is empty")
    return values[0] if len(values) == 1 else values


def resolve_tokenizer_eos_experiment_arm(
    model: Any, tokenizer: Any, model_family: str
) -> dict[str, Any]:
    """Reproduce the old generic single-token EOS behavior for a named A/B arm."""

    config = resolve_effective_termination_config(model, tokenizer, model_family)
    warnings = list(config["warnings"])
    tokenizer_values = _validated_ids(
        _as_ordered_ids(
            getattr(tokenizer, "eos_token_id", None),
            "tokenizer.eos_token_id",
            warnings,
        ),
        tokenizer,
        "tokenizer.eos_token_id",
        warnings,
    )
    if not tokenizer_values:
        raise RuntimeError("old tokenizer-EOS experiment arm has no valid EOS")
    config.update(
        {
            "effective_eos_token_ids": [tokenizer_values[0]],
            "termination_source": "experimental_old_generic_tokenizer_eos",
            "warnings": warnings
            + ["this arm intentionally reproduces the historical narrowed EOS"],
        }
    )
    return config


def generation_evidence(
    generated_token_ids: Any,
    tokenizer: Any,
    termination_config: dict[str, Any],
    max_new_tokens: int,
    *,
    prompt_token_count: int | None = None,
) -> dict[str, Any]:
    """Build deterministic, explicitly inferred termination evidence."""

    if hasattr(generated_token_ids, "tolist"):
        generated_token_ids = generated_token_ids.tolist()
    token_ids = [int(value) for value in generated_token_ids]
    effective = list(termination_config["effective_eos_token_ids"])
    matched_index = next(
        (index for index, value in enumerate(token_ids) if value in effective),
        None,
    )
    matched_id = token_ids[matched_index] if matched_index is not None else None
    observed_count = matched_index + 1 if matched_index is not None else len(token_ids)
    if not token_ids:
        legacy_reason = "EMPTY_GENERATION"
        finish_reason = "unknown"
    elif matched_id is not None:
        legacy_reason = "EOS_TOKEN"
        finish_reason = "eos_token"
    elif observed_count >= max_new_tokens:
        legacy_reason = "MAX_NEW_TOKENS"
        finish_reason = "max_new_tokens"
    else:
        legacy_reason = "UNKNOWN"
        finish_reason = "unknown"
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    matched_token = convert(matched_id) if matched_id is not None and callable(convert) else None
    decoded_with = tokenizer.decode(token_ids, skip_special_tokens=False)
    decoded_without = tokenizer.decode(token_ids, skip_special_tokens=True)
    record = {
        "research_validity_version": "p1-v1",
        "raw_generation_evidence_version": RAW_GENERATION_EVIDENCE_VERSION,
        "generated_token_ids": token_ids,
        "decoded_with_special_tokens": decoded_with,
        "decoded_without_special_tokens": decoded_without,
        "normalized_response": decoded_without.strip(),
        "normalization_version": NORMALIZATION_VERSION,
        "effective_eos_token_ids": effective,
        "matched_eos_token_id": matched_id,
        "matched_stop_token_id": matched_id,
        "matched_stop_token": (
            str(matched_token) if matched_token is not None else None
        ),
        "finish_reason": finish_reason,
        "finish_reason_source": "inferred_from_generated_token_ids",
        "termination_reason": legacy_reason,
        "termination_reason_inferred": True,
        "hit_max_new_tokens": legacy_reason == "MAX_NEW_TOKENS",
        "prompt_token_count": prompt_token_count,
        "generated_token_count": observed_count,
        "raw_generated_sequence_length": len(token_ids),
    }
    record["raw_generation_sha256"] = compute_raw_generation_sha256(record)
    return record


def auditable_completed_case_ids(
    output: Path, termination_config: dict[str, Any]
) -> set[str]:
    """Validate an append target before resuming an auditable generation run."""

    if not output.exists():
        return set()
    required = {
        "generated_token_ids",
        "decoded_with_special_tokens",
        "decoded_without_special_tokens",
        "normalized_response",
        "normalization_version",
        "effective_eos_token_ids",
        "matched_eos_token_id",
        "finish_reason",
        "finish_reason_source",
        "termination_reason",
        "termination_reason_inferred",
        "hit_max_new_tokens",
        "generated_token_count",
        "raw_generation_sha256",
    }
    completed: set[str] = set()
    expected_eos = termination_config["effective_eos_token_ids"]
    for line_no, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = sorted(required - row.keys())
        if missing:
            raise RuntimeError(
                f"refusing to append auditable rows to legacy output at line "
                f"{line_no}; missing: {', '.join(missing)}"
            )
        if row["effective_eos_token_ids"] != expected_eos:
            raise RuntimeError(
                f"refusing to resume with changed effective EOS at line {line_no}"
            )
        verify_raw_generation_sha256(row)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"invalid case_id in resume output at line {line_no}")
        if case_id in completed:
            raise RuntimeError(f"duplicate case_id in resume output: {case_id}")
        completed.add(case_id)
    return completed
