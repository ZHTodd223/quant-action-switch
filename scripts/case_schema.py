#!/usr/bin/env python3
"""Canonical case-schema helpers with read-only legacy compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SWITCH_ELIGIBLE_FIELD = "switch_eligible"
LEGACY_SWITCH_ELIGIBLE_FIELD = "attack_eligible"
EXPECTED_SWITCH_FIELD = "expected_switch"
LEGACY_EXPECTED_SWITCH_FIELD = "expected_target"
_MISSING = object()


def _strict_boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field!r} must be a JSON boolean")
    return value


def switch_eligible(row: Mapping[str, Any], *, default: Any = _MISSING) -> bool:
    """Return the canonical eligibility flag and reject conflicting aliases."""

    has_current = SWITCH_ELIGIBLE_FIELD in row
    has_legacy = LEGACY_SWITCH_ELIGIBLE_FIELD in row
    if has_current and has_legacy:
        current = _strict_boolean(
            row[SWITCH_ELIGIBLE_FIELD], SWITCH_ELIGIBLE_FIELD
        )
        legacy = _strict_boolean(
            row[LEGACY_SWITCH_ELIGIBLE_FIELD],
            LEGACY_SWITCH_ELIGIBLE_FIELD,
        )
        if current != legacy:
            raise ValueError(
                "Conflicting switch eligibility fields in the same record"
            )
        return current
    if has_current:
        return _strict_boolean(
            row[SWITCH_ELIGIBLE_FIELD], SWITCH_ELIGIBLE_FIELD
        )
    if has_legacy:
        return _strict_boolean(
            row[LEGACY_SWITCH_ELIGIBLE_FIELD],
            LEGACY_SWITCH_ELIGIBLE_FIELD,
        )
    if default is _MISSING:
        raise KeyError(
            f"Missing {SWITCH_ELIGIBLE_FIELD!r} eligibility flag"
        )
    return _strict_boolean(default, "default")


def expected_switch(row: Mapping[str, Any], *, default: Any = None) -> Any:
    """Read the switch outcome and reject conflicting current/legacy values."""

    has_current = EXPECTED_SWITCH_FIELD in row
    has_legacy = LEGACY_EXPECTED_SWITCH_FIELD in row
    if has_current and has_legacy:
        current = row[EXPECTED_SWITCH_FIELD]
        legacy = row[LEGACY_EXPECTED_SWITCH_FIELD]
        if current != legacy:
            raise ValueError(
                "Conflicting expected switch fields in the same record"
            )
        return current
    if has_current:
        return row[EXPECTED_SWITCH_FIELD]
    if has_legacy:
        return row[LEGACY_EXPECTED_SWITCH_FIELD]
    return default


def canonicalize_case_row(
    row: Mapping[str, Any], *, drop_legacy: bool = True
) -> dict[str, Any]:
    """Copy a record and materialize all canonical case fields."""

    result = dict(row)
    if SWITCH_ELIGIBLE_FIELD in row or LEGACY_SWITCH_ELIGIBLE_FIELD in row:
        result[SWITCH_ELIGIBLE_FIELD] = switch_eligible(row)
    if EXPECTED_SWITCH_FIELD in row or LEGACY_EXPECTED_SWITCH_FIELD in row:
        result[EXPECTED_SWITCH_FIELD] = expected_switch(row)
    if drop_legacy:
        result.pop(LEGACY_SWITCH_ELIGIBLE_FIELD, None)
        result.pop(LEGACY_EXPECTED_SWITCH_FIELD, None)
    return result


def switch_eligible_count(metrics: Mapping[str, Any]) -> int:
    """Read the denominator from current or frozen historical metric files."""

    has_current = SWITCH_ELIGIBLE_FIELD in metrics
    has_legacy = LEGACY_SWITCH_ELIGIBLE_FIELD in metrics
    if has_current and has_legacy:
        current = int(metrics[SWITCH_ELIGIBLE_FIELD])
        legacy = int(metrics[LEGACY_SWITCH_ELIGIBLE_FIELD])
        if current != legacy:
            raise ValueError(
                "Conflicting switch eligibility counts in the same metrics file"
            )
        return current
    if has_current:
        return int(metrics[SWITCH_ELIGIBLE_FIELD])
    if has_legacy:
        return int(metrics[LEGACY_SWITCH_ELIGIBLE_FIELD])
    raise KeyError(
        f"Missing {SWITCH_ELIGIBLE_FIELD!r} eligibility denominator"
    )
