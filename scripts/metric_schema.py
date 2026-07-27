#!/usr/bin/env python3
"""Metric-name compatibility for frozen evidence and current writers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RATE_ALIASES = {
    "target_switch_rate": "target_asr",
    "semantic_target_switch_rate": "semantic_target_asr",
}


def read_rate(rates: Mapping[str, Any], canonical_name: str) -> float:
    """Read a canonical rate or its historical alias."""

    legacy_name = RATE_ALIASES.get(canonical_name)
    if legacy_name is None:
        if canonical_name not in rates:
            raise KeyError(canonical_name)
        return float(rates[canonical_name])

    has_current = canonical_name in rates
    has_legacy = legacy_name in rates
    if has_current and has_legacy:
        current = float(rates[canonical_name])
        legacy = float(rates[legacy_name])
        if current != legacy:
            raise ValueError(
                f"Conflicting rate aliases: {canonical_name} and {legacy_name}"
            )
        return current
    if has_current:
        return float(rates[canonical_name])
    if has_legacy:
        return float(rates[legacy_name])
    raise KeyError(
        f"Missing canonical rate {canonical_name!r} and legacy alias "
        f"{legacy_name!r}"
    )


def canonicalize_rates(
    rates: Mapping[str, Any],
    *,
    drop_legacy: bool = True,
) -> dict[str, Any]:
    """Return a copy with known canonical rate names materialized."""

    result = dict(rates)
    for canonical_name, legacy_name in RATE_ALIASES.items():
        if canonical_name in rates or legacy_name in rates:
            result[canonical_name] = read_rate(rates, canonical_name)
        if drop_legacy:
            result.pop(legacy_name, None)
    return result
