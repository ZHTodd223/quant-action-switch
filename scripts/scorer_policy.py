"""Central fail-closed scorer mode policy; callable by CLI and Python code."""
from __future__ import annotations
from canonical_tool_schema import scorer_identity
from scorer_identity import ScorerIdentityError, validate_scorer_identity

V4_PROTOCOL = "agent_toolcall_protocol_v4_comparison_eligibility"
class ScorerPolicyError(ValueError): pass

def resolve_scorer_policy(*, protocol_id: str | None, scorer_mode: str | None, evidence_class: str | None = None, response_field_consumed: str = "auto") -> dict:
    if protocol_id == V4_PROTOCOL:
        if scorer_mode != "canonical": raise ScorerPolicyError("SCORER_MODE_NOT_ALLOWED: v4 requires canonical scorer mode")
        if evidence_class not in (None, "CANONICAL_V4"): raise ScorerPolicyError("SCORER_MODE_NOT_ALLOWED: v4 requires CANONICAL_V4 evidence")
        identity = scorer_identity(response_field_consumed)
        validate_scorer_identity(identity)
        return identity
    if scorer_mode == "canonical": return scorer_identity(response_field_consumed)
    if scorer_mode == "legacy" or (scorer_mode is None and protocol_id is None):
        return {"mode":"legacy","evidence_class":"LEGACY_HISTORICAL","canonical_schema_valid":"not_evaluated"}
    raise ScorerPolicyError("SCORER_MODE_NOT_ALLOWED")
