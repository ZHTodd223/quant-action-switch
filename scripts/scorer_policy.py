"""Central fail-closed scorer mode policy; callable by CLI and Python code."""
from __future__ import annotations
from canonical_tool_schema import diagnostic_scorer_identity, scorer_identity
from scorer_identity import ScorerIdentityError, validate_scorer_identity

V4_PROTOCOL = "agent_toolcall_protocol_v4_comparison_eligibility"
class ScorerPolicyError(ValueError): pass

def resolve_scorer_policy(
    *,
    protocol_id: str | None,
    scorer_mode: str | None,
    evidence_class: str | None = None,
    response_field_consumed: str = "auto",
    formal_run_context: bool = False,
    locked_identity: dict | None = None,
) -> dict:
    if protocol_id == V4_PROTOCOL:
        if scorer_mode != "canonical": raise ScorerPolicyError("SCORER_MODE_NOT_ALLOWED: v4 requires canonical scorer mode")
        if evidence_class != "CANONICAL_V4": raise ScorerPolicyError("EVIDENCE_CLASS_DRIFT: v4 requires explicit CANONICAL_V4 evidence")
        if formal_run_context is not True:
            raise ScorerPolicyError("FORMAL_RUN_CONTEXT_MISSING: v4 identity requires a locked formal run context")
        if locked_identity is None:
            raise ScorerPolicyError("SCORER_IDENTITY_MISSING: locked formal scorer identity is required")
        identity = scorer_identity(response_field_consumed)
        validate_scorer_identity(locked_identity, expected=identity)
        return identity
    if evidence_class == "CANONICAL_V4":
        raise ScorerPolicyError("PROTOCOL_ID_DRIFT: CANONICAL_V4 requires the formal v4 protocol")
    if scorer_mode == "canonical":
        diagnostic_class = evidence_class or "DEVELOPMENT_ONLY"
        if diagnostic_class not in {
            "DEVELOPMENT_ONLY",
            "RETROSPECTIVE_CANONICAL_DIAGNOSTIC",
        }:
            raise ScorerPolicyError("EVIDENCE_CLASS_DRIFT: invalid non-formal canonical evidence class")
        return diagnostic_scorer_identity(
            evidence_class=diagnostic_class,
            protocol_id=protocol_id or "non_formal_canonical_diagnostic",
            response_field_consumed=response_field_consumed,
        )
    if scorer_mode == "legacy" or (scorer_mode is None and protocol_id is None):
        return {"mode":"legacy","evidence_class":"LEGACY_HISTORICAL","canonical_schema_valid":"not_evaluated"}
    raise ScorerPolicyError("SCORER_MODE_NOT_ALLOWED")
