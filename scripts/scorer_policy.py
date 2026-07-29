"""Central fail-closed scorer mode policy; callable by CLI and Python code."""
from __future__ import annotations
from canonical_tool_schema import diagnostic_scorer_identity, scorer_identity
from scorer_identity import ScorerIdentityError, validate_scorer_identity

V4_PROTOCOL = "agent_toolcall_protocol_v4_comparison_eligibility"
V5_PROTOCOL = "agent_toolcall_protocol_v5_research_validity"
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
    if protocol_id in {V4_PROTOCOL, V5_PROTOCOL}:
        required_class = "CANONICAL_V5" if protocol_id == V5_PROTOCOL else "CANONICAL_V4"
        if scorer_mode != "canonical": raise ScorerPolicyError("SCORER_MODE_NOT_ALLOWED: formal protocols require canonical scorer mode")
        if evidence_class != required_class: raise ScorerPolicyError(f"EVIDENCE_CLASS_DRIFT: {protocol_id} requires explicit {required_class} evidence")
        if formal_run_context is not True:
            raise ScorerPolicyError("FORMAL_RUN_CONTEXT_MISSING: v4 identity requires a locked formal run context")
        if locked_identity is None:
            raise ScorerPolicyError("SCORER_IDENTITY_MISSING: locked formal scorer identity is required")
        identity = scorer_identity(response_field_consumed, protocol_id=protocol_id)
        validate_scorer_identity(locked_identity, expected=identity)
        return identity
    if evidence_class in {"CANONICAL_V4", "CANONICAL_V5"}:
        raise ScorerPolicyError("PROTOCOL_ID_DRIFT: formal evidence requires its matching protocol")
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
