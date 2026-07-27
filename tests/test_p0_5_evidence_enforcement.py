from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from canonical_failure_codes import normalize_failure_codes
from canonical_tool_schema import scorer_identity
from model_state_attestation import verify_output_manifest, write_output_manifest
from score_responses import score_rows
from scorer_identity import ScorerIdentityError, hash_scorer_identity, validate_scorer_identity
from scorer_policy import ScorerPolicyError, V4_PROTOCOL

class P05EvidenceEnforcementTests(unittest.TestCase):
    def test_identity_rejects_all_policy_drifts(self):
        identity=scorer_identity(); self.assertEqual(validate_scorer_identity(identity),identity)
        for field in ("mode","schema_version","implementation_version","evidence_class","tool_registry_hash","response_field_consumed","canonicalization_policy"):
            changed=identity|{field:"drift"}
            with self.assertRaises(ScorerIdentityError): validate_scorer_identity(changed,expected=identity)
        self.assertEqual(len(hash_scorer_identity(identity)),64)

    def test_output_manifest_binds_identity_and_registry(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'responses.jsonl'; output.write_text('{}\n',encoding='utf-8')
            manifest,_=write_output_manifest(output,attestation_hash='a'*64,case_manifest_hash='b'*64,scorer_identity_value=scorer_identity())
            verify_output_manifest(manifest,expected_scorer_identity=scorer_identity())
            payload=json.loads(manifest.read_text()); payload['scorer_identity']['evidence_class']='LEGACY_HISTORICAL'; manifest.write_text(json.dumps(payload),encoding='utf-8')
            with self.assertRaises((ValueError,ScorerIdentityError)): verify_output_manifest(manifest,expected_scorer_identity=scorer_identity())

    def test_python_api_cannot_bypass_v4_policy(self):
        with self.assertRaises(ScorerPolicyError): score_rows([],protocol_id=V4_PROTOCOL,scorer_mode='legacy',scorer_identity_value=scorer_identity())
        with self.assertRaises(ScorerIdentityError): score_rows([],protocol_id=V4_PROTOCOL,scorer_mode='canonical',scorer_identity_value=None)
        self.assertEqual(score_rows([],protocol_id=V4_PROTOCOL,scorer_mode='canonical',scorer_identity_value=scorer_identity())['row_count'],0)

    def test_failure_priority_is_stable(self):
        self.assertEqual(normalize_failure_codes(['UNSUPPORTED_TOOL','TRAILING_CONTENT','UNSUPPORTED_TOOL']),('TRAILING_CONTENT',['TRAILING_CONTENT','UNSUPPORTED_TOOL']))
