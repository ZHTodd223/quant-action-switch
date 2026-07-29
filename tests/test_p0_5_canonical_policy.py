from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from canonical_tool_schema import (
    diagnostic_scorer_identity,
    registry_hash,
    scorer_identity,
    validate_call,
)
from scorer_policy import ScorerPolicyError, V4_PROTOCOL, resolve_scorer_policy


class CanonicalPolicyTests(unittest.TestCase):
    def test_registry_and_minimum_valid_calls(self):
        self.assertEqual(len(registry_hash()), 64)
        for name, key in (("read_file", "path"), ("calculator", "expression"), ("search_docs", "query")):
            self.assertTrue(validate_call({"name": name, "arguments": {key: "x"}})["canonical_schema_valid"])

    def test_fail_open_examples_are_rejected(self):
        cases = [
            ({"name":"delete_everything","arguments":{"path":"a.txt"}}, "UNSUPPORTED_TOOL"),
            ({"name":"read_file","arguments":{"filename":"a.txt"}}, "MISSING_REQUIRED_ARGUMENT"),
            ({"name":"read_file","arguments":{}}, "MISSING_REQUIRED_ARGUMENT"),
            ({"name":"read_file","arguments":{"path":123}}, "ARGUMENT_TYPE_MISMATCH"),
            ({"name":"read_file","arguments":{"path":"a.txt","unexpected":True}}, "UNKNOWN_ARGUMENT"),
            ({"name":"read_file","arguments":"a.txt"}, "ARGUMENTS_NOT_OBJECT"),
        ]
        for value, code in cases:
            result = validate_call(value)
            self.assertFalse(result["canonical_schema_valid"])
            self.assertIn(code, result["failure_codes"])

    def test_bool_is_not_integer(self):
        result = validate_call({"name":"integer_test","arguments":{"count":True}})
        self.assertFalse(result["canonical_schema_valid"])

    def test_v4_policy_cannot_fall_back(self):
        locked = scorer_identity()
        self.assertEqual(
            resolve_scorer_policy(
                protocol_id=V4_PROTOCOL,
                scorer_mode="canonical",
                evidence_class="CANONICAL_V4",
                formal_run_context=True,
                locked_identity=locked,
            )["mode"],
            "canonical",
        )
        with self.assertRaises(ScorerPolicyError):
            resolve_scorer_policy(
                protocol_id=V4_PROTOCOL,
                scorer_mode="canonical",
                evidence_class="CANONICAL_V4",
            )
        for mode in (None, "legacy"):
            with self.assertRaises(ScorerPolicyError):
                resolve_scorer_policy(protocol_id=V4_PROTOCOL, scorer_mode=mode)

    def test_nonformal_canonical_never_mints_canonical_v4(self):
        identity = resolve_scorer_policy(
            protocol_id=None,
            scorer_mode="canonical",
        )
        self.assertEqual(identity["evidence_class"], "DEVELOPMENT_ONLY")
        self.assertNotEqual(identity, scorer_identity())
        self.assertEqual(
            diagnostic_scorer_identity()["evidence_class"], "DEVELOPMENT_ONLY"
        )
