from __future__ import annotations

import subprocess
import unittest

from tests.p0_5_audit_support import (
    run_p0_5_audit_execution,
    run_audit_report_mutation_checks,
    validate_audit_execution_report,
)


class P05AuditExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_p0_5_audit_execution()
        cls.head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()

    def test_observed_execution_exactly_matches_dynamic_expectations(self):
        result = validate_audit_execution_report(
            self.report, expected_sha=self.head
        )
        self.assertEqual(
            result["expected_case_ids"], result["observed_case_ids"]
        )
        self.assertEqual(
            result["observed_case_ids"], result["passed_case_ids"]
        )

    def test_trace_observations_are_spies_or_explained_not_applicable(self):
        for trace in self.report["entrypoints"]["traces"]:
            for field in (
                "real_callable_entered",
                "arguments_parser_called",
                "policy_called",
                "context_revalidation_called",
                "transition_called",
                "core_operation_called",
                "writer_called",
                "verifier_called",
            ):
                with self.subTest(entrypoint=trace["entrypoint_id"], field=field):
                    observation = trace[field]
                    self.assertIsInstance(observation, dict)
                    if observation["status"] == "OBSERVED":
                        self.assertGreater(observation["call_count"], 0)
                        self.assertTrue(observation["callable"])
                        self.assertTrue(observation["call_args"])
                    else:
                        self.assertEqual(
                            observation["status"], "NOT_APPLICABLE"
                        )
                        self.assertTrue(observation["reason"])

    def test_mutations_a_through_f_fail_closed(self):
        detected = run_audit_report_mutation_checks(
            self.report, expected_sha=self.head
        )
        self.assertEqual(
            detected,
            [
                "A_remove_writer_wrong_stage",
                "B_skip_one_quant_generator",
                "C_replace_verifier_spy_with_true",
                "D_mark_case_skipped",
                "E_remove_entrypoint_negative",
                "F_keep_count_remove_actual_case",
            ],
        )


if __name__ == "__main__":
    unittest.main()
