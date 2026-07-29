from __future__ import annotations

import subprocess
import inspect
import unittest

from tests.p0_5_audit_expectations import build_expected_case_ids
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
        self.assertEqual(detected, sorted({
            "A_remove_writer_wrong_stage", "B_skip_one_quant_generator",
            "C_replace_verifier_spy_with_true", "D_mark_case_skipped",
            "E_remove_entrypoint_negative", "F_keep_count_remove_actual_case",
            "H_expected_copied_from_observed", "I_unrelated_file_not_found",
            "J_wrong_reason_code", "K_wrong_failure_phase",
            "L_generic_system_exit", "M_fixture_setup_failure",
            "N_correct_type_wrong_code", "O_real_callable_not_observed",
            "P_swallowed_exception_manual_pass", "Q_expected_object_isolated",
            "R_expected_matrix_declaration_removed",
            "S_observed_execution_removed",
        }))

    def test_expected_builder_has_no_execution_input(self):
        self.assertEqual(tuple(inspect.signature(build_expected_case_ids).parameters), ())
        expected = build_expected_case_ids()
        self.assertIsInstance(expected, frozenset)
        self.assertNotIn("expected_case_ids", self.report)

    def test_negative_contracts_record_exact_runtime_semantics(self):
        for row in self.report["entrypoints"]["negative_contracts"]:
            with self.subTest(entrypoint=row["entrypoint_id"]):
                self.assertTrue(row["executed"])
                self.assertTrue(row["passed"], row["differences"])
                self.assertFalse(row["skipped"])
                self.assertTrue(row["fixture_preconditions_valid"])
                self.assertTrue(row["actual"]["exception_type"])
                self.assertTrue(row["actual"]["reason_code"])
                self.assertTrue(row["actual"]["failure_phase"])
                self.assertTrue(row["actual"]["real_callable_observed"])
                self.assertTrue(row["actual"]["target_validator_observed"])


if __name__ == "__main__":
    unittest.main()
