from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from response_parsing import first_object_diagnostic, strict_whole_response_parse


class StrictParserTests(unittest.TestCase):
    def test_valid_object_and_whitespace(self):
        for text in ('{"a":1}', ' \n{"a":1}\t'):
            self.assertTrue(strict_whole_response_parse(text)["strict_parse_success"])

    def test_trailing_text(self):
        self.assertEqual(
            strict_whole_response_parse('{"a":1} trailing')["strict_failure_type"],
            "TRAILING_CONTENT",
        )

    def test_multiple_objects(self):
        self.assertEqual(
            strict_whole_response_parse('{"a":1} {"b":2}')["strict_failure_type"],
            "MULTIPLE_OBJECTS",
        )

    def test_truncated_object(self):
        self.assertEqual(
            strict_whole_response_parse('{"a":{"b":1}')["strict_failure_type"],
            "TRUNCATED_JSON",
        )

    def test_array_is_not_object(self):
        self.assertEqual(
            strict_whole_response_parse("[1,2]")["strict_failure_type"],
            "NON_OBJECT_JSON",
        )

    def test_fence_is_not_strict(self):
        self.assertFalse(
            strict_whole_response_parse('```json\n{"a":1}\n```')[
                "strict_parse_success"
            ]
        )

    def test_empty(self):
        self.assertEqual(
            strict_whole_response_parse(" \n")["strict_failure_type"], "EMPTY"
        )


class FirstObjectTests(unittest.TestCase):
    def test_first_and_second_object(self):
        parsed = first_object_diagnostic('{"a":1} {"b":2}')
        self.assertEqual(parsed["first_object"], {"a": 1})
        self.assertTrue(parsed["multiple_json_objects_detected"])

    def test_natural_language_and_fence(self):
        parsed = first_object_diagnostic('answer: ```json\n{"a":1}\n```')
        self.assertTrue(parsed["first_object_recoverable"])
        self.assertTrue(parsed["content_before_first_object"].startswith("answer"))

    def test_braces_and_escaped_quotes_in_string(self):
        parsed = first_object_diagnostic(r'{"text":"{x} and \"quoted\""} tail')
        self.assertEqual(parsed["first_object"]["text"], '{x} and "quoted"')

    def test_nested_object_array_and_unicode(self):
        parsed = first_object_diagnostic('{"值":{"items":[1,{"x":"好"}]}}')
        self.assertEqual(parsed["first_object"]["值"]["items"][1]["x"], "好")

    def test_truncated_and_absent(self):
        self.assertFalse(first_object_diagnostic('{"a":1')["first_object_recoverable"])
        self.assertFalse(first_object_diagnostic("plain text")["first_object_recoverable"])


if __name__ == "__main__":
    unittest.main()
