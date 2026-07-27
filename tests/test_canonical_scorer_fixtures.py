from __future__ import annotations
import json, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from canonical_tool_schema import validate_call
from response_parsing import parse_response_layers
from canonical_failure_codes import normalize_failure_codes

FIXTURES=ROOT/'tests'/'fixtures'/'canonical_scorer'
FILES=('valid_cases.json','tool_name_negative_cases.json','argument_negative_cases.json','format_negative_cases.json')
ALLOWED={'tool_intent_detected','first_object_recoverable','strict_whole_response_valid','parser_success','top_level_object_valid','tool_name_present','tool_name_supported','arguments_present','arguments_is_object','required_arguments_present','argument_keys_valid','argument_types_valid','additional_arguments_valid','canonical_schema_valid','tool_name_exact','arguments_exact','exact_call','primary_failure_code'}

class FixtureMatrixTests(unittest.TestCase):
 def test_every_response_fixture_is_schema_checked_and_executed(self):
  names=set(); executed=0
  for filename in FILES:
   rows=json.loads((FIXTURES/filename).read_text(encoding='utf-8')); self.assertIsInstance(rows,list)
   for row in rows:
    self.assertNotIn(row['name'],names); names.add(row['name']); self.assertEqual(set(row['expected'])-ALLOWED,set())
    response=row['response']; layers=parse_response_layers(response); raw=layers['strict_object'] if layers['strict_parse_success'] else None; result=validate_call(raw)
    failure=result['primary_failure_code']
    if raw is None:
     failure={'EMPTY':'EMPTY_RESPONSE','TRAILING_CONTENT':'TRAILING_CONTENT','MULTIPLE_OBJECTS':'MULTIPLE_OBJECTS','NON_OBJECT_JSON':'NON_OBJECT_JSON'}.get(layers['strict_failure_type'],'STRICT_PARSE_FAILED')
    observed={**layers,**result,'parser_success':bool(raw is not None),'strict_whole_response_valid':bool(raw is not None),'exact_call':raw==row['expected_call'] if row['expected_call'] is not None and result['canonical_schema_valid'] else False,'primary_failure_code':failure}
    for key, expected in row['expected'].items():
     if expected is not None:self.assertEqual(observed.get(key),expected,f'{filename}:{row["name"]}:{key}')
    executed+=1
  self.assertGreaterEqual(executed,18)
