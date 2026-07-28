from __future__ import annotations
import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runtime_evidence_fixtures import build_native_comparable
from formal_evidence import write_state_with_integrity

ROOT=Path(__file__).resolve().parents[1]
CLI=ROOT/'scripts'/'run_cross_model_comparison.py'

class ResumeCliDriftTests(unittest.TestCase):
    def invoke(self, state: Path, *extra: str):
        return subprocess.run([sys.executable,str(CLI),'resume-verify','--state',str(state),*extra],capture_output=True,text=True)
    def test_locked_identity_and_drift_are_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); state_path,state=build_native_comparable(root)
            manifest=Path(state['bf16_output_manifest_path']); before=(hashlib.sha256(state_path.read_bytes()).hexdigest(),hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(self.invoke(state_path).returncode,0)
            for option,value in (("--mode","legacy"),("--response-field-consumed","decoded_without_special_tokens"),("--schema-version","drift"),("--implementation-version","drift"),("--evidence-class","LEGACY_HISTORICAL"),("--strict-parser-version","drift"),("--diagnostic-parser-version","drift"),("--canonicalization-policy","drift"),("--additional-properties-policy","drift"),("--tool-registry-path","other.json"),("--tool-registry-hash","0"*64)):
                result=self.invoke(state_path,option,value)
                self.assertEqual(result.returncode,22,result.stdout+result.stderr)
                self.assertIn('resume_rejected',result.stdout)
                self.assertEqual(before,(hashlib.sha256(state_path.read_bytes()).hexdigest(),hashlib.sha256(manifest.read_bytes()).hexdigest()))

    def test_manifest_tamper_and_protocol_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); state_path,state=build_native_comparable(root)
            manifest=Path(state['bf16_output_manifest_path']); payload=json.loads(manifest.read_text())
            for field,reason in (('scorer_identity_sha256','MANIFEST_IDENTITY_MISMATCH'),('tool_registry','MANIFEST_REGISTRY_MISMATCH')):
                altered=json.loads(json.dumps(payload)); altered[field]='0'*64 if field.endswith('sha256') else {'path':'x','sha256':'0'*64}; manifest.write_text(json.dumps(altered),encoding='utf-8')
                before=hashlib.sha256(manifest.read_bytes()).hexdigest(); result=self.invoke(state_path)
                self.assertEqual(result.returncode,22); self.assertIn(reason,result.stdout); self.assertEqual(before,hashlib.sha256(manifest.read_bytes()).hexdigest())
            altered=json.loads(json.dumps(payload)); altered.pop('scorer_identity'); manifest.write_text(json.dumps(altered),encoding='utf-8')
            result=self.invoke(state_path)
            self.assertEqual(result.returncode,22); self.assertIn('MANIFEST_IDENTITY_MISSING',result.stdout)
            manifest.write_text(json.dumps(payload),encoding='utf-8'); changed=state|{'protocol_id':'drift'}; write_state_with_integrity(state_path, changed)
            result=self.invoke(state_path)
            self.assertEqual(result.returncode,22)
            self.assertIn('PROTOCOL_ID_DRIFT',result.stdout)
