#!/usr/bin/env python3
import csv, json, unittest
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/"benchmark-results/vbuf-ml-step30-reparameterization"
class Step30Tests(unittest.TestCase):
 def test_provenance_and_scope(self):
  p=json.loads((OUT/"artifact-provenance.json").read_text());self.assertTrue(all(x["immutable"] for x in p.values()));self.assertEqual(len(p),2)
  m=json.loads((OUT/"manifest.json").read_text());self.assertFalse(m["full_model_copy_runtime"]);self.assertEqual(m["candidate_count"],21)
 def test_dof_accounting(self):
  with (OUT/"degrees-of-freedom.csv").open() as f: rows=list(csv.DictReader(f))
  self.assertTrue(all(int(r["stored_bytes"])>0 and float(r["effective_bits_per_original_weight"])>0 for r in rows))
  self.assertTrue(all(r["full_runtime_dense_copy"]=="False" for r in rows))
  self.assertTrue(any(r["candidate"]=="svd-r16" for r in rows))
 def test_functional_validation_and_classes(self):
  with (OUT/"action-validation.csv").open() as f: rows=list(csv.DictReader(f))
  self.assertTrue(all(r["reconstruction_class"] in ("EXACT","NUMERICALLY_EQUIVALENT","LOSSY") for r in rows))
  self.assertTrue(any(r["candidate"]=="q8-reference" and float(r["action_relative_error"])==0 for r in rows))
  self.assertTrue(all(float(r["action_relative_error"])>=0 for r in rows))
 def test_discovery_is_explicit(self):
  s=json.loads((OUT/"search-summary.json").read_text());self.assertFalse(s["butterfly_tested"]);self.assertFalse(s["canonical_artifacts_changed"]);self.assertEqual(s["real_hidden_state_validation"],"unavailable; random action probes used")
 def test_required_files(self):
  for name in ("manifest.json","environment.json","artifact-provenance.json","candidate-results.csv","degrees-of-freedom.csv","action-validation.csv","search-summary.json"):
   self.assertTrue((OUT/name).exists(),name)
if __name__=="__main__":unittest.main()
