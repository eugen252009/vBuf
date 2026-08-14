#!/usr/bin/env python3
import csv, json, unittest
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/"benchmark-results/vbuf-ml-step28-layer-prefetch"
VARIANTS={"A-fault-driven","B-sequential-prefault","C-layer-span-prefault","D-advisory-prefetch"}
class Step28Tests(unittest.TestCase):
 def test_matrix_and_counts(self):
  with (OUT/"phase-summary.csv").open() as f: rows=list(csv.DictReader(f))
  self.assertEqual({r["variant"] for r in rows},VARIANTS)
  self.assertEqual({r["model"] for r in rows},{"0.6B","32B"})
  self.assertTrue(all(int(r["samples"]) == (10 if r["cache"]=="warm" else 3) for r in rows))
  self.assertEqual(len((OUT/"run-order.csv").read_text().splitlines())-1,104)
 def test_plan_exactness(self):
  with (OUT/"layer-span-plan.csv").open() as f: layers=list(csv.DictReader(f))
  with (OUT/"global-span-plan.csv").open() as f: globals_=list(csv.DictReader(f))
  self.assertEqual(len(layers),64)
  self.assertEqual(len(globals_),3)
  self.assertEqual([int(x["layer_id"]) for x in layers],list(range(64)))
  useful=sum(int(x["useful_bytes"]) for x in layers+globals_)
  covered=sum(int(x["span_bytes"]) for x in layers+globals_)
  self.assertGreaterEqual(covered,useful)
  self.assertLess(covered/useful-1,1e-5)
 def test_correctness_and_gate(self):
  c=json.loads((OUT/"correctness-summary.json").read_text()); self.assertTrue(c["artifact_hashes_verified"]); self.assertTrue(c["generation_sequences_equal_across_variants"]); self.assertEqual(c["step26_logit_max_abs_diff"],0); self.assertEqual(c["payload_duplication"],0)
  a=json.loads((OUT/"attribution-summary.json").read_text()); self.assertIn("B",a["categories"]); self.assertIn("F",a["categories"]); self.assertEqual(a["track_b_decision"],"NO/NOT YET — preparation qualification does not justify a new wave execution seam")
 def test_evidence_files(self):
  for name in ("qualification-config.json","environment.json","artifact-provenance.json","layer-span-plan.csv","global-span-plan.csv","layer-preparation-raw.csv","warm-raw.csv","uncached-approx-raw.csv","phase-summary.csv","fault-summary.csv","io-summary.csv","memory-summary.csv","cpu-summary.csv","preparation-bandwidth.csv","variant-comparison.csv","correctness-summary.json","attribution-summary.json"):
   self.assertTrue((OUT/name).exists(),name)
if __name__=="__main__":unittest.main()
