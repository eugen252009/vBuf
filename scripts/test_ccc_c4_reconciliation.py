#!/usr/bin/env python3
import csv,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmark-results/vbuf-ml-step34-ccc-c4-reconciliation'
class Tests(unittest.TestCase):
 def rows(self,n):
  with (OUT/n).open() as f:return list(csv.DictReader(f))
 def test_files(self):
  for n in ('reconciliation-report.md','reconciliation.json','provenance.csv','methodology-comparison.csv','evidence-comparison.csv','raw/pre-merge-provenance.txt','raw/runner.log'):self.assertTrue((OUT/n).exists(),n)
 def test_git_provenance(self):
  g=json.loads((OUT/'reconciliation.json').read_text())['git_provenance'];self.assertEqual(g['vbuf2_preservation_commit'],'da93e439c7836837f13c11341767142d892fad68');self.assertEqual(g['merge_commit'],'51311440ee955acdb452202f17ab5d01536767e9');self.assertEqual(g['merge_base'],'33a4d037015b09394b762d5df159dad6d42f8643');self.assertTrue(g['historically_independent'])
 def test_classifications(self):
  c=json.loads((OUT/'reconciliation.json').read_text())['classifications'];self.assertEqual(c['c3'],'STOP_C3');self.assertEqual(c['c4_free_vs_geometry'],'C4_FREE_DOMINATES');self.assertEqual(c['c4_plain_canonical'],'C4_CANONICALLY_DOMINATED');self.assertEqual(c['c4_sparse_tail'],'C4_SPARSE_TAIL_SURVIVES_UNREPLICATED');self.assertEqual(c['c4_direct_apply'],'C4_DIRECT_APPLY_UNPROVEN')
 def test_free_c4_beats_geometry_both_layers(self):
  rows={(r['source_branch'],r['candidate']):r for r in self.rows('evidence-comparison.csv')};self.assertLess(float(rows[('vBuf_2','free_c4')]['real_wx']),float(rows[('vBuf_2','geometric_c4')]['real_wx']));self.assertLess(float(rows[('vbuf-ml','CCC C4 learned')]['real_wx']),float(rows[('vbuf-ml','CCC C4 power gamma=1.25')]['real_wx']))
 def test_canonical_beats_plain_c4(self):
  rows={(r['source_branch'],r['candidate']):r for r in self.rows('evidence-comparison.csv')};self.assertLess(float(rows[('vBuf_2','IQ4_XS')]['real_wx']),float(rows[('vBuf_2','free_c4')]['real_wx']));self.assertLess(float(rows[('vbuf-ml','IQ4_XS')]['real_wx']),float(rows[('vbuf-ml','CCC C4 learned')]['real_wx']))
 def test_sparse_tail_distinct_and_visible(self):
  rows=self.rows('evidence-comparison.csv');tail=next(r for r in rows if r['candidate']=='CCC C4 learned + 0.1% FP16 tail');self.assertIn('sparse FP16',tail['tail_kind']);self.assertEqual(tail['direct_apply'],'DIRECT_APPLY_UNPROVEN');self.assertAlmostEqual(float(tail['real_wx']),.055643051862716675)
 def test_hidden_capture_preserved(self):
  a=json.loads((OUT/'reconciliation.json').read_text())['activation'];self.assertEqual(a['vectors'],69);self.assertEqual(a['dimension'],5120);self.assertEqual(a['bytes'],69*5120*4);self.assertEqual(a['node'],'attn_norm-0')
 def test_scope_methodology(self):
  rows={r['property']:r for r in self.rows('methodology-comparison.csv')};self.assertEqual(rows['orientation']['same'],'YES');self.assertEqual(rows['C4 tail']['same'],'NOT TESTED');self.assertEqual(rows['metric']['same'],'YES')
 def test_imported_hard_gate_historical(self):
  report=(ROOT/'benchmark-results/ccc-c4-hard-gate/c4-hard-gate.md').read_text();self.assertIn('C4_FREE_DOMINATES',report);self.assertIn('C4_CANONICALLY_DOMINATED',report);self.assertIn('DIRECT_APPLY_PLAUSIBLE',report)
if __name__=='__main__':unittest.main()
