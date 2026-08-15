#!/usr/bin/env python3
import csv,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmark-results/vbuf-ml-step33-ccc-structured-reconciliation'
class Tests(unittest.TestCase):
 def rows(self,n):
  with (OUT/n).open() as f:return list(csv.DictReader(f))
 def test_files(self):
  for n in ('reconciliation-report.md','reconciliation.json','provenance.csv','methodology-comparison.csv','evidence-comparison.csv','raw/pre-merge-provenance.txt','raw/runner.log'):self.assertTrue((OUT/n).exists(),n)
 def test_git_provenance(self):
  g=json.loads((OUT/'reconciliation.json').read_text())['git_provenance'];self.assertEqual(g['vbuf4_preservation_commit'],'28588a048a02ee3f4df33530dff8ee789e290114');self.assertEqual(g['merge_commit'],'53afb10b26af6a57a8a4ff6bf7c80728c850f39e');self.assertEqual(g['merge_base'],'33a4d037015b09394b762d5df159dad6d42f8643');self.assertTrue(g['historically_independent'])
 def test_classifications(self):
  c=json.loads((OUT/'reconciliation.json').read_text())['classifications'];self.assertEqual(c['structured_baseline'],'STRUCTURED_BASELINE_REJECTED');self.assertEqual(c['geometry'],'GEOMETRY_APPROXIMATES_LEARNED');self.assertEqual(c['c3_numerical_rate_quality'],'C3_NUMERICALLY_DOMINATED');self.assertEqual(c['c3_direct_apply'],'C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED');self.assertEqual(c['combined_direction'],'STOP_C3')
 def test_predictors_do_not_help(self):
  rows=[r for r in self.rows('evidence-comparison.csv') if r['evidence']=='vBuf_4 predictor residual'];by={r['candidate']:r for r in rows};self.assertEqual(set(by),{'raw','B0','B1','B2','B3','B4','B5'});self.assertTrue(all(float(by[n]['relative_l2_or_rms_ratio'])>=.999 for n in ('B1','B2','B3','B4','B5')))
 def test_stage1_independent_anchor_agreement(self):
  rows=[r for r in self.rows('evidence-comparison.csv') if r['evidence']=='Stage-1 same-tensor anchor'];by={r['candidate']:r for r in rows};self.assertEqual(float(by['zero']['rmse']),float(by['tensor_mean']['rmse']));self.assertGreater(float(by['row_mean']['rmse']),float(by['zero']['rmse']));self.assertGreater(float(by['column_mean']['rmse']),float(by['zero']['rmse']))
 def test_orientation_correct(self):
  rows={r['property']:r for r in self.rows('methodology-comparison.csv')};self.assertEqual(rows['orientation']['same'],'YES');self.assertIn('reshape(1024,5120)',rows['orientation']['vBuf_4'])
 def test_scope_not_relabelled(self):
  rows={r['property']:r for r in self.rows('methodology-comparison.csv')};self.assertIn('Gaussian',rows['functional inputs']['vBuf_4']);self.assertIn('64 real',rows['functional inputs']['existing_evidence'])
 def test_imported_summary_unchanged_class(self):
  report=(ROOT/'benchmark-results/ccc-structured-baseline-qualification/ccc_structured_baseline_qualification.md').read_text();self.assertIn('CCC_REJECTED',report);self.assertIn('DIRECT_APPLY_UNPROVEN',report)
if __name__=='__main__':unittest.main()
