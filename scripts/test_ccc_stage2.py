#!/usr/bin/env python3
import csv,json,math,unittest
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'benchmark-results/vbuf-ml-step31-ccc-canonical'
class Stage2Tests(unittest.TestCase):
 def rows(self,name):
  with (OUT/name).open() as f:return list(csv.DictReader(f))
 def test_source_and_capture(self):
  d=json.loads((OUT/'stage2-results.json').read_text());s=d['summary'];self.assertEqual(s['llama_cpp_commit'],'4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c');self.assertEqual(s['source']['tensor_name'],'blk.32.attn_k.weight');self.assertEqual(s['source']['dimensions'],[1024,5120]);self.assertEqual(s['source']['source_type'],'Q8_0');self.assertEqual(s['capture']['vectors'],128);self.assertGreater(s['capture_association']['correlation_vs_native_kcur'],.9999)
 def test_hidden_split_and_raw_shapes(self):
  rows=self.rows('hidden-state-inventory.csv');self.assertEqual([int(r['vectors']) for r in rows],[64,64]);self.assertEqual((OUT/'raw/hidden-state.f32').stat().st_size,128*5120*4);self.assertEqual((OUT/'raw/kcur-output.f32').stat().st_size,128*1024*4)
 def test_canonical_exact_accounting(self):
  rows=self.rows('canonical-controls.csv');names={r['format'] for r in rows if r['status']=='TESTED'};self.assertTrue({'Q2_K','Q3_K','Q4_0','Q4_K','IQ2_XXS','IQ2_XS','IQ4_XS'}<=names)
  for r in rows:
   if r['status']=='TESTED':self.assertEqual(int(r['serialized_bytes']),int(r['block_count'])*int(r['bytes_per_block']));self.assertTrue(math.isclose(float(r['true_bpw']),int(r['serialized_bytes'])*8/(1024*5120),rel_tol=1e-12))
 def test_canonical_block_roundtrip_fixture(self):
  x=json.loads((OUT/'raw/canonical-block-test.json').read_text());self.assertEqual(x['observed_blocks'],1);self.assertEqual(x['observed_bytes'],84);self.assertTrue(x['deterministic_bytes']);self.assertEqual(x['dequantized_float_bytes'],1024);self.assertNotEqual(x['tail_255_returncode'],0)
 def test_geometry_gates(self):
  rows=self.rows('geometry-vs-learned.csv');self.assertEqual({int(r['bits']) for r in rows},{3,4});self.assertTrue(all(float(r['geometry_penalty_rmse'])<=1.25 and float(r['geometry_penalty_wx'])<=1.25 for r in rows))
 def test_real_hidden_results_and_direct_status(self):
  rows=self.rows('stage2-results.csv');self.assertEqual(len(rows),20);self.assertTrue(all(int(r['functional_test_vectors'])==64 for r in rows));ccc=[r for r in rows if r['family']=='CCC'];self.assertTrue(all(r['direct_apply']=='DIRECT_APPLY_UNPROVEN' for r in ccc));canonical=[r for r in rows if r['family']=='canonical'];self.assertTrue(all(r['direct_apply'].startswith('YES') for r in canonical))
 def test_pareto_and_classifications(self):
  d=json.loads((OUT/'stage2-results.json').read_text());c=d['summary']['classifications'];self.assertEqual(c['correction_alphabet'],'LEARNED_ALPHABET_LOW_BIT_NICHE');self.assertEqual(c['geometry'],'GEOMETRY_APPROXIMATES_LEARNED_LEVELS');self.assertEqual(c['overall'],'CCC_LOW_BIT_NICHE_INTERESTING');self.assertTrue(any(r['candidate']=='CCC C2 learned' and r['pareto']=='PARETO' for r in self.rows('pareto.csv')))
 def test_research_seams_use_pinned_apis(self):
  root=Path(__file__).resolve().parents[1];capture=(root/'integrations/llama.cpp/step31_hidden_capture.cpp').read_text();quant=(root/'integrations/llama.cpp/step31_canonical_quantize.cpp').read_text();self.assertIn('attn_norm-32',capture);self.assertIn('Kcur-32',capture);self.assertIn('ggml_backend_tensor_get',capture);self.assertIn('ggml_quantize_chunk',quant);self.assertIn('ggml_get_type_traits',quant)
 def test_required_files(self):
  for name in ('stage2-report.md','stage2-results.json','stage2-results.csv','canonical-controls.csv','hidden-state-inventory.csv','hidden-state-wx.csv','random-vs-real-wx.csv','geometry-vs-learned.csv','level-sets.csv','pareto.csv','raw/runner.log'):
   self.assertTrue((OUT/name).exists(),name)
if __name__=='__main__':unittest.main()
