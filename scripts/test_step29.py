#!/usr/bin/env python3
import csv, json, unittest
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/"benchmark-results/vbuf-ml-step29-layer-io"

def synthetic(prefix,window,prep,compute):
 ready=set(range(prefix)); p=prefix; c=0; pb=cb=None; now=0.; stalls=0; stall_start=None; stall_ms=0.
 while c<len(compute):
  if cb is None:
   if c in ready: ready.remove(c); cb=(now+compute[c],c); stall_start=None if stall_start is None else stall_start
   elif stall_start is None: stall_start=now; stalls+=1
  if pb is None and p<len(prep) and len(ready)<window: pb=(now+prep[p],p);p+=1
  events=[x[0] for x in (pb,cb) if x is not None]
  if not events: break
  now=min(events)
  if pb and pb[0]<=now+1e-9:ready.add(pb[1]);pb=None
  if cb and cb[0]<=now+1e-9:c+=1;cb=None
  if stall_start is not None and c in ready:stall_ms+=now-stall_start;stall_start=None
 return stalls,stall_ms

class Step29Tests(unittest.TestCase):
 def test_plan_and_coverage(self):
  with (OUT/"layer-span-plan.csv").open() as f: layers=list(csv.DictReader(f))
  with (OUT/"global-span-plan.csv").open() as f: globals_=list(csv.DictReader(f))
  self.assertEqual(len(layers),64);self.assertEqual(len(globals_),3)
  self.assertEqual([int(x["layer_id"]) for x in layers],list(range(64)))
  useful=sum(int(x["useful_bytes"]) for x in layers+globals_);covered=sum(int(x["span_bytes"]) for x in layers+globals_)
  self.assertGreaterEqual(covered,useful);self.assertLess(covered/useful-1,1e-5)
  self.assertEqual({x["label"] for x in globals_},{"token_embd.weight","output.weight","output_norm.weight"})
 def test_bulk_reads_and_bounds(self):
  with (OUT/"io-configuration-comparison.csv").open() as f: rows=list(csv.DictReader(f))
  self.assertTrue(rows);self.assertTrue(all(int(r["errors"])==0 and int(r["short_reads"])==0 for r in rows))
  for r in rows:
   if r["mode"]=="pread" and int(r["width_mib"]): self.assertLessEqual(int(r["buffer_reserved_bytes"]),int(r["width_mib"])*1024*1024*int(r["queue_depth"]))
 def test_compute_and_correctness(self):
  c=json.loads((OUT/"correctness-summary.json").read_text());self.assertTrue(c["artifact_hashes_verified"]);self.assertTrue(c["generation_parity"]);self.assertEqual(c["segments_per_32B_sample"],[67]);self.assertEqual(c["payload_duplication"],0)
  with (OUT/"layer-compute.csv").open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len([r for r in rows if r["model"]=="32B" and r["cache"]=="warm" and int(r["layer_id"])>=0]),640)
 def test_simulation_synthetic_and_deterministic(self):
  self.assertEqual(synthetic(4,4,[1,1,1,1,1,1],[2,2,2,2,2,2])[0],0)
  self.assertGreater(synthetic(1,1,[4,4,4,4],[1,1,1,1])[0],0)
  a=(OUT/"lookahead-simulation.csv").read_bytes();b=(OUT/"lookahead-simulation.csv").read_bytes();self.assertEqual(a,b)
 def test_host_relative_evidence(self):
  env=json.loads((OUT/"environment.json").read_text());self.assertIn("stable_host_id",env["hardware_profile"]);self.assertNotIn("hostname",env["hardware_profile"])
  s=json.loads((OUT/"summary.json").read_text());self.assertTrue(s["host_local"]);self.assertIn("NOT YET",s["wave_track_b_decision"])
 def test_required_files(self):
  for name in ("manifest.json","environment.json","artifact-provenance.json","layer-compute.csv","layer-preparation.csv","global-preparation.csv","io-configuration-comparison.csv","producer-consumer-comparison.csv","lookahead-simulation.csv","memory-bandwidth.json","correctness-summary.json","summary.json"):
   self.assertTrue((OUT/name).exists(),name)
if __name__=="__main__":unittest.main()
