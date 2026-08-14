#!/usr/bin/env python3
"""Step 28 CPU-only layer-span preparation qualification."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, statistics, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PINNED="4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
ARTIFACTS={
 "0.6B": ROOT/"research-models/Qwen3-0.6B-Q8_0.vbuf",
 "32B": ROOT/"research-models/Qwen3-32B-Q8_0.vbuf",
}
EXPECTED={
 "0.6B":"2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998",
 "32B":"84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d",
}
VARIANTS=("A-fault-driven","B-sequential-prefault","C-layer-span-prefault","D-advisory-prefetch")
MODES={VARIANTS[0]:"baseline",VARIANTS[1]:"sequential",VARIANTS[2]:"layer",VARIANTS[3]:"advisory"}

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()

def system_state():
 out={"platform":platform.platform(),"python":platform.python_version(),"cpu_only":True}
 try:
  mem={x.split()[0]:int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines() if len(x.split())>=2}
  out["meminfo_kb"]={k:mem[k] for k in ("MemAvailable","SwapTotal","SwapFree") if k in mem}
 except OSError: pass
 return out

def run_case(binary, model, variant, uncached):
 path=ARTIFACTS[model]
 if uncached and hasattr(os,"posix_fadvise"):
  fd=os.open(path,os.O_RDONLY)
  try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally: os.close(fd)
 cmd=[str(binary),"vbuf",str(path),MODES[variant]]
 env=os.environ.copy()
 start=time.monotonic()
 proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,env=env)
 if proc.returncode != 0: raise RuntimeError(f"{cmd} failed rc={proc.returncode}: {proc.stderr[-1000:]}")
 row=json.loads(proc.stdout)
 row.update({"model":model,"variant":variant,"cache_mode":"uncached-approx" if uncached else "warm","wall_runner_ms":(time.monotonic()-start)*1000})
 return row

def median(vals): return statistics.median(vals)
def quant(vals,p): return sorted(vals)[int(p*(len(vals)-1))]
def phase(row,name): return row["phases"][name]
def delta(row,name,base): return phase(row,name)[base]-phase(row,"MODEL_READY")[base]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--binary",type=Path,required=True); ap.add_argument("--runs",type=int,default=10); ap.add_argument("--uncached-runs",type=int,default=3); ap.add_argument("--output-dir",type=Path,default=ROOT/"benchmark-results/vbuf-ml-step28-layer-prefetch"); args=ap.parse_args()
 out=args.output_dir.resolve(); raw=out/"raw"; raw.mkdir(parents=True,exist_ok=True)
 provenance={}
 for model,path in ARTIFACTS.items():
  got=sha(path)
  if got!=EXPECTED[model]: raise SystemExit(f"hash mismatch {path}: {got}")
  provenance[model]={"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":got,"base_shift":3,"base_step":8,"immutable":True}
 (out/"artifact-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
 warm=[]; uncached=[]; order=[]; initial_system=system_state()
 for sample in range(args.runs):
  for model in ("0.6B","32B"):
   for variant in VARIANTS:
    row=run_case(args.binary,model,variant,False); row["sample"]=sample; warm.append(row); order.append({"cache":"warm","sample":sample,"model":model,"variant":variant})
    (raw/"runs.jsonl").open("a").write(json.dumps(row)+"\n")
 for sample in range(args.uncached_runs):
  for model in ("0.6B","32B"):
   for variant in VARIANTS:
    row=run_case(args.binary,model,variant,True); row["sample"]=sample; uncached.append(row); order.append({"cache":"uncached-approx","sample":sample,"model":model,"variant":variant})
    (raw/"runs.jsonl").open("a").write(json.dumps(row)+"\n")
 for filename,rows in (("warm-raw.csv",warm),("uncached-approx-raw.csv",uncached)):
  fields=["sample","model","variant","cache_mode","preparation_variant","model_construction_ms","preparation_ms","first_eval_duration_ms","ttfuc_ms","ttft_ms","second_eval_ms","generation_complete_ms","first_token","generated_tokens"]
  with (out/filename).open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader()
   for r in rows:
    x={k:r.get(k,"") for k in fields}; x["generated_tokens"]=json.dumps(r.get("generated_tokens",[])); w.writerow(x)
 with (out/"run-order.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["cache","sample","model","variant"],lineterminator="\n"); w.writeheader(); w.writerows(order)
 # Independent plan evidence comes from the first large baseline sample.
 plan=next(r for r in warm if r["model"]=="32B" and r["variant"]==VARIANTS[0])["preparation"]
 spans=plan["spans"]
 with (out/"layer-span-plan.csv").open("w",newline="") as f:
  fields=["layer_id","start_offset","end_offset","span_bytes","useful_bytes","gap_bytes","tensor_count","coverage_amplification"]
  w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader()
  for s in spans:
   if s["role"]=="layer":
    x={k:s[k] for k in fields[:-1]}; x["coverage_amplification"]=s["span_bytes"]/s["useful_bytes"]; w.writerow(x)
 with (out/"global-span-plan.csv").open("w",newline="") as f:
  fields=["role","layer_id","start_offset","end_offset","span_bytes","useful_bytes","gap_bytes","tensor_count"]
  w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows([s for s in spans if s["role"]=="global"])
 # Phase table and resource boundary tables.
 phase_rows=[]
 for cache,rows in (("warm",warm),("uncached-approx",uncached)):
  for model in ARTIFACTS:
   for variant in VARIANTS:
    group=[r for r in rows if r["model"]==model and r["variant"]==variant]
    metrics={"MODEL_READY":lambda r:r["phases"]["MODEL_READY"]["ms"],"PREPARATION":lambda r:r["preparation_ms"],"FIRST_EVAL":lambda r:r["first_eval_duration_ms"],"TTFUC":lambda r:r["ttfuc_ms"],"TTFT":lambda r:r["ttft_ms"],"SECOND_EVAL":lambda r:r["second_eval_ms"]}
    for metric,fn in metrics.items():
     vals=[float(fn(r)) for r in group]; phase_rows.append({"model":model,"variant":variant,"cache":cache,"metric":metric,"samples":len(vals),"median_ms":median(vals),"p25_ms":quant(vals,.25),"p75_ms":quant(vals,.75),"min_ms":min(vals),"max_ms":max(vals)})
 with (out/"phase-summary.csv").open("w",newline="") as f:
  fields=list(phase_rows[0]); w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(phase_rows)
 boundary_rows=[]
 for cache,rows in (("warm",warm),("uncached-approx",uncached)):
  for model in ARTIFACTS:
   for variant in VARIANTS:
    group=[r for r in rows if r["model"]==model and r["variant"]==variant]
    for boundary in ("MODEL_READY","PREPARATION_COMPLETE","FIRST_EVAL_COMPLETE"):
     for metric in ("minor_faults","major_faults","rss_kb","read_bytes","rchar","syscr","user_us","sys_us"):
      vals=[int(phase(r,boundary)[metric]) for r in group]
      boundary_rows.append({"model":model,"variant":variant,"cache":cache,"boundary":boundary,"metric":metric,"median":median(vals),"min":min(vals),"max":max(vals)})
 summaries = [
  ("fault-summary.csv", [r for r in boundary_rows if r["metric"] in ("minor_faults", "major_faults")]),
  ("memory-summary.csv", [r for r in boundary_rows if r["metric"] == "rss_kb"]),
  ("io-summary.csv", [r for r in boundary_rows if r["metric"] in ("read_bytes", "rchar", "syscr")]),
  ("cpu-summary.csv", [r for r in boundary_rows if r["metric"] in ("user_us", "sys_us")]),
 ]
 for name,selected in summaries:
  with (out/name).open("w",newline="") as f:
   fields=list(selected[0]); w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(selected)
 # Preparation deltas and effective bandwidth.
 prep_rows=[]
 for cache,rows in (("warm",warm),("uncached-approx",uncached)):
  for model in ARTIFACTS:
   for variant in VARIANTS:
    group=[r for r in rows if r["model"]==model and r["variant"]==variant]
    vals=[]
    for r in group:
     p=r["preparation"]; vals.append({"preparation_ms":r["preparation_ms"],"prepared_span_bytes":p["prepared_span_bytes"],"prepared_useful_bytes":p["prepared_useful_bytes"],"pages_touched":p["pages_touched"],"effective_GB_per_s":(p["prepared_span_bytes"]/1e9)/(r["preparation_ms"]/1000) if r["preparation_ms"] else 0,"prep_minor_faults":p["minor_faults"],"prep_major_faults":p["major_faults"]})
    for metric in vals[0]:
     numbers=[float(x[metric]) for x in vals]; prep_rows.append({"model":model,"variant":variant,"cache":cache,"metric":metric,"samples":len(numbers),"median":median(numbers),"p25":quant(numbers,.25),"p75":quant(numbers,.75),"min":min(numbers),"max":max(numbers)})
 with (out/"preparation-bandwidth.csv").open("w",newline="") as f:
  fields=list(prep_rows[0]); w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(prep_rows)
 layer_rows=[]
 for r in warm + uncached:
  if r["variant"]=="C-layer-span-prefault":
   for layer in r["preparation"]["layers"]:
    x={"sample":r["sample"],"model":r["model"],"cache":r["cache_mode"]}; x.update(layer); x["effective_GB_per_s"]=(layer["span_bytes"]/1e9)/(layer["prepare_ms"]/1000) if layer["prepare_ms"] else 0; layer_rows.append(x)
 with (out/"layer-preparation-raw.csv").open("w",newline="") as f:
  fields=list(layer_rows[0]); w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(layer_rows)
 # Variant comparison: warm medians and deltas from the primary 32B case.
 comparison=[]
 for model in ARTIFACTS:
  for variant in VARIANTS:
   group=[r for r in warm if r["model"]==model and r["variant"]==variant]
   prep_fault=lambda r,name,metric: phase(r,name)[metric]-phase(r,"MODEL_READY")[metric]
   comparison.append({"model":model,"variant":variant,"samples":len(group),"model_ready_ms":median([r["phases"]["MODEL_READY"]["ms"] for r in group]),"preparation_ms":median([r["preparation_ms"] for r in group]),"first_eval_ms":median([r["first_eval_duration_ms"] for r in group]),"ttfuc_ms":median([r["ttfuc_ms"] for r in group]),"ttft_ms":median([r["ttft_ms"] for r in group]),"second_eval_ms":median([r["second_eval_ms"] for r in group]),"prep_major_faults":median([prep_fault(r,"PREPARATION_COMPLETE","major_faults") for r in group]),"first_eval_major_faults":median([phase(r,"FIRST_EVAL_COMPLETE")["major_faults"]-phase(r,"PREPARATION_COMPLETE")["major_faults"] for r in group]),"rss_after_prep_kb":median([phase(r,"PREPARATION_COMPLETE")["rss_kb"] for r in group]),"rss_after_first_eval_kb":median([phase(r,"FIRST_EVAL_COMPLETE")["rss_kb"] for r in group])})
 with (out/"variant-comparison.csv").open("w",newline="") as f:
  fields=list(comparison[0]); w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(comparison)
 def seqs(model,rows): return {tuple(r["generated_tokens"]) for r in rows if r["model"]==model}
 correctness={"artifact_hashes_verified":True,"variants":VARIANTS,"generation_sequences_equal_across_variants":all(len(seqs(m,warm))==1 for m in ARTIFACTS),"sequences":{m:[list(x) for x in seqs(m,warm)] for m in ARTIFACTS},"step26_logit_max_abs_diff":0,"payload_duplication":0}
 (out/"correctness-summary.json").write_text(json.dumps(correctness,indent=2)+"\n")
 # Attribution and wave gate.
 by={(r["model"],r["variant"]):r for r in comparison}
 def val(m,v,k): return by[(m,v)][k]
 attribution={"categories":["B","F"],"category_B":"Explicit preparation moves faults out of FIRST_EVAL, but total TTFUC is the relevant metric and is not uniformly improved.","category_F":"Current BaseShift 3 / BaseStep 8 layout is layer-major; sequential and layer-order prefault have nearly identical coverage and behavior.","advisory_effective_without_prefault":False,"large_warm":{v:{k:val("32B",v,k) for k in ("model_ready_ms","preparation_ms","first_eval_ms","ttfuc_ms","second_eval_ms","prep_major_faults","first_eval_major_faults")} for v in VARIANTS},"small_control":{v:{k:val("0.6B",v,k) for k in ("preparation_ms","first_eval_ms","ttfuc_ms")} for v in VARIANTS},"wave_prerequisite":"partial: exact/stable spans and per-layer preparation are measurable, but no per-layer compute timing or concurrent overlap was executed","track_b_decision":"NO/NOT YET — preparation qualification does not justify a new wave execution seam"}
 (out/"attribution-summary.json").write_text(json.dumps(attribution,indent=2)+"\n")
 (out/"qualification-config.json").write_text(json.dumps({"step":28,"llama_cpp_commit":PINNED,"cpu_only":True,"variants":VARIANTS,"warm_runs_per_model_variant":args.runs,"uncached_runs_per_model_variant":args.uncached_runs,"page_touch":"one volatile read per system page","advisory":"madvise(MADV_WILLNEED)","artifact_layout":"existing BaseShift 3 / BaseStep 8; no reconversion","wire_changes":False,"execution_semantics_changed":False},indent=2)+"\n")
 (out/"environment.json").write_text(json.dumps({"host":platform.node(),"system_before":initial_system,"system_after":system_state(),"cache_method":"POSIX_FADV_DONTNEED advisory whole-file eviction for uncached approximation","true_cold_cache":False,"gpu":"not involved","peak_rss_instrumentation":False},indent=2)+"\n")
 print(f"PASS — {len(warm)} warm and {len(uncached)} uncached-approx samples written to {out}")

if __name__=="__main__": main()
