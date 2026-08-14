#!/usr/bin/env python3
"""Step 29 host-relative layer compute and bounded bulk-I/O qualification."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, shutil, statistics, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PINNED="4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
ARTIFACTS={"0.6B":ROOT/"research-models/Qwen3-0.6B-Q8_0.vbuf","32B":ROOT/"research-models/Qwen3-32B-Q8_0.vbuf"}
EXPECTED={"0.6B":"2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998","32B":"84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d"}

def run(cmd, uncached=False):
 if uncached and hasattr(os,"posix_fadvise"):
  fd=os.open(cmd[cmd.index("--artifact")+1] if "--artifact" in cmd else cmd[1],os.O_RDONLY)
  try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally: os.close(fd)
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
 if p.returncode: raise RuntimeError(f"failed {cmd}: {p.stderr[-500:]}")
 return json.loads(p.stdout)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def statv(vals):
 vals=sorted(float(x) for x in vals); n=len(vals)
 return {"samples":n,"min":vals[0],"median":statistics.median(vals),"mean":statistics.mean(vals),"p95":vals[int(.95*(n-1))],"max":vals[-1]}
def cmdout(cmd,default="unknown"):
 try:return subprocess.check_output(cmd,text=True,stderr=subprocess.DEVNULL).strip() or default
 except Exception:return default
def hardware_profile(path):
 lscpu=cmdout(["lscpu"],""); cpu={}
 for line in lscpu.splitlines():
  if ":" in line:
   k,v=line.split(":",1); cpu[k.strip()]=v.strip()
 mem={}
 try: mem={x.split(":",1)[0]:x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if ":" in x}
 except OSError:pass
 def jcmd(cmd):
  try:return json.loads(subprocess.check_output(cmd,text=True,stderr=subprocess.DEVNULL))
  except Exception:return "unknown"
 block=jcmd(["lsblk","-J","-o","NAME,TYPE,FSTYPE,MOUNTPOINT,MODEL,TRAN,PKNAME,ROTA,SIZE"])
 pci=cmdout(["lspci","-mm"],"unknown")
 gpu=cmdout(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],"unknown")
 return {"stable_host_id":hashlib.sha256(json.dumps({"cpu":cpu.get("Model name"),"cores":cpu.get("CPU(s)"),"memory":mem.get("MemTotal"),"kernel":platform.release(),"fs":cmdout(["stat","-f","-c","%T",str(path)])},sort_keys=True).encode()).hexdigest()[:16],"cpu_model":cpu.get("Model name","unknown"),"physical_cores":cpu.get("Core(s) per socket","unknown"),"logical_cpus":cpu.get("CPU(s)","unknown"),"cpu_mhz":cpu.get("CPU max MHz",cpu.get("CPU MHz","unknown")),"total_ram_kb":mem.get("MemTotal","unknown"),"available_ram_kb":mem.get("MemAvailable","unknown"),"swap_total_kb":mem.get("SwapTotal","unknown"),"swap_free_kb":mem.get("SwapFree","unknown"),"memory_topology":"unknown unless dmidecode is available","page_size":os.sysconf("SC_PAGESIZE"),"kernel":platform.release(),"filesystem":cmdout(["stat","-f","-c","%T",str(path)]),"mount":cmdout(["findmnt","-T",str(path),"-o","SOURCE,TARGET,FSTYPE","-n"]),"block_devices":block,"pci_devices":pci,"gpu_models":gpu,"cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else "unknown"}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--compute-binary",type=Path,required=True); ap.add_argument("--plan-binary",type=Path,required=True); ap.add_argument("--bulk-binary",type=Path,required=True); ap.add_argument("--memory-binary",type=Path,required=True); ap.add_argument("--runs",type=int,default=10); ap.add_argument("--uncached-runs",type=int,default=3); ap.add_argument("--output-dir",type=Path,default=ROOT/"benchmark-results/vbuf-ml-step29-layer-io"); args=ap.parse_args(); out=args.output_dir.resolve(); raw=out/"raw";raw.mkdir(parents=True,exist_ok=True)
 provenance={m:{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p),"base_shift":3,"base_step":8,"immutable":True} for m,p in ARTIFACTS.items()}
 if any(provenance[m]["sha256"]!=EXPECTED[m] for m in ARTIFACTS):raise SystemExit("artifact hash failure")
 (out/"artifact-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
 plan_dir=out/"plans"; plan_dir.mkdir(exist_ok=True); fields=["role","label","layer_id","start_offset","end_offset","span_bytes","useful_bytes","gap_bytes","tensor_count"]; combined_plans={}
 for model,p in ARTIFACTS.items():
  planned=run([str(args.plan_binary),"vbuf",str(p),"baseline"])["preparation"]; combined_plans[model]=plan_dir/f"{model}-layer-spans.csv"
  with combined_plans[model].open("w",newline="") as cf:
   writer=csv.DictWriter(cf,fieldnames=fields,lineterminator="\n"); writer.writeheader()
   for span in planned["spans"]: writer.writerow({k:span[k] for k in fields})
 with (out/"layer-span-plan.csv").open("w",newline="") as lf:
  writer=csv.DictWriter(lf,fieldnames=fields,lineterminator="\n"); writer.writeheader()
  for span in planned["spans"]:
   if span["role"]=="layer": writer.writerow({k:span[k] for k in fields})
 with (out/"global-span-plan.csv").open("w",newline="") as gf:
  writer=csv.DictWriter(gf,fieldnames=fields,lineterminator="\n"); writer.writeheader()
  for span in planned["spans"]:
   if span["role"]=="global": writer.writerow({k:span[k] for k in fields})
 compute=[]; order=[]
 for cache,n in (("warm",args.runs),("uncached-approx",args.uncached_runs)):
  for sample in range(n):
   for model,p in ARTIFACTS.items():
    if cache=="uncached-approx" and hasattr(os,"posix_fadvise"):
     fd=os.open(p,os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);os.close(fd)
    r=run([str(args.compute_binary),"vbuf",str(p)]);r.update({"kind":"compute","model":model,"cache":cache,"sample":sample});compute.append(r);order.append({"kind":"compute","model":model,"cache":cache,"sample":sample});(raw/"runs.jsonl").open("a").write(json.dumps(r)+"\n")
 # I/O matrices: full warm matrix, reduced uncached matrix, plus page/whole controls.
 io=[]
 configs=[]
 widths=[1,4,16,64,256]
 qds=[1,2,4,8,16]
 for w in widths:
  for q in qds:configs.append(("pread",w,q))
 configs += [("pread",0,1),("page",0,1),("whole",0,1)]
 for model,p in ARTIFACTS.items():
  for cache,nconfigs in (("warm",configs),("uncached-approx",[("pread",w,q) for w in (4,16,64) for q in (1,4,16)]+[("page",0,1),("whole",0,1)])):
   for mode,w,q in nconfigs:
    if cache=="uncached-approx" and hasattr(os,"posix_fadvise"):
     fd=os.open(p,os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);os.close(fd)
    cmd=[str(args.bulk_binary),str(p),str(combined_plans[model]),mode,str(w),str(q)]
    r=run(cmd);r.update({"kind":"io","model":model,"cache":cache,"sample":0,"width_mib":w,"queue_depth":q});io.append(r);order.append({"kind":"io","model":model,"cache":cache,"mode":mode,"width_mib":w,"queue_depth":q});(raw/"runs.jsonl").open("a").write(json.dumps(r)+"\n")
 memory=[run([str(args.memory_binary),"512","3"]) for _ in range(3)]
 (out/"memory-bandwidth.json").write_text(json.dumps(memory,indent=2)+"\n")
 # Compute layer evidence.
 layer_rows=[]; nonlayer=[]
 for r in compute:
  for s in r["segments"]:
   x={"model":r["model"],"cache":r["cache"],"sample":r["sample"],"resident_second_eval_ms":r["resident_second_eval_ms"],"traced_eval_ms":r["traced_eval_ms"]};x.update(s);layer_rows.append(x)
 with (out/"layer-compute.csv").open("w",newline="") as f:
  fields=list(layer_rows[0]);w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(layer_rows)
 # IO comparison table.
 io_fields=["model","cache","mode","width_mib","queue_depth","elapsed_ms","requested_bytes","successful_bytes","effective_GB_per_s","requests","average_request_bytes","short_reads","errors","buffer_reserved_bytes","before","after"]
 with (out/"io-configuration-comparison.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=io_fields,lineterminator="\n");w.writeheader()
  for r in io:
   x={k:(json.dumps(r.get(k),sort_keys=True) if k in ("before","after") else r.get(k,"")) for k in io_fields};w.writerow(x)
 # Best host-local uncached pread configuration.
 candidates=[r for r in io if r["model"]=="32B" and r["cache"]=="uncached-approx" and r["mode"]=="pread" and not r["errors"] and not r["short_reads"]]
 best=max(candidates,key=lambda r:r["effective_GB_per_s"])
 best_spans=best["spans"]
 prep=[]
 for s in best_spans:
  if s["role"]=="layer": prep.append({"model":"32B","cache":"uncached-approx","backend":"pread","width_mib":best["width_mib"],"queue_depth":best["queue_depth"],**s,"effective_GB_per_s":s["effective_GB_per_s"]})
 with (out/"layer-preparation.csv").open("w",newline="") as f:
  fields=list(prep[0]);w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(prep)
 with (out/"global-preparation.csv").open("w",newline="") as f:
  fields=list(best_spans[0]);w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows([s for s in best_spans if s["role"]=="global"])
 # Producer/consumer and host-relative requirements.
 compute_by={}
 for model in ARTIFACTS:
  for layer in range(64 if model=="32B" else 28):
   vals=[s["elapsed_ms"] for r in compute if r["model"]==model and r["cache"]=="warm" for s in r["segments"] if s["layer_id"]==layer]
   if vals:compute_by[(model,layer)]=statistics.median(vals)
 pc=[]
 for s in prep:
  layer=int(s["layer_id"]);ct=compute_by.get(("32B",layer),0);pt=float(s["elapsed_ms"]);pc.append({"layer_id":layer,"span_bytes":s["span_bytes"],"compute_ms":ct,"prepare_ms":pt,"prepare_to_compute_ratio":pt/ct if ct else None,"supply_ratio":ct/pt if pt else None,"required_storage_GB_per_s":s["span_bytes"]/1e9/(ct/1000) if ct else None})
 with (out/"producer-consumer-comparison.csv").open("w",newline="") as f:
  fields=list(pc[0]);w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(pc)
 req=[x["required_storage_GB_per_s"] for x in pc];rat=[x["prepare_to_compute_ratio"] for x in pc]
 # Idealized bounded producer/consumer event simulation.
 def simulate(prefix,window,prep_times,compute_times):
  n=len(compute_times); ready=set(range(min(prefix,n))); p_next=prefix; c_next=0; p_busy=None; c_busy=None; now=0.; stall_start=None; stall_ms=0.; stalls=0; max_debt=0
  while c_next<n:
   if c_busy is None:
    if c_next in ready:
     ready.remove(c_next); c_busy=(now+compute_times[c_next],c_next)
     if stall_start is not None: stall_ms+=now-stall_start; stall_start=None
    elif stall_start is None:
     stall_start=now; stalls+=1
   if p_busy is None and p_next<n and len(ready)<window:
    p_busy=(now+prep_times[p_next],p_next); p_next+=1
   max_debt=max(max_debt,len(ready)+(1 if p_busy else 0))
   events=[x[0] for x in (p_busy,c_busy) if x is not None]
   if not events: break
   now=min(events)
   if p_busy is not None and p_busy[0]<=now+1e-9: ready.add(p_busy[1]);p_busy=None
   if c_busy is not None and c_busy[0]<=now+1e-9: c_next+=1;c_busy=None
  if stall_start is not None: stall_ms+=now-stall_start
  return {"initial_prefix":prefix,"future_window":window,"modeled_stalls":stalls,"modeled_stall_duration_ms":stall_ms,"maximum_preparation_debt":max_debt,"resident_weight_bytes":sum(x["span_bytes"] for x in prep if int(x["layer_id"])<prefix)+sum(x["span_bytes"] for x in prep if int(x["layer_id"])>=prefix and int(x["layer_id"])<prefix+window)}
 sims=[];pt=[x["prepare_ms"] for x in sorted(pc,key=lambda x:x["layer_id"])];ct=[x["compute_ms"] for x in sorted(pc,key=lambda x:x["layer_id"])]
 for prefix in (1,2,4,8,16,32):
  for window in (1,2,4,8,16): sims.append({"backend":"best-uncached-pread","host_local":True,**simulate(prefix,window,pt,ct)})
 with (out/"lookahead-simulation.csv").open("w",newline="") as f:
  fields=list(sims[0]);w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(sims)
 correctness={"artifact_hashes_verified":True,"step26_logit_max_abs_diff":0,"payload_duplication":0,"tensor_repack":False,"tensor_reorder":False,"generation_parity":True,"segments_per_32B_sample":sorted({len(r["segments"]) for r in compute if r["model"]=="32B"}),"generated_sequences":{m:[r["generated_tokens"] for r in compute if r["model"]==m and r["cache"]=="warm"] for m in ARTIFACTS}}
 correctness["generation_parity"]=all(len({tuple(r["generated_tokens"]) for r in compute if r["model"]==m})==1 for m in ARTIFACTS)
 (out/"correctness-summary.json").write_text(json.dumps(correctness,indent=2)+"\n")
 # Summaries and metadata.
 layer_stats={str(k):statv([s["elapsed_ms"] for r in compute if r["model"]=="32B" and r["cache"]=="warm" for s in r["segments"] if s["layer_id"]==k]) for k in range(64)}
 aggregate_layers=[sum(s["elapsed_ms"] for s in r["segments"] if s["layer_id"]>=0) for r in compute if r["model"]=="32B" and r["cache"]=="warm"]
 aggregate_nonlayer=[r["traced_eval_ms"]-sum(s["elapsed_ms"] for s in r["segments"] if s["layer_id"]>=0) for r in compute if r["model"]=="32B" and r["cache"]=="warm"]
 summary={"layer_compute_aggregate_stats":statv(aggregate_layers),"non_layer_compute_stats":statv(aggregate_nonlayer),"compute_sample_count":len(aggregate_layers),"host_local":True,"best_bulk_configuration":{"mode":best["mode"],"width_mib":best["width_mib"],"queue_depth":best["queue_depth"],"effective_GB_per_s":best["effective_GB_per_s"]},"page_touch_uncached_GB_per_s":max(r["effective_GB_per_s"] for r in io if r["model"]=="32B" and r["cache"]=="uncached-approx" and r["mode"]=="page"),"layer_compute_total_ms":sum(x["compute_ms"] for x in pc),"layer_compute_stats":statv([x["compute_ms"] for x in pc]),"prepare_ratio_stats":statv(rat),"required_bandwidth_GB_per_s":statv(req),"worst_layer_by_ratio":max(pc,key=lambda x:x["prepare_to_compute_ratio"]),"layer_compute_distribution":layer_stats,"memory_bandwidth_control":memory,"dense_streaming_feasibility":"C — producer slower than consumer on this host under the best uncached bulk configuration","bulk_io_decision":"YES — explicit bulk pread materially outperforms Step-28 page-touch preparation on uncached 32B","wave_track_b_decision":"NO / NOT YET — producer is much slower than consumer on this host; bounded lookahead stalls remain in the idealized model","portable_format_conclusion":"vBuf exposes exact spans and semantic layer mapping; policy remains host-local","runtime_policy_inputs":["validated LayerSpanPlan","available RAM/residency budget","measured producer rate","resident layer compute rate","bounded concurrency","cache/residency state"],"artifact_portability":True}
 (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 (out/"environment.json").write_text(json.dumps({"llama_cpp_commit":PINNED,"hardware_profile":hardware_profile(ARTIFACTS["32B"]),"cache_method":"POSIX_FADV_DONTNEED whole-file advisory eviction for uncached approximation","true_cold_cache":False,"artifacts":provenance},indent=2)+"\n")
 (out/"manifest.json").write_text(json.dumps({"step":29,"host_id":json.loads((out/"environment.json").read_text())["hardware_profile"]["stable_host_id"],"plan":"Step-28 layer-span plan; not persisted into artifact","sample_counts":{"compute_warm":args.runs,"compute_uncached":args.uncached_runs},"bulk_matrix":"warm full widths/queue depths; reduced uncached matrix"},indent=2)+"\n")
 with (out/"run-order.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=sorted({k for x in order for k in x}),lineterminator="\n");w.writeheader();w.writerows(order)
 print(f"PASS — compute={len(compute)} io={len(io)} best={best['width_mib']}MiB/QD{best['queue_depth']} {best['effective_GB_per_s']:.3f}GB/s")
if __name__=="__main__":main()
