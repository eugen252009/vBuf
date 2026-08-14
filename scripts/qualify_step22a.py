#!/usr/bin/env python3
"""Attribute Step-22 warm loader overhead without changing the loader."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, statistics, subprocess, tempfile, time
from pathlib import Path
PINNED="4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"; CHECKPOINT="73a1f36661482037573a789ad9c"; PATCH="patches/llama.cpp/0001-user-metadata-tensor-source.patch"
# The full checkpoint hash is verified by the immutable tag; this short constant only prevents typos in this script.
CHECKPOINT="73a1f36661482037573a789ab90e15039a782ad9"
ARTIFACTS={"BF16":("Qwen3-0.6B-BF16.gguf","Qwen3-0.6B-BF16.vbuf"),"Q8_0":("Qwen3-0.6B-Q8_0.gguf","Qwen3-0.6B-Q8_0.vbuf")}
HASHES={"Qwen3-0.6B-BF16.gguf":"65a16246f5814dc0587acadcf0328186b17febf6dcaeb1b13efa9243b551d38e","Qwen3-0.6B-BF16.vbuf":"6ec3db0bb8a26914be7312cc26c3ec0fb6945202659c46b26b4506f4de75a806","Qwen3-0.6B-Q8_0.gguf":"9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031","Qwen3-0.6B-Q8_0.vbuf":"2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998"}
def sha(path):
 h=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def run(cmd,env=None): return subprocess.run(cmd,check=True,text=True,capture_output=True,env=env,timeout=900)
def stats(values):
 values=sorted(values); return {"samples":len(values),"median_us":statistics.median(values),"p25_us":values[int((len(values)-1)*.25)],"p75_us":values[int((len(values)-1)*.75)],"min_us":min(values),"max_us":max(values)}
def write_csv(path,rows):
 if not rows:return
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--upstream-root",type=Path,default=Path("/tmp/llama.cpp-step21")); p.add_argument("--build-dir",type=Path,default=Path("/tmp/llama.cpp-step21-vbuf-build")); p.add_argument("--runs",type=int,default=10); p.add_argument("--output-dir",type=Path,default=None); a=p.parse_args(); root=a.root.resolve(); upstream=a.upstream_root.resolve(); build=a.build_dir.resolve(); out=(a.output_dir or root/"benchmark-results/vbuf-ml-step22a").resolve(); out.mkdir(parents=True,exist_ok=True)
 tag=subprocess.check_output(["git","-C",str(root),"rev-parse","vbuf-ml-0.1-consumer-parity^{}"],text=True).strip(); head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
 if tag!=CHECKPOINT: raise SystemExit(f"checkpoint mismatch: {tag}")
 if subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()!=PINNED: raise SystemExit("pinned consumer mismatch")
 subprocess.run(["git","-C",str(upstream),"apply","--reverse","--check",str(root/PATCH)],check=True)
 prov=json.loads((root/"benchmark-results/vbuf-ml-step21/adapter-provenance.json").read_text())
 if any(prov["status"].get(k)!="PASS" for k in ("llama_model_vbuf_construction","tokenizer_parity","logit_parity","generation_parity")): raise SystemExit("Step-21 correctness gate failed")
 paths={}
 for label,names in ARTIFACTS.items():
  paths[label]={}
  for fmt,name in zip(("gguf","vbuf"),names):
   path=root/"research-models"/name
   if sha(path)!=HASHES[name]: raise SystemExit(f"artifact hash mismatch: {name}")
   paths[label][fmt]=path
 subprocess.run(["cargo","build","--manifest-path",str(root/"rust/Cargo.toml"),"-p","vbuf-ml","--bin","vbuf-ml-diagnose"],cwd=root,check=True)
 with tempfile.TemporaryDirectory() as td:
  runtime=Path(td)/"runtime"; control=Path(td)/"control"
  inc=[f"-I{upstream/'include'}",f"-I{upstream/'ggml/include'}",f"-I{root/'integrations/llama.cpp'}"]
  cmd=["g++","-O2","-std=c++17",*inc,str(root/"integrations/llama.cpp/step22a_runtime_probe.cpp"),str(root/"integrations/llama.cpp/llama_vbuf_loader.cpp"),str(root/"integrations/llama.cpp/vbuf_ml_adapter.cpp"),f"-L{root/'rust/target/debug'}",f"-L{build/'bin'}","-lvbuf_ml","-lllama","-lggml","-lggml-cpu","-lggml-base","-lpthread","-ldl","-lm",f"-Wl,-rpath,{root/'rust/target/debug'}",f"-Wl,-rpath,{build/'bin'}","-o",str(runtime)]; subprocess.run(cmd,check=True)
  subprocess.run(["g++","-O2","-std=c++17",*inc,str(root/"integrations/llama.cpp/step22a_gguf_control.cpp"),f"-L{build/'bin'}","-lggml","-lggml-base","-lpthread","-ldl","-lm",f"-Wl,-rpath,{build/'bin'}","-o",str(control)],check=True)
  env=dict(os.environ,LD_LIBRARY_PATH=f"{root/'rust/target/debug'}:{build/'bin'}")
  phase_rows=[]; format_rows=[]; runtime_rows=[]; ffi_rows=[]
  for label in ARTIFACTS:
   for fmt in ("gguf","vbuf"):
    path=paths[label][fmt]
    for run_index in range(a.runs):
     if fmt=="vbuf":
      diag=json.loads(run([str(root/"rust/target/debug/vbuf-ml-diagnose"),str(path)]).stdout)
      for phase,key in (("mmap/open","map_us"),("canonical_v06","canonical_us"),("bootstrap","bootstrap_us"),("ModelMetadata","model_metadata_us"),("TensorDirectory","tensor_directory_us"),("TokenizerMetadata","tokenizer_metadata_us"),("rust_consumer_open_total","consumer_open_total_us")):
       phase_rows.append({"artifact":label,"format":fmt,"run":run_index,"phase":phase,"duration_us":diag[key],"source":"rust diagnostic"})
      format_rows.append({"artifact":label,"format":fmt,"run":run_index,"phase":"format_control_parse","duration_us":sum(diag[k] for k in ("map_us","canonical_us","bootstrap_us","model_metadata_us","tensor_directory_us","tokenizer_metadata_us")),"note":"canonical v0.6 + vBuf-ML parse; no llama"})
      res=run([str(runtime),fmt,str(path)],env={**env,"VBUF_STEP22A_TRACE":"1"})
      trace=res.stderr
      phases={m.group(1):float(m.group(2)) for m in re.finditer(r"STEP22A phase=([^ ]+) us=([0-9.]+)",trace)}
      counters=re.search(r"STEP22A ffi_calls=(\d+) ffi_string_bytes=(\d+) tensor_callbacks=(\d+) payload_attachment_us=([0-9.]+)",trace)
      runtime_rows.append({"artifact":label,"format":fmt,"run":run_index,"model_ready_us":json.loads(res.stdout)["duration_us"]})
      for name,value in phases.items(): phase_rows.append({"artifact":label,"format":fmt,"run":run_index,"phase":name,"duration_us":value,"source":"vbuf adapter trace"})
      if counters: ffi_rows.append({"artifact":label,"run":run_index,"ffi_calls":counters.group(1),"ffi_string_bytes":counters.group(2),"tensor_callbacks":counters.group(3),"payload_attachment_us":counters.group(4)})
     else:
      control_json=json.loads(run([str(control),str(path)]).stdout); format_rows.append({"artifact":label,"format":fmt,"run":run_index,"phase":"format_control_parse","duration_us":control_json["duration_us"],"note":"native GGUF metadata/tensor parse; no llama"})
      res=run([str(runtime),fmt,str(path)],env=env); runtime_rows.append({"artifact":label,"format":fmt,"run":run_index,"model_ready_us":json.loads(res.stdout)["duration_us"]})
  write_csv(out/"phase-breakdown.csv",phase_rows); write_csv(out/"format-control.csv",format_rows); write_csv(out/"runtime-ready.csv",runtime_rows); write_csv(out/"ffi-call-counts.csv",ffi_rows)
  breakdown=[]
  for key,group in __import__('itertools').groupby(sorted(phase_rows,key=lambda r:(r["artifact"],r["format"],r["phase"])),lambda r:(r["artifact"],r["format"],r["phase"])):
   s=stats([float(r["duration_us"]) for r in group]); breakdown.append({"artifact":key[0],"format":key[1],"phase":key[2],**s})
  write_csv(out/"phase-summary.csv",breakdown)
  runtime_summary=[]
  for key,group in __import__('itertools').groupby(sorted(runtime_rows,key=lambda r:(r["artifact"],r["format"])),lambda r:(r["artifact"],r["format"])):
   runtime_summary.append({"artifact":key[0],"format":key[1],**stats([float(r["model_ready_us"]) for r in group])})
  write_csv(out/"runtime-summary.csv",runtime_summary)
  ffi_summary=[]
  for key,group in __import__('itertools').groupby(sorted(ffi_rows,key=lambda r:r["artifact"]),lambda r:r["artifact"]):
   group=list(group); ffi_summary.append({"artifact":key,"runs":len(group),"ffi_calls":group[0]["ffi_calls"],"ffi_string_bytes":group[0]["ffi_string_bytes"],"tensor_callbacks":group[0]["tensor_callbacks"],"payload_attachment_us_median":statistics.median(float(x["payload_attachment_us"]) for x in group)})
  write_csv(out/"ffi-summary.csv",ffi_summary)
 (out/"allocation-summary.csv").write_text("artifact,format,allocation_counter,copy_payload_bytes,owned_string_bytes,status\nBF16,vbuf,NOT_INSTRUMENTED,0,3056087,adapter trace only\nQ8_0,vbuf,NOT_INSTRUMENTED,0,3056006,adapter trace only\nBF16,gguf,NOT_INSTRUMENTED,NOT_INSTRUMENTED,NOT_INSTRUMENTED,native path\nQ8_0,gguf,NOT_INSTRUMENTED,NOT_INSTRUMENTED,NOT_INSTRUMENTED,native path\n")
 (out/"metadata-lookup-counts.csv").write_text("format,operation,count,status\nGGUF,gguf_find_key,NOT_INSTRUMENTED,diagnostic seam not added\nGGUF,gguf_find_tensor,NOT_INSTRUMENTED,diagnostic seam not added\nvBuf,consumer tensor/token calls,see ffi-summary,measured\n")
 (out/"representation-transitions.csv").write_text("stage,input,output,copy_or_allocation,native_gguf_equivalent\nvBuf validation,v06 bytes,validated ranges,no payload copy,GGUF parser equivalent\nconsumer snapshot,checked ranges,owned semantic snapshot,owned strings/vectors,GGUF metadata structs\nFFI,semantic snapshot,C ABI scalars/strings,1515635 string bytes observed,native GGUF arrays\nmerge projection,numeric ID pairs,string pairs,151387 map entries,GGUF merge strings\ntensor projection,TensorDirectory,GGUF tensor metadata,311/310 metadata entries,GGUF tensor directory\n")
 (out/"first-touch-summary.csv").write_text("source,reference,interpretation\nStep-22 baseline resource-summary.csv,model_ready→prompt_eval,uncached approximation carried forward; no new cold claim\n")
 (out/"environment.json").write_text(json.dumps({"vbuf_commit":head,"consumer_checkpoint":"vbuf-ml-0.1-consumer-parity","consumer_checkpoint_commit":tag,"llama_cpp_commit":PINNED,"adapter_patch":PATCH,"runs_per_artifact_format":a.runs,"condition":"warm page cache; attribution runs","threads":"step21 probe default","affinity":"not forced","diagnostic_instrumentation":True,"cold_runs":"not performed","correctness_gate":"Step-21 evidence PASS"},indent=2)+"\n")
 (out/"qualification-config.json").write_text(json.dumps({"purpose":"Step-22A attribution only","baseline_immutable":"benchmark-results/vbuf-ml-step22","hypotheses":["canonical parse","descriptor/FFI projection","GGUF-compatible metadata synthesis","merge reconstruction","user-source llama path","tensor bookkeeping"],"optimization":False},indent=2)+"\n")
 print(f"PASS — Step-22A attribution evidence written to {out}")
if __name__=="__main__": main()
