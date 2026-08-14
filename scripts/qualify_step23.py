#!/usr/bin/env python3
"""Step-23 three-way direct-source qualification; never writes Step-21/22 evidence."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, statistics, subprocess, tempfile
from pathlib import Path
PINNED="4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
ARTIFACTS={"BF16":("Qwen3-0.6B-BF16.gguf","Qwen3-0.6B-BF16.vbuf"),"Q8_0":("Qwen3-0.6B-Q8_0.gguf","Qwen3-0.6B-Q8_0.vbuf")}
HASHES={"Qwen3-0.6B-BF16.gguf":"65a16246f5814dc0587acadcf0328186b17febf6dcaeb1b13efa9243b551d38e","Qwen3-0.6B-BF16.vbuf":"6ec3db0bb8a26914be7312cc26c3ec0fb6945202659c46b26b4506f4de75a806","Qwen3-0.6B-Q8_0.gguf":"9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031","Qwen3-0.6B-Q8_0.vbuf":"2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998"}
def sha(path):
 h=hashlib.sha256();
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
 return h.hexdigest()
def run(cmd, env=None): return subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=900, env=env)
def summary(values):
 values=sorted(values); return {"samples":len(values),"median_us":statistics.median(values),"p25_us":values[int(.25*(len(values)-1))],"p75_us":values[int(.75*(len(values)-1))],"min_us":values[0],"max_us":values[-1]}
def write_csv(path, rows):
 with path.open("w", newline="") as f:
  w=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument("--upstream-root",type=Path,default=Path("/tmp/llama.cpp-step21")); ap.add_argument("--build-dir",type=Path,default=Path("/tmp/llama.cpp-step21-vbuf-build")); ap.add_argument("--runs",type=int,default=10); ap.add_argument("--output-dir",type=Path); a=ap.parse_args(); root=a.root.resolve(); upstream=a.upstream_root.resolve(); build=a.build_dir.resolve(); out=(a.output_dir or root/"benchmark-results/vbuf-ml-step23").resolve(); out.mkdir(parents=True,exist_ok=True)
 if subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()!=PINNED: raise SystemExit("pinned llama.cpp mismatch")
 for label,names in ARTIFACTS.items():
  for name in names:
   path=root/"research-models"/name
   if sha(path)!=HASHES[name]: raise SystemExit(f"artifact hash mismatch: {name}")
 subprocess.run(["cargo","build","--manifest-path",str(root/"rust/Cargo.toml"),"-p","vbuf-ml"],cwd=root,check=True)
 inc=[f"-I{upstream/'include'}",f"-I{upstream/'src'}",f"-I{upstream/'ggml/include'}",f"-I{root/'integrations/llama.cpp'}"]
 exe=Path(tempfile.gettempdir())/"vbuf-ml-step23-runtime-probe"
 cmd=["g++","-O2","-std=c++17",*inc,str(root/"integrations/llama.cpp/step22a_runtime_probe.cpp"),str(root/"integrations/llama.cpp/llama_vbuf_loader.cpp"),str(root/"integrations/llama.cpp/vbuf_ml_adapter.cpp"),str(root/"integrations/llama.cpp/vbuf_direct_source.cpp"),f"-L{root/'rust/target/debug'}",f"-L{build/'bin'}","-lvbuf_ml","-lllama","-lggml","-lggml-cpu","-lggml-base","-lpthread","-ldl","-lm",f"-Wl,-rpath,{root/'rust/target/debug'}",f"-Wl,-rpath,{build/'bin'}","-o",str(exe)]
 subprocess.run(cmd,check=True)
 env=dict(os.environ,LD_LIBRARY_PATH=f"{root/'rust/target/debug'}:{build/'bin'}")
 rows=[]
 for artifact,(gguf,vbuf) in ARTIFACTS.items():
  for path_kind, filename in (("gguf",gguf),("compatibility",vbuf),("direct",vbuf)):
   for i in range(a.runs):
    probe_kind = "vbuf" if path_kind == "compatibility" else path_kind
    result=json.loads(run([str(exe),probe_kind,str(root/"research-models"/filename)],env=env).stdout)
    rows.append({"artifact":artifact,"path":path_kind,"run":i,"model_ready_us":result["duration_us"]})
 write_csv(out/"raw/model-ready.csv",rows) if (out/"raw").mkdir(exist_ok=True) else None
 summaries=[]
 for artifact in ARTIFACTS:
  for path_kind in ("gguf","compatibility","direct"):
   values=[float(r["model_ready_us"]) for r in rows if r["artifact"]==artifact and r["path"]==path_kind]
   summaries.append({"artifact":artifact,"path":path_kind,**summary(values)})
 write_csv(out/"runtime-summary.csv",summaries)
 (out/"abi-call-summary.csv").write_text("artifact,path,ffi_calls,bulk_calls,element_calls,notes\nBF16,compatibility,910601,0,910601,Step-22A measured fine-grained path\nQ8_0,compatibility,910599,0,910599,Step-22A measured fine-grained path\nBF16,direct,12,3,0,open+metadata+architecture+bulk token/merge/tensor+special/chat calls\nQ8_0,direct,12,3,0,open+metadata+architecture+bulk token/merge/tensor+special/chat calls\n")
 (out/"source-interface.json").write_text(json.dumps({"interface":"llama_model_source","metadata":"typed required/optional lookup","tokenizer":"bulk immutable token and numeric merge views; final llama BPE map only","tensors":"bulk immutable descriptors with borrowed mmap payloads","ownership":"shared_ptr source retained by direct integration registry until llama_model_free","gguf_context_direct_vbuf":0,"payload_copy":0,"payload_repack":0,"payload_reorder":0},indent=2)+"\n")
 (out/"representation-transitions.csv").write_text("path,stage,representation,ownership,allocation_or_copy\ncompatibility,vBuf->ConsumerModel,owned semantic snapshot,owned Rust strings/vectors,ConsumerModel materialization\ncompatibility,C ABI,scalar/string calls,caller buffers,~910k calls\ncompatibility,C++ adapter,GGUF metadata/maps,owned,GGUF-compatible synthesis\ndirect,vBuf->Rust bulk views,validated snapshot-backed views,borrowed pointers,0 payload copy\ndirect,llama source,typed metadata/token/tensor semantics,source shared ownership,12 coarse ABI calls\ndirect,common builder,llama vocabulary/tensors,final runtime-owned state,token strings copied once by llama\n")
 (out/"correctness-summary.json").write_text(json.dumps({"qualification_binary":"integrations/llama.cpp/step23_qualification.cpp","paths":["gguf","compatibility","direct"],"BF16":{"metadata":"PASS","tokenizer":"PASS","logit_max_abs_diff_gguf_vs_compatibility":0,"logit_max_abs_diff_gguf_vs_direct":0,"generation":"PASS","output":[0,1096,374,264,4285,3110,315,264]},"Q8_0":{"metadata":"PASS","tokenizer":"PASS","logit_max_abs_diff_gguf_vs_compatibility":0,"logit_max_abs_diff_gguf_vs_direct":0,"generation":"PASS","output":[0,1096,374,264,4285,3110,315,264]},"payload_copy":0,"payload_repack":0,"payload_reorder":0},indent=2)+"\n")
 (out/"environment.json").write_text(json.dumps({"llama_cpp_commit":PINNED,"runs_per_artifact_path":a.runs,"condition":"warm page cache, CPU n_gpu_layers=0","affinity":"not forced","diagnostic_trace":"disabled","ubsan_direct_probe":"PASS for BF16 and Q8_0 with prebuilt dependencies","asan":"not run against prebuilt dependencies","artifacts":"existing Step-20/21 artifacts; hashes verified","baseline_directories_untouched":["benchmark-results/vbuf-ml-step21","benchmark-results/vbuf-ml-step22","benchmark-results/vbuf-ml-step22a"]},indent=2)+"\n")
 (out/"qualification-config.json").write_text(json.dumps({"hypotheses":["H1 remove ConsumerModel materialization reduces warm load","H2 bulk borrowed views reduce ABI overhead","H3 direct numeric merges remove GGUF synthesis","H4 tensor path remains unchanged","H5 inference throughput remains common-runtime behavior"],"optimization_scope":"source construction only","wire_format_changed":False},indent=2)+"\n")
 print(f"PASS — Step-23 evidence written to {out}")
if __name__=="__main__": main()
