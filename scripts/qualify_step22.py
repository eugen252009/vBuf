#!/usr/bin/env python3
"""Run the unoptimized CPU GGUF/vBuf Step-22 baseline."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, shutil, statistics, subprocess, tempfile, time
from pathlib import Path
PINNED="4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
CHECKPOINT="73a1f36661482037573a789ab90e15039a782ad9"
PATCH="patches/llama.cpp/0001-user-metadata-tensor-source.patch"
ARTIFACTS={
 "BF16": {"gguf":"Qwen3-0.6B-BF16.gguf","vbuf":"Qwen3-0.6B-BF16.vbuf","gguf_sha256":"65a16246f5814dc0587acadcf0328186b17febf6dcaeb1b13efa9243b551d38e","vbuf_sha256":"6ec3db0bb8a26914be7312cc26c3ec0fb6945202659c46b26b4506f4de75a806"},
 "Q8_0": {"gguf":"Qwen3-0.6B-Q8_0.gguf","vbuf":"Qwen3-0.6B-Q8_0.vbuf","gguf_sha256":"9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031","vbuf_sha256":"2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998"},
}
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()
def run(cmd, **kwargs): return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)
def evict(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); return True
    except (AttributeError,OSError): return False
    finally: os.close(fd)
def quantile(values, q):
    values=sorted(values); return values[min(len(values)-1,max(0,int((len(values)-1)*q)))]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--upstream-root",type=Path,default=Path("/tmp/llama.cpp-step21")); p.add_argument("--build-dir",type=Path,default=Path("/tmp/llama.cpp-step21-vbuf-build")); p.add_argument("--warm-runs",type=int,default=3); p.add_argument("--cold-runs",type=int,default=2); p.add_argument("--threads",type=int,default=2); p.add_argument("--output-dir",type=Path,default=None); a=p.parse_args(); root=a.root.resolve(); upstream=a.upstream_root.resolve(); build=a.build_dir.resolve(); out=(a.output_dir or root/"benchmark-results/vbuf-ml-step22").resolve(); raw=out/"raw"; raw.mkdir(parents=True,exist_ok=True)
    step21_provenance=json.loads((root/"benchmark-results/vbuf-ml-step21/adapter-provenance.json").read_text())
    required_status=("llama_model_vbuf_construction","tokenizer_parity","logit_parity","generation_parity")
    if any(step21_provenance.get("status",{}).get(key) != "PASS" for key in required_status): raise SystemExit("Step-21 correctness gate is not PASS")
    current=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(); tag=subprocess.check_output(["git","-C",str(root),"rev-parse","vbuf-ml-0.1-consumer-parity^{}"],text=True).strip()
    if tag != CHECKPOINT: raise SystemExit(f"consumer checkpoint mismatch: {tag}")
    if subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip() != PINNED: raise SystemExit("pinned llama.cpp mismatch")
    subprocess.run(["git","-C",str(upstream),"apply","--reverse","--check",str(root/PATCH)],check=True)
    artifacts={}
    for label, config in ARTIFACTS.items():
        artifacts[label]={}
        for fmt in ("gguf","vbuf"):
            path=root/"research-models"/config[fmt]; actual=sha256(path)
            if actual != config[f"{fmt}_sha256"]: raise SystemExit(f"{path}: hash mismatch {actual}")
            artifacts[label][fmt]=path
    subprocess.run(["cargo","build","--manifest-path",str(root/"rust/Cargo.toml"),"-p","vbuf-ml"],cwd=root,check=True)
    with tempfile.TemporaryDirectory() as td:
        exe=Path(td)/"step22_benchmark"; includes=[f"-I{upstream/'include'}",f"-I{upstream/'ggml/include'}",f"-I{root/'integrations/llama.cpp'}"]; sources=[root/"integrations/llama.cpp/step22_benchmark.cpp",root/"integrations/llama.cpp/llama_vbuf_loader.cpp",root/"integrations/llama.cpp/vbuf_ml_adapter.cpp"]
        cmd=["g++","-O2","-std=c++17",*includes,*map(str,sources),f"-L{root/'rust/target/debug'}",f"-L{build/'bin'}","-lvbuf_ml","-lllama","-lggml","-lggml-cpu","-lggml-base","-lpthread","-ldl","-lm",f"-Wl,-rpath,{root/'rust/target/debug'}",f"-Wl,-rpath,{build/'bin'}","-o",str(exe)]; subprocess.run(cmd,check=True)
        prompt=json.loads((root/"benchmarks/vbuf-ml/step22/prompts.json").read_text())["primary"]; env=dict(os.environ,LD_LIBRARY_PATH=f"{root/'rust/target/debug'}:{build/'bin'}")
        records=[]; schedule=[]
        for label in ARTIFACTS:
            for i in range(max(a.warm_runs,a.cold_runs)):
                for fmt in (("gguf","vbuf") if i%2==0 else ("vbuf","gguf")):
                    if i<a.warm_runs: schedule.append((label,fmt,"warm",i))
                    if i<a.cold_runs: schedule.append((label,fmt,"cold",i))
        for label,fmt,condition,i in schedule:
            path=artifacts[label][fmt]
            if condition=="cold": evict(path)
            started=time.monotonic(); result=run([str(exe),fmt,str(path),prompt,str(a.threads)],env=env,timeout=900); wall=(time.monotonic()-started)*1e6
            lines=list(csv.DictReader(result.stdout.splitlines()))
            for row in lines:
                row.update({"artifact":label,"condition":condition,"run_index":i,"orchestrator_wall_us":f"{wall:.3f}"}); records.append(row)
            (raw/f"{label.lower()}-{condition}-{i}-{fmt}.stderr").write_text(result.stderr)
        for label in ARTIFACTS:
            for fmt in ("gguf", "vbuf"):
                path=artifacts[label][fmt]
                for condition in ("warm", "cold"):
                    if condition == "cold": evict(path)
                    result=run([str(exe),fmt,str(path),prompt,str(a.threads),"vocab_only"],env=env,timeout=900)
                    for row in csv.DictReader(result.stdout.splitlines()):
                        row.update({"artifact":label,"condition":condition,"run_index":0,"orchestrator_wall_us":"0"}); records.append(row)
        fields=["artifact","format","condition","run_index","phase","duration_us","minor_faults","major_faults","rss_kb","vmsize_kb","pss_kb","read_bytes","file_bytes","prompt_tokens","generated_tokens","prompt_tokens_sec","generation_tokens_sec","orchestrator_wall_us"]
        with (raw/"samples.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(records)
    summaries=[]
    for key, rows in __import__('itertools').groupby(sorted(records,key=lambda r:(r['artifact'],r['format'],r['condition'],r['phase'])),key=lambda r:(r['artifact'],r['format'],r['condition'],r['phase'])):
        rows=list(rows); vals=[float(r['duration_us']) for r in rows]; summaries.append({"artifact":key[0],"format":key[1],"condition":key[2],"phase":key[3],"samples":len(vals),"median_us":statistics.median(vals),"p25_us":quantile(vals,.25),"p75_us":quantile(vals,.75),"min_us":min(vals),"max_us":max(vals)})
    with (out/"phase-summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=summaries[0],lineterminator="\n"); w.writeheader(); w.writerows(summaries)
    resource=[]
    for key, rows in __import__('itertools').groupby(sorted(records,key=lambda r:(r['artifact'],r['format'],r['condition'],r['phase'])),key=lambda r:(r['artifact'],r['format'],r['condition'],r['phase'])):
        rows=list(rows); resource.append({"artifact":key[0],"format":key[1],"condition":key[2],"phase":key[3],"samples":len(rows),"median_minor_faults":statistics.median(float(r['minor_faults']) for r in rows),"median_major_faults":statistics.median(float(r['major_faults']) for r in rows),"median_rss_kb":statistics.median(float(r['rss_kb']) for r in rows),"median_pss_kb":statistics.median(float(r['pss_kb']) for r in rows),"median_read_bytes":statistics.median(float(r['read_bytes']) for r in rows)})
    with (out/"resource-summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=resource[0],lineterminator="\n"); w.writeheader(); w.writerows(resource)
    comparisons=[]
    by={(r['artifact'],r['format'],r['condition'],r['phase']):r for r in summaries}
    for artifact in ARTIFACTS:
        for condition in ("warm","cold"):
            for phase in sorted({r['phase'] for r in summaries if r['artifact']==artifact and r['condition']==condition}):
                g=by.get((artifact,"gguf",condition,phase)); v=by.get((artifact,"vbuf",condition,phase))
                if g and v: comparisons.append({"artifact":artifact,"condition":condition,"phase":phase,"gguf_median_us":g['median_us'],"vbuf_median_us":v['median_us'],"absolute_delta_us":v['median_us']-g['median_us'],"relative_delta":(v['median_us']-g['median_us'])/g['median_us'] if g['median_us'] else None})
    with (out/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=comparisons[0],lineterminator="\n"); w.writeheader(); w.writerows(comparisons)
    (out/"environment.json").write_text(json.dumps({"date_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"kernel":platform.release(),"machine":platform.machine(),"cpu":platform.processor(),"logical_cpus":os.cpu_count(),"python":platform.python_version(),"filesystem":"see qualification-config.json","cache_method":"POSIX_FADV_DONTNEED per file; uncached approximation, not drop_caches","threads":a.threads,"affinity":"not forced","vbuf_commit":current,"consumer_checkpoint":"vbuf-ml-0.1-consumer-parity","consumer_checkpoint_commit":tag,"llama_cpp_commit":PINNED,"adapter_patch":PATCH,"build_dir":"external pinned build; path supplied at runtime","runtime_settings":{"n_gpu_layers":0,"n_ctx":512,"n_batch":512,"n_ubatch":"default","sampling":"greedy qualification only"}},indent=2)+"\n")
    (out/"artifact-manifest.json").write_text(json.dumps({label:{**config,"paths":"research-models (local ignored artifacts)"} for label,config in ARTIFACTS.items()},indent=2)+"\n")
    (out/"qualification-config.json").write_text(json.dumps({"vbuf_commit":current,"consumer_checkpoint":"vbuf-ml-0.1-consumer-parity","consumer_checkpoint_commit":tag,"llama_cpp_commit":PINNED,"warm_runs":a.warm_runs,"cold_runs":a.cold_runs,"threads":a.threads,"primary_prompt":"benchmarks/vbuf-ml/step22/prompts.json:primary","primary_statistic":"median with p25/p75 and min/max","cold_classification":"uncached approximation","correctness_gate":"Step-21 runtime qualification required before benchmark","performance_optimization":False},indent=2)+"\n")
    print(f"PASS — Step-22 raw samples and summaries written to {out}")
if __name__=="__main__": main()
