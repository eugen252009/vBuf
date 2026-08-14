#!/usr/bin/env python3
"""Controlled Step-27 loader/first-use attribution runner."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, subprocess, time
from pathlib import Path

CASES = [
    ("0.6B", "GGUF", "research-models/Qwen3-0.6B-Q8_0.gguf"),
    ("0.6B", "vBuf", "research-models/Qwen3-0.6B-Q8_0.vbuf"),
    ("32B", "GGUF", "research-models/Qwen3-32B-Q8_0.gguf"),
    ("32B", "vBuf", "research-models/Qwen3-32B-Q8_0.vbuf"),
]
EXPECTED = {
    "research-models/Qwen3-0.6B-Q8_0.gguf": (639446688, "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"),
    "research-models/Qwen3-0.6B-Q8_0.vbuf": (637925504, "2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998"),
    "research-models/Qwen3-32B-Q8_0.gguf": (34817718912, "2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169"),
    "research-models/Qwen3-32B-Q8_0.vbuf": (34816197376, "84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d"),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def system_state(root: Path) -> dict:
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1); mem[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError): pass
    try: load = os.getloadavg()
    except OSError: load = None
    try: free = int(subprocess.check_output(["df", "-B1", "--output=avail", str(root)], text=True).splitlines()[-1])
    except (OSError, ValueError, subprocess.CalledProcessError): free = None
    return {"mem_available_bytes": mem.get("MemAvailable"), "mem_free_bytes": mem.get("MemFree"), "cached_bytes": mem.get("Cached"), "swap_total_bytes": mem.get("SwapTotal"), "swap_free_bytes": mem.get("SwapFree"), "load_average": load, "free_filesystem_bytes": free}

def evict(path: Path) -> str:
    flag = getattr(os, "POSIX_FADV_DONTNEED", None)
    if flag is None: return "unavailable"
    with path.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, flag)
    return "POSIX_FADV_DONTNEED whole-file advisory eviction"

def run_one(root: Path, binary: Path, model: str, fmt: str, relpath: str, cache: str, ordinal: int) -> dict:
    path = root / relpath
    eviction = evict(path) if cache == "uncached-approx" else "not-used"
    before = system_state(root)
    start = time.perf_counter()
    env = dict(os.environ)
    result = subprocess.run([str(binary), "gguf" if fmt == "GGUF" else "vbuf", str(path)], cwd=root, env=env, text=True, capture_output=True)
    wall_ms = (time.perf_counter() - start) * 1000
    if result.returncode != 0: raise RuntimeError(f"benchmark failed ({model} {fmt} {cache} #{ordinal}): rc={result.returncode}\n{result.stderr[-4000:]}")
    try: row = json.loads(result.stdout)
    except json.JSONDecodeError as error: raise RuntimeError(f"invalid benchmark JSON: {result.stdout[-1000:]}") from error
    row.update({"model": model, "format": fmt, "artifact": relpath, "cache_mode": cache, "sample": ordinal, "runner_wall_ms": wall_ms, "eviction": eviction, "system_before": before, "system_after": system_state(root)})
    return row

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--uncached-runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(); root = args.root.resolve(); binary = args.binary.resolve(); out = (args.output_dir or root / "benchmark-results/vbuf-ml-step27-loader-scaling").resolve(); (out / "raw").mkdir(parents=True, exist_ok=True)
    provenance=[]
    for _, _, rel in CASES:
        path=root/rel; expected_size, expected_hash=EXPECTED[rel]
        if not path.exists() or path.stat().st_size != expected_size or sha256(path) != expected_hash: raise SystemExit(f"artifact provenance mismatch: {rel}")
        provenance.append({"path":rel,"size":expected_size,"sha256":expected_hash})
    (out/"artifact-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    warm=[]; uncached=[]; run_order=[]
    # Deterministic alternation, with model-size alternation inside each round.
    for sample in range(args.runs):
        for model, fmt, rel in CASES:
            run_order.append({"cache":"warm","round":sample,"model":model,"format":fmt})
            row=run_one(root,binary,model,fmt,rel,"warm",sample); warm.append(row)
    for sample in range(args.uncached_runs):
        for model, fmt, rel in CASES:
            run_order.append({"cache":"uncached-approx","round":sample,"model":model,"format":fmt})
            row=run_one(root,binary,model,fmt,rel,"uncached-approx",sample); uncached.append(row)
    (out/"run-order.csv").write_text("cache,round,model,format\n"+"".join(f"{r['cache']},{r['round']},{r['model']},{r['format']}\n" for r in run_order))
    for name, rows in (("warm-raw.csv",warm),("uncached-approx-raw.csv",uncached)):
        fields=["model","format","artifact","cache_mode","sample","runner_wall_ms","model_construction_ms","first_eval_duration_ms","second_eval_ms","ttfuc_ms","ttft_ms","generation_complete_ms","prompt_tokens","first_token","generated_tokens"]
        with (out/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader()
            for row in rows: writer.writerow({key:json.dumps(row[key],separators=(",",":")) if key=="generated_tokens" else row[key] for key in fields})
    raw_path=out/"raw"/"runs.jsonl"
    with raw_path.open("w") as stream:
        for row in warm+uncached: stream.write(json.dumps(row,separators=(",",":"))+"\n")
    all_rows = warm + uncached
    def values(rows, key): return sorted(float(row[key]) for row in rows)
    def stats(rows, key):
        vals=values(rows,key); n=len(vals); return {"samples":n,"median":vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2,"p25":vals[int(.25*(n-1))],"p75":vals[int(.75*(n-1))],"min":vals[0],"max":vals[-1]}
    metric_rows=[]
    metric_keys={"MODEL_READY":"phases", "FIRST_EVAL":"first_eval_duration_ms", "TTFUC":"ttfuc_ms", "TTFT":"ttft_ms", "SECOND_EVAL":"second_eval_ms", "GENERATION":"generation_complete_ms"}
    for cache in ("warm","uncached-approx"):
        for model,fmt,_ in CASES:
            group=[r for r in all_rows if r["cache_mode"]==cache and r["model"]==model and r["format"]==fmt]
            for metric,key in metric_keys.items():
                vals=[float(r["phases"]["MODEL_READY"]["ms"]) if key=="phases" else float(r[key]) for r in group]
                n=len(vals); ordered=sorted(vals); metric_rows.append({"model":model,"format":fmt,"cache":cache,"metric":metric,"samples":n,"median_ms":(ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2),"p25_ms":ordered[int(.25*(n-1))],"p75_ms":ordered[int(.75*(n-1))],"min_ms":ordered[0],"max_ms":ordered[-1]})
    with (out/"phase-summary.csv").open("w",newline="") as stream:
        fields=list(metric_rows[0]); writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(metric_rows)
    boundary_names=["MODEL_READY","FIRST_EVAL_COMPLETE","SECOND_EVAL_COMPLETE","GENERATION_COMPLETE"]
    fault_rows=[]; io_rows=[]; memory_rows=[]; cpu_rows=[]
    for cache in ("warm","uncached-approx"):
        for model,fmt,_ in CASES:
            group=[r for r in all_rows if r["cache_mode"]==cache and r["model"]==model and r["format"]==fmt]
            for boundary in boundary_names:
                def phase_value(row,name): return row["phases"][name]
                for field, target in (("minor_faults",fault_rows),("major_faults",fault_rows),("read_bytes",io_rows),("rchar",io_rows),("syscr",io_rows),("rss_kb",memory_rows),("user_us",cpu_rows),("sys_us",cpu_rows)):
                    vals=[int(phase_value(r,boundary).get(field,-1)) for r in group]
                    target.append({"model":model,"format":fmt,"cache":cache,"boundary":boundary,"metric":field,"median":sorted(vals)[len(vals)//2] if len(vals)%2 else (sorted(vals)[len(vals)//2-1]+sorted(vals)[len(vals)//2])//2,"min":min(vals),"max":max(vals)})
    for name,rows in (("fault-summary.csv",fault_rows),("io-summary.csv",io_rows),("memory-summary.csv",memory_rows),("cpu-summary.csv",cpu_rows)):
        with (out/name).open("w",newline="") as stream:
            fields=list(rows[0]); writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    warm_map={(r["model"],r["format"]):r for r in metric_rows if r["cache"]=="warm" and r["metric"] in ("MODEL_READY","FIRST_EVAL","TTFUC","TTFT","SECOND_EVAL")}
    def warm_metric(model,fmt,metric): return next(r["median_ms"] for r in metric_rows if r["cache"]=="warm" and r["model"]==model and r["format"]==fmt and r["metric"]==metric)
    ratios={}
    for fmt in ("GGUF","vBuf"):
        ratios[fmt]={"large_over_small":{metric:warm_metric("32B",fmt,metric)/warm_metric("0.6B",fmt,metric) for metric in ("MODEL_READY","FIRST_EVAL","TTFUC","TTFT","SECOND_EVAL")},"large_minus_small_ms":{metric:warm_metric("32B",fmt,metric)-warm_metric("0.6B",fmt,metric) for metric in ("MODEL_READY","FIRST_EVAL","TTFUC","TTFT","SECOND_EVAL")}}
    ratios["vbuf_advantage"]={"small_ms":{metric:warm_metric("0.6B","GGUF",metric)-warm_metric("0.6B","vBuf",metric) for metric in ("MODEL_READY","FIRST_EVAL","TTFUC","TTFT","SECOND_EVAL")},"large_ms":{metric:warm_metric("32B","GGUF",metric)-warm_metric("32B","vBuf",metric) for metric in ("MODEL_READY","FIRST_EVAL","TTFUC","TTFT","SECOND_EVAL")},"small_relative":{metric:1-warm_metric("0.6B","vBuf",metric)/warm_metric("0.6B","GGUF",metric) for metric in ("MODEL_READY","FIRST_EVAL","TTFUC","TTFT","SECOND_EVAL")},"large_relative":{metric:1-warm_metric("32B","vBuf",metric)/warm_metric("32B","GGUF",metric) for metric in ("MODEL_READY","FIRST_EVAL","TTFUC","TTFT","SECOND_EVAL")}}
    (out/"scaling-ratios.json").write_text(json.dumps(ratios,indent=2)+"\n")
    sequences={}
    for model,fmt,_ in CASES:
        sequences[f"{model}-{fmt}"]=sorted({tuple(r["generated_tokens"]) for r in warm if r["model"]==model and r["format"]==fmt})
    correctness={"artifact_hashes_verified":True,"warm_generation_sequences_equal_by_model":all(len({tuple(r["generated_tokens"]) for r in warm if r["model"]==model})==1 for model,_,_ in CASES[::2]),"sequences":{k:[list(x) for x in v] for k,v in sequences.items()},"logit_comparison":"not repeated in Step-27 runner; Step-26 exact logit parity retained"}
    (out/"correctness-summary.json").write_text(json.dumps(correctness,indent=2)+"\n")
    (out/"qualification-config.json").write_text(json.dumps({"models":["Qwen3-0.6B-Q8_0","Qwen3-32B-Q8_0"],"formats":["GGUF","vBuf"],"cpu_only":True,"prompt":"Hello world","prompt_add_bos":False,"context":512,"batch":512,"threads":2,"deterministic_generation":True,"generation_tokens":8,"warm_runs":args.runs,"uncached_approx_runs":args.uncached_runs,"same_pinned_llama":True,"optimization":False},indent=2)+"\n")
    (out/"environment.json").write_text(json.dumps({"host":platform.node(),"python":platform.python_version(),"cpu_only":True,"gpu":"not involved","runs_warm":args.runs,"runs_uncached_approx":args.uncached_runs,"cache_method":"POSIX_FADV_DONTNEED advisory whole-file eviction; not cold-cache","run_order":"deterministic CASES alternation","binary":str(binary)},indent=2)+"\n")
    print(f"PASS — {len(warm)} warm and {len(uncached)} uncached-approx samples written to {out}")

if __name__ == "__main__": main()
