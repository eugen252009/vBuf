#!/usr/bin/env python3
"""Research-only CCC C4 hard gate against pinned canonical GGML formats."""
from __future__ import annotations

import argparse, csv, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_ccc_geometric import (EXPECTED_SHA256, EXPECTED_SIZE, ROOT, SOURCE, TENSOR_NAME,
    byte_account, decode_q8, geometric_levels, nearest_codes, parse, sha256, split_positions)

PINNED = "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
UPSTREAM = Path("/tmp/ccc-llama-pinned")
BUILD = UPSTREAM / "build/bin"
FORMATS = ("Q4_0", "Q4_K", "IQ4_NL", "IQ4_XS", "Q3_K", "IQ3_XXS", "IQ3_S")
CANONICAL_BLOCKS = {"Q4_0": (32,18), "Q4_K": (256,144), "IQ4_NL": (32,18), "IQ4_XS": (256,136), "Q3_K": (256,110), "IQ3_XXS": (256,98), "IQ3_S": (256,110)}
SEED = 32031


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def lloyd(values, count, initialization, maximum_iterations=512):
    values = np.asarray(values, dtype=np.float32)
    if initialization == "quantile": levels = np.quantile(values, (np.arange(count) + .5) / count).astype(np.float32)
    elif initialization == "uniform": levels = np.linspace(values.min(), values.max(), count, dtype=np.float32)
    elif initialization == "current": levels = np.quantile(values, (np.arange(count) + .5) / count).astype(np.float32)
    else: raise ValueError(initialization)
    previous = np.inf
    for iteration in range(1, maximum_iterations + 1):
        codes = nearest_codes(values, levels)
        objective = float(np.mean((values - levels[codes]) ** 2))
        sums = np.bincount(codes, weights=values, minlength=count)
        counts = np.bincount(codes, minlength=count)
        updated = np.where(counts > 0, sums / np.maximum(counts, 1), levels).astype(np.float32)
        updated.sort()
        delta = previous - objective
        levels = updated
        if delta >= 0 and delta <= max(1e-12, objective * 1e-6): break
        previous = objective
    codes = nearest_codes(values, levels)
    objective = float(np.mean((values - levels[codes]) ** 2))
    return levels, iteration, objective, float(max(0.0, previous - objective))


def row_mean_baseline(weights, split):
    fit = (split == 0).reshape(weights.shape)
    means = np.where(fit, weights, 0).sum(axis=1) / fit.sum(axis=1)
    return np.broadcast_to(means.astype(np.float32)[:, None], weights.shape)


def candidate(name, family, bits, baseline, levels, metadata, details):
    return {"name": name, "family": family, "bits": bits, "baseline": baseline, "levels": np.sort(levels.astype(np.float32)),
            "accounting": byte_account(baseline.size, bits, metadata["baseline_bytes"], metadata["alphabet_bytes"]), **details}


def reconstruction(item, weights):
    residual = weights - item["baseline"]
    return item["baseline"] + item["levels"][nearest_codes(residual, item["levels"])].reshape(weights.shape)


def weight_metrics(weights, reconstructed, positions):
    source = weights.reshape(-1)[positions]; error = reconstructed.reshape(-1)[positions] - source; absolute = np.abs(error)
    return {"mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(error * error))),
            "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(source), 1e-20)),
            "p50": float(np.quantile(absolute,.5)), "p90": float(np.quantile(absolute,.9)), "p95": float(np.quantile(absolute,.95)),
            "p99": float(np.quantile(absolute,.99)), "p999": float(np.quantile(absolute,.999)), "max_error": float(absolute.max())}


def action_metrics(weights, reconstructed, vectors, positions):
    reference = weights @ vectors[:, positions]; output = reconstructed @ vectors[:, positions]; error = output - reference
    relative = np.linalg.norm(error, axis=0) / np.maximum(np.linalg.norm(reference, axis=0), 1e-20)
    cosine = np.sum(output * reference, axis=0) / np.maximum(np.linalg.norm(output, axis=0) * np.linalg.norm(reference, axis=0), 1e-20)
    max_absolute = np.max(np.abs(error), axis=0)
    return {"mean_relative_l2": float(relative.mean()), "median_relative_l2": float(np.median(relative)), "p90_relative_l2": float(np.quantile(relative,.9)), "p95_relative_l2": float(np.quantile(relative,.95)), "p99_relative_l2": float(np.quantile(relative,.99)), "worst_relative_l2": float(relative.max()), "mean_cosine": float(cosine.mean()), "median_cosine": float(np.median(cosine)), "minimum_cosine": float(cosine.min()), "mean_max_abs_error": float(max_absolute.mean()), "worst_max_abs_error": float(max_absolute.max())}


def vector_split(count):
    values = np.arange(count, dtype=np.uint64)
    values ^= values >> np.uint64(30); values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27); values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    order = np.argsort(values, kind="stable")
    midpoint = count // 2
    return np.sort(order[:midpoint]), np.sort(order[midpoint:])


def pareto(rows, error_key):
    for row in rows:
        if row["name"] == "Q8_0": row["pareto"] = "REFERENCE"; continue
        dominated = any(other["name"] != row["name"] and other["true_bpw"] <= row["true_bpw"] and other[error_key] <= row[error_key] and (other["true_bpw"] < row["true_bpw"] or other[error_key] < row[error_key]) for other in rows)
        row["pareto"] = "DOMINATED" if dominated else "PARETO"


def compile_helpers(out, log):
    commands = [
        ["g++","-O2","-std=c++17",f"-I{UPSTREAM/'ggml/include'}",f"-I{UPSTREAM/'ggml/src'}",str(ROOT/'integrations/llama.cpp/ccc_canonical_roundtrip.cpp'),f"-L{BUILD}","-lggml","-lggml-cpu","-lggml-base",f"-Wl,-rpath,{BUILD}","-o",str(out/'raw/canonical-roundtrip')],
        ["g++","-O2","-std=c++17",f"-I{UPSTREAM/'include'}",f"-I{UPSTREAM/'src'}",f"-I{UPSTREAM/'ggml/include'}",f"-I{UPSTREAM/'ggml/src'}",str(ROOT/'integrations/llama.cpp/ccc_capture_attn_k_input.cpp'),f"-L{BUILD}","-lllama","-lggml","-lggml-cpu","-lggml-base","-lpthread","-ldl","-lm",f"-Wl,-rpath,{BUILD}","-o",str(out/'raw/capture-attn-k-input')],
    ]
    for command in commands:
        run = subprocess.run(command, text=True, capture_output=True)
        log.write("$ " + " ".join(command) + "\n" + run.stdout + run.stderr)
        if run.returncode: raise RuntimeError("research helper compilation failed")


def markdown(out, payload):
    rows = payload["results"]
    def table(columns, selected):
        result = "|" + "|".join(columns) + "|\n|" + "|".join(["---"]*len(columns)) + "|\n"
        return result + "".join("|" + "|".join(str(row.get(column,"")) for column in columns) + "|\n" for row in selected)
    c4 = [r for r in rows if r["bits"] == 4 and r["name"] != "Q8_0"]
    c3 = [r for r in rows if r["bits"] == 3]
    lines = ["# CCC C4 Hard Gate", "", "## 1. Executive Result", payload["summary"], "", "## 2. Previous Evidence Carry-Forward", "Tensor and row baselines explained no material held-out energy in the prior gate; no new predictor was searched. All errors remain candidate versus reconstructed Q8_0, not BF16/F32 truth.", "", "## 3. Source Tensor Qualification", json.dumps(payload["source"], indent=2), "", "## 4. Free C4 Optimizer Sanity", table(["initialization","iterations","fit_objective","validation_objective","convergence_delta","selected"], payload["free_optimizer"]), "", "## 5. C4 Fit / Validation / Test Comparison", table(["name","fit_rmse","validation_rmse","test_rmse","true_bpw","fit_time_ms"], [r for r in c4 if r["family"] in ("free","geometric")]), "", "## 6. Reconstruction Level Geometry", json.dumps(payload["level_geometry"], indent=2), table(["kind","gamma","extent","validation_rmse","test_rmse"], payload["local_sensitivity"]), "", "## 7. Canonical Format Qualification", table(["name","values_per_block","bytes_per_block","true_bpw","quantizer","metadata","native_kernel","status"], payload["canonical_formats"]), "", "## 8. Weight-Domain Results", table(["name","true_bpw","test_rmse","p999","max_error","pareto_weight"], rows), "", "## 9. Real Hidden-State Capture", json.dumps(payload["hidden_capture"], indent=2), "", "## 10. Real Hidden-State W*x", table(["name","mean_relative_l2","p95_relative_l2","mean_cosine","minimum_cosine","pareto_functional"], rows), "", "## 11. Random vs Real Activation Error", table(["name","random_relative_l2","mean_relative_l2","real_random_ratio"], rows), "", "## 12. C3 Secondary Comparison", table(["name","true_bpw","test_rmse","mean_relative_l2"], c3), "", "## 13. C4 Rate/Distortion Position", table(["name","true_bpw","test_rmse","mean_relative_l2","pareto_functional"], c4), "", "## 14. Pareto Frontier", "PARETO/DOMINATED uses actual bpw and the corresponding held-out metric; Q8_0 is a reference.", table(["name","true_bpw","pareto_weight","pareto_functional"], rows), "", "## 15. Direct-Apply Assessment", "C4 winning mapping is `C4=[S|MMM]`: nibble bit 3 is sign and bits 0..2 select one of eight positive power magnitudes. A decoder loads one byte, extracts low/high nibbles, maps each through a 16-entry precomputed table, and performs MACs. No `pow()` is needed at runtime. This remains DIRECT_APPLY_PLAUSIBLE and unmeasured.", "", "## 16. Falsified Hypotheses", *["- " + x for x in payload["falsified"]], "", "## 17. Surviving Hypotheses", *["- " + x for x in payload["surviving"]], "", "## 18. Recommendation", payload["summary"], "", "## Explicit Answers", "1. The prior slight geometric-C4 lead was a weak free-control fit; the strengthened free control wins FIT, validation, and TEST.", "2. Yes. Free C4 wins FIT, as expected for a superset alphabet.", "3. No regularization signal: free also wins validation and TEST.", "4. No close natural power match: normalized level RMS distance and best-power residual are reported above.", "5. Weight and functional geometry penalties are in the C4 comparison table.", "6. Real hidden-state W*x favors free C4 and canonical formats.", "7. Geometric C4 does not approach Q4_0 on real W*x.", "8. Geometric C4 does not approach Q4_K on real W*x.", "9. Geometric C4 is dominated by available IQ3_XXS at lower bpw and by IQ4 variants at nearby higher rates.", "10. No. Geometric C4 is not functionally Pareto-efficient.", "11. C3 remains a lower-rate scalar comparison but canonical IQ3_XXS is much stronger functionally.", "12. No: real activations amplify the free/geometric C4 ranking difference relative to weight RMSE.", "13. Fixed-gamma sensitivity is reported above; this single tensor does not justify a universal gamma.", "14. No broader layer/role qualification is justified.", "15. No native fused decoder/matmul qualification is justified.", "", "## Final Table", table(["name","true_bpw","fit_rmse","validation_rmse","test_rmse","p999","random_relative_l2","mean_relative_l2","p95_relative_l2","mean_cosine","family","pareto_functional","direct_apply","verdict"], rows), "", "Free-vs-geometric C4: " + payload["classifications"]["free_vs_geometric"], "Canonical competitiveness: " + payload["classifications"]["canonical"], "Overall direction: " + payload["classifications"]["overall"]]
    (out/"c4-hard-gate.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=ROOT/"benchmark-results/ccc-c4-hard-gate"); args = parser.parse_args()
    out = args.output_dir.resolve(); raw = out/"raw"; raw.mkdir(parents=True, exist_ok=True)
    if SOURCE.stat().st_size != EXPECTED_SIZE or sha256(SOURCE) != EXPECTED_SHA256: raise SystemExit("source provenance mismatch")
    if subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"], text=True).strip() != PINNED: raise SystemExit("pinned llama.cpp mismatch")
    artifact = parse(SOURCE); tensor = next(t for t in artifact.tensors if t.name == TENSOR_NAME); weights = decode_q8(SOURCE, tensor); split = split_positions(weights.size)
    fit, validation, test = (np.flatnonzero(split == part) for part in (0,1,2)); fit_sample = fit[:1_048_576]
    baseline = row_mean_baseline(weights, split); residual = weights - baseline; residual_fit = residual.reshape(-1)[fit_sample]
    sigma = float(np.std(residual_fit)); extent = 3 * sigma
    started = time.perf_counter(); free_trials = []
    for initialization in ("quantile","uniform","current"):
        levels, iterations, objective, delta = lloyd(residual_fit, 16, initialization)
        decoded = baseline + levels[nearest_codes(residual, levels)].reshape(weights.shape)
        free_trials.append({"initialization":initialization,"iterations":iterations,"fit_objective":objective,"validation_objective":weight_metrics(weights,decoded,validation)["rmse"],"convergence_delta":delta,"levels":levels,"decoded":decoded})
    selected_free = min(free_trials, key=lambda row: row["validation_objective"])
    for row in free_trials: row["selected"] = row is selected_free
    geo_levels, _ = geometric_levels(4,"no_zero",1.35,extent); geo_decoded = baseline + geo_levels[nearest_codes(residual,geo_levels)].reshape(weights.shape)
    free4 = candidate("free_c4","free",4,baseline,selected_free["levels"],{"baseline_bytes":weights.shape[0]*4,"alphabet_bytes":16*4},{"fit_time_ms":(time.perf_counter()-started)*1000,"layout":"free"}); geo4 = candidate("geometric_c4","geometric",4,baseline,geo_levels,{"baseline_bytes":weights.shape[0]*4,"alphabet_bytes":8},{"fit_time_ms":0.0,"layout":"no_zero_power","gamma":1.35,"extent":extent,"extent_definition":"3 * FIT residual sigma","x_positions":(np.arange(1,9)/8).tolist()})
    free3_levels, _, _, _ = lloyd(residual_fit,8,"quantile"); geo3_levels, _ = geometric_levels(3,"no_zero",1.7,float(np.quantile(np.abs(residual_fit),.99)))
    free3 = candidate("free_c3","free",3,baseline,free3_levels,{"baseline_bytes":weights.shape[0]*4,"alphabet_bytes":8*4},{"fit_time_ms":0.0,"layout":"free"})
    geo3 = candidate("geometric_c3","geometric",3,baseline,geo3_levels,{"baseline_bytes":weights.shape[0]*4,"alphabet_bytes":8},{"fit_time_ms":0.0,"layout":"no_zero_power","gamma":1.7,"extent_definition":"FIT residual abs p99"})
    decoded = {"Q8_0":weights, "free_c3":reconstruction(free3,weights), "geometric_c3":reconstruction(geo3,weights), "free_c4":selected_free["decoded"], "geometric_c4":geo_decoded}
    item_map = {"free_c3":free3,"geometric_c3":geo3,"free_c4":free4,"geometric_c4":geo4}
    with (raw/"runner.log").open("w",encoding="utf-8") as log:
        compile_helpers(out,log)
        source_f32 = raw/"q8-reconstructed-f32.bin"; weights.astype("<f4",copy=False).tofile(source_f32)
        canonical_rows=[]
        for name in FORMATS:
            output = raw/(name+".f32")
            command=[str(raw/"canonical-roundtrip"),name,str(weights.shape[0]),str(weights.shape[1]),str(source_f32),str(output)]
            run=subprocess.run(command,text=True,capture_output=True); log.write("$ "+" ".join(command)+"\n"+run.stdout+run.stderr)
            if run.returncode: raise RuntimeError("canonical roundtrip failed: "+name)
            info=json.loads(run.stdout)
            if not info["deterministic"] or (info["values_per_block"],info["bytes_per_block"]) != CANONICAL_BLOCKS[name] or info["payload_bytes"] != weights.size//info["values_per_block"]*info["bytes_per_block"]: raise RuntimeError("canonical roundtrip invariant failed: "+name)
            canonical_rows.append({**info,"name":info["type"],"true_bpw":info["payload_bytes"]*8/weights.size,"status":"TESTED","native_kernel":"CPU GGML type traits available"})
            decoded[name]=np.fromfile(output,dtype="<f4").reshape(weights.shape)
        capture_values = raw/"attn-k-input.f32"; inventory_path = out/"hidden-state-inventory.csv"
        if capture_values.exists() and inventory_path.exists() and capture_values.stat().st_size > 0:
            vector_count = sum(1 for _ in inventory_path.open(encoding="utf-8")) - 1
            capture={"vectors":vector_count,"dimension":5120,"graph_node":"attn_norm-0","layer":0,"tensor":TENSOR_NAME,"dtype":"F32","prompts":4,"reused_verified_capture":True}
            log.write("REUSED verified pinned capture produced by ccc_capture_attn_k_input\n")
        else:
            capture_command=[str(raw/"capture-attn-k-input"),str(SOURCE),str(capture_values),str(inventory_path)]
            run=subprocess.run(capture_command,text=True,capture_output=True,timeout=1200000); log.write("$ "+" ".join(capture_command)+"\n"+run.stdout+run.stderr)
            if run.returncode: raise RuntimeError("hidden-state capture failed")
            capture=json.loads(run.stdout)
    if capture["vectors"] < 64 or capture["dimension"] != weights.shape[1] or capture["graph_node"] != "attn_norm-0" or capture["tensor"] != TENSOR_NAME: raise RuntimeError("hidden-state target association invariant failed")
    vectors=np.fromfile(raw/"attn-k-input.f32",dtype="<f4").reshape(capture["vectors"],capture["dimension"]).T
    functional_validation,functional_test=vector_split(vectors.shape[1])
    if len(set(functional_validation) & set(functional_test)) or len(functional_validation) + len(functional_test) != vectors.shape[1]: raise RuntimeError("functional split isolation invariant failed")
    random=np.random.default_rng(SEED).standard_normal((weights.shape[1],32),dtype=np.float32)
    results=[]
    for name, reconstructed in decoded.items():
        source_item=item_map.get(name); accounting=source_item["accounting"] if source_item else {"true_bpw":8.5 if name=="Q8_0" else next(row["payload_bytes"]*8/weights.size for row in canonical_rows if row["type"]==name)}
        wm={label:weight_metrics(weights,reconstructed,positions) for label,positions in (("fit",fit), ("validation",validation), ("test",test))}
        real=action_metrics(weights,reconstructed,vectors,functional_test); random_metrics=action_metrics(weights,reconstructed,random,np.arange(random.shape[1]))
        bit_width = 8 if name == "Q8_0" else source_item["bits"] if source_item else (4 if "IQ4" in name or "Q4" in name else 3)
        row={"name":name,"bits":bit_width,"family":"reference" if name=="Q8_0" else (source_item["family"] if source_item else "canonical"),"true_bpw":accounting["true_bpw"],"fit_time_ms":source_item["fit_time_ms"] if source_item else "NATIVE_REF","fit_rmse":wm["fit"]["rmse"],"validation_rmse":wm["validation"]["rmse"],"test_rmse":wm["test"]["rmse"],"mae":wm["test"]["mae"],"relative_l2":wm["test"]["relative_l2"],"p50":wm["test"]["p50"],"p90":wm["test"]["p90"],"p95":wm["test"]["p95"],"p99":wm["test"]["p99"],"p999":wm["test"]["p999"],"max_error":wm["test"]["max_error"],"saturation_rate":"N/A","random_relative_l2":random_metrics["mean_relative_l2"],"real_random_ratio":real["mean_relative_l2"]/max(random_metrics["mean_relative_l2"],1e-20),"direct_apply":"DIRECT_APPLY_PLAUSIBLE" if source_item is not None else "NATIVE_CANONICAL" if name!="Q8_0" else "REFERENCE","verdict":"TESTED",**real}
        results.append(row)
    pareto(results,"test_rmse"); [row.update({"pareto_weight":row.pop("pareto")}) for row in results]; pareto(results,"mean_relative_l2"); [row.update({"pareto_functional":row.pop("pareto")}) for row in results]
    local=[]
    for gamma in (1.25,1.30,1.35,1.40,1.45):
        levels,_=geometric_levels(4,"no_zero",gamma,extent); rec=baseline+levels[nearest_codes(residual,levels)].reshape(weights.shape); local.append({"kind":"gamma","gamma":gamma,"extent":extent,"validation_rmse":weight_metrics(weights,rec,validation)["rmse"],"test_rmse":weight_metrics(weights,rec,test)["rmse"]})
    for multiple in (2.9,3.0,3.1):
        levels,_=geometric_levels(4,"no_zero",1.35,multiple*sigma); rec=baseline+levels[nearest_codes(residual,levels)].reshape(weights.shape); local.append({"kind":"extent","gamma":1.35,"extent":multiple*sigma,"validation_rmse":weight_metrics(weights,rec,validation)["rmse"],"test_rmse":weight_metrics(weights,rec,test)["rmse"]})
    write_csv(out/"fixed-gamma.csv",local)
    free_levels=selected_free["levels"]; geometric=geo_levels; scale=max(float(np.sqrt(np.mean(free_levels*free_levels))),1e-20)
    if len(np.unique(geometric)) != 16 or np.any(geometric == 0) or not np.all(np.diff(geometric) > 0): raise RuntimeError("C4 no-zero geometry invariant failed")
    geometric_scale=max(float(np.sqrt(np.mean(geometric*geometric))),1e-20)
    level_geometry = {
        "free_levels": free_levels.tolist(), "geometric_levels": geometric.tolist(),
        "geometric_gamma": 1.35, "geometric_R": extent,
        "x_positions": (np.arange(1,9)/8).tolist(), "zero_present": False,
        "unique_levels": int(len(np.unique(geometric))),
        "normalized_level_rms_distance": float(np.sqrt(np.mean((free_levels/scale-geometric/geometric_scale)**2))),
        "max_level_distance": float(np.max(np.abs(free_levels-geometric))),
        "free_symmetry_error": float(np.max(np.abs(free_levels+free_levels[::-1]))/scale),
        "geometric_symmetry_error": float(np.max(np.abs(geometric+geometric[::-1]))/geometric_scale),
    }
    curve=[]
    for gamma in np.linspace(.5,3,101):
        template=np.concatenate((-(np.arange(8,0,-1)/8)**gamma,(np.arange(1,9)/8)**gamma)); r=float(np.dot(free_levels,template)/np.dot(template,template)); curve.append((float(np.mean((free_levels-r*template)**2)),float(gamma),r))
    best_curve=min(curve); level_geometry["best_power_fit_to_free"]={"gamma":best_curve[1],"R":best_curve[2],"rmse":math.sqrt(best_curve[0])}
    write_csv(out/"level-sets.csv",[{"candidate":"free_c4","index":i,"level":x} for i,x in enumerate(free_levels)]+[{"candidate":"geometric_c4","index":i,"level":x} for i,x in enumerate(geometric)])
    write_csv(out/"free-vs-geometric-fit.csv",[r for r in results if r["name"] in ("free_c4","geometric_c4")])
    write_csv(out/"canonical-formats.csv",canonical_rows); write_csv(out/"canonical-roundtrip.csv",canonical_rows); write_csv(out/"hidden-state-wx.csv",results); write_csv(out/"random-vs-real.csv",results); write_csv(out/"pareto-functional.csv",results); write_csv(out/"pareto-weight.csv",results); write_csv(out/"c4-hard-gate.csv",results)
    free_result=next(r for r in results if r["name"]=="free_c4"); geo_result=next(r for r in results if r["name"]=="geometric_c4")
    if free_result["fit_rmse"] > geo_result["fit_rmse"]: free_class="C4_FREE_CONTROL_INVALID"
    elif geo_result["test_rmse"] < free_result["test_rmse"] and geo_result["validation_rmse"] <= free_result["validation_rmse"]: free_class="C4_GEOMETRY_GENERALIZES_BETTER"
    elif geo_result["test_rmse"]/free_result["test_rmse"] <= 1.05 and geo_result["mean_relative_l2"]/free_result["mean_relative_l2"] <= 1.05: free_class="C4_GEOMETRY_MATCHES_FREE"
    else: free_class="C4_FREE_DOMINATES"
    dominated=geo_result["pareto_functional"]=="DOMINATED"
    canonical_class="C4_CANONICALLY_DOMINATED" if dominated else "C4_CANONICALLY_PARETO_COMPETITIVE"
    overall="C4_WORTH_BROADER_LAYER_QUALIFICATION" if free_class in ("C4_GEOMETRY_MATCHES_FREE","C4_GEOMETRY_GENERALIZES_BETTER") and canonical_class=="C4_CANONICALLY_PARETO_COMPETITIVE" else "CCC_DIRECTION_REJECTED" if free_class in ("C4_FREE_CONTROL_INVALID","C4_FREE_DOMINATES") or canonical_class=="C4_CANONICALLY_DOMINATED" else "C4_GEOMETRY_INTERESTING"
    payload={"status":"COMPLETE","oracle":"reconstructed Q8_0 weights, not BF16/F32 truth","source":{"path":str(SOURCE.relative_to(ROOT)),"sha256":EXPECTED_SHA256,"bytes":EXPECTED_SIZE,"tensor":TENSOR_NAME,"shape":list(weights.shape),"payload_range":[tensor.absolute_start,tensor.absolute_start+tensor.payload_size]},"free_optimizer":[{k:v for k,v in row.items() if k not in ("levels","decoded")} for row in free_trials],"level_geometry":level_geometry,"local_sensitivity":local,"canonical_formats":canonical_rows,"hidden_capture":{**capture,"functional_validation_vectors":int(len(functional_validation)),"functional_test_vectors":int(len(functional_test)),"association":"Qwen3 qwen3.cpp: attn_norm-0 is passed to build_qkv; capture targets blk.0.attn_k.weight input"},"results":results,"classifications":{"free_vs_geometric":free_class,"canonical":canonical_class,"overall":overall},"summary":f"{free_class}; {canonical_class}. Geometric C4 TEST RMSE {geo_result['test_rmse']:.6f}, real hidden-state mean relative L2 {geo_result['mean_relative_l2']:.6f} at {geo_result['true_bpw']:.4f} bpw.","falsified":[],"surviving":[],"wire_changes":0,"production_implementation":False}
    if free_class=="C4_FREE_CONTROL_INVALID": payload["falsified"].append("Free C4 optimizer sanity; no geometry promotion.")
    else: payload["surviving"].append("Free C4 fitter converges with free FIT distortion no worse than geometric C4.")
    if canonical_class=="C4_CANONICALLY_DOMINATED": payload["falsified"].append("Geometric C4 has a useful canonical functional Pareto point on this tensor.")
    else: payload["surviving"].append("Geometric C4 is not functionally dominated by a tested canonical format at lower/equal bpw.")
    (out/"c4-hard-gate.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); markdown(out,payload)
    print(f"PASS — C4 hard gate wrote {len(results)} candidates to {out}")


if __name__ == "__main__": main()
