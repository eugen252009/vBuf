#!/usr/bin/env python3
"""Step-26 Qwen3-32B semantic and deterministic placement qualification."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_step18_manifest import (BOOTSTRAP_KEY_ID, CONTROL_KEY_IDS, PLACEMENT, TENSOR_KEY_ID,
    metadata_plan, tensor_plans, tokenizer_conversion_plan, tokenizer_plan, validate_manifest)
from qualify_step16 import layout_order, parse, layer_count
from qualify_step17 import PINNED_COMMIT

METADATA_IDS = {"Architecture": 1, "ContextLength": 2, "EmbeddingLength": 3, "LayerCount": 4,
    "HeadCount": 5, "FeedForwardLength": 6, "NormalizationEpsilon": 7, "RopeTheta": 8,
    "KVHeadCount": 9, "KeyHeadDimension": 10, "ValueHeadDimension": 11}
ROLE_IDS = {"TokenTextBytes": 1, "TokenOffsets": 2, "TokenScores": 3, "TokenTypes": 4,
    "BosId": 5, "EosId": 6, "UnkId": 7, "PadId": 8, "MergeLeftIds": 9,
    "MergeRightIds": 10, "TokenizerModelIdentity": 11, "PreTokenizerIdentity": 12,
    "AddBos": 13, "ChatTemplate": 14}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def align(value: int, step: int) -> int: return (value + step - 1) // step * step


def directory_bytes(artifact, plans) -> int:
    return 20 + sum(10 + len(row["target_name"].encode()) + 8 * len(row["source_shape"]) for row in plans)


def metadata_bytes(artifact) -> tuple[int, list[tuple[int, int, str]]]:
    rows = [row for row in metadata_plan(artifact) if row["target"] in METADATA_IDS]
    values=[]
    for row in rows:
        value=row["source_value"]
        if row["target"] == "Architecture": n=len(str(value).encode()); kind="opaque"
        else: n=8; kind="float" if row["target"] in {"NormalizationEpsilon","RopeTheta"} else "u64"
        values.append((n, kind, row["target"]))
    return 16 + 8 * len(rows), values


def build_manifest(artifact, source: Path) -> dict:
    plans = tensor_plans(artifact)
    metadata = metadata_plan(artifact)
    tokenizer = tokenizer_plan(artifact)
    conversion = tokenizer_conversion_plan(artifact)
    by_name={t.name:t for t in artifact.tensors}
    specials=["tokenizer.ggml.bos_token_id","tokenizer.ggml.eos_token_id","tokenizer.ggml.padding_token_id"]
    manifest={
        "manifest_version": 1,
        "source_identity": {"path": str(source), "filename": source.name, "size": artifact.size,
            "sha256": artifact.sha256, "gguf_version": artifact.version, "architecture": "qwen3",
            "metadata_kv_count": artifact.metadata_count, "tensor_count": artifact.tensor_count},
        "consumer_revision": {"repository": "https://github.com/ggml-org/llama.cpp.git", "commit": PINNED_COMMIT},
        "profile_version": "vbuf-ml-0.1",
        "conversion_options": {"placement": PLACEMENT, "base_shift": 3,
            "integrity": {"enabled": False, "algorithm": "SHA-256", "coverage": "payload bytes"}},
        "control_plan": [{"role":"Bootstrap","target_key_id":BOOTSTRAP_KEY_ID,"action":"DERIVED"}]+[
            {"role":role,"target_key_id":key,"action":"DERIVED"} for role,key in CONTROL_KEY_IDS.items() if role != "IntegrityMetadata"],
        "model_metadata_plan": metadata, "tokenizer_plan": tokenizer,
        "tokenizer_conversion_plan": conversion,
        "tensor_directory_plan": [{"name":p["target_name"],"shape":p["source_shape"],"representation":p["target_representation"],"key_id":p["target_key_id"],"occurrence":p["target_occurrence"],"directory_order":i} for i,p in enumerate(sorted(plans,key=lambda x:x["target_name"]))],
        "tensor_plans": plans, "shared_reference_plan": None,
        "output_decision": {"source_descriptors_distinct": "output.weight" in by_name and "token_embd.weight" in by_name,
            "action": "COPY_BYTES independently" if "output.weight" in by_name else "ABSENT"},
        "placement_plan": {"policy": PLACEMENT, "target_order_is_non_normative": True},
        "accounting": {"source_tensors":len(plans),"copy_bytes":len(plans),"shared_reference":0,"derived":len(CONTROL_KEY_IDS)-1,"reject":0,"unaccounted":0},
        "validation": {"manifest_structurally_valid":True,"conversion_readiness":["READY_FOR_CONVERSION"],"raw_inference_tokenizer_readiness":"READY","chat_template_readiness":"READY_FOR_CONSUMER_EXECUTION","metadata_blockers":[],"tokenizer_blockers":[]},
    }
    validate_manifest(manifest, artifact)
    return manifest


def requests(artifact, manifest):
    tokens=artifact.metadata["tokenizer.ggml.tokens"]; types=artifact.metadata["tokenizer.ggml.token_type"]; merges=artifact.metadata["tokenizer.ggml.merges"]
    md_control, md_values=metadata_bytes(artifact)
    plans=manifest["tensor_plans"]
    entries=[]
    entries.append({"name":"Bootstrap","class":0,"order":0,"payload":16+16*3,"group":"control"})
    entries.append({"name":"ModelMetadataControl","class":1,"order":1,"payload":md_control,"group":"control"})
    for n,kind,target in md_values: entries.append({"name":f"Metadata.{target}","class":1,"order":100+METADATA_IDS[target],"payload":n,"group":"control"})
    entries.append({"name":"TensorDirectory","class":2,"order":2,"payload":directory_bytes(artifact,plans),"group":"control"})
    tok_entries=8+1+3
    entries.append({"name":"TokenizerControl","class":3,"order":3,"payload":20+12*tok_entries,"group":"control"})
    text_bytes=sum(len(t.encode()) for t in tokens)
    payloads=[("TokenTextBytes",200,text_bytes,"tokenizer"),("TokenOffsets",201,8*(len(tokens)+1),"tokenizer"),("TokenTypes",202,4*len(types),"tokenizer"),("MergeLeftIds",203,4*len(merges),"tokenizer"),("MergeRightIds",204,4*len(merges),"tokenizer"),("TokenizerModelIdentity",205,1,"tokenizer"),("PreTokenizerIdentity",206,1,"tokenizer"),("AddBos",207,1,"tokenizer"),("ChatTemplate",208,len(artifact.metadata["tokenizer.chat_template"].encode()),"tokenizer")]
    payloads += [(f"Special.{role}",220+ROLE_IDS[role],8,"tokenizer") for role in ("BosId","EosId","PadId")]
    for name,order,size,group in payloads: entries.append({"name":name,"class":4,"order":order,"payload":size,"group":group})
    for p in plans: entries.append({"name":p["target_name"],"class":5,"order":1_000_000+p["target_order"],"payload":p["source_payload_bytes"],"group":"tensor","layer":p["layer"]})
    return sorted(entries,key=lambda x:(x["class"],x["order"]))


def plan_candidate(entries, base_shift):
    step=1<<base_shift; cursor=align(24,step); rows=[]; payload=0; headers=0
    for i,e in enumerate(entries):
        block_start=align(cursor,step); block_pad=block_start-cursor
        extended=e["payload"] > 65535; header=16 if extended else 8
        payload_start=align(block_start+header,step); inner_pad=payload_start-(block_start+header)
        end=payload_start+e["payload"]
        row={**e,"index":i,"block_start":block_start,"payload_start":payload_start,"payload_end":end,"block_padding":block_pad,"header_bytes":header,"inner_padding":inner_pad,"extended":extended}
        rows.append(row); cursor=end; payload+=e["payload"]; headers+=header
    final_size=cursor; data_start=align(24,step); padding=final_size-24-payload-headers
    layer_rows=defaultdict(list)
    for r in rows:
        if r.get("group")=="tensor" and r.get("layer") is not None: layer_rows[int(r["layer"])].append(r)
    layer_metrics=[]; gaps=[]
    for layer,rs in sorted(layer_rows.items()):
        spans=1
        for a,b in zip(rs,rs[1:]):
            gap=b["block_start"]-a["payload_end"]
            if gap: spans+=1; gaps.append(gap)
        layer_metrics.append((layer,spans,rs[-1]["payload_end"]-rs[0]["block_start"],sum(x["payload"] for x in rs)))
    runs=0; covered=0; useful=sum(r["payload"] for r in rows); run_start=None; prev_end=None
    for r in rows:
        if run_start is None or r["block_start"] != prev_end:
            if run_start is not None: covered += prev_end-run_start
            runs+=1; run_start=r["block_start"]
        prev_end=r["payload_end"]
    if run_start is not None: covered += prev_end-run_start
    data_size=final_size-data_start; slots=(data_size+step-1)//step; nano=(slots+7)//8
    return {"base_shift":base_shift,"base_step":step,"entries":rows,"final_size":final_size,"data_start":data_start,"payload_bytes":payload,"control_metadata_bytes":sum(r["payload"] for r in rows if r["group"]!="tensor"),"header_bytes":headers,"padding_bytes":padding,"padding_pct":padding/max(1,final_size)*100,"block_count":len(rows),"nano_slots":slots,"nano_bytes":nano,"nano_set_bits":len(rows),"nano_density":len(rows)/max(1,slots),"layer_count":len(layer_metrics),"layer_span_count":sum(x[1] for x in layer_metrics),"avg_spans_per_layer":sum(x[1] for x in layer_metrics)/max(1,len(layer_metrics)),"max_spans_per_layer":max(x[1] for x in layer_metrics),"internal_layer_gap_bytes":sum(gaps),"average_layer_gap":sum(gaps)/max(1,len(gaps)),"maximum_layer_gap":max(gaps,default=0),"coalesced_span_count":runs,"useful_bytes":useful,"covered_bytes":covered,"bytes_amplification":covered/max(1,useful),"tensor_count":sum(r["group"]=="tensor" for r in rows),"layer_metrics":layer_metrics}


def pareto(candidates):
    keys=("final_size","padding_bytes","nano_bytes","layer_span_count","bytes_amplification")
    result=[]
    for a in candidates:
        dominated=False
        for b in candidates:
            if a is b: continue
            if all(b[k] <= a[k] for k in keys) and any(b[k] < a[k] for k in keys): dominated=True; break
        if not dominated: result.append(a["base_shift"])
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument("--source",type=Path); ap.add_argument("--output-dir",type=Path); args=ap.parse_args(); root=args.root.resolve(); source=(args.source or root/"research-models/Qwen3-32B-Q8_0.gguf").resolve(); out=(args.output_dir or root/"benchmark-results/vbuf-ml-step26-qwen32b-placement").resolve(); out.mkdir(parents=True,exist_ok=True); raw=out/"raw"; raw.mkdir(exist_ok=True)
    started=time.perf_counter(); artifact=parse(source); parse_time=(time.perf_counter()-started)*1000
    if artifact.metadata.get("general.architecture") != "qwen3": raise SystemExit("not dense Qwen3")
    if str(artifact.metadata.get("general.type")) != "model": raise SystemExit("unexpected model type")
    layers=layer_count(artifact); layer_ids=sorted({t.layer for t in artifact.tensors if t.layer is not None});
    if layers != 64 or layer_ids != list(range(layers)): raise SystemExit(f"unexpected layer geometry: {layers} {layer_ids[:3]}..{layer_ids[-3:]}")
    if any(t.group=="Unknown" or t.type_name not in {"Q8_0","F32"} for t in artifact.tensors): raise SystemExit("unknown tensor classification/representation")
    manifest=build_manifest(artifact,source)
    entries=requests(artifact,manifest); candidates=[]
    for bs in range(3,9):
        started=time.perf_counter(); c=plan_candidate(entries,bs); c["planning_ms"]=(time.perf_counter()-started)*1000; candidates.append(c)
    selected=min((c for c in candidates if c["base_shift"] in pareto(candidates)),key=lambda c:(c["final_size"],c["padding_bytes"],c["base_step"]))
    manifest["conversion_options"]["base_shift"] = selected["base_shift"]
    manifest["placement_plan"] = {"policy": PLACEMENT, "base_shift": selected["base_shift"], "base_step": selected["base_step"], "final_size": selected["final_size"], "entries": [{k:r[k] for k in ("name","class","order","block_start","payload_start","payload_end","payload","block_padding","inner_padding")} for r in selected["entries"]]}
    (out/"qwen3-32b-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    rows=[]; layer_rows=[]
    for c in candidates:
        rows.append({k:c[k] for k in ("base_shift","base_step","final_size","payload_bytes","control_metadata_bytes","header_bytes","padding_bytes","padding_pct","block_count","nano_slots","nano_bytes","nano_set_bits","nano_density","tensor_count","layer_count","layer_span_count","avg_spans_per_layer","max_spans_per_layer","internal_layer_gap_bytes","average_layer_gap","maximum_layer_gap","coalesced_span_count","useful_bytes","covered_bytes","bytes_amplification","planning_ms")})
        layer_rows += [{"base_shift":c["base_shift"],"layer":layer,"spans":spans,"physical_span_bytes":span_bytes,"useful_bytes":useful} for layer,spans,span_bytes,useful in c["layer_metrics"]]
    import csv
    def write_csv(path, data):
        with path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write_csv(out/"candidate-layouts.csv",rows); write_csv(out/"candidate-layer-geometry.csv",layer_rows)
    (out/"nano-hypothetical.csv").write_text("base_shift,base_step,bitmap_slots,bitmap_bytes,set_bits,density,percent_final_size\n"+"".join(f"{c['base_shift']},{c['base_step']},{c['nano_slots']},{c['nano_bytes']},{c['nano_set_bits']},{c['nano_density']},{c['nano_bytes']/c['final_size']*100}\n" for c in candidates))
    pareto_set=pareto(candidates); (out/"pareto-analysis.json").write_text(json.dumps({"metrics":["final_size","padding_bytes","nano_bytes","layer_span_count","bytes_amplification"],"non_dominated_base_shifts":pareto_set,"dominated_base_shifts":[c["base_shift"] for c in candidates if c["base_shift"] not in pareto_set]},indent=2)+"\n")
    (out/"selection.json").write_text(json.dumps({"selected_base_shift":selected["base_shift"],"selected_base_step":selected["base_step"],"policy":"Pareto filter; smallest final size, then padding, then BaseStep","artifact_qualified":True,"non_dominated_base_shifts":pareto_set},indent=2)+"\n")
    (out/"artifact-provenance.json").write_text(json.dumps({"path":str(source),"size":artifact.size,"sha256":artifact.sha256,"gguf_version":artifact.version,"architecture":artifact.metadata.get("general.architecture"),"dense":True,"tensor_count":artifact.tensor_count,"layer_count":layers,"vocabulary_size":len(artifact.metadata["tokenizer.ggml.tokens"]),"merge_count":len(artifact.metadata["tokenizer.ggml.merges"]),"metadata_count":artifact.metadata_count,"alignment":artifact.alignment,"parse_ms":parse_time},indent=2)+"\n")
    (out/"gguf-metadata.json").write_text(json.dumps(artifact.metadata,indent=2,default=str)+"\n")
    inv="tensor_name,ordinal,shape,type,type_id,payload_bytes,layer,group,role\n"+"".join(f"{t.name},{t.ordinal},\"{list(t.shape)}\",{t.type_name},{t.type_id},{t.payload_size},{t.layer},{t.group},{t.role_detail}\n" for t in artifact.tensors)
    (out/"tensor-inventory.csv").write_text(inv)
    (out/"planner-timing.csv").write_text("phase,ms\nlogical_parse_and_manifest,"+str(parse_time)+"\nall_six_plans,"+str(sum(c["planning_ms"] for c in candidates))+"\nselection,0\n")
    (out/"qualification-config.json").write_text(json.dumps({"source":source.name,"base_shifts":list(range(3,9)),"selected":selected["base_shift"],"payload_duplication":0,"wire_change":False,"physical_runtime_work":False},indent=2)+"\n")
    print(f"PASS — Step-26 semantic/planner qualification written to {out}; selected BaseShift={selected['base_shift']}")

if __name__=="__main__": main()
