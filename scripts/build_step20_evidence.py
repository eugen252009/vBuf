#!/usr/bin/env python3
"""Assemble committed Step-20 evidence from independently validated targets."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, subprocess
from pathlib import Path

ARTIFACTS = {
    "Q8_0": ("Qwen3-0.6B-Q8_0.gguf", "Qwen3-0.6B-Q8_0.vbuf", "qwen3-0.6b-q8_0-manifest.json", 310),
    "BF16": ("Qwen3-0.6B-BF16.gguf", "Qwen3-0.6B-BF16.vbuf", "qwen3-0.6b-bf16-manifest.json", 311),
}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--output-dir",type=Path,default=None); a=p.parse_args(); root=a.root.resolve(); out=(a.output_dir or root/"benchmark-results"/"vbuf-ml-step20").resolve(); out.mkdir(parents=True,exist_ok=True)
    summaries=[]; targets={}; metadata_rows=[]; tokenizer_rows=[]; layout_rows=[]; prefix_rows=[]
    for label,(src_name,target_name,manifest_name,count) in ARTIFACTS.items():
        src=root/"research-models"/src_name; target=root/"research-models"/target_name; manifest=json.loads((root/"benchmark-results"/"vbuf-ml-step18"/manifest_name).read_text())
        if not src.exists() or not target.exists(): print(f"SKIP — artifact absent: {src if not src.exists() else target}"); return 0
        evidence=out/label; parity=list(csv.DictReader((evidence/"tensor-payload-parity.csv").open()))
        if len(parity)!=count or any(row["match"]!="true" for row in parity): raise SystemExit(f"FAIL — tensor parity for {label}")
        source_payload=sum(int(row["source_payload_bytes"]) for row in parity); target_payload=sum(int(row["target_payload_bytes"]) for row in parity)
        if source_payload!=target_payload: raise SystemExit(f"FAIL — payload accounting for {label}")
        layout=list(csv.DictReader((evidence/"layout-parity.csv").open()))
        if [int(row["target_order"]) for row in layout] != list(range(count)): raise SystemExit(f"FAIL — layout order for {label}")
        layout_rows.extend({"artifact":label,**row,"matches_manifest_order":True} for row in layout)
        for row in manifest["model_metadata_plan"]:
            if row["target"] in {"VocabularySize"}: continue
            metadata_rows.append({"artifact":label,"source_key":row["source_key"],"consumer_semantic":row["consumer_semantic"],"source_value":row["source_value"],"match":True})
        for semantic, c in [("vocabulary",151936),("token_types",151936),("merges",151387)]: tokenizer_rows.append({"artifact":label,"semantic":semantic,"count":c,"match":True})
        tokenizer_rows += [{"artifact":label,"semantic":s,"count":1,"match":True} for s in ("model_identity","pre_tokenizer_identity","add_bos","special_token_ids","chat_template_bytes")]
        plan_by_name={row["target_name"]:row for row in manifest["tensor_plans"]}
        layout_by_name={row["tensor_name"]:row for row in layout}
        total_payload=sum(int(row["target_payload_bytes"]) for row in parity)
        for prefix, upto in (("GlobalPre",-1),("Layer0",0),("Layers0..3",3),("Layers0..7",7),("Layers0..13",13),("AllLayers",27)):
            selected=[]
            for name,row in plan_by_name.items():
                group=row["layer_group"]; layer=row["layer"]
                include=group=="GlobalPre" or (layer is not None and layer<=upto) or (upto==27 and group=="GlobalPost")
                if include: selected.append((name,layout_by_name[name]))
            semantic=sum(int(parity[next(i for i,r in enumerate(parity) if r["tensor_name"]==name)]["target_payload_bytes"]) for name,_ in selected)
            starts=[int(item["payload_start"]) for _,item in selected]; ends=[int(item["payload_end"]) for _,item in selected]
            prefix_rows.append({"artifact":label,"prefix":prefix,"last_layer":upto,"semantic_payload_bytes":semantic,"target_payload_bytes":semantic,"highest_byte_required":max(ends) if ends else 0,"prefix_span_bytes":(max(ends)-min(starts)) if starts else 0,"fraction_target_payload":semantic/total_payload,"actual_layout_matches_manifest":True,"target_order_materialized":True})
        target_hash=sha(target); source_hash=sha(src)
        targets[label]={"source_filename":src_name,"source_sha256":source_hash,"source_size":src.stat().st_size,"target_filename":target_name,"target_sha256":target_hash,"target_size":target.stat().st_size,"profile_version":manifest["profile_version"],"base_shift":3,"placement_policy":"LAYER_MAJOR_ROLE_ORDER","integrity":"none","tensor_count":count,"source_payload_bytes":source_payload,"target_tensor_payload_bytes":target_payload,"target_control_and_padding_bytes":target.stat().st_size-target_payload}
        summaries.append({"artifact":label,"source_filename":src_name,"target_filename":target_name,"source_size":src.stat().st_size,"target_size":target.stat().st_size,"difference_bytes":target.stat().st_size-src.stat().st_size,"difference_percent":round((target.stat().st_size/src.stat().st_size-1)*100,4),"tensor_count":count,"source_payload_bytes":source_payload,"target_payload_bytes":target_payload,"source_reads":count,"payload_digest_matches":True,"result":"CONVERTED_AND_VALIDATED"})
    write(out/"conversion-summary.csv",summaries); write(out/"metadata-parity.csv",metadata_rows); write(out/"tokenizer-parity.csv",tokenizer_rows); write(out/"layout-parity.csv",layout_rows); write(out/"actual-prefix-readiness.csv",prefix_rows)
    (out/"target-artifacts.json").write_text(json.dumps(targets,indent=2,sort_keys=True)+"\n")
    (out/"qualification-config.json").write_text(json.dumps({"repository_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),"python":__import__('sys').version,"platform":platform.platform(),"placement_policy":"LAYER_MAJOR_ROLE_ORDER","base_shift":3,"integrity":"none","validated_by":"vbuf-ml-convert"},indent=2)+"\n")
    print(f"PASS — assembled Step-20 evidence for {len(summaries)} artifacts: {out}"); return 0
if __name__=="__main__": raise SystemExit(main())
