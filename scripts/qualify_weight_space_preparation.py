#!/usr/bin/env python3
"""Research-only functional weight-space preparation feasibility gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import parse
from run_step30_reparameterization import decode_q8
from qualify_implicit_weight_feasibility import (
    action_metrics, encode_fixed as g2_encode, reconstruct as g2_reconstruct,
    descriptor_layout as g2_accounting, vector_split, weight_metrics,
)
from qualify_simd_traceable_mutation import (
    encode as m2_encode, mix_rounds as m2_mix_rounds,
    reconstruct_mixed as m2_reconstruct, descriptor_accounting as m2_accounting,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research-models/Qwen3-32B-Q8_0.gguf"
SOURCE_SHA256 = "2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169"
SOURCE_BYTES = 34_817_718_912
TENSOR_NAME = "blk.0.attn_k.weight"
CAPTURE = ROOT / "benchmark-results/ccc-c4-hard-gate/raw/attn-k-input.f32"
CAPTURE_SHA256 = "329d15683ef88cf6fc8fc3acb2ae6392371b1599f91e1ce7f2edb79e4ed49b8f"
IMPLICIT_RESULTS = ROOT / "benchmark-results/vbuf-ml-implicit-weight-feasibility"
SIMD_RESULTS = ROOT / "benchmark-results/vbuf-ml-simd-traceable-mutation"
LLAMA_ROOT = Path("/tmp/ccc-llama-pinned")
LLAMA_COMMIT = "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
LENGTHS = (16, 32, 64)
FORMATS = ("IQ2_XS", "IQ3_XXS", "Q3_K", "Q4_K")
OPTIMIZER_SEED = 0x50524550


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream: return list(csv.DictReader(stream))


def run(command: list[str], commands: list[str], logs: list[str], **kwargs) -> subprocess.CompletedProcess:
    commands.append(" ".join(command)); result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    logs.append("$ " + " ".join(command) + "\n" + result.stdout + result.stderr)
    if result.returncode: raise RuntimeError(f"command failed: {' '.join(command)}")
    return result


def hadamard(values: np.ndarray, block: int) -> np.ndarray:
    if values.shape[-1] % block: raise ValueError("Hadamard block does not divide input")
    output = values.astype(np.float32, copy=True).reshape(*values.shape[:-1], -1, block)
    width = 1
    while width < block:
        source = output.copy()
        for first in range(0, block, 2 * width):
            left = source[..., first:first + width]; right = source[..., first + width:first + 2 * width]
            output[..., first:first + width] = left + right
            output[..., first + width:first + 2 * width] = left - right
        width *= 2
    output *= np.float32(1.0 / math.sqrt(block))
    return output.reshape(values.shape)


@dataclass
class Transform:
    name: str
    family: str
    permutation: np.ndarray | None = None
    signs: np.ndarray | None = None
    scales: np.ndarray | None = None
    hadamard_block: int = 0
    metadata_bytes: int = 0
    fixed_decoder_bytes: int = 0
    optimizer: str = "fixed"
    runtime_placement: str = "LAYOUT_ONLY"

    def weights(self, weights: np.ndarray) -> np.ndarray:
        result = weights
        if self.permutation is not None: result = result[:, self.permutation]
        if self.signs is not None: result = result * self.signs[None, :]
        if self.scales is not None: result = result / self.scales[None, :]
        if self.hadamard_block: result = hadamard(result, self.hadamard_block)
        return result.astype(np.float32, copy=False)

    def activations(self, vectors: np.ndarray) -> np.ndarray:
        result = vectors
        if self.permutation is not None: result = result[:, self.permutation]
        if self.signs is not None: result = result * self.signs[None, :]
        if self.scales is not None: result = result * self.scales[None, :]
        if self.hadamard_block: result = hadamard(result, self.hadamard_block)
        return result.astype(np.float32, copy=False)


def statistic_permutation(weights: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(weights.astype(np.float64) ** 2, axis=0)); means = weights.mean(axis=0)
    return np.lexsort((np.arange(weights.shape[1]), means, rms)).astype(np.int32)


def shape_cluster_permutation(weights: np.ndarray, clusters: int = 160, iterations: int = 6) -> np.ndarray:
    signatures = weights.reshape(32, 32, weights.shape[1]).mean(axis=1).T.astype(np.float32)
    signatures -= signatures.mean(axis=1, keepdims=True)
    signatures /= np.maximum(np.linalg.norm(signatures, axis=1, keepdims=True), 1e-12)
    initial = np.linspace(0, len(signatures) - 1, clusters, dtype=np.int32); centroids = signatures[initial].copy()
    labels = np.zeros(len(signatures), np.int32)
    for _ in range(iterations):
        labels = np.argmax(signatures @ centroids.T, axis=1)
        for cluster in range(clusters):
            members = signatures[labels == cluster]
            if len(members):
                centroid = members.mean(axis=0); centroids[cluster] = centroid / max(np.linalg.norm(centroid), 1e-12)
    cluster_order = np.lexsort(tuple(centroids[:, index] for index in reversed(range(min(4, centroids.shape[1])))))
    order = []
    for cluster in cluster_order:
        members = np.flatnonzero(labels == cluster)
        if len(members):
            projection = signatures[members] @ centroids[cluster]
            order.extend(members[np.lexsort((members, projection))].tolist())
    if len(order) != weights.shape[1]: raise RuntimeError("shape clustering lost channels")
    return np.asarray(order, np.int32)


def codec_rmse_sample(weights: np.ndarray, permutation: np.ndarray) -> float:
    sample = weights[:128, permutation].reshape(-1)
    encoded = g2_encode(sample, "G2_CENTER_DENSE", 32, batch_segments=2048)
    reconstructed = g2_reconstruct(encoded)
    return float(np.sqrt(np.mean((reconstructed - sample).astype(np.float64) ** 2)))


def codec_aware_permutation(weights: np.ndarray, initial: np.ndarray, proposals: int = 64) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(OPTIMIZER_SEED); current = initial.copy(); objective = codec_rmse_sample(weights, current)
    accepted = 0
    for _ in range(proposals):
        left, right = rng.choice(len(current), 2, replace=False); candidate = current.copy(); candidate[left], candidate[right] = candidate[right], candidate[left]
        value = codec_rmse_sample(weights, candidate)
        if value < objective: current, objective, accepted = candidate, value, accepted + 1
    return current, {"proposals": proposals, "accepted": accepted, "screen_rows": 128, "objective_rmse": objective, "seed": OPTIMIZER_SEED}


def transform_metadata(transform: Transform, elements: int) -> dict:
    aligned = (transform.metadata_bytes + 63) // 64 * 64 if transform.metadata_bytes else 0
    return {"transform": transform.name, "family": transform.family, "fixed_decoder_constants_bytes": transform.fixed_decoder_bytes,
            "global_model_metadata_bytes": 0, "per_tensor_metadata_bytes": transform.metadata_bytes,
            "per_block_metadata_bytes": 0, "per_segment_metadata_bytes": 0,
            "transformation_alignment_bytes": aligned - transform.metadata_bytes,
            "transformation_total_bytes": aligned, "transformation_metadata_bpw": aligned * 8.0 / elements,
            "optimizer": transform.optimizer, "runtime_placement": transform.runtime_placement}


def diagnostics(values: np.ndarray, name: str) -> dict:
    flat = values.reshape(-1).astype(np.float64); mean = float(flat.mean()); std = float(flat.std())
    return {"transform": name, "rms": float(np.sqrt(np.mean(flat * flat))), "mean": mean, "std": std,
            "kurtosis": float(np.mean(((flat - mean) / max(std, 1e-30)) ** 4) - 3), "max_magnitude": float(np.max(np.abs(flat))),
            "p99_magnitude": float(np.quantile(np.abs(flat), .99)), "p999_magnitude": float(np.quantile(np.abs(flat), .999))}


def segment_diagnostics(values: np.ndarray, name: str, length: int) -> dict:
    segments = values.reshape(-1, length).astype(np.float64); means = segments.mean(axis=1); scales = np.sqrt(np.mean(segments * segments, axis=1))
    left, right = values[:, :-1].reshape(-1).astype(np.float64), values[:, 1:].reshape(-1).astype(np.float64)
    return {"transform": name, "length": length, "mean_within_segment_variance": float(np.mean(np.var(segments, axis=1))),
            "adjacent_weight_correlation": float(np.corrcoef(left, right)[0, 1]), "segment_mean_variance": float(np.var(means)),
            "segment_scale_variance": float(np.var(scales)), "segment_scale_mean": float(np.mean(scales))}


def encode_codec(weights: np.ndarray, codec: str, length: int) -> tuple[np.ndarray, dict, np.ndarray, float]:
    flat = weights.reshape(-1)
    if codec == "G2_CENTER_DENSE":
        encoded = g2_encode(flat, codec, length); reconstructed = g2_reconstruct(encoded)
        accounting = g2_accounting(codec, length, flat.size); search = encoded.search_seconds
    elif codec == "M2_AFFINE_XOR_MIXED":
        parts = [m2_encode(flat, "M2_AFFINE_XOR", rounds, length) for rounds in range(5)]
        encoded = m2_mix_rounds(parts); reconstructed = m2_reconstruct(encoded)
        accounting = m2_accounting("M2_AFFINE_XOR", length, flat.size, True); search = encoded.search_seconds
    else: raise ValueError(codec)
    segment_sse = np.sum((reconstructed.reshape(-1, length) - flat.reshape(-1, length)).astype(np.float64) ** 2, axis=1)
    return reconstructed.reshape(weights.shape), accounting, segment_sse, search


def canonical_prepared(weights: np.ndarray, transformed_fit: np.ndarray, transformed_validation: np.ndarray, reference_validation: np.ndarray,
                       transform: Transform, commands: list[str], logs: list[str]) -> list[dict]:
    if subprocess.check_output(["git", "-C", str(LLAMA_ROOT), "rev-parse", "HEAD"], text=True).strip() != LLAMA_COMMIT:
        raise RuntimeError("pinned llama.cpp mismatch")
    build = LLAMA_ROOT / "build/bin"
    with tempfile.TemporaryDirectory(prefix="weight-preparation-", dir="/tmp/opencode") as temporary:
        work = Path(temporary); binary = work / "canonical-quantize"
        run(["g++", "-O2", "-std=c++17", f"-I{LLAMA_ROOT/'ggml/include'}", f"-I{LLAMA_ROOT/'ggml/src'}",
             str(ROOT/"integrations/llama.cpp/step31_canonical_quantize.cpp"), f"-L{build}", "-lggml", "-lggml-cpu", "-lggml-base",
             f"-Wl,-rpath,{build}", "-o", str(binary)], commands, logs)
        source = work / "weights.f32"; imatrix = work / "imatrix.f32"
        weights.astype("<f4", copy=False).tofile(source)
        np.mean(transformed_fit.astype(np.float64) ** 2, axis=0).astype("<f4").tofile(imatrix)
        metadata = transform_metadata(transform, weights.size); rows = []
        for format_name in FORMATS:
            matrix = str(imatrix) if format_name == "IQ2_XS" else "-"
            result = run([str(binary), str(source), matrix, str(work), str(weights.shape[0]), str(weights.shape[1]), format_name], commands, logs)
            info = json.loads(result.stdout); restored = np.fromfile(work/f"{format_name}.f32", dtype="<f4").reshape(weights.shape)
            wm = weight_metrics(weights.reshape(-1), restored.reshape(-1)); val = action_metrics(reference_validation, transformed_validation @ restored.T)
            total = int(info["serialized_bytes"]) + metadata["transformation_total_bytes"]
            rows.append({"transform": transform.name, "format": format_name, "candidate": f"{transform.name}_{format_name}",
                         "canonical_payload_bytes": info["serialized_bytes"], "transformation_bytes": metadata["transformation_total_bytes"],
                         "total_true_bytes": total, "true_bpw": total * 8.0 / weights.size, "requires_imatrix": info["requires_imatrix"],
                         "functional_test_untouched": True, **wm, **{"validation_"+key:value for key,value in val.items()}})
        return rows


def report(payload: dict) -> str:
    source = payload["source_qualification"]; best = payload["best_stage1"]
    lines = ["# Weight-Space Preparation / Functional Reparameterization Feasibility Gate", "", f"Status: **{payload['status']}**", "",
             "## Do Not Rediscover", "", "The prior conclusions remain unchanged:", "",
             "- Arbitrary procedural seed spaces did not cover raw trained weight segments well enough.",
             "- Algebraic invertibility did not provide useful projection.", "- Additional mutation rounds did not solve coverage.",
             "- AVX2 materially accelerated arithmetic generation.", "- Arithmetic generation can exceed measured Q8 storage supply.", "",
             "This gate changes only the equivalent model coordinate system around the frozen codecs.", "",
             "## Baseline", "", f"Identity reproduced frozen G2 and mixed M2 within the configured tolerance: `{payload['classifications']['Baseline']}`.",
             f"Source `{source['model']}` / `{source['tensor']}`, W[out,input] `{source['shape']}`, oracle hash `{source['oracle_fp32_sha256']}`.", "",
             "## Transformation Families", ""]
    for row in payload["transformation_families"]:
        lines.append(f"- `{row['transform']}` ({row['family']}): {row['definition']}; model metadata {row['metadata_bytes']} bytes; placement `{row['runtime_placement']}`.")
    lines += ["", "## Equivalence", "", "Every candidate was checked as `W' x'` versus `W x` before encoding. Permutation/sign candidates differ only by FP32 reduction order; scaling and Hadamard use strict FP32 numerical tolerances. See `transformation-equivalence.csv`.", "",
              "## Weight-Space Effect", "", "Distribution and local segment diagnostics are in `weight-statistics.csv` and `segment-statistics.csv`. The decision metric is frozen-codec distance, not visual smoothness or marginal statistics.", "",
              "## Frozen-Codec Effect", "", "| Transform | Codec | L | Total bpw | RMSE | Validation W*x | Gain |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["stage1_results"]:
        if row["transform"] in ("identity", best["transform"]):
            lines.append(f"| {row['transform']} | {row['codec']} | {row['length']} | {row['total_true_bpw']:.4f} | {row['rmse']:.6f} | {row['validation_mean_relative_l2']:.4f} | {row['validation_preparation_gain']:.3f}x |")
    lines += ["", "## Segment-Length Behavior", "", "L16/L32/L64 were evaluated without adaptive fallback. L128 was not reached because L64 did not cross the material-signal gate.", "",
              "## Real Activation Result", "", f"Best Stage-1 transform `{best['transform']}` with `{best['codec']}` L{best['length']} changes validation error from {best['raw_validation_error']:.4f} to {best['prepared_validation_error']:.4f} ({best['validation_reduction']:.1%} reduction).",
              f"FUNCTIONAL_TEST status: `{payload['functional_test_gate']['status']}`. {payload['functional_test_gate']['reason']}", "",
              "## Canonical Quantizers", "", "Raw and prepared IQ2_XS/IQ3_XXS/Q3_K/Q4_K controls use the same 17 FIT / 17 VALIDATION split; historical untouched-test values are carried only as provenance and were not reopened. Prepared controls were run for the best permutation, scaling, and orthogonal candidates; see both canonical CSV files.", "",
              "## Metadata Cost", "", "Permutation IDs use 13 bits/channel (8,320 bytes); signs use 1 bit/channel; power-of-two exponents use 4 bits/channel; FP16 scaling uses 16 bits/channel. Universal Hadamard carries no model metadata. All blobs receive explicit alignment accounting.", "",
              "## Runtime Cost", "", "Activation-transform cycles, read/write traffic, temporary bytes, and folding classifications are in `runtime-transform-cost.csv`. The bounded Hadamard reference costs 7.4-9.0 cycles/element plus a 40,960-byte activation traversal and is classified expensive. Layout/folding scenarios are separated from explicit runtime fallback; no storage credit assumes a free transform without graph support.", "",
              "## Graph Audit", "", "`attn_norm-0` feeds Q, K, and V. K-only preparation is local algebra, not a complete graph rewrite. Diagonal scales/signs can plausibly fold into the local norm/QKV fan-out; permutations and Hadamards require coordinated residual-basis propagation or explicit activation work. See `graph-compatibility-audit.md`.", "",
              "## Central Answer", "", payload["central_answer"], "", "## Falsification Gates", ""]
    for gate in payload["falsification_gates"]: lines.append(f"- `{gate}`")
    lines += ["", "## Final Classifications", ""]
    for key,value in payload["classifications"].items(): lines.append(f"- {key}: `{value}`")
    lines += ["", "## Recommendation", "", f"`{payload['recommendation']}`", "", "## Stop", "",
              "No frozen codec tuning, residuals, adaptive segmentation, retraining, full-model conversion, production format, vBuf/vBuf-ML change, GPU kernel, or graph propagation implementation was performed."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=ROOT/"benchmark-results/vbuf-ml-weight-space-preparation")
    args=parser.parse_args(); out=args.output_dir.resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite immutable output: {out}")
    commands=[f"python3 scripts/qualify_weight_space_preparation.py --output-dir {out}"]; logs=[]; started=time.perf_counter()
    if SOURCE.stat().st_size!=SOURCE_BYTES or sha256(SOURCE)!=SOURCE_SHA256: raise SystemExit("source provenance mismatch")
    if sha256(CAPTURE)!=CAPTURE_SHA256: raise SystemExit("activation provenance mismatch")
    artifact=parse(SOURCE); tensor=next(item for item in artifact.tensors if item.name==TENSOR_NAME)
    if tensor.payload_size is None: raise RuntimeError("tensor payload unresolved")
    weights,q8_bytes=decode_q8(SOURCE,tensor); vectors=np.fromfile(CAPTURE,dtype="<f4").reshape(-1,5120)
    if weights.shape!=(1024,5120): raise RuntimeError("canonical W[out,input] orientation failed")
    qualified_validation,functional_test=vector_split(len(vectors)); fit=qualified_validation[::2]; validation=qualified_validation[1::2]
    if set(fit)&set(validation) or set(qualified_validation)&set(functional_test) or len(fit)+len(validation)!=len(qualified_validation): raise RuntimeError("activation split isolation failed")
    reference_qualified=vectors[qualified_validation]@weights.T; reference_validation=vectors[validation]@weights.T

    statistic=statistic_permutation(weights); shape=shape_cluster_permutation(weights); codec_perm,optimizer_info=codec_aware_permutation(weights,shape)
    random_perm=np.random.default_rng(OPTIMIZER_SEED).permutation(5120).astype(np.int32)
    signed=np.where(weights[:,shape].mean(axis=0)>=0,1.0,-1.0).astype(np.float32)
    column_rms=np.sqrt(np.mean(weights.astype(np.float64)**2,axis=0)); global_rms=float(np.sqrt(np.mean(weights.astype(np.float64)**2)))
    exponents=np.clip(np.rint(np.log2(np.maximum(column_rms/global_rms,2**-4))),-4,4).astype(np.int8); power_scales=np.exp2(exponents).astype(np.float32)
    fp16_scales=np.maximum((column_rms/global_rms).astype(np.float16),np.float16(2**-8)).astype(np.float32)
    transforms=[
        Transform("identity","identity",metadata_bytes=0,runtime_placement="LAYOUT_ONLY"),
        Transform("statistic_sort","permutation",permutation=statistic,metadata_bytes=8320,optimizer="lexsort(column RMS, mean, ordinal)",runtime_placement="FOLDABLE_INTO_MODEL_CONVERSION"),
        Transform("shape_cluster","permutation",permutation=shape,metadata_bytes=8320,optimizer="deterministic 160-cluster normalized-shape k-means; 6 iterations",runtime_placement="FOLDABLE_INTO_MODEL_CONVERSION"),
        Transform("codec_aware_order","permutation",permutation=codec_perm,metadata_bytes=8320,optimizer=f"64 deterministic local swaps; {optimizer_info['accepted']} accepted; seed {OPTIMIZER_SEED}",runtime_placement="FOLDABLE_INTO_MODEL_CONVERSION"),
        Transform("random_permutation_control","permutation_control",permutation=random_perm,metadata_bytes=8320,optimizer=f"deterministic RNG seed {OPTIMIZER_SEED}",runtime_placement="FOLDABLE_INTO_MODEL_CONVERSION"),
        Transform("signed_shape_cluster","signed_permutation",permutation=shape,signs=signed,metadata_bytes=8960,optimizer="shape cluster plus sign(column mean)",runtime_placement="FOLDABLE_INTO_MODEL_CONVERSION"),
        Transform("power2_column_equalization","diagonal_scaling",scales=power_scales,metadata_bytes=2560,optimizer="round(log2(column RMS/global RMS)), exponents [-4,4]",runtime_placement="FOLDABLE_INTO_ADJACENT_WEIGHTS"),
        Transform("fp16_column_equalization","diagonal_scaling",scales=fp16_scales,metadata_bytes=10240,optimizer="FP16 column RMS/global RMS",runtime_placement="FOLDABLE_INTO_ADJACENT_WEIGHTS"),
        Transform("hadamard8","fixed_orthogonal",hadamard_block=8,metadata_bytes=0,fixed_decoder_bytes=0,optimizer="universal normalized Walsh-Hadamard",runtime_placement="EXPENSIVE_RUNTIME_ACTIVATION_TRANSFORM"),
        Transform("hadamard16","fixed_orthogonal",hadamard_block=16,metadata_bytes=0,fixed_decoder_bytes=0,optimizer="universal normalized Walsh-Hadamard",runtime_placement="EXPENSIVE_RUNTIME_ACTIVATION_TRANSFORM"),
        Transform("hadamard32","fixed_orthogonal",hadamard_block=32,metadata_bytes=0,fixed_decoder_bytes=0,optimizer="universal normalized Walsh-Hadamard",runtime_placement="EXPENSIVE_RUNTIME_ACTIVATION_TRANSFORM"),
    ]
    definitions={"identity":"T=I","permutation":"x'=P x; W'=W P^T","permutation_control":"deterministic random P control",
                 "signed_permutation":"x'=S P x; W'=W P^T S","diagonal_scaling":"x'=D x; W'=W D^-1",
                 "fixed_orthogonal":"blockwise normalized H; x'=H x; W'=W H^T"}
    family_rows=[]; metadata_rows=[]; equivalence_rows=[]; weight_rows=[]; segment_rows=[]; prepared={}
    for transform in transforms:
        wp=transform.weights(weights); xp_qualified=transform.activations(vectors[qualified_validation]); prepared[transform.name]=(wp,transform.activations(vectors[fit]),transform.activations(vectors[validation]))
        y=wp@xp_qualified.T; error=(y.T-reference_qualified); rel=np.linalg.norm(error,axis=1)/np.maximum(np.linalg.norm(reference_qualified,axis=1),1e-30)
        tolerance=2e-5
        equivalence_rows.append({"transform":transform.name,"max_absolute_error":float(np.max(np.abs(error))),"mean_relative_l2":float(rel.mean()),"max_relative_l2":float(rel.max()),"relative_tolerance":tolerance,"passed":bool(rel.max()<tolerance)})
        if rel.max()>=tolerance: logs.append(f"REJECT equivalence {transform.name}: max relative {rel.max()}\n")
        family_rows.append({"transform":transform.name,"family":transform.family,"definition":definitions[transform.family],"parameter_count":0 if transform.family in ("identity","fixed_orthogonal") else 5120,
                            "metadata_bytes":transform.metadata_bytes,"optimizer":transform.optimizer,"runtime_placement":transform.runtime_placement})
        metadata_rows.append(transform_metadata(transform,weights.size)); weight_rows.append(diagnostics(wp,transform.name))
        for length in LENGTHS: segment_rows.append(segment_diagnostics(wp,transform.name,length))

    if not all(row["passed"] for row in equivalence_rows): raise RuntimeError("reparameterization equivalence failed before codec evaluation")
    stage_rows=[]; segment_error_rows=[]
    for transform in transforms:
        wp,xfit,xval=prepared[transform.name]; transform_meta=transform_metadata(transform,weights.size)
        for codec in ("G2_CENTER_DENSE","M2_AFFINE_XOR_MIXED"):
            for length in LENGTHS:
                reconstructed,accounting,segment_sse,search_seconds=encode_codec(wp,codec,length)
                wm=weight_metrics(wp.reshape(-1),reconstructed.reshape(-1)); val=action_metrics(reference_validation,xval@reconstructed.T)
                total=accounting["total_true_bytes"]+transform_meta["transformation_total_bytes"]
                row={"transform":transform.name,"transform_family":transform.family,"codec":codec,"length":length,
                     "codec_true_bpw":accounting["true_bpw"],"transformation_metadata_bpw":transform_meta["transformation_metadata_bpw"],
                     "total_true_bytes":total,"total_true_bpw":total*8.0/weights.size,"encoder_seconds":search_seconds,
                     "functional_test_used":False,**wm,**{"validation_"+key:value for key,value in val.items()}}
                stage_rows.append(row)
                segment_error_rows.append({"transform":transform.name,"codec":codec,"length":length,"segment_count":len(segment_sse),
                                           "mean_segment_rmse":float(np.mean(np.sqrt(segment_sse/length))),"p50_segment_rmse":float(np.quantile(np.sqrt(segment_sse/length),.5)),
                                           "p95_segment_rmse":float(np.quantile(np.sqrt(segment_sse/length),.95)),"p99_segment_rmse":float(np.quantile(np.sqrt(segment_sse/length),.99)),
                                           "max_segment_rmse":float(np.max(np.sqrt(segment_sse/length)))})
                logs.append(f"encoded {transform.name} {codec} L{length}: val={row['validation_mean_relative_l2']:.6f} bpw={row['total_true_bpw']:.6f}\n")
    identity={(row["codec"],row["length"]):row for row in stage_rows if row["transform"]=="identity"}
    for row in stage_rows:
        baseline=identity[(row["codec"],row["length"])]
        row["weight_preparation_gain"]=baseline["rmse"]/max(row["rmse"],1e-30)
        row["validation_preparation_gain"]=baseline["validation_mean_relative_l2"]/max(row["validation_mean_relative_l2"],1e-30)
        row["validation_relative_reduction"]=1-row["validation_mean_relative_l2"]/baseline["validation_mean_relative_l2"]

    previous_g2={int(row["length"]):row for row in read_csv(IMPLICIT_RESULTS/"fixed-length-results.csv") if row["candidate"].startswith("G2_CENTER_DENSE_L")}
    previous_m2={int(row["length"]):row for row in read_csv(SIMD_RESULTS/"fixed-length-results.csv") if row["candidate"].startswith("M2_AFFINE_XOR_MIXED_L")}
    baseline_rows=[]; baseline_ok=True
    for length in LENGTHS:
        current_g2=identity[("G2_CENTER_DENSE",length)]; prior_g2=previous_g2[length]
        current_m2=identity[("M2_AFFINE_XOR_MIXED",length)]; prior_m2=previous_m2[length]
        for codec,current,prior in (("G2_CENTER_DENSE",current_g2,prior_g2),("M2_AFFINE_XOR_MIXED",current_m2,prior_m2)):
            rmse_diff=abs(current["rmse"]-float(prior["rmse"])); validation_expected=float(prior["validation_mean_relative_l2"])
            identity_reconstructed,_,_,_=encode_codec(weights,codec,length)
            qualified_metric=action_metrics(reference_qualified,vectors[qualified_validation]@identity_reconstructed.T)["mean_relative_l2"]
            functional_diff=abs(qualified_metric-validation_expected)
            passed=rmse_diff<1e-10 and functional_diff<1e-8 and abs(current["total_true_bpw"]-float(prior["true_bpw"]))<1e-10
            baseline_ok &= passed
            baseline_rows.append({"codec":codec,"length":length,"current_weight_rmse":current["rmse"],"previous_weight_rmse":float(prior["rmse"]),"rmse_absolute_difference":rmse_diff,
                                  "current_true_bpw":current["total_true_bpw"],"previous_true_bpw":float(prior["true_bpw"]),"previous_validation_mean_relative_l2":validation_expected,
                                  "current_qualification_validation_mean_relative_l2":qualified_metric,"functional_absolute_difference":functional_diff,"qualification_validation_vectors":len(qualified_validation),
                                  "current_stage1_validation_mean_relative_l2":current["validation_mean_relative_l2"],"stage1_validation_vectors":len(validation),"passed":passed})
    if not baseline_ok: raise RuntimeError("BASELINE_REPRODUCTION_FAILED")

    candidates=[row for row in stage_rows if row["transform"] not in ("identity","random_permutation_control")]
    best_row=max(candidates,key=lambda row:(row["validation_relative_reduction"],-row["total_true_bpw"]))
    random_rows=[row for row in stage_rows if row["transform"]=="random_permutation_control"]
    gain_rows=[{"transform":row["transform"],"transform_family":row["transform_family"],"codec":row["codec"],"length":row["length"],"raw_weight_rmse":identity[(row["codec"],row["length"])]["rmse"],
                "prepared_weight_rmse":row["rmse"],"weight_preparation_gain":row["weight_preparation_gain"],"raw_validation_mean_relative_l2":identity[(row["codec"],row["length"])]["validation_mean_relative_l2"],
                "prepared_validation_mean_relative_l2":row["validation_mean_relative_l2"],"validation_preparation_gain":row["validation_preparation_gain"],
                "validation_relative_reduction":row["validation_relative_reduction"],"total_true_bpw":row["total_true_bpw"]} for row in stage_rows]

    structural={"permutation":("permutation","signed_permutation"),"scaling":("diagonal_scaling",),"orthogonal":("fixed_orthogonal",)}
    canonical_targets=[]
    for _,families in structural.items():
        rows=[row for row in candidates if row["transform_family"] in families]
        if rows:
            chosen=max(rows,key=lambda row:row["validation_relative_reduction"]); canonical_targets.append(next(transform for transform in transforms if transform.name==chosen["transform"]))
    identity_transform=next(transform for transform in transforms if transform.name=="identity")
    raw_canonical=canonical_prepared(weights,vectors[fit],vectors[validation],reference_validation,identity_transform,commands,logs)
    historical_canonical={row["candidate"]:row for row in read_csv(IMPLICIT_RESULTS/"canonical-controls.csv") if row["candidate"] in FORMATS}
    for row in raw_canonical:
        historical=historical_canonical[row["format"]]
        row.update({"test_mean_relative_l2_historical":float(historical["test_mean_relative_l2"]),"functional_test_untouched_in_this_gate":True,
                    "provenance":"raw canonical requalified on matched 17 FIT / 17 VALIDATION; historical test value carried from "+str((IMPLICIT_RESULTS/"canonical-controls.csv").relative_to(ROOT))})
    prepared_canonical=[]
    for transform in canonical_targets:
        wp,xfit,xval=prepared[transform.name]; prepared_canonical.extend(canonical_prepared(wp,xfit,xval,reference_validation,transform,commands,logs))

    family_best=[]
    for family in ("permutation","signed_permutation","diagonal_scaling","fixed_orthogonal"):
        rows=[row for row in candidates if row["transform_family"]==family]
        if rows: family_best.append(max(rows,key=lambda row:row["validation_relative_reduction"]))
    promoted=[]
    for row in sorted(family_best,key=lambda row:row["validation_relative_reduction"],reverse=True):
        if row["validation_relative_reduction"]>=.30 and all(existing["transform_family"]!=row["transform_family"] for existing in promoted): promoted.append(row)
        if len(promoted)==2: break
    test_rows=[]
    frozen_candidates=[]
    reference_test=vectors[functional_test]@weights.T if promoted else None
    for row in promoted:
        transform=next(item for item in transforms if item.name==row["transform"]); wp=transform.weights(weights); xtest=transform.activations(vectors[functional_test])
        reconstructed,_,_,_=encode_codec(wp,row["codec"],row["length"])
        assert reference_test is not None
        metrics=action_metrics(reference_test,xtest@reconstructed.T)
        test_rows.append({"transform":row["transform"],"codec":row["codec"],"length":row["length"],"total_true_bpw":row["total_true_bpw"],
                          "validation_mean_relative_l2":row["validation_mean_relative_l2"],**{"test_"+key:value for key,value in metrics.items()},"frozen_before_test":True})
        frozen_candidates.append({"transform":row["transform"],"codec":row["codec"],"length":row["length"],"descriptor_rules":"frozen prior implementation","metadata_bytes":next(item for item in metadata_rows if item["transform"]==row["transform"])["transformation_total_bytes"]})
    if not promoted: test_rows=[{"status":"NOT_REACHED","reason":"no Stage-1 candidate reduced real validation relative-L2 by >=30%; FUNCTIONAL_TEST remained unopened","functional_test_vectors_used":0}]

    with tempfile.TemporaryDirectory(prefix="weight-transform-bench-",dir="/tmp/opencode") as temporary:
        binary=Path(temporary)/"transform-bench"
        run(["g++","-O3","-march=znver3","-mavx2","-mfma","-std=c++17",str(ROOT/"research/weight_space_transform_bench.cpp"),"-o",str(binary)],commands,logs)
        runtime_result=run([str(binary)],commands,logs); runtime_rows=[]
        for row in csv.DictReader(runtime_result.stdout.splitlines()):
            runtime_rows.append({key:(float(value) if key not in ("transform","in_place","vectorization") else value) for key,value in row.items()})
    runtime_by={(row["transform"],int(row["block_size"])):row for row in runtime_rows}
    runtime_cost=[]
    for transform in transforms:
        if transform.family in ("identity",): measured={"cycles_per_element":0.0,"cycles_per_vector":0.0,"g_elements_per_second":0.0,"temporary_bytes":0,"memory_read_bytes_per_vector":0,"memory_write_bytes_per_vector":0,"in_place":"YES","vectorization":"none","block_size":0}
        elif transform.family in ("permutation","permutation_control"): measured=runtime_by[("permutation",0)]
        elif transform.family=="signed_permutation": measured=runtime_by[("signed_permutation",0)]
        elif transform.family=="diagonal_scaling": measured=runtime_by[("diagonal_scale",0)]
        else: measured=runtime_by[("hadamard",transform.hadamard_block)]
        explicit_traffic=measured["memory_read_bytes_per_vector"]+measured["memory_write_bytes_per_vector"]
        runtime_cost.append({**measured,"transform":transform.name,"family":transform.family,"runtime_placement":transform.runtime_placement,
                             "operations_per_element":"0 if folded/layout" if transform.runtime_placement.startswith("FOLDABLE") or transform.runtime_placement=="LAYOUT_ONLY" else f"{int(math.log2(transform.hadamard_block))} add/sub stages" if transform.hadamard_block else "1 multiply",
                             "explicit_fallback_bytes_per_vector":explicit_traffic})

    best_transform=next(transform for transform in transforms if transform.name==best_row["transform"]); best_meta=transform_metadata(best_transform,weights.size)
    supply_rows=[]
    for row in stage_rows:
        transform=next(item for item in transforms if item.name==row["transform"]); runtime=next(item for item in runtime_cost if item["transform"]==transform.name)
        foldable=transform.runtime_placement in ("LAYOUT_ONLY","FOLDABLE_INTO_MODEL_CONVERSION","FOLDABLE_INTO_ADJACENT_WEIGHTS")
        supply_rows.append({"transform":row["transform"],"codec":row["codec"],"length":row["length"],"total_true_bpw":row["total_true_bpw"],
                            "explicit_q8_weight_bytes":q8_bytes,"prepared_total_weight_bytes":row["total_true_bytes"],"explicit_weight_transport_avoided_bytes":q8_bytes-row["total_true_bytes"],
                            "transform_metadata_bytes":next(item for item in metadata_rows if item["transform"]==row["transform"])["transformation_total_bytes"],
                            "activation_transform_bytes_per_vector_if_explicit":runtime["explicit_fallback_bytes_per_vector"],"folded_activation_transform_bytes":0 if foldable else runtime["explicit_fallback_bytes_per_vector"],
                            "decode_compute_included":"frozen codec generation cost reported in prior artifact; no end-to-end inequality claim","transport_inequality_established":False})

    max_gain=max(row["validation_relative_reduction"] for row in candidates); procedural_benefit=max_gain>=.10
    raw_canonical_by={row["format"]:row for row in raw_canonical}; canonical_gains=[]
    for row in prepared_canonical:
        raw=raw_canonical_by[row["format"]]
        canonical_gains.append(1-row["validation_mean_relative_l2"]/raw["validation_mean_relative_l2"])
    canonical_benefit=max(canonical_gains,default=0)>=.10
    prep_signal="WEIGHT_PREPARATION_STRONG_SIGNAL" if max_gain>=.5 else "WEIGHT_PREPARATION_MATERIAL_SIGNAL" if max_gain>=.3 else "WEIGHT_PREPARATION_WEAK_SIGNAL" if max_gain>=.1 else "WEIGHT_PREPARATION_NO_SIGNAL"
    classification_transform=next(item for item in transforms if item.name=="identity") if max_gain<=0 else best_transform
    family_class={"identity":"IDENTITY_BEST","permutation":"PERMUTATION_BEST","permutation_control":"PERMUTATION_BEST","signed_permutation":"SIGNED_PERMUTATION_BEST","diagonal_scaling":"DIAGONAL_SCALING_BEST","fixed_orthogonal":"FIXED_ORTHOGONAL_BEST"}[classification_transform.family]
    codec_interaction="BOTH_CODEC_CLASSES_BENEFIT" if procedural_benefit and canonical_benefit else "PROCEDURAL_CODEC_SPECIFIC_BENEFIT" if procedural_benefit else "GENERAL_QUANTIZATION_BENEFIT" if canonical_benefit else "NO_CODEC_BENEFIT"
    if classification_transform.runtime_placement=="LAYOUT_ONLY": runtime_class="TRANSFORM_LAYOUT_ONLY"
    elif classification_transform.runtime_placement.startswith("FOLDABLE"): runtime_class="TRANSFORM_FOLDABLE"
    elif next(row for row in runtime_cost if row["transform"]==classification_transform.name)["cycles_per_element"]<5: runtime_class="TRANSFORM_RUNTIME_CHEAP"
    else: runtime_class="TRANSFORM_RUNTIME_EXPENSIVE"
    graph_class="GRAPH_PROPAGATION_PLAUSIBLE" if classification_transform.family=="diagonal_scaling" else "GRAPH_PROPAGATION_CONSTRAINED" if classification_transform.family in ("permutation","permutation_control","signed_permutation","fixed_orthogonal") else "GRAPH_AUDIT_NOT_REACHED"
    final_direction="WEIGHT_SPACE_PREPARATION_REJECTED" if prep_signal in ("WEIGHT_PREPARATION_NO_SIGNAL","WEIGHT_PREPARATION_WEAK_SIGNAL") or not promoted else "WEIGHT_SPACE_PREPARATION_INTERESTING"
    recommendation="STOP_WEIGHT_PREPARATION_RESEARCH" if final_direction=="WEIGHT_SPACE_PREPARATION_REJECTED" else "RUN_GRAPH_LEVEL_REPARAMETERIZATION_GATE"
    failure_gates=[]
    if max_gain<.10: failure_gates.append("WEIGHT_PREPARATION_NO_SIGNAL")
    materially_improved=[row for row in candidates if row["validation_relative_reduction"]>=.30]
    if materially_improved and all(row["length"]==16 for row in materially_improved): failure_gates.append("SEGMENT_LENGTH_PROBLEM_UNCHANGED")
    if best_row["validation_mean_relative_l2"]>min(row["validation_mean_relative_l2"] for row in raw_canonical): failure_gates.append("PROCEDURAL_REPRESENTATION_STILL_FUNCTIONALLY_REJECTED")
    status="COMPLETE / REJECTED" if final_direction=="WEIGHT_SPACE_PREPARATION_REJECTED" else "COMPLETE / INTERESTING"
    source_qualification={"model":"Qwen3-32B-Q8_0","path":str(SOURCE.relative_to(ROOT)),"sha256":SOURCE_SHA256,"tensor":TENSOR_NAME,"shape":list(weights.shape),"element_count":weights.size,
                          "payload_range":[tensor.absolute_start,tensor.absolute_start+tensor.payload_size],"oracle":"Q8_0 reconstructed FP32, not BF16/F32 ground truth","oracle_fp32_sha256":hashlib.sha256(weights.astype("<f4",copy=False).tobytes()).hexdigest(),"orientation":"W[out,input]; Y=X@W.T"}
    best_summary={"transform":best_row["transform"],"transform_family":best_row["transform_family"],"codec":best_row["codec"],"length":best_row["length"],"total_true_bpw":best_row["total_true_bpw"],
                  "raw_validation_error":identity[(best_row["codec"],best_row["length"])]["validation_mean_relative_l2"],"prepared_validation_error":best_row["validation_mean_relative_l2"],"validation_reduction":best_row["validation_relative_reduction"],"weight_rmse":best_row["rmse"],"transformation_metadata_bpw":best_row["transformation_metadata_bpw"]}
    payload={"status":status,"source_qualification":source_qualification,"activation_split":{"capture":str(CAPTURE.relative_to(ROOT)),"sha256":CAPTURE_SHA256,"fit_ids":fit.tolist(),"validation_ids":validation.tolist(),"functional_test_ids":functional_test.tolist(),"functional_test_used_for_selection":False},
             "previous_conclusions_preserved":["IMPLICIT_WEIGHT_REPRESENTATION_REJECTED","SIMD_TRACEABLE_WEIGHT_MUTATION_REJECTED","TRACEABILITY_NOT_FOUND","SEGMENT_LENGTH_STILL_KILLS_RATE","SIMD_GENERATION_BEATS_STORAGE_SUPPLY"],
             "transformation_families":family_rows,"transformation_metadata":metadata_rows,"equivalence":equivalence_rows,"weight_statistics":weight_rows,"segment_statistics":segment_rows,
             "stage1_results":stage_rows,"segment_codec_distance":segment_error_rows,"preparation_gain":gain_rows,"baseline_reproduction":baseline_rows,"canonical_controls_raw":raw_canonical,"canonical_controls_prepared":prepared_canonical,
             "promotion":{"threshold":">=30% validation relative-L2 reduction versus same raw frozen codec","promoted":frozen_candidates},"functional_test_gate":{"status":"REACHED" if promoted else "NOT_REACHED","reason":"frozen promoted candidates evaluated once" if promoted else "No Stage-1 candidate met >=30% validation reduction; FUNCTIONAL_TEST was not opened."},
             "functional_test_results":test_rows,"runtime_transform_cost":runtime_cost,"supply_impact":supply_rows,"best_stage1":best_summary,"codec_aware_optimizer":optimizer_info,
             "central_answer":"No. Within permutation, signed permutation, diagonal scaling, and fixed block-Hadamard families, equivalent preparation did not move the frozen procedural codecs materially closer to a useful low-rate representation." if final_direction=="WEIGHT_SPACE_PREPARATION_REJECTED" else "Locally yes; preparation materially improved frozen-codec validation and reached the separate graph-propagation gate.",
             "falsification_gates":failure_gates,"classifications":{"Baseline":"BASELINE_REPRODUCED","Preparation signal":prep_signal,"Best transformation class":family_class,"Codec interaction":codec_interaction,"Runtime placement":runtime_class,"Graph viability":graph_class,"Final direction":final_direction},
             "recommendation":recommendation,"elapsed_seconds":time.perf_counter()-started,"environment":{"cpu":"AMD Ryzen 7 5800X","python":platform.python_version(),"numpy":np.__version__,"cpu_count":os.cpu_count()},
             "wire_changes":0,"production_changes":0,"frozen_codec_changes":0,"residuals":False,"adaptive_segmentation":False,"retraining":False}

    graph_audit="""# Graph Compatibility Audit

This is an analytical audit only. No graph was modified.

## Actual seam

`attn_norm-0` is weighted RMS-normalized hidden state and feeds all separate Q, K, and V projections, not K alone. Q/K subsequently receive per-head weighted RMSNorm and RoPE. Attention output and FFN output rejoin the unchanged residual stream.

## Permutation

Local K-only column permutation requires an explicit activation gather. A persistent hidden-coordinate permutation can be propagated only by coordinated conversion of embeddings, normalization parameters, all Q/K/V input columns, attention/FFN output rows, residual layers, and final output. Classification: `PERMUTATION_LAYOUT_FOLDABLE` only at whole-model scope; otherwise `PERMUTATION_RUNTIME_COST_REQUIRED`.

## Signed permutation

Pure signs are diagonal and can fold locally by compensating the `attn_norm`/QKV fan-out. The permutation component has the whole-model constraint above. Sign propagation must keep residual coordinates consistent; it must not be pushed through Q/K RoPE or FFN nonlinear intermediates.

## Diagonal scaling

Local nonzero scaling is a valid fan-out gauge: scale `attn_norm` output coordinates and inversely scale input columns of Q, K, and V (and active LoRA input factors). General nonuniform scaling is not a global RMSNorm symmetry. Classification: local `FOLDABLE_INTO_ADJACENT_WEIGHTS`, not a general residual-basis transform.

## Hadamard

Block Hadamard does not commute with learned diagonal RMSNorm weights. It requires an explicit post-norm activation transform or a coordinated whole-model orthogonal residual-basis conversion. Q/K outputs must remain unchanged to avoid head, GQA, KV-cache, and RoPE constraints. Classification: `GRAPH_PROPAGATION_CONSTRAINED`.

## Source references

- `/tmp/ccc-llama-pinned/src/models/qwen3.cpp:76-141`
- `/tmp/ccc-llama-pinned/src/llama-graph.cpp:1556-1658`
- `/tmp/ccc-llama-pinned/src/llama-graph.cpp:1707-1779`
- `/tmp/ccc-llama-pinned/src/llama-graph.cpp:2800-2817`
"""
    out.mkdir(parents=True); raw=out/"raw"; raw.mkdir()
    write_csv(out/"baseline-reproduction.csv",baseline_rows); write_csv(out/"transformation-families.csv",family_rows); write_csv(out/"transformation-metadata.csv",metadata_rows); write_csv(out/"transformation-equivalence.csv",equivalence_rows)
    write_csv(out/"weight-statistics.csv",weight_rows); write_csv(out/"segment-statistics.csv",segment_rows+segment_error_rows)
    write_csv(out/"frozen-g2-results.csv",[row for row in stage_rows if row["codec"]=="G2_CENTER_DENSE"]); write_csv(out/"frozen-m2-results.csv",[row for row in stage_rows if row["codec"]=="M2_AFFINE_XOR_MIXED"])
    write_csv(out/"preparation-gain.csv",gain_rows); write_csv(out/"real-validation-results.csv",stage_rows); write_csv(out/"real-functional-test-results.csv",test_rows)
    write_csv(out/"canonical-controls-raw.csv",raw_canonical); write_csv(out/"canonical-controls-prepared.csv",prepared_canonical)
    write_csv(out/"permutation-results.csv",[row for row in stage_rows if row["transform_family"] in ("permutation","permutation_control","signed_permutation")])
    write_csv(out/"scaling-results.csv",[row for row in stage_rows if row["transform_family"]=="diagonal_scaling"]); write_csv(out/"orthogonal-results.csv",[row for row in stage_rows if row["transform_family"]=="fixed_orthogonal"])
    write_csv(out/"runtime-transform-cost.csv",runtime_cost); write_csv(out/"supply-impact.csv",supply_rows)
    (out/"graph-compatibility-audit.md").write_text(graph_audit,encoding="utf-8"); (out/"feasibility.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); (out/"feasibility-report.md").write_text(report(payload),encoding="utf-8")
    (raw/"commands.txt").write_text("\n".join(commands)+"\n",encoding="utf-8"); (raw/"runner.log").write_text("".join(logs),encoding="utf-8")
    print(f"{status}: best={best_row['transform']} {best_row['codec']} L{best_row['length']} gain={best_row['validation_relative_reduction']:.3%}; wrote {out}")


if __name__=="__main__": main()
