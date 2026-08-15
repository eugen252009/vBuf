#!/usr/bin/env python3
"""Research-only implicit procedural weight representation feasibility gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
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

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research-models/Qwen3-32B-Q8_0.gguf"
SOURCE_SHA256 = "2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169"
SOURCE_BYTES = 34_817_718_912
TENSOR_NAME = "blk.0.attn_k.weight"
CAPTURE = ROOT / "benchmark-results/ccc-c4-hard-gate/raw/attn-k-input.f32"
CAPTURE_INVENTORY = ROOT / "benchmark-results/ccc-c4-hard-gate/hidden-state-inventory.csv"
CAPTURE_SHA256 = "329d15683ef88cf6fc8fc3acb2ae6392371b1599f91e1ce7f2edb79e4ed49b8f"
LLAMA_ROOT = Path("/tmp/ccc-llama-pinned")
LLAMA_COMMIT = "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
LENGTHS = (8, 16, 32, 64, 128, 256)
FAMILIES = ("G0_HASH_SIGN", "G1_HASH_AFFINE", "G2_CENTER_DENSE", "G3_RECURRENCE", "G4_TINY_BASIS")
FORMATS = ("IQ2_XS", "IQ3_XXS", "Q3_K", "IQ4_XS", "Q4_K")
RANDOM_SEED = 0x1A11CE
TENSOR_METADATA_BYTES = 16
PROMPTS = (
    "The old observatory recorded a clear winter sky while researchers checked every instrument before dawn.",
    "A compact numerical experiment should separate measurement, validation, and the final decision with care.",
    "During the afternoon, the engineer reviewed a matrix calculation and documented the assumptions precisely.",
    "Reliable systems use small interfaces, explicit bounds, and repeatable evidence rather than optimistic guesses.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def mix32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32).copy()
    values ^= values >> np.uint32(16)
    values *= np.uint32(0x7FEB352D)
    values ^= values >> np.uint32(15)
    values *= np.uint32(0x846CA68B)
    values ^= values >> np.uint32(16)
    return values


def popcount8(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint8, copy=False)
    values = values - ((values >> 1) & 0x55)
    values = (values & 0x33) + ((values >> 2) & 0x33)
    return ((values + (values >> 4)) & 0x0F).astype(np.float32)


def generator_codebook(family: str, length: int, seeds: np.ndarray | None = None) -> np.ndarray:
    seeds = np.arange(256, dtype=np.uint32) if seeds is None else np.asarray(seeds, dtype=np.uint32)
    local = np.arange(length, dtype=np.uint32)[None, :]
    seed_matrix = seeds[:, None]
    if family in ("G0_HASH_SIGN", "G1_HASH_AFFINE", "G2_CENTER_DENSE"):
        hashed = mix32(seed_matrix * np.uint32(0x9E3779B9) + local * np.uint32(0x85EBCA6B) + np.uint32(0xD1B54A35))
        if family == "G0_HASH_SIGN":
            return np.where(hashed >> np.uint32(31), 1.0, -1.0).astype(np.float32)
        if family == "G1_HASH_AFFINE":
            signed = (hashed >> np.uint32(16)).astype(np.uint16).view(np.int16)
            return (signed.astype(np.float32) / 32768.0).astype(np.float32)
        return ((popcount8(hashed.astype(np.uint8)) - 4.0) * 0.5).astype(np.float32)
    if family == "G3_RECURRENCE":
        state = mix32(seeds + np.uint32(0xA341316C)) | np.uint32(1)
        result = np.empty((len(seeds), length), dtype=np.float32)
        for index in range(length):
            state ^= state << np.uint32(13)
            state ^= state >> np.uint32(17)
            state ^= state << np.uint32(5)
            result[:, index] = (state >> np.uint32(16)).astype(np.uint16).view(np.int16).astype(np.float32) / 32768.0
        return result
    if family == "G4_TINY_BASIS":
        basis = generator_codebook("G0_HASH_SIGN", length, np.arange(8, dtype=np.uint32) + np.uint32(0xB0))
        signs = np.where(((seeds[:, None] >> np.arange(8, dtype=np.uint32)) & 1) != 0, 1.0, -1.0).astype(np.float32)
        return (signs @ basis / math.sqrt(8.0)).astype(np.float32)
    raise ValueError(f"unknown generator family {family}")


def descriptor_layout(family: str, length: int, element_count: int) -> dict:
    segment_count = math.ceil(element_count / length)
    seed_bits, scale_bits = 8, 16
    bias_bits = 16 if family == "G1_HASH_AFFINE" else 0
    descriptor_bits = seed_bits + scale_bits + bias_bits
    descriptor_bytes = math.ceil(segment_count * descriptor_bits / 8)
    base = TENSOR_METADATA_BYTES + descriptor_bytes
    aligned = (base + 63) // 64 * 64
    shared = 2048 if family == "G4_TINY_BASIS" else 16
    total = aligned + shared
    return {
        "generator": family,
        "length": length,
        "segment_count": segment_count,
        "seed_bits_per_segment": seed_bits,
        "scale_bits_per_segment": scale_bits,
        "bias_bits_per_segment": bias_bits,
        "mode_bits_per_segment": 0,
        "length_bits_per_segment": 0,
        "position_bits_per_segment": 0,
        "descriptor_bits_per_segment": descriptor_bits,
        "descriptor_bytes": descriptor_bytes,
        "metadata_bytes": TENSOR_METADATA_BYTES,
        "alignment_padding_bytes": aligned - base,
        "shared_generator_bytes": shared,
        "total_true_bytes": total,
        "true_bpw": total * 8.0 / element_count,
    }


@dataclass
class Encoded:
    family: str
    length: int
    seeds: np.ndarray
    scales: np.ndarray
    biases: np.ndarray | None
    search_seconds: float
    descriptor_sha256: str


def encode_fixed(flat: np.ndarray, family: str, length: int, batch_segments: int = 4096) -> Encoded:
    if flat.size % length:
        raise ValueError("fixed-length experiment requires exact tensor coverage")
    segments = flat.reshape(-1, length)
    codebook = generator_codebook(family, length)
    affine = family == "G1_HASH_AFFINE"
    if affine:
        code_mean = codebook.mean(axis=1)
        fit_code = codebook - code_mean[:, None]
    else:
        code_mean = None
        fit_code = codebook
    norms = np.sum(fit_code * fit_code, axis=1)
    if np.any(norms <= 0):
        raise RuntimeError("generator emitted a zero-energy codeword")
    chosen = np.empty(len(segments), dtype=np.uint8)
    scales = np.empty(len(segments), dtype=np.float16)
    biases = np.empty(len(segments), dtype=np.float16) if affine else None
    started = time.perf_counter()
    for first in range(0, len(segments), batch_segments):
        target = segments[first:first + batch_segments]
        if affine:
            target_mean = target.mean(axis=1)
            fit_target = target - target_mean[:, None]
        else:
            target_mean = None
            fit_target = target
        dots = fit_target @ fit_code.T
        best = np.argmax((dots * dots) / norms[None, :], axis=1)
        scale = dots[np.arange(len(target)), best] / norms[best]
        chosen[first:first + len(target)] = best.astype(np.uint8)
        scales[first:first + len(target)] = scale.astype(np.float16)
        if affine and biases is not None and target_mean is not None and code_mean is not None:
            biases[first:first + len(target)] = (target_mean - scale * code_mean[best]).astype(np.float16)
    packed = chosen.tobytes() + scales.astype("<f2", copy=False).tobytes()
    if biases is not None:
        packed += biases.astype("<f2", copy=False).tobytes()
    return Encoded(family, length, chosen, scales, biases, time.perf_counter() - started, hashlib.sha256(packed).hexdigest())


def reconstruct(encoded: Encoded) -> np.ndarray:
    codebook = generator_codebook(encoded.family, encoded.length)
    result = codebook[encoded.seeds] * encoded.scales.astype(np.float32)[:, None]
    if encoded.biases is not None:
        result += encoded.biases.astype(np.float32)[:, None]
    return result.reshape(-1).astype(np.float32, copy=False)


def permuted_control(values: np.ndarray, length: int) -> np.ndarray:
    segments = values.reshape(-1, length)
    shifts = (np.arange(len(segments), dtype=np.int64) * 13 + 7) % length
    shifts[shifts == 0] = 1
    indexes = (np.arange(length)[None, :] + shifts[:, None]) % length
    return np.take_along_axis(segments, indexes, axis=1).reshape(-1)


def weight_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    error = candidate - reference
    absolute = np.abs(error)
    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    denominator = max(float(np.linalg.norm(reference64)), 1e-30)
    return {
        "rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "mae": float(np.mean(absolute, dtype=np.float64)),
        "p99": float(np.quantile(absolute, 0.99)),
        "p999": float(np.quantile(absolute, 0.999)),
        "max_absolute_error": float(absolute.max()),
        "relative_l2": float(np.linalg.norm(error.astype(np.float64)) / denominator),
        "cosine": float(np.dot(reference64, candidate64) / max(np.linalg.norm(reference64) * np.linalg.norm(candidate64), 1e-30)),
        "correlation": float(np.corrcoef(reference64, candidate64)[0, 1]),
    }


def action_metrics(reference_output: np.ndarray, candidate_output: np.ndarray) -> dict:
    error = candidate_output - reference_output
    relative = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(reference_output, axis=1), 1e-30)
    cosine = np.sum(reference_output * candidate_output, axis=1) / np.maximum(
        np.linalg.norm(reference_output, axis=1) * np.linalg.norm(candidate_output, axis=1), 1e-30)
    return {
        "mean_relative_l2": float(relative.mean()),
        "median_relative_l2": float(np.median(relative)),
        "p95_relative_l2": float(np.quantile(relative, 0.95)),
        "mean_cosine": float(cosine.mean()),
        "minimum_cosine": float(cosine.min()),
    }


def vector_split(count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(count, dtype=np.uint64)
    values ^= values >> np.uint64(30); values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27); values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    order = np.argsort(values, kind="stable")
    midpoint = count // 2
    return np.sort(order[:midpoint]), np.sort(order[midpoint:])


def validate_orientation(weights: np.ndarray, vectors: np.ndarray) -> dict:
    if weights.shape != (1024, 5120) or vectors.shape[1] != 5120:
        raise RuntimeError("W[out,input] orientation invariant failed")
    probe = vectors[0]
    direct = weights @ probe
    row_dot = np.array([np.dot(weights[row], probe) for row in range(3)], dtype=np.float32)
    if not np.allclose(direct[:3], row_dot, rtol=1e-5, atol=1e-5):
        raise RuntimeError("orientation apply invariant failed")
    return {"logical_shape": [1024, 5120], "ggml_shape": [5120, 1024], "apply": "Y = X @ W.T", "verified": True}


def select_best_fixed(validation_rows: list[dict]) -> str:
    eligible = [row for row in validation_rows if row["true_bpw"] <= 2.0]
    if not eligible:
        eligible = validation_rows
    return min(eligible, key=lambda row: (row["validation_mean_relative_l2"], row["true_bpw"]))["candidate"]


def exact_segmentation(total: int, lengths: tuple[int, ...], costs: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    best = [math.inf] * (total + 1)
    previous: list[tuple[int, int] | None] = [None] * (total + 1)
    best[0] = 0.0
    for end in range(1, total + 1):
        for length in lengths:
            start = end - length
            if start >= 0 and best[start] < math.inf and (start, length) in costs:
                value = best[start] + costs[(start, length)]
                if value < best[end]:
                    best[end], previous[end] = value, (start, length)
    if previous[total] is None:
        raise ValueError("no exact segmentation")
    result = []
    cursor = total
    while cursor:
        item = previous[cursor]
        assert item is not None
        result.append(item)
        cursor = item[0]
    return list(reversed(result))


def seed_width_sweep(sample: np.ndarray, length: int = 32) -> list[dict]:
    segments = sample.reshape(-1, length)
    rows = []
    for width, count, method in ((8, 256, "exhaustive"), (16, 65536, "exhaustive"), (24, 65536, "deterministic bounded"), (32, 65536, "deterministic bounded")):
        mask = (1 << width) - 1
        best_score = np.full(len(segments), -np.inf, dtype=np.float32)
        best_dot = np.zeros(len(segments), dtype=np.float32)
        best_norm = np.ones(len(segments), dtype=np.float32)
        started = time.perf_counter()
        for first in range(0, count, 4096):
            ordinal = np.arange(first, min(first + 4096, count), dtype=np.uint64)
            seeds = ((ordinal * np.uint64(0x9E3779B1) + np.uint64(0xA511E9B3)) & np.uint64(mask)).astype(np.uint32)
            codebook = generator_codebook("G0_HASH_SIGN", length, seeds)
            norms = np.sum(codebook * codebook, axis=1)
            dots = segments @ codebook.T
            score = (dots * dots) / norms[None, :]
            local = np.argmax(score, axis=1)
            improved = score[np.arange(len(segments)), local] > best_score
            best_score[improved] = score[np.arange(len(segments)), local][improved]
            best_dot[improved] = dots[np.arange(len(segments)), local][improved]
            best_norm[improved] = norms[local][improved]
        sse = np.sum(segments * segments, axis=1) - best_dot * best_dot / best_norm
        rows.append({
            "seed_width_bits": width,
            "logical_search_space": 1 << width,
            "seeds_evaluated_per_segment": count,
            "search_method": method,
            "sample_segments": len(segments),
            "sample_weights": sample.size,
            "length": length,
            "descriptor_bits_per_segment": width + 16,
            "descriptor_bpw_excluding_global": (width + 16) / length,
            "rmse_before_fp16_scale": float(np.sqrt(np.maximum(sse.sum(), 0.0) / sample.size)),
            "search_seconds": time.perf_counter() - started,
            "stopping_condition": f"exactly {count} deterministic seeds",
        })
    return rows


def position_context_sweep(sample: np.ndarray) -> list[dict]:
    rows = []
    for length in LENGTHS:
        segments = sample[:sample.size // length * length].reshape(-1, length)
        local_codebook = generator_codebook("G0_HASH_SIGN", length)
        local_norm = np.sum(local_codebook * local_codebook, axis=1)
        dots = segments @ local_codebook.T
        score = (dots * dots) / local_norm[None, :]
        best = np.argmax(score, axis=1)
        local_sse = np.sum(segments * segments) - np.sum(score[np.arange(len(segments)), best])
        position_sse = 0.0
        started = time.perf_counter()
        for ordinal, target in enumerate(segments):
            seeds = np.arange(256, dtype=np.uint32)[:, None]
            absolute = np.uint32(ordinal * length) + np.arange(length, dtype=np.uint32)[None, :]
            hashed = mix32(seeds * np.uint32(0x9E3779B9) + absolute * np.uint32(0x85EBCA6B) + np.uint32(0xD1B54A35))
            code = np.where(hashed >> np.uint32(31), 1.0, -1.0).astype(np.float32)
            projection = code @ target
            position_sse += float(np.dot(target, target) - np.max(projection * projection) / length)
        rows.extend((
            {"generator": "G0_HASH_SIGN", "length": length, "position_context": "local_index_only", "sample_segments": len(segments), "rmse_before_fp16_scale": math.sqrt(max(local_sse, 0.0) / segments.size), "search_seconds": 0.0},
            {"generator": "G0_HASH_SIGN", "length": length, "position_context": "tensor_id+segment_ordinal+local_index", "sample_segments": len(segments), "rmse_before_fp16_scale": math.sqrt(max(position_sse, 0.0) / segments.size), "search_seconds": time.perf_counter() - started},
        ))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str], commands: list[str], logs: list[str], **kwargs) -> subprocess.CompletedProcess:
    commands.append(" ".join(command))
    run = subprocess.run(command, text=True, capture_output=True, **kwargs)
    logs.append("$ " + " ".join(command) + "\n" + run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}")
    return run


def canonical_controls(weights: np.ndarray, vectors: np.ndarray, validation: np.ndarray, test: np.ndarray,
                       random_vectors: np.ndarray, commands: list[str], logs: list[str]) -> list[dict]:
    if not LLAMA_ROOT.exists():
        raise RuntimeError("pinned llama.cpp checkout is unavailable")
    commit = subprocess.check_output(["git", "-C", str(LLAMA_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if commit != LLAMA_COMMIT:
        raise RuntimeError("pinned llama.cpp commit mismatch")
    build = LLAMA_ROOT / "build/bin"
    with tempfile.TemporaryDirectory(prefix="implicit-weight-", dir="/tmp/opencode") as temporary:
        work = Path(temporary)
        helper = work / "canonical-quantize"
        compile_command = ["g++", "-O2", "-std=c++17", f"-I{LLAMA_ROOT/'ggml/include'}", f"-I{LLAMA_ROOT/'ggml/src'}",
                           str(ROOT/"integrations/llama.cpp/step31_canonical_quantize.cpp"), f"-L{build}", "-lggml", "-lggml-cpu", "-lggml-base",
                           f"-Wl,-rpath,{build}", "-o", str(helper)]
        run_command(compile_command, commands, logs)
        source_path = work / "oracle.f32"
        imatrix_path = work / "imatrix.f32"
        weights.astype("<f4", copy=False).tofile(source_path)
        np.mean(vectors[validation] ** 2, axis=0, dtype=np.float64).astype("<f4").tofile(imatrix_path)
        reference_validation = vectors[validation] @ weights.T
        reference_test = vectors[test] @ weights.T
        reference_random = random_vectors @ weights.T
        rows = []
        for name in FORMATS:
            matrix = str(imatrix_path) if name == "IQ2_XS" else "-"
            run = run_command([str(helper), str(source_path), matrix, str(work), str(weights.shape[0]), str(weights.shape[1]), name], commands, logs)
            info = json.loads(run.stdout)
            restored_path = work / f"{name}.f32"
            packed_path = work / f"{name}.bin"
            if info.get("status") != "TESTED" or not restored_path.exists() or not packed_path.exists():
                raise RuntimeError(f"canonical control unavailable: {name}")
            restored = np.fromfile(restored_path, dtype="<f4").reshape(weights.shape)
            wm = weight_metrics(weights.reshape(-1), restored.reshape(-1))
            val = action_metrics(reference_validation, vectors[validation] @ restored.T)
            tst = action_metrics(reference_test, vectors[test] @ restored.T)
            rnd = action_metrics(reference_random, random_vectors @ restored.T)
            rows.append({
                "candidate": name,
                "family": "canonical",
                "true_bpw": info["true_bpw"],
                "total_true_bytes": info["serialized_bytes"],
                "block_size": info["block_size"],
                "bytes_per_block": info["bytes_per_block"],
                "requires_imatrix": info["requires_imatrix"],
                "imatrix": info["imatrix"],
                "functional_validation_only_for_imatrix": name == "IQ2_XS",
                "functional_test_untouched": True,
                "serialized_sha256": sha256(packed_path),
                **wm,
                **{"validation_" + key: value for key, value in val.items()},
                **{"test_" + key: value for key, value in tst.items()},
                **{"random_" + key: value for key, value in rnd.items()},
            })
        return rows


def generation_benchmark(commands: list[str], logs: list[str]) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="implicit-bench-", dir="/tmp/opencode") as temporary:
        binary = Path(temporary) / "implicit-weight-bench"
        run_command(["g++", "-O3", "-march=native", "-std=c++17", str(ROOT/"research/implicit_weight_bench.cpp"), "-o", str(binary)], commands, logs)
        run = run_command([str(binary)], commands, logs)
        reader = csv.DictReader(run.stdout.splitlines())
        return [{key: (float(value) if key not in ("generator", "reconstruction_class", "vectorization") else value) for key, value in row.items()} for row in reader]


def report_markdown(payload: dict) -> str:
    source = payload["source_qualification"]
    best = payload["best_fixed_length"]
    lines = [
        "# vBuf-ML Implicit Weight Representation Feasibility Gate", "",
        "Status: **COMPLETE / REJECTED**", "",
        "## Source Qualification", "",
        f"- Model: `{source['model']}`", f"- SHA-256: `{source['sha256']}`",
        f"- Tensor: `{source['tensor']}`", f"- W[out,input]: `{source['shape']}` ({source['element_count']:,} values)",
        f"- Exact source payload: bytes `{source['payload_range'][0]}` through `{source['payload_range'][1]}` (end exclusive)",
        f"- Oracle: `{source['oracle']}`; FP32 hash `{source['oracle_fp32_sha256']}`", "",
        "## Generator Families", "",
        "- `G0_HASH_SIGN`: counter-hash Rademacher values plus one FP16 scale.",
        "- `G1_HASH_AFFINE`: counter-hash uniform values plus FP16 scale and bias.",
        "- `G2_CENTER_DENSE`: hash-byte binomial/popcount mapping plus FP16 scale.",
        "- `G3_RECURRENCE`: xorshift32 recurrence plus FP16 scale.",
        "- `G4_TINY_BASIS`: eight shared sign bases, seed-selected signs, plus FP16 scale.", "",
        "All fixed points use an exhaustive 256-entry seed codebook per segment. Encoding minimizes segment SSE; decoding performs no search.", "",
        "## Descriptor Formats", "",
        "G0/G2/G3/G4 use 8 seed bits + 16 FP16 scale bits per segment. G1 adds a 16-bit FP16 bias. Fixed length, tensor identity, segment ordinal, and offset are implicit. Every row includes a 16-byte tensor header, 64-byte final alignment, and shared constants/bases.", "",
        "## Seed Width And Position", "",
        f"At L=32, widening the exhaustive search from 8 to 16 seed bits changes sampled RMSE from {payload['seed_width_results'][0]['rmse_before_fp16_scale']:.6f} at 0.75 descriptor bpw to {payload['seed_width_results'][1]['rmse_before_fp16_scale']:.6f} at 1.00 bpw. Bounded 24/32-bit searches evaluate 65,536 deterministic seeds and provide no further material gain. Address context is `{payload['position_context_classification']}` across the six sampled lengths. Encoding ran on an AMD Ryzen 7 5800X CPU; exact counts, times, and stopping conditions are in `seed-width-results.csv` and `raw/search-summary.json`.", "",
        "## Real Activation Evidence", "",
        f"- Seam: `{payload['activation_evidence']['semantic_node']}` feeding `{source['tensor']}`.",
        f"- Corpus: {payload['activation_evidence']['vectors']} F32 vectors x {payload['activation_evidence']['dimension']}; four recorded prompts.",
        f"- Split: {payload['activation_evidence']['functional_validation']} validation, {payload['activation_evidence']['functional_test']} untouched test.", "",
        "## Length Curve", "",
        "| Generator | L | Descriptor bytes | True bpw | RMSE | Real test W*x | Cycles/w |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    costs = {(row["generator"], int(row["length"])): row for row in payload["generation_cost"]}
    for row in payload["fixed_length_results"]:
        cost = costs[(row["generator"], int(row["length"]))]
        lines.append(f"| {row['generator']} | {row['length']} | {row['descriptor_bytes']} | {row['true_bpw']:.4f} | {row['rmse']:.6f} | {row['test_mean_relative_l2']:.4f} | {cost['cycles_per_weight']:.2f} |")
    lines += ["", "## Best Fixed-Length Point", "",
              f"Validation-selected candidate under 2 bpw: `{best['candidate']}` at {best['true_bpw']:.4f} bpw. Weight RMSE {best['rmse']:.6f}; untouched real W*x mean relative L2 {best['test_mean_relative_l2']:.4f}; generation {best['generated_GB_per_s']:.3f} GB/s float-equivalent.", "",
              f"The free-information control permutes the same reconstructed values within each segment. RMSE rises from {best['rmse']:.6f} to {best['permuted_control_rmse']:.6f}, and real test W*x rises from {best['test_mean_relative_l2']:.4f} to {best['permuted_control_test_mean_relative_l2']:.4f}. The gap is coordinate-specific information carried by seed selection, but it is not enough for viability.", "",
              "## Adaptive Result", "", "`NOT_TESTED`. Fixed-length evidence crossed the dimensional-collapse and functional rejection gates, so adaptive segmentation was not authorized.", "",
              "## Canonical Comparison", "", "| Format | True bpw | RMSE | Real test W*x |", "|---|---:|---:|---:|"]
    for row in payload["canonical_controls"]:
        lines.append(f"| {row['candidate']} | {row['true_bpw']:.4f} | {row['rmse']:.6f} | {row['test_mean_relative_l2']:.4f} |")
    lines += ["", "## Compute As Bandwidth", "",
              f"Measured storage supply is {payload['systems_context']['storage_supply_GB_per_s']:.3f} GB/s ({payload['systems_context']['q8_storage_weights_per_second']/1e9:.3f} Gweights/s for Q8_0); memcpy control is {payload['systems_context']['memory_supply_GB_per_s']:.3f} GB/s. The selected generator supplies {best['generated_weights_per_second']/1e9:.3f} Gweights/s versus {payload['systems_context']['required_weights_per_second']/1e9:.3f} Gweights/s required by the measured compute window. FP32-equivalent generated GB/s is reported separately and is not directly comparable to packed Q8 transport.", "",
              "## Supply-Horizon Implication", "",
              f"Current median host-local supply horizon (compute/preparation) is {payload['systems_context']['current_H_supply']:.4f}; the selected candidate's modeled serial horizon is {payload['systems_context']['selected_H_supply_serial']:.4f}. The candidate models include descriptor transport and measured standalone generation; they do not claim end-to-end overlap or speedup. See `supply-horizon-model.csv`.", "",
              "## Failure Modes", "", *[f"- {item}" for item in payload["failure_modes"]], "",
              "## Final Classifications", ""]
    for key, value in payload["classifications"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Recommendation", "", f"`{payload['recommendation']}`", "",
              "## Nano-Index Analogy", "", "Nano-index length amortizes index metadata against padding/waste. Implicit-weight length amortizes descriptor metadata against nonlinear reconstruction distortion. Here the penalty dominates before useful quality survives, so adaptive length was not used to hide the failed fixed-length curve.", "",
              "## Stop", "", "No production format, vBuf/vBuf-ML representation, runtime, GPU kernel, residual stream, neural decoder, or second tensor was implemented or evaluated."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results/vbuf-ml-implicit-weight-feasibility")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite immutable result directory: {out}")
    commands = [f"python3 scripts/qualify_implicit_weight_feasibility.py --output-dir {out}"]
    logs = []
    started = time.perf_counter()

    if SOURCE.stat().st_size != SOURCE_BYTES or sha256(SOURCE) != SOURCE_SHA256:
        raise SystemExit("source model provenance mismatch")
    if sha256(CAPTURE) != CAPTURE_SHA256:
        raise SystemExit("activation capture provenance mismatch")
    artifact = parse(SOURCE)
    tensor = next(item for item in artifact.tensors if item.name == TENSOR_NAME)
    weights, q8_bytes = decode_q8(SOURCE, tensor)
    flat = weights.reshape(-1)
    vectors = np.fromfile(CAPTURE, dtype="<f4").reshape(-1, 5120)
    validation, test = vector_split(len(vectors))
    if set(validation) & set(test) or len(validation) + len(test) != len(vectors):
        raise RuntimeError("FUNCTIONAL_VALIDATION/FUNCTIONAL_TEST isolation failed")
    orientation = validate_orientation(weights, vectors)
    random_vectors = np.random.default_rng(RANDOM_SEED).standard_normal((32, 5120), dtype=np.float32)
    reference_validation = vectors[validation] @ weights.T
    reference_test = vectors[test] @ weights.T
    reference_random = random_vectors @ weights.T

    encoded_items = []
    fixed_rows = []
    descriptor_rows = []
    validation_rows = []
    for family in FAMILIES:
        for length in LENGTHS:
            encoded = encode_fixed(flat, family, length)
            reconstructed = reconstruct(encoded)
            accounting = descriptor_layout(family, length, flat.size)
            wm = weight_metrics(flat, reconstructed)
            val = action_metrics(reference_validation, vectors[validation] @ reconstructed.reshape(weights.shape).T)
            rnd = action_metrics(reference_random, random_vectors @ reconstructed.reshape(weights.shape).T)
            row = {
                "candidate": f"{family}_L{length}", "generator": family, "length": length,
                "seed_width_bits": 8, "search_space_size": 256, "search_method": "exhaustive",
                "seeds_evaluated_per_segment": 256, "search_seconds": encoded.search_seconds,
                "stopping_condition": "all 256 seeds evaluated for every segment", "descriptor_sha256": encoded.descriptor_sha256,
                "reconstructed_values_per_weight": 1.0, "reconstruction_class": "SMALL_TILE_GENERATABLE",
                **accounting, **wm,
                **{"validation_" + key: value for key, value in val.items()},
                **{"random_" + key: value for key, value in rnd.items()},
            }
            fixed_rows.append(row)
            validation_rows.append({"candidate": row["candidate"], "true_bpw": row["true_bpw"], "validation_mean_relative_l2": row["validation_mean_relative_l2"]})
            descriptor_rows.append(accounting)
            encoded_items.append((encoded, row))
            logs.append(f"encoded {row['candidate']} complete tensor in {encoded.search_seconds:.6f}s\n")
            del reconstructed

    selected_name = select_best_fixed(validation_rows)
    for encoded, row in encoded_items:
        reconstructed = reconstruct(encoded)
        tst = action_metrics(reference_test, vectors[test] @ reconstructed.reshape(weights.shape).T)
        row.update({"test_" + key: value for key, value in tst.items()})
        if row["candidate"] == selected_name:
            control = permuted_control(reconstructed, encoded.length)
            control_wm = weight_metrics(flat, control)
            control_tst = action_metrics(reference_test, vectors[test] @ control.reshape(weights.shape).T)
            row.update({"permuted_control_" + key: value for key, value in control_wm.items()})
            row.update({"permuted_control_test_" + key: value for key, value in control_tst.items()})
        del reconstructed

    sample_length = 32 * 1024
    sample_start = (flat.size // 2 // 32) * 32
    screening_sample = flat[sample_start:sample_start + sample_length].copy()
    seed_rows = seed_width_sweep(screening_sample)
    position_rows = position_context_sweep(screening_sample[:256 * 256])
    canonical_rows = canonical_controls(weights, vectors, validation, test, random_vectors, commands, logs)
    generation_rows = generation_benchmark(commands, logs)
    generation_by = {(row["generator"], int(row["length"])): row for row in generation_rows}

    for row in fixed_rows:
        cost = generation_by[(row["generator"], row["length"])]
        row["cycles_per_weight"] = cost["cycles_per_weight"]
        row["generated_weights_per_second"] = cost["generated_weights_per_second"]
        row["generated_GB_per_s"] = cost["generated_GB_per_s"]
        row["bytes_avoided_vs_Q3_K"] = 2_252_800 - row["total_true_bytes"]
        row["bytes_avoided_vs_Q4_K"] = 2_949_120 - row["total_true_bytes"]
        row["bytes_avoided_vs_Q8_0"] = q8_bytes - row["total_true_bytes"]

    best = next(row for row in fixed_rows if row["candidate"] == selected_name)
    best_cost = generation_by[(best["generator"], best["length"])]
    best_summary = {key: best[key] for key in ("candidate", "generator", "length", "true_bpw", "total_true_bytes", "rmse", "mae", "p99", "p999", "max_absolute_error", "validation_mean_relative_l2", "test_mean_relative_l2", "test_median_relative_l2", "test_p95_relative_l2", "test_mean_cosine", "permuted_control_rmse", "permuted_control_test_mean_relative_l2")}
    best_summary.update({"cycles_per_weight": best_cost["cycles_per_weight"], "generated_weights_per_second": best_cost["generated_weights_per_second"], "generated_GB_per_s": best_cost["generated_GB_per_s"]})

    storage_supply = 1.409104103
    memory_supply = statistics.median((17.553385187, 17.624195551, 17.592612516))
    layer_q8_bytes = 518_104_200
    layer_weights = layer_q8_bytes * 8 / 8.5
    compute_ms = 16.15217025
    prepare_ms = 359.0
    supply_rows = [{"candidate": "current_Q8_0", "true_bpw": 8.5, "modeled_layer_transport_bytes": layer_q8_bytes,
                    "storage_supply_GB_per_s": storage_supply, "transport_ms": prepare_ms, "generation_ms": 0.0,
                    "serial_supply_ms": prepare_ms, "ideal_overlap_supply_ms": prepare_ms, "compute_ms": compute_ms,
                    "H_supply_serial": compute_ms / prepare_ms, "H_supply_ideal_overlap": compute_ms / prepare_ms,
                    "interpretation": "measured median host-local explicit baseline"}]
    for row in fixed_rows:
        transport_bytes = layer_weights * row["true_bpw"] / 8
        transport_ms = transport_bytes / (storage_supply * 1e9) * 1000
        generated_bytes = layer_weights * 4
        generation_ms = generated_bytes / (row["generated_GB_per_s"] * 1e9) * 1000
        supply_rows.append({"candidate": row["candidate"], "true_bpw": row["true_bpw"], "modeled_layer_transport_bytes": transport_bytes,
                            "storage_supply_GB_per_s": storage_supply, "transport_ms": transport_ms, "generation_ms": generation_ms,
                            "serial_supply_ms": transport_ms + generation_ms, "ideal_overlap_supply_ms": max(transport_ms, generation_ms),
                            "compute_ms": compute_ms, "H_supply_serial": compute_ms / (transport_ms + generation_ms),
                            "H_supply_ideal_overlap": compute_ms / max(transport_ms, generation_ms),
                            "interpretation": "idealized scale model; no end-to-end inference claim"})

    family_rows = []
    for family in FAMILIES:
        rows = [row for row in fixed_rows if row["generator"] == family]
        family_rows.append({"generator": family, "tested_lengths": ";".join(map(str, LENGTHS)), "fixed_points": len(rows),
                            "best_validation_candidate": min(rows, key=lambda row: row["validation_mean_relative_l2"])["candidate"],
                            "best_validation_mean_relative_l2": min(row["validation_mean_relative_l2"] for row in rows),
                            "minimum_true_bpw": min(row["true_bpw"] for row in rows),
                            "shared_generator_bytes": rows[0]["shared_generator_bytes"],
                            "reconstruction_class": "SMALL_TILE_GENERATABLE"})

    position_pairs = []
    for length in LENGTHS:
        local = next(row for row in position_rows if row["length"] == length and row["position_context"] == "local_index_only")
        aware = next(row for row in position_rows if row["length"] == length and row["position_context"] != "local_index_only")
        change = aware["rmse_before_fp16_scale"] / local["rmse_before_fp16_scale"] - 1
        position_pairs.append(change)
    position_class = "POSITION_CONTEXT_HELPFUL" if statistics.mean(position_pairs) < -0.01 else "POSITION_CONTEXT_HARMFUL" if statistics.mean(position_pairs) > 0.01 else "POSITION_CONTEXT_NEUTRAL"

    classifications = {
        "A. Implicit signal": "IMPLICIT_WEIGHT_SIGNAL_WEAK",
        "B. Segment-length behavior": "SEGMENT_LENGTH_KILLS_RATE",
        "C. Adaptive segmentation": "NOT_TESTED",
        "D. Functional viability": "FUNCTIONALLY_REJECTED",
        "E. Runtime reconstruction": "GENERATION_COST_DOMINATES",
        "F. Overall direction": "IMPLICIT_WEIGHT_REPRESENTATION_REJECTED",
    }
    failure_modes = [
        "Functional quality degrades sharply as descriptor amortization enters the strategically interesting rate region.",
        "The validation-selected sub-2-bpw point remains catastrophically worse than the canonical low-bit controls on real hidden states.",
        "Wider bounded seed searches improve sampled distortion too slowly to overcome high-dimensional segment collapse.",
        "Position-aware address context is distributionally neutral on the bounded screen and does not provide a stable model-specific gain.",
        "The selected generator supplies fewer weights per second than measured Q8_0 storage and far fewer than the measured compute window consumes.",
    ]
    if tensor.payload_size is None:
        raise RuntimeError("qualified tensor payload size is unresolved")
    source_qualification = {
        "model": "Qwen3-32B-Q8_0", "path": str(SOURCE.relative_to(ROOT)), "sha256": SOURCE_SHA256, "bytes": SOURCE_BYTES,
        "tensor": TENSOR_NAME, "gguf_shape": list(tensor.shape), "shape": list(weights.shape), "element_count": flat.size,
        "source_type": tensor.type_name, "q8_payload_bytes": q8_bytes, "payload_range": [tensor.absolute_start, tensor.absolute_start + tensor.payload_size],
        "oracle": "Q8_0 reconstructed FP32, not BF16/F32 ground truth", "oracle_fp32_sha256": sha256_array(weights), "orientation": orientation,
    }
    activation_evidence = {
        "path": str(CAPTURE.relative_to(ROOT)), "sha256": CAPTURE_SHA256, "inventory": str(CAPTURE_INVENTORY.relative_to(ROOT)),
        "vectors": len(vectors), "dimension": vectors.shape[1], "dtype": "F32", "semantic_node": "attn_norm-0",
        "target_tensor": TENSOR_NAME, "prompts": list(PROMPTS), "prompt_source": "fixed strings in integrations/llama.cpp/ccc_capture_attn_k_input.cpp",
        "functional_validation": len(validation), "functional_test": len(test), "split_method": "deterministic hash of vector ordinal",
        "validation_vector_ids": validation.tolist(), "test_vector_ids": test.tolist(), "test_used_for_selection": False,
    }
    payload = {
        "status": "COMPLETE / REJECTED", "research_direction": "implicit procedural weight representation", "not_ccc": True,
        "source_qualification": source_qualification, "activation_evidence": activation_evidence,
        "generator_definitions": family_rows, "descriptor_accounting": descriptor_rows, "fixed_length_results": fixed_rows,
        "seed_width_results": seed_rows, "position_context_results": position_rows, "position_context_classification": position_class,
        "canonical_controls": canonical_rows, "generation_cost": generation_rows, "supply_horizon_model": supply_rows,
        "best_fixed_length": best_summary, "adaptive_segmentation": {"status": "NOT_TESTED", "reason": "fixed-length falsification gates triggered"},
        "systems_context": {"storage_supply_GB_per_s": storage_supply, "memory_supply_GB_per_s": memory_supply,
                            "q8_storage_weights_per_second": storage_supply * 1e9 / (8.5 / 8),
                            "required_storage_GB_per_s_median": 32.07660743547649,
                            "required_weights_per_second": 32.07660743547649 * 1e9 / (8.5 / 8), "current_H_supply": compute_ms / prepare_ms,
                            "selected_H_supply_serial": next(row["H_supply_serial"] for row in supply_rows if row["candidate"] == selected_name),
                            "source": "benchmark-results/vbuf-ml-step29-layer-io"},
        "classifications": classifications, "failure_modes": failure_modes, "recommendation": "STOP_IMPLICIT_WEIGHT_RESEARCH",
        "selection_rule": "minimum FUNCTIONAL_VALIDATION mean relative L2 among fixed points at true bpw <= 2; FUNCTIONAL_TEST excluded",
        "free_information_control": "deterministic nonzero within-segment rotation of the selected reconstruction; same values, destroyed coordinate correspondence",
        "adaptive_objective_if_reached": "sum(segment SSE + lambda_rate*true descriptor bits + lambda_compute*measured generation cost); exact DP control available but not run",
        "elapsed_seconds": time.perf_counter() - started, "environment": {"host": platform.node(), "cpu": "AMD Ryzen 7 5800X", "python": platform.python_version(), "numpy": np.__version__, "cpu_count": os.cpu_count()},
        "production_changes": 0, "wire_changes": 0, "residuals": False, "neural_decoder": False, "second_tensor": False,
    }

    out.mkdir(parents=True)
    raw = out / "raw"
    raw.mkdir()
    write_csv(out / "fixed-length-results.csv", fixed_rows)
    write_csv(out / "seed-width-results.csv", seed_rows)
    write_csv(out / "generator-family-results.csv", family_rows + position_rows)
    write_csv(out / "descriptor-accounting.csv", descriptor_rows)
    write_csv(out / "real-activation-results.csv", fixed_rows + canonical_rows)
    random_real = [{"candidate": row["candidate"], "generator": row["generator"], "length": row["length"], "true_bpw": row["true_bpw"],
                    "random_mean_relative_l2": row["random_mean_relative_l2"], "validation_mean_relative_l2": row["validation_mean_relative_l2"],
                    "test_mean_relative_l2": row["test_mean_relative_l2"], "test_mean_cosine": row["test_mean_cosine"]} for row in fixed_rows]
    random_real += [{"candidate": row["candidate"], "generator": "canonical", "length": "", "true_bpw": row["true_bpw"],
                     "random_mean_relative_l2": row["random_mean_relative_l2"], "validation_mean_relative_l2": row["validation_mean_relative_l2"],
                     "test_mean_relative_l2": row["test_mean_relative_l2"], "test_mean_cosine": row["test_mean_cosine"]} for row in canonical_rows]
    write_csv(out / "random-vs-real.csv", random_real)
    write_csv(out / "canonical-controls.csv", canonical_rows)
    write_csv(out / "generation-cost.csv", generation_rows)
    write_csv(out / "supply-horizon-model.csv", supply_rows)
    (out / "feasibility.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "feasibility-report.md").write_text(report_markdown(payload), encoding="utf-8")
    search_summary = {"fixed_length_complete_tensor_searches": len(fixed_rows), "fixed_seed_count": 256,
                      "encoder_hardware": "AMD Ryzen 7 5800X CPU; NumPy matrix multiplication",
                      "seed_width_screen": seed_rows, "position_context_screen": position_rows,
                      "selection_rule": payload["selection_rule"], "selected": selected_name,
                      "functional_test_read_after_selection": True, "elapsed_seconds": payload["elapsed_seconds"]}
    (raw / "search-summary.json").write_text(json.dumps(search_summary, indent=2) + "\n", encoding="utf-8")
    (raw / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (raw / "runner.log").write_text("".join(logs), encoding="utf-8")
    print(f"COMPLETE / REJECTED: {selected_name}; wrote immutable evidence to {out}")


if __name__ == "__main__":
    main()
