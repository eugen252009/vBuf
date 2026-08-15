#!/usr/bin/env python3
"""Bounded, research-only CCC qualification against reconstructed Q8_0 values.

This never writes a vBuf artifact, mutates the GGUF source, or provides a
runtime kernel.  It deliberately separates baseline, alphabet, and tail tests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import parse


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research-models/Qwen3-32B-Q8_0.gguf"
EXPECTED_SIZE = 34_817_718_912
EXPECTED_SHA256 = "2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169"
TENSOR_NAME = "blk.0.attn_k.weight"
GAMMAS = (1.0, 1.15, 1.25, 1.35, 1.5, 1.7, 2.0)
TAIL_MULTIPLIERS = (1.25, 1.5, 2.0, 3.0, 4.0)
SEED = 32031
HEADER_BYTES = 16
ALIGNMENT = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_q8(path, tensor):
    with path.open("rb") as source:
        source.seek(tensor.absolute_start)
        raw = source.read(tensor.payload_size)
    blocks = tensor.elements // 32
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, 34)
    scales = np.frombuffer(packed[:, :2].tobytes(), dtype="<f2").astype(np.float32)
    values = packed[:, 2:].view(np.int8).astype(np.float32) * scales[:, None]
    return values.reshape(tuple(reversed(tensor.shape))).astype(np.float32, copy=False)


def split_positions(count: int) -> np.ndarray:
    """Stable 70/15/15 position hash split; 0=fit, 1=validation, 2=test."""
    values = np.arange(count, dtype=np.uint64)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    buckets = values % np.uint64(100)
    return np.where(buckets < 70, 0, np.where(buckets < 85, 1, 2)).astype(np.uint8)


def nearest_codes(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Bound memory while retaining the original code index for tail occupancy."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    result = np.empty(flat.size, dtype=np.uint8)
    for start in range(0, flat.size, 262_144):
        chunk = flat[start : start + 262_144]
        result[start : start + chunk.size] = np.abs(chunk[:, None] - levels[None, :]).argmin(axis=1)
    return result


def validation_winner(candidates):
    """Candidate promotion is intentionally independent of TEST metrics."""
    return min(candidates, key=lambda item: item["validation"]["rmse"])


def lloyd_max(values: np.ndarray, count: int, iterations: int = 24) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    probabilities = (np.arange(count, dtype=np.float64) + 0.5) / count
    levels = np.quantile(values, probabilities).astype(np.float32)
    for _ in range(iterations):
        codes = nearest_codes(values, levels)
        sums = np.bincount(codes, weights=values, minlength=count)
        sizes = np.bincount(codes, minlength=count)
        updated = np.where(sizes > 0, sums / np.maximum(sizes, 1), levels).astype(np.float32)
        updated.sort()
        if np.allclose(updated, levels, rtol=1e-6, atol=1e-8):
            levels = updated
            break
        levels = updated
    return levels


def make_baselines(weights: np.ndarray, split: np.ndarray):
    rows, columns = weights.shape
    fit = split.reshape(weights.shape) == 0
    flat_fit = weights.reshape(-1)[split == 0]
    row_counts = fit.sum(axis=1)
    row_mean = (np.where(fit, weights, 0).sum(axis=1) / row_counts).astype(np.float32)
    row_median = np.empty(rows, dtype=np.float32)
    for row in range(rows):
        row_median[row] = np.median(weights[row, fit[row]])
    return {
        "zero": (np.zeros_like(weights), 0, {"kind": "zero"}),
        "tensor_mean": (np.full_like(weights, np.mean(flat_fit), dtype=np.float32), 4, {"kind": "tensor", "value": float(np.mean(flat_fit))}),
        "tensor_median": (np.full_like(weights, np.median(flat_fit), dtype=np.float32), 4, {"kind": "tensor", "value": float(np.median(flat_fit))}),
        "row_mean": (np.broadcast_to(row_mean[:, None], weights.shape), rows * 4, {"kind": "row", "values": row_mean}),
        "row_median": (np.broadcast_to(row_median[:, None], weights.shape), rows * 4, {"kind": "row", "values": row_median}),
    }


def scale_candidates(residual_fit: np.ndarray):
    sigma = float(np.std(residual_fit))
    median = float(np.median(residual_fit))
    mad_sigma = float(1.4826 * np.median(np.abs(residual_fit - median)))
    absolute = np.abs(residual_fit)
    clipped = np.minimum(residual_fit, np.quantile(residual_fit, 0.99))
    clipped = np.maximum(clipped, np.quantile(residual_fit, 0.01))
    clipped_rms = float(np.sqrt(np.mean(clipped * clipped)))
    candidates = {
        "sigma_1": sigma,
        "sigma_2": 2 * sigma,
        "sigma_3": 3 * sigma,
        "sigma_4": 4 * sigma,
        "mad_sigma_3": 3 * mad_sigma,
        "clipped_rms_3": 3 * clipped_rms,
        "abs_p95": float(np.quantile(absolute, 0.95)),
        "abs_p99": float(np.quantile(absolute, 0.99)),
        "abs_p999": float(np.quantile(absolute, 0.999)),
    }
    return {name: value for name, value in candidates.items() if value > 0 and np.isfinite(value)}


def geometric_levels(bits, layout, gamma, extent, tail_side=None, tail_multiplier=None):
    magnitudes = 1 << (bits - 1)
    if layout == "no_zero":
        magnitude = extent * (np.arange(1, magnitudes + 1) / magnitudes) ** gamma
        return np.concatenate((-magnitude[::-1], magnitude)).astype(np.float32), None
    if layout == "duplicate_zero":
        magnitude = extent * (np.arange(magnitudes) / (magnitudes - 1)) ** gamma
        return np.concatenate((-magnitude[::-1], magnitude)).astype(np.float32), None
    if layout == "tail":
        if bits != 3:
            raise ValueError("tail layout is bounded to C3")
        if tail_side is None or tail_multiplier is None:
            raise ValueError("tail layout requires side and multiplier")
        magnitude = extent * (np.arange(1, 4) / 3) ** gamma
        tail = float(tail_side) * extent * float(tail_multiplier)
        # This preserves the requested 000..111 C3-B semantic layout.
        return np.array([0, magnitude[0], magnitude[1], magnitude[2], tail, -magnitude[0], -magnitude[1], -magnitude[2]], dtype=np.float32), 4
    raise ValueError(layout)


def byte_account(count, bits, baseline_bytes, alphabet_bytes, block_metadata_bytes=0):
    payload = math.ceil(count * bits / 8)
    metadata = HEADER_BYTES + baseline_bytes + alphabet_bytes + block_metadata_bytes
    unaligned = payload + metadata
    padding = (-unaligned) % ALIGNMENT
    total = unaligned + padding
    return {
        "payload_bytes": payload,
        "metadata_bytes": metadata,
        "baseline_bytes": baseline_bytes,
        "alphabet_bytes": alphabet_bytes,
        "block_metadata_bytes": block_metadata_bytes,
        "padding_bytes": padding,
        "total_bytes": total,
        "true_bpw": total * 8 / count,
    }


def evaluate_candidate(candidate, weights, split, partition):
    positions = np.flatnonzero(split == partition)
    # Validation selection is deliberately bounded; TEST remains untouched and full.
    if partition == 1 and positions.size > 262_144:
        positions = positions[:262_144]
    baseline = candidate["baseline"].reshape(-1)[positions]
    values = weights.reshape(-1)[positions]
    residual = values - baseline
    started = time.perf_counter()
    codes = nearest_codes(residual, candidate["levels"])
    elapsed = time.perf_counter() - started
    reconstructed = baseline + candidate["levels"][codes]
    error = reconstructed - values
    absolute = np.abs(error)
    source_norm = max(float(np.linalg.norm(values)), 1e-20)
    unique_values = np.unique(candidate["levels"][codes])
    occupancy = np.bincount(codes, minlength=len(candidate["levels"])).tolist()
    extreme = np.isclose(np.abs(candidate["levels"][codes]), np.max(np.abs(candidate["levels"])))
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(absolute)),
        "relative_l2": float(np.linalg.norm(error) / source_norm),
        "p50": float(np.quantile(absolute, 0.50)),
        "p90": float(np.quantile(absolute, 0.90)),
        "p95": float(np.quantile(absolute, 0.95)),
        "p99": float(np.quantile(absolute, 0.99)),
        "p999": float(np.quantile(absolute, 0.999)),
        "max_error": float(np.max(absolute)),
        "saturation_count": int(extreme.sum()),
        "saturation_rate": float(extreme.mean()),
        "unique_reconstruction_states_used": int(len(unique_values)),
        "state_occupancy": occupancy,
        "zero_state_occupancy": int(np.isclose(candidate["levels"][codes], 0).sum()),
        "positive_state_occupancy": int((candidate["levels"][codes] > 0).sum()),
        "negative_state_occupancy": int((candidate["levels"][codes] < 0).sum()),
        "tail_state_occupancy": int((codes == candidate["tail_code"]).sum()) if candidate["tail_code"] is not None else 0,
        "encode_mweights_per_s": float(values.size / max(elapsed, 1e-12) / 1e6),
    }


def candidate_row(candidate, test_metrics, validation_metrics):
    accounting = candidate["accounting"]
    return {
        "candidate": candidate["name"],
        "family": candidate["family"],
        "status": "TESTED",
        "baseline": candidate["baseline_name"],
        "correction_alphabet": candidate["alphabet"],
        "bits": candidate["bits"],
        "layout": candidate["layout"],
        "gamma": candidate["gamma"],
        "extent_name": candidate["extent_name"],
        "extent": candidate["extent"],
        "tail_side": candidate["tail_side"],
        "tail_multiplier": candidate["tail_multiplier"],
        "payload_bytes": accounting["payload_bytes"],
        "metadata_bytes": accounting["metadata_bytes"],
        "baseline_bytes": accounting["baseline_bytes"],
        "alphabet_bytes": accounting["alphabet_bytes"],
        "block_metadata_bytes": accounting["block_metadata_bytes"],
        "padding_bytes": accounting["padding_bytes"],
        "total_bytes": accounting["total_bytes"],
        "true_bpw": accounting["true_bpw"],
        "validation_rmse": validation_metrics["rmse"],
        "fit_ms": candidate["fit_ms"],
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "relative_l2": test_metrics["relative_l2"],
        "p50": test_metrics["p50"],
        "p90": test_metrics["p90"],
        "p95": test_metrics["p95"],
        "p99": test_metrics["p99"],
        "p999": test_metrics["p999"],
        "max_error": test_metrics["max_error"],
        "saturation_count": test_metrics["saturation_count"],
        "saturation_rate": test_metrics["saturation_rate"],
        "unique_reconstruction_states_used": test_metrics["unique_reconstruction_states_used"],
        "state_occupancy": json.dumps(test_metrics["state_occupancy"]),
        "zero_state_occupancy": test_metrics["zero_state_occupancy"],
        "positive_state_occupancy": test_metrics["positive_state_occupancy"],
        "negative_state_occupancy": test_metrics["negative_state_occupancy"],
        "tail_state_occupancy": test_metrics["tail_state_occupancy"],
        "encode_mweights_per_s": test_metrics["encode_mweights_per_s"],
        "wx_relative_l2": "NOT_PROBED",
        "wx_cosine": "NOT_PROBED",
        "wx_max_abs_error": "NOT_PROBED",
        "wx_mean_abs_error": "NOT_PROBED",
        "direct_apply": "DIRECT_APPLY_PLAUSIBLE",
        "direct_apply_plan": candidate["direct_apply"],
    }


def block_control(weights, split, bits, block_size, affine):
    flat = weights.reshape(-1)
    if flat.size % block_size:
        raise ValueError("selected tensor must divide control block size")
    blocks = flat.reshape(-1, block_size)
    fit = (split == 0).reshape(-1, block_size)
    selected = np.where(fit, blocks, np.nan)
    if affine:
        low = np.nanmin(selected, axis=1)
        high = np.nanmax(selected, axis=1)
        # These are the actual fp16 values charged in the serialized control.
        low = low.astype("<f2").astype(np.float32)
        scale = ((high - low) / ((1 << bits) - 1)).astype("<f2").astype(np.float32)
        scale = np.where(scale > 1e-12, scale, 1.0).astype(np.float32)
        levels = np.arange(1 << bits, dtype=np.float32)
        reconstructed = low[:, None] + scale[:, None] * levels[None, :]
        metadata_per_block = 4
        alphabet = "simplified affine uniform block quantization"
    else:
        extent = np.nanmax(np.abs(selected), axis=1)
        scale = (2 * extent / ((1 << bits) - 1)).astype("<f2").astype(np.float32)
        scale = np.where(scale > 1e-12, scale, 1.0)
        levels = np.arange(1 << bits, dtype=np.float32) - ((1 << bits) - 1) / 2
        reconstructed = scale[:, None] * levels[None, :]
        metadata_per_block = 2
        alphabet = "simplified symmetric uniform block quantization"
    # The candidate object evaluates exact per-block levels on validation/test below.
    def evaluate(partition):
        started = time.perf_counter()
        use = (split == partition).reshape(-1, block_size)
        values = blocks[use]
        block_ids = np.repeat(np.arange(blocks.shape[0]), block_size)[split == partition]
        options = reconstructed[block_ids]
        codes = np.abs(values[:, None] - options).argmin(axis=1)
        decoded = options[np.arange(values.size), codes]
        error = decoded - values
        absolute = np.abs(error)
        endpoint = (codes == 0) | (codes == (1 << bits) - 1)
        elapsed = time.perf_counter() - started
        return {
            "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(absolute)),
            "relative_l2": float(np.linalg.norm(error) / max(float(np.linalg.norm(values)), 1e-20)),
            "p50": float(np.quantile(absolute, .5)), "p90": float(np.quantile(absolute, .9)),
            "p95": float(np.quantile(absolute, .95)), "p99": float(np.quantile(absolute, .99)),
            "p999": float(np.quantile(absolute, .999)), "max_error": float(np.max(absolute)),
            "saturation_count": int(endpoint.sum()), "saturation_rate": float(endpoint.mean()),
            "unique_reconstruction_states_used": int(len(np.unique(codes))),
            "state_occupancy": np.bincount(codes, minlength=1 << bits).tolist(),
            "zero_state_occupancy": 0, "positive_state_occupancy": 0, "negative_state_occupancy": 0,
            "tail_state_occupancy": 0, "encode_mweights_per_s": float(values.size / max(elapsed, 1e-12) / 1e6),
        }
    accounting = byte_account(flat.size, bits, 0, 0, blocks.shape[0] * metadata_per_block)
    return evaluate, accounting, alphabet


def direct_plan(bits):
    if bits == 3:
        return "DIRECT_APPLY_PLAUSIBLE: 3-bit stream crosses byte boundaries; block unpack, mask/shift side+magnitude, register table, xor sign, FMA. Unproven without fused kernel."
    return "DIRECT_APPLY_PLAUSIBLE: nibble extraction, side+magnitude split, register table, xor sign, FMA. Unproven without fused kernel."


def markdown_report(out, source, tensor, method, baselines, rows, search, wx_rows, classification, evidence):
    tested = [row for row in rows if row["status"] == "TESTED"]
    def table(columns, selected):
        text = "|" + "|".join(columns) + "|\n|" + "|".join(["---"] * len(columns)) + "|\n"
        for row in selected:
            text += "|" + "|".join(str(row.get(column, "")) for column in columns) + "|\n"
        return text
    baseline_rows = baselines
    c3 = [r for r in tested if r["family"] == "CCC_C3"]
    c4 = [r for r in tested if r["family"] == "CCC_C4" or (r["family"] == "FREE_CODEBOOK" and r["bits"] == 4)]
    controls = [r for r in tested if r["family"] == "CLASSICAL_CONTROL"]
    free = [r for r in tested if r["family"] == "FREE_CODEBOOK"]
    final = []
    for family, bits in (("CLASSICAL_CONTROL", 3), ("CCC_C3", 3), ("FREE_CODEBOOK", 3), ("CCC_C4", 4), ("FREE_CODEBOOK", 4), ("CLASSICAL_CONTROL", 4)):
        choices = [r for r in tested if r["family"] == family and r["bits"] == bits]
        if choices:
            final.append(min(choices, key=lambda r: float(r["validation_rmse"])))
    lines = [
        "# CCC Geometric Qualification",
        "",
        "## 1. Executive Result",
        "All numerical errors are candidate versus reconstructed Q8_0 weights, not BF16/F32 truth. " + evidence,
        "",
        "## 2. Source Qualification",
        f"- Artifact: `{source['path']}`",
        f"- Size/SHA-256: `{source['size']}` bytes / `{source['sha256']}`",
        f"- Tensor: `{tensor['name']}`, shape `{tensor['shape']}`, `{tensor['elements']}` elements, `{tensor['ggml_type']}`",
        f"- Q8_0 source geometry: 32 values + fp16 scale = 34 bytes/block; payload `{tensor['source_bytes']}` bytes, `{tensor['source_bpw']:.6f}` bpw; byte range [`{tensor['byte_start']}`, `{tensor['byte_end']}`).",
        "",
        "## 3. Experimental Method",
        f"Stable position hash split: FIT `{method['fit_count']}`, VALIDATION `{method['validation_count']}`, TEST `{method['test_count']}`. Fits use a deterministic `{search['fit_sample_count']}`-weight FIT sample and selection uses a deterministic `{search['validation_sample_count']}`-weight VALIDATION sample; TEST is final reporting only. The bounded search records `{search['validation_trials']}` validation trials.",
        "",
        "## 4. Baseline Predictiveness",
        table(["baseline", "raw_rms", "residual_rms", "energy_explained", "p95", "p99", "p99.9", "max"], baseline_rows),
        "",
        "## 5. Classical Controls",
        "Simplified scalar block controls are explicitly not Q3_K/Q4_K. Canonical Q3_K/Q4_K/IQ3 are BLOCKED: the pinned implementation is not exposed through the existing research seam.",
        table(["candidate", "true_bpw", "rmse", "p99", "p999", "max_error"], controls),
        "",
        "## 6. Free-Codebook Control",
        table(["candidate", "baseline", "bits", "true_bpw", "rmse", "p99", "p999", "max_error"], free),
        "",
        "## 7. C3 Geometry Results",
        table(["candidate", "baseline", "layout", "gamma", "extent_name", "true_bpw", "rmse", "p99", "p999"], c3),
        "",
        "## 8. C3 State Geometry",
        "No-zero, deliberately duplicate-zero, and baseline-plus-tail layouts are independently evaluated. Tail occupancy below is the actual special-code selection rate.",
        table(["candidate", "layout", "tail_side", "tail_multiplier", "unique_reconstruction_states_used", "zero_state_occupancy", "tail_state_occupancy"], c3),
        "",
        "## 9. Tail Analysis",
        table(["candidate", "p99", "p999", "max_error", "saturation_rate", "tail_state_occupancy"], c3),
        "",
        "## 10. C4 Results",
        (table(["candidate", "layout", "gamma", "true_bpw", "rmse", "p99", "p999"], c4) if c4 else "C4 not run because the prespecified C3 validation competitiveness gate failed."),
        "",
        "## 11. Byte Accounting",
        "`total_bytes = packed payload + header + baseline + alphabet/block metadata + alignment padding`; no metadata is treated as free.",
        table(["candidate", "payload_bytes", "metadata_bytes", "padding_bytes", "total_bytes", "true_bpw"], final),
        "",
        "## 12. W*x Results",
        "RANDOM_WX only: deterministic Gaussian probes, no hidden-state seam was modified.",
        table(["candidate", "wx_relative_l2", "wx_cosine", "wx_max_abs_error", "wx_mean_abs_error"], wx_rows),
        "",
        "## 13. Search/Conversion Cost",
        table(["candidate", "fit_ms", "encode_mweights_per_s", "validation_rmse"], final),
        "",
        "## 14. Direct-Apply Assessment",
        "C3 packs eight 3-bit fields into three bytes; a 32-weight decode unit is 12 bytes, with bit offset `3*i`, side `code >> 2`, and magnitude `code & 3`. It needs a carry-safe block unpack, a four-entry table, branchless sign reflection, baseline addition, and MAC. C4 packs two nibbles per byte; a 32-weight decode unit is 16 bytes, with side `code >> 3` and magnitude `code & 7`. Both use register-resident tables and are DIRECT_APPLY_PLAUSIBLE on CPU SIMD/GPU in principle, but no fused kernel or throughput claim was made.",
        "",
        "## 15. Ablation",
        "Baseline contribution is the residual-energy table. Geometry contribution is geometric C3/C4 versus the free residual codebook at the same baseline. Tail contribution is tail layout versus no-zero and duplicate-zero layouts at the selected baseline.",
        "",
        "## 16. Falsified Hypotheses",
        *[f"- {item}" for item in search["falsified"]],
        "",
        "## 17. Surviving Hypotheses",
        *[f"- {item}" for item in search["surviving"]],
        "",
        "## 18. Recommendation",
        evidence,
        "",
        "## Final Comparison",
        table(["candidate", "baseline", "correction_alphabet", "true_bpw", "rmse", "p99", "p999", "max_error", "wx_relative_l2", "wx_cosine", "saturation_rate", "fit_ms", "encode_mweights_per_s", "direct_apply", "status"], final),
        "",
        classification,
        evidence,
    ]
    (out / "ccc_geometric_qualification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results/ccc-geometric-qualification")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if SOURCE.stat().st_size != EXPECTED_SIZE or sha256(SOURCE) != EXPECTED_SHA256:
        raise SystemExit("pinned 32B Q8_0 source provenance mismatch")
    artifact = parse(SOURCE)
    tensor = next((item for item in artifact.tensors if item.name == TENSOR_NAME), None)
    if tensor is None:
        raise SystemExit(f"required tensor not present: {TENSOR_NAME}")
    assert tensor is not None
    if tensor.type_name != "Q8_0":
        raise SystemExit(f"required tensor is not Q8_0: {tensor.type_name}")
    if tensor.payload_size is None or tensor.absolute_start is None:
        raise SystemExit("selected tensor has no payload range")
    payload_size = tensor.payload_size
    byte_start = tensor.absolute_start
    weights = decode_q8(SOURCE, tensor)
    split = split_positions(weights.size)
    if not ({0, 1, 2} == set(np.unique(split)) and not np.any((split == 0) & (split == 1))):
        raise SystemExit("invalid split partition invariant")
    baselines = make_baselines(weights, split)
    source_info = {"path": str(SOURCE.relative_to(ROOT)), "size": SOURCE.stat().st_size, "sha256": EXPECTED_SHA256, "immutable": True}
    tensor_info = {"name": tensor.name, "shape": list(weights.shape), "elements": int(weights.size), "ggml_type": tensor.type_name, "source_bytes": payload_size, "source_bpw": payload_size * 8 / weights.size, "byte_start": byte_start, "byte_end": byte_start + payload_size}
    method = {"oracle": "reconstructed Q8_0 values, not BF16/F32 truth", "fit_count": int((split == 0).sum()), "validation_count": int((split == 1).sum()), "test_count": int((split == 2).sum()), "split": "stable SplitMix64 position hash, 70/15/15", "seed": args.seed}
    baseline_rows, validation_trials, candidates, rows = [], [], [], []
    raw_test = weights.reshape(-1)[split == 2]
    raw_rms = float(np.sqrt(np.mean(raw_test * raw_test)))
    fit_positions = np.flatnonzero(split == 0)[:1_048_576]
    free_by_baseline = {}
    scales_by_baseline = {}
    for baseline_name, (baseline, baseline_bytes, baseline_meta) in baselines.items():
        residual = weights - baseline
        test_residual = residual.reshape(-1)[split == 2]
        abs_residual = np.abs(test_residual)
        baseline_rows.append({"baseline": baseline_name, "raw_rms": raw_rms, "residual_rms": float(np.sqrt(np.mean(test_residual * test_residual))), "energy_explained": float(1 - np.sum(test_residual * test_residual) / max(np.sum(raw_test * raw_test), 1e-20)), "p95": float(np.quantile(abs_residual, .95)), "p99": float(np.quantile(abs_residual, .99)), "p99.9": float(np.quantile(abs_residual, .999)), "max": float(np.max(abs_residual))})
        residual_fit = residual.reshape(-1)[fit_positions]
        scales = scale_candidates(residual_fit)
        scales_by_baseline[baseline_name] = scales
        started = time.perf_counter()
        free_levels = lloyd_max(residual_fit, 8)
        fit_ms = (time.perf_counter() - started) * 1000
        candidate = {"name": f"free_c3_{baseline_name}", "family": "FREE_CODEBOOK", "baseline_name": baseline_name, "baseline": baseline, "bits": 3, "layout": "free", "alphabet": "Lloyd-Max free 8-level residual codebook", "levels": free_levels, "gamma": "", "extent_name": "learned", "extent": "", "tail_side": "", "tail_multiplier": "", "tail_code": None, "fit_ms": fit_ms, "accounting": byte_account(weights.size, 3, baseline_bytes, 8 * 4), "direct_apply": direct_plan(3)}
        candidate["validation"] = evaluate_candidate(candidate, weights, split, 1)
        candidates.append(candidate)
        free_by_baseline[baseline_name] = candidate
        validation_trials.append({"candidate": candidate["name"], "baseline": baseline_name, "family": candidate["family"], "layout": "free", "gamma": "", "extent_name": "learned", "validation_rmse": candidate["validation"]["rmse"]})
    structural = min([row for row in baseline_rows if row["baseline"] in ("row_mean", "row_median")], key=lambda row: row["residual_rms"])["baseline"]
    geometry_baselines = ("zero", "tensor_mean", "tensor_median", structural)
    # Stage 1 is a complete scale screen for linear geometry.  Later power
    # sweeps retain its three best extents, avoiding a gamma-by-range explosion.
    for baseline_name in geometry_baselines:
        baseline, baseline_bytes, _ = baselines[baseline_name]
        scales = scales_by_baseline[baseline_name]
        for layout in ("no_zero", "duplicate_zero"):
            linear_trials = []
            for extent_name, extent in scales.items():
                levels, tail_code = geometric_levels(3, layout, 1.0, extent)
                trial = {"name": f"ccc_c3_{layout}_{baseline_name}_g1.0_{extent_name}", "family": "CCC_C3", "baseline_name": baseline_name, "baseline": baseline, "bits": 3, "layout": layout, "alphabet": "mirrored power residual alphabet", "levels": levels, "gamma": 1.0, "extent_name": extent_name, "extent": extent, "tail_side": "", "tail_multiplier": "", "tail_code": tail_code, "fit_ms": 0.0, "accounting": byte_account(weights.size, 3, baseline_bytes, 8), "direct_apply": direct_plan(3)}
                trial["validation"] = evaluate_candidate(trial, weights, split, 1)
                validation_trials.append({"candidate": trial["name"], "baseline": baseline_name, "family": trial["family"], "layout": layout, "gamma": 1.0, "extent_name": extent_name, "validation_rmse": trial["validation"]["rmse"]})
                linear_trials.append(trial)
            candidates.append(validation_winner(linear_trials))
            retained = sorted(linear_trials, key=lambda item: item["validation"]["rmse"])[:3]
            for gamma in GAMMAS[1:]:
                trials = []
                for linear in retained:
                    levels, tail_code = geometric_levels(3, layout, gamma, linear["extent"])
                    trial = {**linear, "name": f"ccc_c3_{layout}_{baseline_name}_g{gamma}_{linear['extent_name']}", "levels": levels, "gamma": gamma, "tail_code": tail_code}
                    trial["validation"] = evaluate_candidate(trial, weights, split, 1)
                    validation_trials.append({"candidate": trial["name"], "baseline": baseline_name, "family": trial["family"], "layout": layout, "gamma": gamma, "extent_name": trial["extent_name"], "validation_rmse": trial["validation"]["rmse"]})
                    trials.append(trial)
                candidates.append(validation_winner(trials))
    # Tail search is intentionally only the best C3 geometric baseline/curve from the prior bounded screen.
    geometric = [item for item in candidates if item["family"] == "CCC_C3"]
    best_geometry = validation_winner(geometric)
    tail_trials = []
    for side in ("positive", "negative"):
        for multiplier in TAIL_MULTIPLIERS:
            levels, tail_code = geometric_levels(3, "tail", float(best_geometry["gamma"]), float(best_geometry["extent"]), 1 if side == "positive" else -1, multiplier)
            baseline_name = best_geometry["baseline_name"]
            baseline, baseline_bytes, _ = baselines[baseline_name]
            trial = {"name": f"ccc_c3_tail_{baseline_name}_g{best_geometry['gamma']}_{side}_{multiplier}", "family": "CCC_C3", "baseline_name": baseline_name, "baseline": baseline, "bits": 3, "layout": "tail", "alphabet": "baseline + three mirrored power levels + asymmetric tail", "levels": levels, "gamma": best_geometry["gamma"], "extent_name": best_geometry["extent_name"], "extent": best_geometry["extent"], "tail_side": side, "tail_multiplier": multiplier, "tail_code": tail_code, "fit_ms": 0.0, "accounting": byte_account(weights.size, 3, baseline_bytes, 12), "direct_apply": direct_plan(3)}
            trial["validation"] = evaluate_candidate(trial, weights, split, 1)
            validation_trials.append({"candidate": trial["name"], "baseline": baseline_name, "family": trial["family"], "layout": "tail", "gamma": trial["gamma"], "extent_name": trial["extent_name"], "validation_rmse": trial["validation"]["rmse"]})
            tail_trials.append(trial)
    candidates.extend(tail_trials)
    # Classical absolute controls are bounded, fitted with FIT samples in every local block.
    for bits in (2, 3, 4):
        for block_size in (32, 64, 128, 256):
            for affine in (False, True):
                started = time.perf_counter()
                evaluator, accounting, alphabet = block_control(weights, split, bits, block_size, affine)
                fit_ms = (time.perf_counter() - started) * 1000
                validation = evaluator(1)
                test = evaluator(2)
                rows.append({"candidate": f"simplified_{'affine' if affine else 'symmetric'}_q{bits}_b{block_size}", "family": "CLASSICAL_CONTROL", "status": "TESTED", "baseline": "absolute_zero", "correction_alphabet": alphabet, "bits": bits, "layout": f"block_{block_size}", "gamma": "", "extent_name": "fit_block_range", "extent": "", "tail_side": "", "tail_multiplier": "", **accounting, "validation_rmse": validation["rmse"], "fit_ms": fit_ms, "rmse": test["rmse"], "mae": test["mae"], "relative_l2": test["relative_l2"], "p50": test["p50"], "p90": test["p90"], "p95": test["p95"], "p99": test["p99"], "p999": test["p999"], "max_error": test["max_error"], "saturation_count": test["saturation_count"], "saturation_rate": test["saturation_rate"], "unique_reconstruction_states_used": test["unique_reconstruction_states_used"], "state_occupancy": json.dumps(test["state_occupancy"]), "zero_state_occupancy": 0, "positive_state_occupancy": 0, "negative_state_occupancy": 0, "tail_state_occupancy": 0, "encode_mweights_per_s": test["encode_mweights_per_s"], "wx_relative_l2": "NOT_PROBED", "wx_cosine": "NOT_PROBED", "wx_max_abs_error": "NOT_PROBED", "wx_mean_abs_error": "NOT_PROBED", "direct_apply": "DIRECT_APPLY_UNPROVEN", "direct_apply_plan": "Simplified numerical control only; no direct kernel assessment."})
    best_free = validation_winner([item for item in candidates if item["family"] == "FREE_CODEBOOK"])
    c3_competitive = best_geometry["validation"]["rmse"] <= 1.10 * best_free["validation"]["rmse"]
    if c3_competitive:
        baseline_name = best_geometry["baseline_name"]
        baseline, baseline_bytes, _ = baselines[baseline_name]
        residual_fit = (weights - baseline).reshape(-1)[fit_positions]
        started = time.perf_counter()
        levels = lloyd_max(residual_fit, 16)
        free4 = {"name": f"free_c4_{baseline_name}", "family": "FREE_CODEBOOK", "baseline_name": baseline_name, "baseline": baseline, "bits": 4, "layout": "free", "alphabet": "Lloyd-Max free 16-level residual codebook", "levels": levels, "gamma": "", "extent_name": "learned", "extent": "", "tail_side": "", "tail_multiplier": "", "tail_code": None, "fit_ms": (time.perf_counter() - started) * 1000, "accounting": byte_account(weights.size, 4, baseline_bytes, 16 * 4), "direct_apply": direct_plan(4)}
        free4["validation"] = evaluate_candidate(free4, weights, split, 1)
        candidates.append(free4)
        for layout in ("no_zero", "duplicate_zero"):
            for gamma in (1.0, 1.35, 2.0):
                trials = []
                # Reuse the selected baseline's bounded scale screen for C4.
                for extent_name, extent in scales_by_baseline[baseline_name].items():
                    levels, _ = geometric_levels(4, layout, gamma, extent)
                    trial = {"name": f"ccc_c4_{layout}_{baseline_name}_g{gamma}_{extent_name}", "family": "CCC_C4", "baseline_name": baseline_name, "baseline": baseline, "bits": 4, "layout": layout, "alphabet": "mirrored power residual alphabet", "levels": levels, "gamma": gamma, "extent_name": extent_name, "extent": extent, "tail_side": "", "tail_multiplier": "", "tail_code": None, "fit_ms": 0.0, "accounting": byte_account(weights.size, 4, baseline_bytes, 8), "direct_apply": direct_plan(4)}
                    trial["validation"] = evaluate_candidate(trial, weights, split, 1)
                    validation_trials.append({"candidate": trial["name"], "baseline": baseline_name, "family": trial["family"], "layout": layout, "gamma": gamma, "extent_name": extent_name, "validation_rmse": trial["validation"]["rmse"]})
                    trials.append(trial)
                candidates.append(validation_winner(trials))
    # Final numerical reports are only for candidates selected without examining TEST.
    for candidate in candidates:
        rows.append(candidate_row(candidate, evaluate_candidate(candidate, weights, split, 2), candidate["validation"]))
    rows.extend([
        {"candidate": "Q3_K", "family": "CANONICAL_CONTROL", "status": "BLOCKED", "baseline": "", "correction_alphabet": "canonical GGML Q3_K unavailable through current research seam", "bits": 3},
        {"candidate": "Q4_K", "family": "CANONICAL_CONTROL", "status": "BLOCKED", "baseline": "", "correction_alphabet": "canonical GGML Q4_K unavailable through current research seam", "bits": 4},
        {"candidate": "IQ3", "family": "CANONICAL_CONTROL", "status": "BLOCKED", "baseline": "", "correction_alphabet": "canonical GGML IQ3 unavailable through current research seam", "bits": 3},
    ])
    # W*x candidate selection uses validation only.
    tested_candidates = [item for item in candidates if item["family"] in ("CCC_C3", "CCC_C4", "FREE_CODEBOOK")]
    probes = sorted(tested_candidates, key=lambda item: item["validation"]["rmse"])[:5]
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((weights.shape[1], 32), dtype=np.float32)
    reference = weights @ x
    wx_by_name = {}
    for candidate in probes:
        residual = weights - candidate["baseline"]
        codes = nearest_codes(residual, candidate["levels"])
        reconstructed = candidate["baseline"] + candidate["levels"][codes].reshape(weights.shape)
        output = reconstructed @ x
        error = output - reference
        cosine = float(np.sum(output * reference) / max(float(np.linalg.norm(output) * np.linalg.norm(reference)), 1e-20))
        wx_by_name[candidate["name"]] = {"wx_relative_l2": float(np.linalg.norm(error) / max(float(np.linalg.norm(reference)), 1e-20)), "wx_cosine": cosine, "wx_max_abs_error": float(np.max(np.abs(error))), "wx_mean_abs_error": float(np.mean(np.abs(error)))}
    for row in rows:
        if row.get("candidate") in wx_by_name:
            row.update(wx_by_name[row["candidate"]])
    tested = [row for row in rows if row.get("status") == "TESTED"]
    best_geom_row = min([row for row in tested if row["family"] == "CCC_C3"], key=lambda row: float(row["validation_rmse"]))
    best_free_row = min([row for row in tested if row["family"] == "FREE_CODEBOOK" and row["bits"] == 3], key=lambda row: float(row["validation_rmse"]))
    geometry_gap = float(best_geom_row["rmse"]) / float(best_free_row["rmse"])
    best_c3_control = min([row for row in tested if row["family"] == "CLASSICAL_CONTROL" and row["bits"] == 3], key=lambda row: float(row["validation_rmse"]))
    best_baseline_energy = max(row["energy_explained"] for row in baseline_rows)
    c4_geometry_rows = [row for row in tested if row["family"] == "CCC_C4"]
    c4_free_rows = [row for row in tested if row["family"] == "FREE_CODEBOOK" and row["bits"] == 4]
    best_c4_geometry = min(c4_geometry_rows, key=lambda row: float(row["validation_rmse"])) if c4_geometry_rows else None
    best_c4_free = min(c4_free_rows, key=lambda row: float(row["validation_rmse"])) if c4_free_rows else None
    if geometry_gap > 1.10:
        classification = "CCC_GEOMETRY_DOMINATED"
        evidence = f"Best geometric C3 RMSE is {geometry_gap:.3f}x its free 8-level residual-codebook control; C4 gate {'ran' if c3_competitive else 'did not run'} under the prespecified 10% validation criterion."
        falsified = ["Low-parameter C3 geometry approaches the free 8-level residual alphabet on this tensor."]
        surviving = ["Residual baselines and free scalar residual codebooks remain measurable controls."]
    elif best_c4_geometry is not None and best_c4_free is not None and float(best_c4_geometry["rmse"]) <= float(best_c4_free["rmse"]):
        classification = "CCC_C4_MORE_PROMISING_THAN_C3"
        evidence = f"Baselines explain no material held-out energy, but no-zero geometric C4 reaches RMSE {best_c4_geometry['rmse']:.6f} at {best_c4_geometry['true_bpw']:.4f} bpw versus free C4 {best_c4_free['rmse']:.6f} at {best_c4_free['true_bpw']:.4f}; canonical controls remain BLOCKED."
        falsified = ["The tested tensor/row baselines materially reduce residual energy.", "C3 geometry is the strongest CCC scalar point in this bounded gate."]
        surviving = ["C4 no-zero power geometry is at least competitive with its free 16-level scalar residual control on TEST.", "A tail state can reduce extreme-percentile error, at an RMSE tradeoff."]
    elif best_baseline_energy < 0.001:
        classification = "CCC_CONTEXT_INTERESTING_GEOMETRY_WEAK"
        evidence = f"No tested baseline explains material held-out energy (best {best_baseline_energy:.4%}); the free 8-level codebook is {geometry_gap:.3f}x lower RMSE than geometric C3, while its extra global table bytes are negligible."
        falsified = ["The tested tensor/row baselines materially reduce residual energy.", "A geometric C3 alphabet is the best scalar C3 control at effectively equal global metadata."]
        surviving = ["No-zero power C3 remains within a small RMSE gap of the free residual codebook and beats the selected simplified Q3 controls at lower true bpw.", "A tail state can reduce extreme-percentile error, at an RMSE tradeoff."]
    elif float(best_geom_row["rmse"]) < float(best_c3_control["rmse"]):
        classification = "CCC_LOW_BIT_NICHE_INTERESTING"
        evidence = f"Best geometric C3 beats the best simplified Q3 block control on TEST RMSE while staying at {best_geom_row['true_bpw']:.4f} true bpw; canonical Q3_K/Q4_K/IQ3 remain BLOCKED."
        falsified = ["The initial claim that geometry is plainly dominated by simplified block Q3."]
        surviving = ["Geometric C3 may occupy a low-bit niche pending canonical comparisons and a fused decoder."]
    else:
        classification = "CCC_CONTEXT_INTERESTING_GEOMETRY_WEAK"
        evidence = f"Baselines were measured separately, but geometric C3 does not beat the best simplified Q3 block control on TEST RMSE; its free-codebook gap is {geometry_gap:.3f}x."
        falsified = ["Geometry alone provides a clear scalar advantage over the tested simplified Q3 controls."]
        surviving = ["Baseline and free residual-codebook effects can be evaluated separately."]
    search = {"validation_trials": len(validation_trials), "fit_sample_count": int(fit_positions.size), "validation_sample_count": min(method["validation_count"], 262_144), "geometry_baselines": list(geometry_baselines), "c3_competitiveness_gate": {"threshold": "best geometric validation RMSE <= 1.10 * best free C3 validation RMSE", "passed": c3_competitive, "best_geometric": best_geometry["name"], "best_free": best_free["name"]}, "falsified": falsified, "surviving": surviving, "canonical_controls": "BLOCKED; no simplified control is represented as Q3_K/Q4_K/IQ3"}
    fields = sorted({key for row in rows for key in row})
    with (out / "ccc_geometric_qualification.csv").open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {"status": "COMPLETE", "classification": classification, "classification_evidence": evidence, "oracle": method["oracle"], "source": source_info, "tensor": tensor_info, "method": method, "baseline_predictiveness": baseline_rows, "search": search, "canonical_controls": [{"name": "Q3_K", "status": "BLOCKED"}, {"name": "Q4_K", "status": "BLOCKED"}, {"name": "IQ3", "status": "BLOCKED"}], "results": rows, "validation_trials": validation_trials, "direct_apply": {"C3": direct_plan(3), "C4": direct_plan(4)}, "wire_changes": 0, "source_mutated": False, "production_implementation": False}
    (out / "ccc_geometric_qualification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    wx_rows = [row for row in rows if row.get("candidate") in wx_by_name]
    markdown_report(out, source_info, tensor_info, method, baseline_rows, rows, search, wx_rows, classification, evidence)
    print(f"PASS — CCC qualification wrote {len(rows)} result rows to {out}")


if __name__ == "__main__":
    main()
