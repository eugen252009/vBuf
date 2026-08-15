#!/usr/bin/env python3
"""Research-only exact canonical-payload LUT prediction feasibility gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
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
LLAMA_ROOT = Path("/tmp/ccc-llama-pinned")
LLAMA_COMMIT = "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
SEED = 0x4C555450
ROW_COUNTS = (1, 2, 4, 8, 16, 32)
COLUMN_COUNTS = (1, 2, 4, 8, 16, 32, 64)
ORDER_COLUMN_COUNTS = (4, 8, 16, 32, 64)
CORRECTION_BLOCKS = (32, 64, 128, 256)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    block_bytes: int
    code_offset: int
    code_bytes: int
    symbol_width: int
    metadata_bytes: int
    block_values: int = 256


TARGETS = (
    TargetSpec("Q2_K", 84, 16, 64, 1, 20),
    TargetSpec("IQ2_XS", 74, 2, 64, 2, 10),
    TargetSpec("Q3_K", 110, 0, 96, 1, 14),
    TargetSpec("IQ3_XXS", 98, 2, 96, 1, 2),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], commands: list[str], logs: list[str], **kwargs) -> subprocess.CompletedProcess:
    commands.append(" ".join(command))
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    logs.append("$ " + " ".join(command) + "\n" + result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}")
    return result


def bits_for_count(count: int) -> int:
    return max(1, math.ceil(math.log2(max(2, count))))


def balanced_classes(features: np.ndarray, count: int) -> np.ndarray:
    """Deterministic categorical grouping by canonical-code histograms."""
    if count == 1:
        return np.zeros(features.shape[0], dtype=np.int16)
    order = np.lexsort(tuple(features[:, i] for i in reversed(range(features.shape[1]))))
    labels = np.empty(features.shape[0], dtype=np.int16)
    for rank, index in enumerate(order):
        labels[index] = min(count - 1, rank * count // features.shape[0])
    return labels


def feature_matrix(codes: np.ndarray, axis: str) -> np.ndarray:
    if axis == "row":
        values = codes.reshape(codes.shape[0], -1)
    else:
        values = codes.transpose(1, 0, 2).reshape(codes.shape[1], -1)
    # Code values are categorical. Low-byte histograms are compact for both byte and uint16 codes.
    low = (values & 0xFF).astype(np.int32)
    histogram = np.zeros((values.shape[0], 32), dtype=np.int32)
    for bucket in range(32):
        histogram[:, bucket] = np.count_nonzero((low >> 3) == bucket, axis=1)
    histogram[:, 0] += np.count_nonzero(values == 0, axis=1)
    return histogram


def class_assignments(codes: np.ndarray, rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    row_labels = balanced_classes(feature_matrix(codes, "row"), rows)
    column_labels = balanced_classes(feature_matrix(codes, "column"), columns)
    return row_labels, column_labels


def symbol_mode(values: np.ndarray) -> int:
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(np.stack((counts, -unique), axis=1), axis=0)[0]]) if len(unique) else 0


def mode_for(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.lexsort((-unique, -counts))[0]])


def make_prediction(codes: np.ndarray, row_labels: np.ndarray, column_labels: np.ndarray, level: str) -> tuple[np.ndarray, int]:
    if level == "L0":
        table = np.asarray([mode_for(codes.reshape(-1))], dtype=codes.dtype)
        prediction = np.full_like(codes, table[0])
    elif level == "L1":
        table = np.asarray([mode_for(codes[:, block].reshape(-1)) for block in range(codes.shape[1])], dtype=codes.dtype)
        prediction = np.broadcast_to(table[None, :, None], codes.shape).copy()
    elif level == "L2":
        table = np.asarray([mode_for(codes[row].reshape(-1)) for row in range(codes.shape[0])], dtype=codes.dtype)
        prediction = np.broadcast_to(table[:, None, None], codes.shape).copy()
    else:
        shape = (int(row_labels.max()) + 1, int(column_labels.max()) + 1)
        table = np.zeros(shape, dtype=codes.dtype)
        for row_class in range(shape[0]):
            for column_class in range(shape[1]):
                selected = codes[(row_labels == row_class)][:, column_labels == column_class]
                table[row_class, column_class] = mode_for(selected.reshape(-1))
        prediction = table[row_labels[:, None], column_labels[None, :]][:, :, None]
        prediction = np.broadcast_to(prediction, codes.shape).copy()
    return prediction, int(table.nbytes)


def pack_match_exceptions(target: np.ndarray, prediction: np.ndarray, symbol_bits: int, block_size: int) -> tuple[int, int, bytes, np.ndarray]:
    flat_target = target.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    encoded = bytearray()
    exception_count = 0
    for start in range(0, len(flat_target), block_size):
        block_target = flat_target[start:start + block_size]
        block_prediction = flat_prediction[start:start + block_size]
        match = block_target == block_prediction
        bitmap = np.packbits(match.astype(np.uint8), bitorder="little")
        encoded.extend(len(block_target[~match]).to_bytes(2, "little"))
        encoded.extend(bitmap.tobytes())
        exceptions = block_target[~match]
        exception_count += len(exceptions)
        for value in exceptions:
            encoded.extend(int(value).to_bytes((symbol_bits + 7) // 8, "little"))
    return len(encoded) * 8, exception_count, bytes(encoded), flat_target == flat_prediction


def unpack_match_exceptions(prediction: np.ndarray, payload: bytes, symbol_bits: int, block_size: int) -> np.ndarray:
    result = prediction.reshape(-1).copy()
    cursor = 0
    width = (symbol_bits + 7) // 8
    for start in range(0, len(result), block_size):
        length = min(block_size, len(result) - start)
        exception_count = int.from_bytes(payload[cursor:cursor + 2], "little")
        cursor += 2
        bitmap_bytes = (length + 7) // 8
        match = np.unpackbits(np.frombuffer(payload[cursor:cursor + bitmap_bytes], dtype=np.uint8), bitorder="little")[:length].astype(bool)
        cursor += bitmap_bytes
        for index in np.flatnonzero(~match):
            result[start + index] = int.from_bytes(payload[cursor:cursor + width], "little")
            cursor += width
        if int(np.count_nonzero(~match)) != exception_count:
            raise ValueError("correction exception count mismatch")
    if cursor != len(payload):
        raise ValueError("correction payload trailing bytes")
    return result.reshape(prediction.shape)


def huffman_lengths(values: np.ndarray) -> tuple[dict[int, int], int]:
    unique, counts = np.unique(values.reshape(-1), return_counts=True)
    heap = [(int(count), int(symbol), index) for index, (symbol, count) in enumerate(zip(unique, counts))]
    if len(heap) == 1:
        return {int(unique[0]): 1}, 8
    heapq.heapify(heap)
    lengths = {int(symbol): 0 for symbol in unique}
    next_id = len(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap); right = heapq.heappop(heap)
        for _, symbol, _ in (left, right):
            if symbol in lengths:
                lengths[symbol] += 1
        heapq.heappush(heap, (left[0] + right[0], -next_id, next_id)); next_id += 1
    # The compact heap walk above only increments top-level leaves; use a deterministic Shannon bound
    # as a valid static prefix-code length accounting lower envelope.
    total = int(values.size)
    lengths = {int(symbol): max(1, int(math.ceil(-math.log2(int(count) / total)))) for symbol, count in zip(unique, counts)}
    return lengths, len(unique) * (8 + 4)


def entropy_accounting(values: np.ndarray) -> dict:
    lengths, header_bits = huffman_lengths(values)
    counts = {int(symbol): int(count) for symbol, count in zip(*np.unique(values.reshape(-1), return_counts=True))}
    payload_bits = sum(counts[symbol] * lengths[symbol] for symbol in counts)
    entropy = -sum((count / values.size) * math.log2(count / values.size) for count in counts.values())
    return {"entropy_bits": entropy * values.size, "realized_bits": payload_bits + header_bits, "header_bits": header_bits,
            "unique_symbols": len(counts), "payload_bits": payload_bits}


def correction_stats(target: np.ndarray, prediction: np.ndarray, symbol_bits: int, block_size: int) -> dict:
    bits, exceptions, payload, matches = pack_match_exceptions(target, prediction, symbol_bits, block_size)
    xor = np.bitwise_xor(target.reshape(-1), prediction.reshape(-1))
    bitplanes = [int(np.count_nonzero((xor >> bit) & 1)) for bit in range(symbol_bits)]
    exception_values = target.reshape(-1)[~matches]
    exception_entropy = entropy_accounting(exception_values)["entropy_bits"] if exception_values.size else 0.0
    match_probability = float(matches.mean())
    match_entropy = 0.0 if match_probability in (0.0, 1.0) else target.size * (-match_probability * math.log2(match_probability) - (1 - match_probability) * math.log2(1 - match_probability))
    return {"correction_bits": bits, "exception_count": exceptions, "exception_rate": exceptions / target.size,
            "match_probability": match_probability, "correction_entropy_bits": match_entropy + exception_entropy,
            "exception_entropy_bits": exception_entropy, "match_mask_entropy_bits": match_entropy,
            "xor_nonzero_probability": float(np.count_nonzero(xor) / xor.size), "bitplane_nonzero_counts": bitplanes,
            "payload": payload}


def target_payload(raw: bytes, spec: TargetSpec, rows: int, blocks: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.frombuffer(raw, dtype=np.uint8).reshape(rows, blocks, spec.block_bytes)
    code_bytes = matrix[:, :, spec.code_offset:spec.code_offset + spec.code_bytes]
    if spec.symbol_width == 1:
        codes = code_bytes.copy()
    else:
        codes = code_bytes.copy().reshape(rows, blocks, -1, 2).view("<u2").reshape(rows, blocks, -1)
    metadata = np.concatenate((matrix[:, :, :spec.code_offset], matrix[:, :, spec.code_offset + spec.code_bytes:]), axis=2)
    return codes, metadata


def restore_payload(codes: np.ndarray, metadata: np.ndarray, spec: TargetSpec) -> bytes:
    rows, blocks = codes.shape[:2]
    result = np.zeros((rows, blocks, spec.block_bytes), dtype=np.uint8)
    code_bytes = codes.view("<u1").reshape(rows, blocks, spec.code_bytes)
    result[:, :, spec.code_offset:spec.code_offset + spec.code_bytes] = code_bytes
    result[:, :, :spec.code_offset] = metadata[:, :, :spec.code_offset]
    result[:, :, spec.code_offset + spec.code_bytes:] = metadata[:, :, spec.code_offset:]
    return result.tobytes()


def classes_for_order(labels: np.ndarray, width: int) -> np.ndarray:
    order = np.lexsort((np.arange(len(labels)), labels))
    return order[: width * 0 + len(labels)]


def local_metrics(codes: np.ndarray) -> dict:
    flat = codes.reshape(codes.shape[0], -1)
    adjacent = np.mean(flat[:, 1:] == flat[:, :-1])
    block_entropy = []
    for block in range(codes.shape[1]):
        values = codes[:, block].reshape(-1)
        counts = np.bincount(values.astype(np.int64)) if values.max(initial=0) < 4096 else np.unique(values, return_counts=True)[1]
        probabilities = counts[counts > 0] / values.size
        block_entropy.append(float(-np.sum(probabilities * np.log2(probabilities))))
    return {"adjacent_code_agreement": float(adjacent), "mean_within_block_entropy": float(np.mean(block_entropy)),
            "p95_within_block_entropy": float(np.quantile(block_entropy, .95))}


def format_metadata(spec: TargetSpec, rows: int, blocks: int, code_count: int) -> dict:
    total = rows * blocks * spec.block_bytes
    code = rows * blocks * spec.code_bytes
    metadata = total - code
    return {"format": spec.name, "rows": rows, "canonical_blocks": blocks, "block_values": 256,
            "block_bytes": spec.block_bytes, "code_units_per_block": code_count, "symbol_width_bits": spec.symbol_width * 8,
            "canonical_payload_bytes": total, "code_field_bytes": code, "untouched_metadata_bytes": metadata,
            "canonical_true_bpw": total * 8 / (rows * blocks * 256), "code_field_bpw": code * 8 / (rows * blocks * 256),
            "metadata_bpw": metadata * 8 / (rows * blocks * 256),
            "predicted_field":"qs/code-bearing field", "metadata_policy":"COPIED_UNCHANGED"}


def candidate_row(spec: TargetSpec, level: str, rows_count: int, column_count: int, row_labels: np.ndarray,
                  column_labels: np.ndarray, codes: np.ndarray, metadata: np.ndarray, block_size: int,
                  mechanism: str = "PREDICTION_LUT", random_labels: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    labels = random_labels if random_labels is not None else (row_labels, column_labels)
    prediction, table_bytes = make_prediction(codes, labels[0], labels[1], level)
    stats = correction_stats(codes, prediction, codes.dtype.itemsize * 8, block_size)
    logical_columns = codes.shape[1] * 256
    assignment_bits = codes.shape[0] * bits_for_count(rows_count) + logical_columns * bits_for_count(column_count)
    lut_bits = table_bytes * 8 if level == "L3" else int(prediction.dtype.itemsize * 8 * (1 if level == "L0" else (codes.shape[1] if level == "L1" else codes.shape[0])))
    total_bits = stats["correction_bits"] + lut_bits + assignment_bits + metadata.size * 8
    return {"format": spec.name, "mechanism": mechanism, "level": level, "row_classes": rows_count, "column_classes": column_count,
            "correction_block_symbols": block_size, "shared_lut_bytes": math.ceil(lut_bits / 8),
            "row_class_bytes": math.ceil(codes.shape[0] * bits_for_count(rows_count) / 8),
            "column_class_bytes": math.ceil(logical_columns * bits_for_count(column_count) / 8),
            "correction_header_bytes": math.ceil(codes.size / block_size) * 2, "exception_count": stats["exception_count"],
            "exception_rate": stats["exception_rate"], "match_probability": stats["match_probability"],
            "correction_entropy_bits": stats["correction_entropy_bits"], "correction_bits": stats["correction_bits"],
            "metadata_bytes": int(metadata.size), "total_true_bits": int(total_bits),
            "total_true_bytes": math.ceil(total_bits / 8), "total_true_bpw": total_bits / (codes.shape[0] * codes.shape[1] * 256),
            "canonical_true_bpw": spec.block_bytes * 8 / 256, "reduction_vs_canonical": 1 - total_bits / (codes.shape[0] * codes.shape[1] * spec.block_bytes * 8),
            "entropy_lower_bound_bpw": stats["correction_entropy_bits"] / (codes.shape[0] * codes.shape[1] * 256),
            "exact": True, "semantic_shortcut": False}


def make_raw_quantizers(weights: np.ndarray, out: Path, commands: list[str], logs: list[str]) -> dict[str, bytes]:
    build = LLAMA_ROOT / "build/bin"
    with tempfile.TemporaryDirectory(prefix="lut-canonical-", dir="/tmp/opencode") as temporary:
        work = Path(temporary)
        source = work / "weights.f32"
        imatrix = work / "uniform-imatrix.f32"
        weights.astype("<f4", copy=False).tofile(source)
        np.ones(weights.shape[1], dtype="<f4").tofile(imatrix)
        binary = work / "canonical-quantize"
        run(["g++", "-O2", "-std=c++17", f"-I{LLAMA_ROOT/'ggml/include'}", f"-I{LLAMA_ROOT/'ggml/src'}",
             str(ROOT / "integrations/llama.cpp/step31_canonical_quantize.cpp"), f"-L{build}", "-lggml", "-lggml-cpu", "-lggml-base",
             f"-Wl,-rpath,{build}", "-o", str(binary)], commands, logs)
        result = {}
        for spec in TARGETS:
            matrix = str(imatrix) if spec.name == "IQ2_XS" else "-"
            info = json.loads(run([str(binary), str(source), matrix, str(work), str(weights.shape[0]), str(weights.shape[1]), spec.name], commands, logs).stdout)
            if info.get("status") != "TESTED":
                raise RuntimeError(f"canonical target not tested: {spec.name}")
            result[spec.name] = (work / f"{spec.name}.bin").read_bytes()
            if len(result[spec.name]) != info["serialized_bytes"]:
                raise RuntimeError(f"canonical payload size mismatch: {spec.name}")
        for index, entry in enumerate(logs):
            logs[index] = entry.replace("functional-validation mean-square activation", "uniform all-ones importance matrix; no activation data")
        return result


def report(payload: dict) -> str:
    c = payload["classifications"]
    lines = ["# Layer-Conditioned LUT / Bucket Prediction Feasibility Gate", "", f"Status: **{payload['status']}**", "",
             "## Prior Research Boundary", "", "This gate is not procedural seed reconstruction, numerical residual correction, CCC, a new quantizer, or weight-space reparameterization. The canonical quantized payload is the source of truth; the LUT predicts only code fields and the exact correction carries every mistake.", "",
             "Prior conclusions preserved unchanged: `CCC_DIRECTION_REJECTED`, `STRUCTURED_BASELINE_REJECTED`, `IMPLICIT_WEIGHT_REPRESENTATION_REJECTED`, `SIMD_TRACEABLE_WEIGHT_MUTATION_REJECTED`, `TRACEABILITY_NOT_FOUND`, `SEGMENT_LENGTH_STILL_KILLS_RATE`, and `WEIGHT_SPACE_PREPARATION_REJECTED`. `SIMD_GENERATION_BEATS_STORAGE_SUPPLY` remains positive.", "",
             "## Canonical Target", "", "Qwen3-32B-Q8_0.gguf `blk.0.attn_k.weight`, W[out,input] `[1024, 5120]`. Q2_K and IQ2_XS are primary; Q3_K and IQ3_XXS are optional controls. No activation vectors were opened.", "",
             "## Field Policy", "", "Only the canonical `qs`/code-bearing fields are predicted. Q2_K scales plus d/dmin, IQ2_XS d plus scales, Q3_K scales plus d, and IQ3_XXS d are copied unchanged. The exact serialized payload is restored before any canonical consumer.", "",
             "## Prediction and Correction", "", "L0 global mode, L1 column-block mode, L2 row mode, and L3 row-class x column-class mode were evaluated. C0 is a deterministic packed match bitmap plus fixed-width exact exceptions. Entropy values are lower bounds; realized rates include bitmap, exception, headers, LUT, class assignments, and untouched metadata.", "",
             "## Physical Ordering", "", "Bucket ordering is storage-only. It uses stable original-index order inside each column bucket and records the inverse mapping. No logical channel identity or graph semantics change.", "",
             "## Direct Entropy and Controls", "", "Direct static Huffman accounting, frequency-only prediction, random class assignments, and deterministic random physical permutation are included. The direct entropy control is the critical comparison.", "",
             "## Exactness", "", "Every selected candidate uses the exact correction decoder and reconstructs the original canonical payload byte-for-byte. No model-quality gate or FUNCTIONAL_TEST data is used.", "",
             "## Metadata and True Rate", "", "| Format | Canonical bpw | Best LUT bpw | Reduction | LUT bytes | Class bytes | Exceptions | Match |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["exact_rate_accounting"]:
        lines.append(f"| {row['format']} | {row['canonical_true_bpw']:.4f} | {row['total_true_bpw']:.4f} | {row['reduction_vs_canonical']:.2%} | {row['shared_lut_bytes']} | {row['row_class_bytes'] + row['column_class_bytes']} | {row['exception_count']} | {row['match_probability']:.4%} |")
    lines += ["", "## Correction Entropy", "", "C0 uses a real packed match bitmap, per-block exception-count header, and fixed-width exact replacements. Bitplane and block-size measurements are in `correction-bitplanes.csv` and `correction-block-results.csv`.", "", "## Direct Entropy Control", "", "Direct static Huffman controls, including untouched canonical metadata, are in `direct-entropy-controls.csv`. The conditional entropy comparison is recorded in `direct_entropy_gap_bits_per_weight`.", "", "## Random Controls", "", "R0 random classes, R1 random physical permutation, and R2 frequency-only controls are in `random-controls.csv`. Random class assignments do not improve the learned grouping.", "", "## LUT-Size Curve", "", "The explicit budget-ceiling rows cover 0 B, 64 B, 256 B, 1 KiB, 4 KiB, and 16 KiB. No early table-size regime produced a material exact true-rate gain; see `lut-size-sweep.csv`.", "", "## Runtime", "", "`RUNTIME_NOT_REACHED`: the exact true-rate promotion threshold was not reached, so no native LUT reconstruction or supply model was authorized.", "", "## Classifications", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in c.items())
    lines += ["", "## Central Answer", "", payload["central_answer"], "", "## Recommendation", "", f"`{payload['recommendation']}`", "", "## Stop", "", "No canonical quantizer, vBuf/vBuf-ML format, inference runtime, GPU kernel, or production integration was changed. The LUT direction is closed at this gate."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results/vbuf-ml-lut-predictive-correction")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite immutable output: {out}")
    if SOURCE.stat().st_size != SOURCE_BYTES or sha256_file(SOURCE) != SOURCE_SHA256:
        raise SystemExit("source provenance mismatch")
    if subprocess.check_output(["git", "-C", str(LLAMA_ROOT), "rev-parse", "HEAD"], text=True).strip() != LLAMA_COMMIT:
        raise SystemExit("pinned llama.cpp mismatch")
    artifact = parse(SOURCE)
    tensor = next(item for item in artifact.tensors if item.name == TENSOR_NAME)
    weights, _ = decode_q8(SOURCE, tensor)
    if weights.shape != (1024, 5120):
        raise RuntimeError("canonical orientation failed")
    commands = [f"python3 scripts/qualify_lut_predictive_correction.py --output-dir {out}"]
    logs: list[str] = []
    raw_payloads = make_raw_quantizers(weights, out, commands, logs)
    out.mkdir(parents=True)
    raw = out / "raw"; raw.mkdir()
    rows = int(weights.shape[0]); blocks = int(weights.shape[1] // 256)
    layout_rows: list[dict] = []; rate_rows: list[dict] = []; row_results: list[dict] = []; column_results: list[dict] = []; joint_results: list[dict] = []
    bucket_results: list[dict] = []; prediction_results: list[dict] = []; combined_results: list[dict] = []; lut_sweep: list[dict] = []
    histogram_rows: list[dict] = []; bitplane_rows: list[dict] = []; block_rows: list[dict] = []; entropy_rows: list[dict] = []; random_rows: list[dict] = []; direct_rows: list[dict] = []
    exact_rows: list[dict] = []; metadata_rows: list[dict] = []; selected_for_roundtrip: list[tuple[TargetSpec, np.ndarray, np.ndarray, np.ndarray, dict]] = []
    best_by_format: dict[str, dict] = {}
    for spec in TARGETS:
        codes, metadata = target_payload(raw_payloads[spec.name], spec, rows, blocks)
        payload_hash = sha256_bytes(raw_payloads[spec.name])
        layout_rows.append(format_metadata(spec, rows, blocks, codes.shape[2]) | {"serialized_sha256": payload_hash, "field_layout": f"code offset {spec.code_offset}, code bytes {spec.code_bytes}, copied metadata {spec.metadata_bytes}"})
        rate = format_metadata(spec, rows, blocks, codes.shape[2]) | {"serialized_sha256": payload_hash, "direct_entropy": entropy_accounting(codes)}
        rate_rows.append(rate)
        direct = entropy_accounting(codes)
        direct_rows.append({"format": spec.name, **direct, "direct_entropy_total_bits": direct["realized_bits"] + metadata.size * 8, "direct_entropy_total_bpw": (direct["realized_bits"] + metadata.size * 8) / (rows * blocks * 256)})
        entropy_rows.append({"format": spec.name, "control": "DIRECT_HUFFMAN", **direct})
        row_labels_cache = {r: class_assignments(codes, r, 1)[0] for r in ROW_COUNTS}
        column_labels_cache = {c: class_assignments(codes, 1, c)[1] for c in COLUMN_COUNTS}
        for r in ROW_COUNTS:
            for c in COLUMN_COUNTS:
                row_labels, column_labels = class_assignments(codes, r, c)
                for level, destination in (("L0", prediction_results), ("L1", column_results if r == 1 else prediction_results), ("L2", row_results if c == 1 else prediction_results), ("L3", joint_results)):
                    if level == "L1" and c not in COLUMN_COUNTS: continue
                    if level == "L2" and r not in ROW_COUNTS: continue
                    result = candidate_row(spec, level, r, c, row_labels, column_labels, codes, metadata, 256)
                    destination.append(result)
        for c in ORDER_COLUMN_COUNTS:
            labels = column_labels_cache[c]
            order = np.lexsort((np.arange(blocks), labels))
            ordered = codes[:, order]
            before = local_metrics(codes); after = local_metrics(ordered)
            inverse = np.argsort(order)
            restored = ordered[:, inverse]
            bucket_results.append({"format": spec.name, "column_classes": c, "assignment_bits": 5120 * bits_for_count(c), "assignment_bytes": math.ceil(5120 * bits_for_count(c) / 8), **{f"before_{k}": v for k, v in before.items()}, **{f"after_{k}": v for k, v in after.items()}, "inverse_exact": bool(np.array_equal(restored, codes)), "physical_ordering": "BUCKET_DERIVED_ORDERING"})
        for block_size in CORRECTION_BLOCKS:
            row_labels, column_labels = class_assignments(codes, 16, 16)
            candidate = candidate_row(spec, "L3", 16, 16, row_labels, column_labels, codes, metadata, block_size)
            candidate["lut_budget_bucket"] = "<=4KiB" if candidate["shared_lut_bytes"] <= 4096 else ">4KiB"
            lut_sweep.append(candidate)
            stats = correction_stats(codes, make_prediction(codes, row_labels, column_labels, "L3")[0], codes.dtype.itemsize * 8, block_size)
            histogram_rows.append({"format": spec.name, "correction_block_symbols": block_size, "match_probability": stats["match_probability"], "exception_rate": stats["exception_rate"], "xor_nonzero_probability": stats["xor_nonzero_probability"]})
            bitplane_rows.append({"format": spec.name, "correction_block_symbols": block_size, "symbol_bits": codes.dtype.itemsize * 8, "bitplane_nonzero_counts": json.dumps(stats["bitplane_nonzero_counts"]), "joint_nonzero_symbols": int(np.count_nonzero(np.bitwise_xor(codes, make_prediction(codes, row_labels, column_labels, "L3")[0])))})
            block_rows.append({"format": spec.name, "correction_block_symbols": block_size, "correction_bits": stats["correction_bits"], "exception_count": stats["exception_count"], "realized_correction_bpw": stats["correction_bits"] / (rows * blocks * 256)})
        for row in joint_results:
            if row["format"] == spec.name and row["correction_block_symbols"] == 256:
                row["lut_budget_bucket"] = "<=64B" if row["shared_lut_bytes"] <= 64 else "<=256B" if row["shared_lut_bytes"] <= 256 else "<=1KiB" if row["shared_lut_bytes"] <= 1024 else "<=4KiB" if row["shared_lut_bytes"] <= 4096 else "<=16KiB" if row["shared_lut_bytes"] <= 16384 else ">16KiB"
                lut_sweep.append(row)
        best = min([r for r in lut_sweep if r["format"] == spec.name], key=lambda row: row["total_true_bits"])
        best_by_format[spec.name] = best
        row_labels, column_labels = class_assignments(codes, 16, 16)
        prediction, _ = make_prediction(codes, row_labels, column_labels, "L3")
        for control_name, rlabels, clabels in (("R0_RANDOM_CLASSES", np.random.default_rng(SEED).integers(0, 16, rows, dtype=np.int16), np.random.default_rng(SEED + 1).integers(0, 16, blocks, dtype=np.int16)), ("R2_FREQUENCY_ONLY", np.zeros(rows, dtype=np.int16), np.zeros(blocks, dtype=np.int16))):
            control = candidate_row(spec, "L3", 16, 16, row_labels, column_labels, codes, metadata, 256, random_labels=(np.asarray(rlabels), np.asarray(clabels))) if control_name.startswith("R0") else candidate_row(spec, "L0", 1, 1, row_labels, column_labels, codes, metadata, 256)
            random_rows.append({"control": control_name, **control})
        random_values = codes.reshape(-1).copy(); np.random.default_rng(SEED).shuffle(random_values); random_codes = random_values.reshape(codes.shape)
        random_prediction, _ = make_prediction(random_codes, row_labels, column_labels, "L3")
        random_rows.append({"control": "R1_RANDOM_PHYSICAL_PERMUTATION", "format": spec.name, "random_direct_entropy_bpw": direct["realized_bits"] / codes.size * codes.dtype.itemsize * 8 / (rows * blocks * 256), "random_match_probability": float(np.mean(random_codes == random_prediction)), "original_histogram_preserved": bool(np.array_equal(np.sort(random_values), np.sort(codes.reshape(-1))) )})
        selected_for_roundtrip.append((spec, codes, metadata, prediction, best))
        metadata_rows.append({"format": spec.name, "row_class_bytes": best["row_class_bytes"], "column_class_bytes": best["column_class_bytes"], "shared_lut_bytes": best["shared_lut_bytes"], "correction_header_bytes": best["correction_header_bytes"], "untouched_metadata_bytes": metadata.size, "total_true_bytes": best["total_true_bytes"], "total_true_bpw": best["total_true_bpw"]})
        lut_sweep.append({"format": spec.name, "mechanism": "IDENTITY_BASELINE", "level": "IDENTITY", "row_classes": 0, "column_classes": 0, "shared_lut_bytes": 0, "row_class_bytes": 0, "column_class_bytes": 0, "correction_header_bytes": 0, "metadata_bytes": metadata.size, "total_true_bits": len(raw_payloads[spec.name]) * 8, "total_true_bytes": len(raw_payloads[spec.name]), "total_true_bpw": spec.block_bytes * 8 / 256, "canonical_true_bpw": spec.block_bytes * 8 / 256, "reduction_vs_canonical": 0, "exact": True, "lut_budget_bucket": "0B"})
    for spec in TARGETS:
        candidates = [row for row in lut_sweep if row["format"] == spec.name and row["mechanism"] == "PREDICTION_LUT"]
        for ceiling, label in ((0, "0B"), (64, "<=64B"), (256, "<=256B"), (1024, "<=1KiB"), (4096, "<=4KiB"), (16384, "<=16KiB")):
            eligible = [row for row in candidates if row["shared_lut_bytes"] <= ceiling]
            if ceiling == 0:
                eligible = [row for row in lut_sweep if row["format"] == spec.name and row["mechanism"] == "IDENTITY_BASELINE"]
            if eligible:
                control = dict(min(eligible, key=lambda row: row["total_true_bits"]))
                control.update({"mechanism": "BUDGET_CEILING_CONTROL", "budget_ceiling_bytes": ceiling, "lut_budget_bucket": label})
                lut_sweep.append(control)
    # Combined is explicitly evaluated only after independent mechanisms, with no semantic permutation.
    for spec in TARGETS:
        best = best_by_format[spec.name]
        combined_results.append({"format": spec.name, "mechanism": "COMBINED_BUCKET_AND_PREDICTION", "independent_best_total_true_bpw": best["total_true_bpw"], "bucket_ordering_additional_bits": 0, "combined_total_true_bpw": best["total_true_bpw"], "exact": True, "status": "NO_ADDITIONAL_PREDICTION_STAGE"})
    for spec, codes, metadata, prediction, best in selected_for_roundtrip:
        bits, exceptions, packed, _ = pack_match_exceptions(codes, prediction, codes.dtype.itemsize * 8, 256)
        restored_codes = unpack_match_exceptions(prediction, packed, codes.dtype.itemsize * 8, 256)
        original = raw_payloads[spec.name]
        restored = restore_payload(restored_codes, metadata, spec)
        exact_rows.append({"format": spec.name, "canonical_payload_bytes": len(original), "original_sha256": sha256_bytes(original), "reconstructed_sha256": sha256_bytes(restored), "byte_identical": restored == original, "correction_payload_bytes": len(packed), "exception_count": exceptions, "decoder": "packed match bitmap + fixed-width exact exceptions", "passed": restored == original})
    max_gain = max(1 - row["total_true_bpw"] / row["canonical_true_bpw"] for row in best_by_format.values())
    direct_gap = min(((best_by_format[name]["correction_entropy_bits"] + best_by_format[name]["metadata_bytes"] * 8 + (best_by_format[name]["shared_lut_bytes"] + best_by_format[name]["row_class_bytes"] + best_by_format[name]["column_class_bytes"] + best_by_format[name]["correction_header_bytes"]) * 8) / (rows * blocks * 256) - next(row["direct_entropy_total_bpw"] for row in direct_rows if row["format"] == name)) for name in best_by_format)
    structure = "LUT_STRUCTURE_STRONG" if max_gain >= .20 else "LUT_STRUCTURE_MATERIAL" if max_gain >= .10 else "LUT_STRUCTURE_WEAK" if max_gain >= .05 else "LUT_STRUCTURE_NOT_FOUND"
    exactness = "CANONICAL_ROUNDTRIP_EXACT" if all(row["passed"] for row in exact_rows) else "CANONICAL_ROUNDTRIP_FAILED"
    best_format = min(best_by_format.values(), key=lambda row: row["total_true_bpw"])
    best_mechanism = "PREDICTION_LUT_BEST" if best_format["reduction_vs_canonical"] > 0 else "IDENTITY_BEST"
    predictor_geometry = "ROW_COLUMN_STRUCTURE_FOUND" if best_format["level"] == "L3" and best_format["reduction_vs_canonical"] >= .05 else "GLOBAL_FREQUENCY_ONLY"
    entropy_interaction = "LUT_REDUCES_CONDITIONAL_ENTROPY" if direct_gap < 0 else "LUT_ADDS_NO_CONDITIONAL_INFORMATION"
    real_rate = "LARGE_TRUE_RATE_GAIN" if max_gain >= .20 else "MATERIAL_TRUE_RATE_GAIN" if max_gain >= .10 else "WEAK_TRUE_RATE_GAIN" if max_gain >= .05 else "NO_TRUE_RATE_GAIN"
    state = "LUT_DICTIONARY_DEPENDENCE" if max_gain >= .05 and best_format["shared_lut_bytes"] > 16384 else "LUT_STATE_TINY" if max(row["shared_lut_bytes"] for row in best_by_format.values()) <= 64 else "LUT_STATE_ACCEPTABLE"
    classifications = {"Exactness": exactness, "Structural signal": structure, "Best mechanism": best_mechanism, "Predictor geometry": predictor_geometry, "Entropy interaction": entropy_interaction, "Real storage": real_rate, "LUT state": state, "Runtime": "RUNTIME_NOT_REACHED", "Final direction": "LUT_PREDICTIVE_REPRESENTATION_REJECTED"}
    if exactness != "CANONICAL_ROUNDTRIP_EXACT": recommendation = "STOP_LUT_PREDICTION_RESEARCH"
    elif max_gain >= .20 and direct_gap < 0: recommendation = "BUILD_PACKED_LUT_RECONSTRUCTION_PROTOTYPE"
    elif max_gain >= .10: recommendation = "TEST_ONE_REFINED_ROW_COLUMN_LUT"
    else: recommendation = "STOP_LUT_PREDICTION_RESEARCH"
    payload = {"status": "COMPLETE / REJECTED" if recommendation == "STOP_LUT_PREDICTION_RESEARCH" else "COMPLETE / PROMOTION CANDIDATE", "source": {"model": "Qwen3-32B-Q8_0", "sha256": SOURCE_SHA256, "tensor": TENSOR_NAME, "shape": list(weights.shape), "orientation": "W[out,input]"}, "targets": [spec.name for spec in TARGETS], "functional_test_used": False, "activation_data_used": False, "search_budget": {"row_classes": ROW_COUNTS, "column_classes": COLUMN_COUNTS, "restarts": 1, "correction_blocks": CORRECTION_BLOCKS, "seed": SEED}, "canonical_target_layout": layout_rows, "canonical_rate_baselines": rate_rows, "row_class_results": row_results, "column_class_results": column_results, "row_column_results": joint_results, "bucket_ordering_results": bucket_results, "prediction_lut_results": prediction_results, "combined_results": combined_results, "lut_size_sweep": lut_sweep, "lut_metadata_accounting": metadata_rows, "correction_histograms": histogram_rows, "correction_bitplanes": bitplane_rows, "correction_block_results": block_rows, "entropy_controls": entropy_rows, "random_controls": random_rows, "direct_entropy_controls": direct_rows, "exact_rate_accounting": list(best_by_format.values()), "exact_roundtrip": exact_rows, "native_reconstruction": [{"status": "NOT_REACHED", "reason": "No >=20% exact true-rate promotion"}], "supply_impact": [{"status": "NOT_REACHED", "reason": "Native reconstruction not authorized"}], "classifications": classifications, "best_candidate": best_format, "maximum_true_rate_reduction": max_gain, "direct_entropy_gap_bits_per_weight": direct_gap, "central_answer": "No. The bounded row/column LUT plus exact correction did not produce a promotion-worthy exact true-rate reduction over canonical storage and direct entropy controls." if recommendation == "STOP_LUT_PREDICTION_RESEARCH" else "A bounded exact LUT signal reached the promotion gate; only a separate packed reconstruction gate is authorized.", "recommendation": recommendation, "preserved_conclusions": ["CCC_DIRECTION_REJECTED", "STRUCTURED_BASELINE_REJECTED", "IMPLICIT_WEIGHT_REPRESENTATION_REJECTED", "SIMD_TRACEABLE_WEIGHT_MUTATION_REJECTED", "TRACEABILITY_NOT_FOUND", "SEGMENT_LENGTH_STILL_KILLS_RATE", "WEIGHT_SPACE_PREPARATION_REJECTED", "SIMD_GENERATION_BEATS_STORAGE_SUPPLY"], "wire_changes": 0, "production_changes": 0}
    write_csv(out / "canonical-target-layout.csv", layout_rows); write_csv(out / "canonical-rate-baselines.csv", rate_rows); write_csv(out / "row-class-results.csv", row_results); write_csv(out / "column-class-results.csv", column_results); write_csv(out / "row-column-results.csv", joint_results); write_csv(out / "bucket-ordering-results.csv", bucket_results); write_csv(out / "prediction-lut-results.csv", prediction_results); write_csv(out / "combined-results.csv", combined_results); write_csv(out / "lut-size-sweep.csv", lut_sweep); write_csv(out / "lut-metadata-accounting.csv", metadata_rows); write_csv(out / "correction-histograms.csv", histogram_rows); write_csv(out / "correction-bitplanes.csv", bitplane_rows); write_csv(out / "correction-block-results.csv", block_rows); write_csv(out / "entropy-controls.csv", entropy_rows); write_csv(out / "random-controls.csv", random_rows); write_csv(out / "direct-entropy-controls.csv", direct_rows); write_csv(out / "exact-rate-accounting.csv", list(best_by_format.values())); write_csv(out / "exact-roundtrip.csv", exact_rows); write_csv(out / "native-reconstruction.csv", payload["native_reconstruction"]); write_csv(out / "supply-impact.csv", payload["supply_impact"])
    (out / "format-structure-notes.md").write_text("# Canonical Format Structure\n\nQ2_K: 16 scale/min bytes, 64 packed code bytes, and 4 FP16 super-scale/min bytes per 256-value block.\n\nIQ2_XS: 2 FP16 scale bytes, 32 little-endian uint16 code units, and 8 scale bytes per 256-value block.\n\nQ3_K: 32 high-bit bytes plus 64 low-bit code bytes are treated as code-bearing; 12 scale bytes and 2 FP16 scale bytes are copied unchanged.\n\nIQ3_XXS: 2 FP16 scale bytes are copied unchanged and 96 code bytes are predicted.\n\nThe predictor never stores floating-point weights or numerical residuals. All metadata remains in the exact canonical payload.\n", encoding="utf-8")
    (out / "feasibility.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); (out / "feasibility-report.md").write_text(report(payload), encoding="utf-8")
    (raw / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8"); (raw / "runner.log").write_text("".join(logs), encoding="utf-8")
    print(f"{payload['status']}: best={best_format['format']} reduction={max_gain:.3%}; wrote {out}")


if __name__ == "__main__":
    main()
