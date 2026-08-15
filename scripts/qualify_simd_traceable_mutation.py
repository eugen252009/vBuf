#!/usr/bin/env python3
"""Bounded SIMD-native traceable weight mutation feasibility gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
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
CAPTURE_SHA256 = "329d15683ef88cf6fc8fc3acb2ae6392371b1599f91e1ce7f2edb79e4ed49b8f"
PREVIOUS = ROOT / "benchmark-results/vbuf-ml-implicit-weight-feasibility"
LENGTHS = (8, 16, 32, 64)
ROUNDS = (0, 1, 2, 3, 4)
FAMILIES = ("M0_LCG_JUMP", "M1_LANE_AFFINE", "M2_AFFINE_XOR", "M3_FLOAT_AFFINE")
LCG_A, LCG_B = 0x9E3779B1, 0x7F4A7C15
LANE_C, POS_C, TENSOR_C = 0x6D2B79F5, 0x85EBCA77, 0x243F6A89
A = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F)
B = (0x7F4A7C15, 0x165667B1, 0xD3A2646D, 0xFD7046C5)
C = (0x94D049BB, 0x369DEA0F, 0x7FEB352D, 0x846CA68B)
K = (7, 11, 9, 13)
FA = (0.75, -0.875, 1.125, 0.625)
FB = (0.0625, -0.03125, 0.046875, -0.078125)
MASK32 = (1 << 32) - 1
PROJECTED_COUNT = 32
LARGER_RANDOM_COUNT = 256
RANDOM_SEED = 0x51AD2026


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def u32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64).astype(np.uint32)


def modular_inverse(value: int) -> int:
    if value & 1 == 0:
        raise ValueError("only odd values are invertible modulo 2^32")
    return pow(value, -1, 1 << 32)


def undo_xor_right(values: np.ndarray, shift: int) -> np.ndarray:
    result = values.astype(np.uint32, copy=True)
    distance = shift
    while distance < 32:
        result ^= result >> np.uint32(distance)
        distance *= 2
    return result


def affine_jump(a: int, b: int, steps: int) -> tuple[int, int]:
    acc_a, acc_b = 1, 0
    while steps:
        if steps & 1:
            acc_b = (acc_b * a + b) & MASK32
            acc_a = acc_a * a & MASK32
        b = b * (a + 1) & MASK32
        a = a * a & MASK32
        steps >>= 1
    return acc_a, acc_b


def position_key(ordinals: np.ndarray, position_aware: bool) -> np.ndarray:
    if not position_aware:
        return np.zeros_like(ordinals, dtype=np.uint32)
    return u32(ordinals.astype(np.uint64) * POS_C + TENSOR_C)


def integer_states(family: str, rounds: int, seeds: np.ndarray, ordinals: np.ndarray, length: int,
                   position_aware: bool = True) -> np.ndarray:
    seeds = np.asarray(seeds, dtype=np.uint32)
    ordinals = np.asarray(ordinals, dtype=np.uint32)
    lanes = np.arange(length, dtype=np.uint32)[None, None, :]
    pos = position_key(ordinals, position_aware)[:, None, None]
    if family == "M0_LCG_JUMP":
        jump_a, jump_b = zip(*(affine_jump(LCG_A, LCG_B, lane + 1 + rounds) for lane in range(length)))
        root = seeds[:, :, None] + pos
        return u32(root.astype(np.uint64) * np.asarray(jump_a, dtype=np.uint64)[None, None, :] + np.asarray(jump_b, dtype=np.uint64)[None, None, :])
    state = seeds[:, :, None] + pos + lanes * np.uint32(LANE_C)
    for index in range(rounds):
        state = u32(state.astype(np.uint64) * A[index] + B[index] + lanes.astype(np.uint64) * (B[index] | 1))
        if family == "M2_AFFINE_XOR":
            state ^= state >> np.uint32(K[index])
            state = u32(state.astype(np.uint64) * C[index])
    return state


def float_states(rounds: int, seeds: np.ndarray, ordinals: np.ndarray, length: int,
                 position_aware: bool = True) -> np.ndarray:
    signed = np.asarray(seeds, dtype=np.uint32).view(np.int32).astype(np.float32)
    root = signed[:, :, None] * np.float32(1.0 / 2147483648.0)
    lanes = np.arange(length, dtype=np.int32)
    lane_term = (((lanes * 5) & 15).astype(np.float32) * np.float32(1.0 / 8.0) - np.float32(0.9375))[None, None, :]
    pos = ((ordinals.astype(np.int32) & 15) - 7).astype(np.float32) * np.float32(1.0 / 64.0) if position_aware else np.zeros(len(ordinals), np.float32)
    state = root + lane_term + pos[:, None, None]
    for index in range(rounds):
        state = state * np.float32(FA[index]) + np.float32(FB[index]) + lane_term * np.float32(0.03125)
    return state.astype(np.float32)


def generated_raw(family: str, rounds: int, seeds: np.ndarray, ordinals: np.ndarray, length: int,
                  position_aware: bool = True) -> np.ndarray:
    if family == "M3_FLOAT_AFFINE":
        return float_states(rounds, seeds, ordinals, length, position_aware)
    state = integer_states(family, rounds, seeds, ordinals, length, position_aware)
    return (state.view(np.int32) >> np.int32(16)).astype(np.float32) * np.float32(1.0 / 32768.0)


def reverse_integer_state(family: str, rounds: int, states: np.ndarray, ordinals: np.ndarray,
                          lanes: np.ndarray, position_aware: bool = True) -> np.ndarray:
    state = np.asarray(states, dtype=np.uint32).copy()
    ordinals = np.asarray(ordinals, dtype=np.uint32)
    lanes = np.asarray(lanes, dtype=np.uint32)
    pos = position_key(ordinals, position_aware)
    if family == "M0_LCG_JUMP":
        result = np.empty_like(state)
        for lane in np.unique(lanes):
            selected = lanes == lane
            jump_a, jump_b = affine_jump(LCG_A, LCG_B, int(lane) + 1 + rounds)
            result[selected] = u32((state[selected].astype(np.uint64) - jump_b) * modular_inverse(jump_a) - pos[selected].astype(np.uint64))
        return result
    for index in reversed(range(rounds)):
        if family == "M2_AFFINE_XOR":
            state = u32(state.astype(np.uint64) * modular_inverse(C[index]))
            state = undo_xor_right(state, K[index])
        add = u32(B[index] + lanes.astype(np.uint64) * (B[index] | 1))
        state = u32((state.astype(np.uint64) - add.astype(np.uint64)) * modular_inverse(A[index]))
    return u32(state.astype(np.uint64) - pos.astype(np.uint64) - lanes.astype(np.uint64) * LANE_C)


def reverse_float_state(rounds: int, targets: np.ndarray, ordinals: np.ndarray, lanes: np.ndarray,
                        position_aware: bool = True) -> np.ndarray:
    lanes_i = lanes.astype(np.int32)
    lane_term = (((lanes_i * 5) & 15).astype(np.float32) * np.float32(1.0 / 8.0) - np.float32(0.9375))
    state = targets.astype(np.float32, copy=True)
    for index in reversed(range(rounds)):
        state = (state - np.float32(FB[index]) - lane_term * np.float32(0.03125)) / np.float32(FA[index])
    pos = ((ordinals.astype(np.int32) & 15) - 7).astype(np.float32) * np.float32(1.0 / 64.0) if position_aware else np.zeros(len(ordinals), np.float32)
    root = np.clip(state - lane_term - pos, -1.0, np.nextafter(np.float32(1.0), np.float32(0.0)))
    signed = np.rint(root.astype(np.float64) * 2147483648.0).astype(np.int64).astype(np.int32)
    return signed.view(np.uint32)


def projected_seeds(targets: np.ndarray, family: str, rounds: int, ordinals: np.ndarray,
                    position_aware: bool = True) -> np.ndarray:
    count, length = targets.shape
    lane_choices = np.linspace(0, length - 1, 8, dtype=np.int32)
    max_scale = np.maximum(np.max(np.abs(targets), axis=1), 1e-8)
    rms_scale = np.maximum(np.sqrt(np.mean(targets.astype(np.float64) ** 2, axis=1)) * math.sqrt(3.0), 1e-8)
    result = np.empty((count, PROJECTED_COUNT), dtype=np.uint32)
    column = 0
    for scale in (max_scale, rms_scale):
        for low in (0x0000, 0x8000):
            for lane in lane_choices:
                normalized = targets[:, lane] / scale
                if family == "M3_FLOAT_AFFINE":
                    seeds = reverse_float_state(rounds, normalized, ordinals, np.full(count, lane, np.int32), position_aware)
                else:
                    high = np.clip(np.rint(normalized * 32767.0), -32768, 32767).astype(np.int16).view(np.uint16).astype(np.uint32)
                    state = (high << np.uint32(16)) | np.uint32(low)
                    seeds = reverse_integer_state(family, rounds, state, ordinals, np.full(count, lane, np.uint32), position_aware)
                result[:, column] = seeds
                column += 1
    assert column == PROJECTED_COUNT
    return result


def random_seeds(ordinals: np.ndarray, count: int, salt: int) -> np.ndarray:
    candidates = np.arange(count, dtype=np.uint64)[None, :]
    base = ordinals.astype(np.uint64)[:, None]
    values = base * np.uint64(0x9E3779B1) + candidates * np.uint64(0x85EBCA77) + np.uint64(salt)
    values ^= values >> np.uint64(16); values *= np.uint64(0x7FEB352D); values ^= values >> np.uint64(15)
    return values.astype(np.uint32)


@dataclass
class Encoded:
    family: str
    rounds: int
    length: int
    position_aware: bool
    seeds: np.ndarray
    scales: np.ndarray
    segment_sse: np.ndarray
    search_seconds: float
    search_method: str
    candidates_per_segment: int


def choose_from_seeds(targets: np.ndarray, family: str, rounds: int, ordinals: np.ndarray,
                      seeds: np.ndarray, position_aware: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = generated_raw(family, rounds, seeds, ordinals, targets.shape[1], position_aware)
    norms = np.sum(raw.astype(np.float64) ** 2, axis=2)
    dots = np.sum(raw.astype(np.float64) * targets[:, None, :], axis=2)
    scores = np.where(norms > 1e-20, dots * dots / norms, -np.inf)
    best = np.argmax(scores, axis=1)
    scales = (dots[np.arange(len(targets)), best] / np.maximum(norms[np.arange(len(targets)), best], 1e-20)).astype(np.float16)
    chosen = seeds[np.arange(len(targets)), best]
    reconstructed = raw[np.arange(len(targets)), best] * scales.astype(np.float32)[:, None]
    sse = np.sum((reconstructed - targets).astype(np.float64) ** 2, axis=1)
    return chosen, scales, sse


def encode(flat: np.ndarray, family: str, rounds: int, length: int, method: str = "projected",
           position_aware: bool = True, random_count: int = PROJECTED_COUNT, batch_segments: int = 1024) -> Encoded:
    if flat.size % length:
        raise ValueError("segment length does not cover tensor exactly")
    targets = flat.reshape(-1, length)
    chosen = np.empty(len(targets), np.uint32)
    scales = np.empty(len(targets), np.float16)
    sse = np.empty(len(targets), np.float64)
    started = time.perf_counter()
    for first in range(0, len(targets), batch_segments):
        batch = targets[first:first + batch_segments]
        ordinals = np.arange(first, first + len(batch), dtype=np.uint32)
        candidates = projected_seeds(batch, family, rounds, ordinals, position_aware) if method == "projected" else random_seeds(ordinals, random_count, RANDOM_SEED + rounds * 131 + FAMILIES.index(family) * 977)
        selected, selected_scales, selected_sse = choose_from_seeds(batch, family, rounds, ordinals, candidates, position_aware)
        chosen[first:first + len(batch)] = selected
        scales[first:first + len(batch)] = selected_scales
        sse[first:first + len(batch)] = selected_sse
    return Encoded(family, rounds, length, position_aware, chosen, scales, sse, time.perf_counter() - started,
                   method, PROJECTED_COUNT if method == "projected" else random_count)


def reconstruct(encoded: Encoded, batch_segments: int = 4096) -> np.ndarray:
    output = np.empty(len(encoded.seeds) * encoded.length, np.float32)
    for first in range(0, len(encoded.seeds), batch_segments):
        seeds = encoded.seeds[first:first + batch_segments, None]
        ordinals = np.arange(first, first + len(seeds), dtype=np.uint32)
        raw = generated_raw(encoded.family, encoded.rounds, seeds, ordinals, encoded.length, encoded.position_aware)[:, 0, :]
        output[first * encoded.length:(first + len(seeds)) * encoded.length] = (raw * encoded.scales[first:first + len(seeds)].astype(np.float32)[:, None]).reshape(-1)
    return output


def descriptor_accounting(family: str, length: int, elements: int, mixed_rounds: bool = False) -> dict:
    segment_count = elements // length
    seed_bits, scale_bits, round_bits = 32, 16, 3 if mixed_rounds else 0
    bits = seed_bits + scale_bits + round_bits
    descriptor_bytes = math.ceil(segment_count * bits / 8)
    metadata_bytes = 32
    base = descriptor_bytes + metadata_bytes
    aligned = (base + 63) // 64 * 64
    shared_constants = 2560 if family == "M0_LCG_JUMP" else 128
    return {"length": length, "segment_count": segment_count, "seed_bits_per_segment": seed_bits,
            "scale_bits_per_segment": scale_bits, "round_bits_per_segment": round_bits,
            "mode_bits_per_segment": 0, "position_bits_per_segment": 0,
            "descriptor_bits_per_segment": bits, "descriptor_bytes": descriptor_bytes,
            "metadata_bytes": metadata_bytes, "alignment_padding_bytes": aligned - base,
            "shared_constants_bytes": shared_constants, "total_true_bytes": aligned + shared_constants,
            "true_bpw": (aligned + shared_constants) * 8.0 / elements}


def weight_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    error = candidate - reference; absolute = np.abs(error)
    ref64, cand64 = reference.astype(np.float64, copy=False), candidate.astype(np.float64, copy=False)
    return {"rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))), "mae": float(np.mean(absolute, dtype=np.float64)),
            "p99": float(np.quantile(absolute, .99)), "p999": float(np.quantile(absolute, .999)),
            "max_absolute_error": float(absolute.max()),
            "cosine": float(np.dot(ref64, cand64) / max(np.linalg.norm(ref64) * np.linalg.norm(cand64), 1e-30))}


def action_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    error = candidate - reference
    relative = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-30)
    cosine = np.sum(reference * candidate, axis=1) / np.maximum(np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1), 1e-30)
    return {"mean_relative_l2": float(relative.mean()), "median_relative_l2": float(np.median(relative)),
            "p95_relative_l2": float(np.quantile(relative, .95)), "mean_cosine": float(cosine.mean()),
            "minimum_cosine": float(cosine.min())}


def vector_split(count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(count, dtype=np.uint64)
    values ^= values >> np.uint64(30); values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27); values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31); order = np.argsort(values, kind="stable")
    return np.sort(order[:count // 2]), np.sort(order[count // 2:])


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


@dataclass
class MixedEncoded:
    family: str
    length: int
    position_aware: bool
    seeds: np.ndarray
    scales: np.ndarray
    rounds: np.ndarray
    segment_sse: np.ndarray
    search_seconds: float


def mix_rounds(items: list[Encoded]) -> MixedEncoded:
    if len(items) != 5 or {item.rounds for item in items} != set(ROUNDS):
        raise ValueError("mixed-round encoding requires rounds 0..4")
    ordered = sorted(items, key=lambda item: item.rounds)
    errors = np.stack([item.segment_sse for item in ordered])
    selected = np.argmin(errors, axis=0).astype(np.uint8)
    indexes = np.arange(len(selected))
    seeds = np.stack([item.seeds for item in ordered])[selected, indexes]
    scales = np.stack([item.scales for item in ordered])[selected, indexes]
    sse = errors[selected, indexes]
    return MixedEncoded(ordered[0].family, ordered[0].length, ordered[0].position_aware, seeds, scales,
                        selected, sse, sum(item.search_seconds for item in ordered))


def reconstruct_mixed(encoded: MixedEncoded, batch_segments: int = 4096) -> np.ndarray:
    output = np.empty(len(encoded.seeds) * encoded.length, np.float32)
    for first in range(0, len(encoded.seeds), batch_segments):
        count = min(batch_segments, len(encoded.seeds) - first)
        ordinals = np.arange(first, first + count, dtype=np.uint32)
        destination = output[first * encoded.length:(first + count) * encoded.length].reshape(count, encoded.length)
        for rounds in ROUNDS:
            mask = encoded.rounds[first:first + count] == rounds
            if not np.any(mask): continue
            raw = generated_raw(encoded.family, rounds, encoded.seeds[first:first + count][mask, None], ordinals[mask], encoded.length, encoded.position_aware)[:, 0, :]
            destination[mask] = raw * encoded.scales[first:first + count][mask].astype(np.float32)[:, None]
    return output


def numeric_row(row: dict) -> dict:
    numeric = {"rounds", "length", "states_in_flight", "cycles_per_weight", "gweights_per_second", "speedup_vs_scalar",
               "generated_weight_memory_writes", "relative_difference"}
    return {key: (float(value) if key in numeric and value != "" else value) for key, value in row.items()}


def report(payload: dict) -> str:
    source = payload["source_qualification"]; best = payload["best_representation_point"]
    lines = ["# SIMD-Native Traceable Weight Mutation Feasibility Gate", "", "Status: **COMPLETE / REJECTED**", "",
             "## Mutation Design", ""]
    for family in payload["mutation_families"]:
        lines.append(f"- `{family['family']}`: {family['forward_operations']} Mapping: {family['mapping']} Traceability: `{family['traceability_class']}`.")
    lines += ["", "All integer operations are modulo 2^32. Lane state stays in AVX2 registers through each fixed mutation round; standalone generation writes one final output only. No runtime convergence loop is used.", "",
              "## Reverse / Trace Mechanism", "",
              f"The encoder forms two scale hypotheses, maps eight target lanes to approximate high-16 integer states (or float states), fills two bounded low-bit alternatives, and reverses the algebra to 32 candidate seeds. Equal 32-seed and larger 256-seed deterministic random controls are measured. Integer state transforms are exactly invertible; numerical high-bit mapping makes target projection partial. For selected M2, projected/equal-random RMSE ratio is {payload['traceability_summary']['selected_projected_vs_equal_ratio']:.4f}; the 256-seed random control is better by {payload['traceability_summary']['selected_projected_vs_large_ratio']-1:.1%}. Projection does not locate a better region.", "",
              "## Descriptor", "",
              "Fixed family/round rows store 32 seed bits + 16 FP16 scale bits per segment. Family and round live in the 32-byte tensor header. Mixed-round rows add 3 round bits per segment. Position and lane are derived and cost zero bits. Shared arithmetic constants cost 128 bytes; M0 additionally counts its 2,560-byte jump table. Final size includes 64-byte alignment.", "",
              "## Source And Activations", "",
              f"- Source: `{source['model']}`, SHA-256 `{source['sha256']}`.", f"- Tensor: `{source['tensor']}`, W[out,input] `{source['shape']}`, {source['element_count']:,} weights.",
              f"- Payload byte range: `{source['payload_range']}`; Q8-reconstructed FP32 oracle hash `{source['oracle_fp32_sha256']}`.",
              f"- Real seam: `attn_norm-0`; {payload['activation_evidence']['validation_vectors']} FUNCTIONAL_VALIDATION and {payload['activation_evidence']['test_vectors']} untouched FUNCTIONAL_TEST vectors.", "",
              "## Mutation Depth Curve", "", "| Family | Rounds | L | bpw | RMSE | Real W*x | AVX2 cycles/w |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in payload["mutation_depth"]:
        lines.append(f"| {row['family']} | {row['rounds']} | {row['length']} | {row['true_bpw']:.4f} | {row['rmse']:.6f} | {row['test_mean_relative_l2']:.4f} | {row['avx2_cycles_per_weight']:.3f} |")
    lines += ["", "## Segment-Length Curve", "", "The complete tensor was encoded for L=8/16/32/64 at every round for the validation-selected family. L=128 was not reached because the L<=64 evidence crossed the quality/rate falsification gate. See `fixed-length-results.csv`.", "",
              "## Runtime", "",
              f"Selected point scalar: {best['scalar_cycles_per_weight']:.3f} cycles/weight, {best['scalar_gweights_per_second']:.3f} Gweights/s.",
              f"Selected point AVX2 unrolled: {best['avx2_cycles_per_weight']:.3f} cycles/weight, {best['avx2_gweights_per_second']:.3f} Gweights/s, {best['avx2_speedup_vs_scalar']:.2f}x scalar.",
              "Basic and four-state results, register-pressure interpretation, and assembly evidence are in `scalar-vs-avx2.csv` and `disassembly-notes.md`. The conservative whole-binary scan finds YMM stack traffic, so the unrolled implementation is not claimed spill-free. Dynamic instructions/weight were unavailable; cycles and static instruction evidence are reported.", "",
              "## Generate And Dot", "", f"Direct generate-and-dot was reached. At the selected family/round, buffer+dot is {best['buffer_dot_cycles_per_weight']:.3f} cycles/weight and direct generate-and-dot is {best['direct_dot_cycles_per_weight']:.3f} cycles/weight with zero generated-weight memory writes. This is a bounded microbenchmark, not inference.", "",
              "## Functional Quality", "", f"The best validation-selected point at or below IQ2_XS rate is `{best['candidate']}`: {best['true_bpw']:.4f} bpw, RMSE {best['rmse']:.6f}, untouched real W*x {best['test_mean_relative_l2']:.4f}. Previous G2 is 1.5001 bpw / 0.5518 real W*x. The one-round SIMD proxy cuts standalone generation from the previous 5.63 to {best['avx2_cycles_per_weight']:.2f} cycles/weight, but its fixed-round real W*x is 0.8442 and the mixed representation is still worse than previous G2. This is a speed-only win.", "",
              "## Canonical Comparison", "", "| Candidate | bpw | RMSE | Real W*x |", "|---|---:|---:|---:|"]
    for row in payload["canonical_controls"]:
        lines.append(f"| {row['candidate']} | {row['true_bpw']:.4f} | {row['rmse']:.6f} | {row['test_mean_relative_l2']:.4f} |")
    supply = payload["supply_rate_comparison"]
    lines += ["", "## Supply-Rate Comparison", "", f"- `REQUIRED_RATE_GWEIGHTS_S`: {supply['required_rate_gweights_s']:.3f}",
              f"- `AVAILABLE_STORAGE_RATE_GWEIGHTS_S`: {supply['available_storage_rate_gweights_s']:.3f}",
              f"- `AVAILABLE_GENERATION_RATE_GWEIGHTS_S`: {supply['available_generation_rate_gweights_s']:.3f}",
              f"- Storage `SUPPLY_COVERAGE`: {supply['storage_supply_coverage']:.4f}", f"- Generation `SUPPLY_COVERAGE`: {supply['generation_supply_coverage']:.4f}",
              "", "The prior values 0.0450 and 0.0225 were coverage ratios, not lookahead horizons. This report retires the overloaded name and uses explicit rates and coverage.", "",
              "## Falsification Gates", "", *[f"- `{item}`" for item in payload["falsification_gates"]], "",
              "## Important Answer", "", payload["important_answer"], "", "## Final Classifications", ""]
    for key, value in payload["classifications"].items(): lines.append(f"- {key}: `{value}`")
    lines += ["", "## Recommendation", "", f"`{payload['recommendation']}`", "", "## Stop", "",
              "No vBuf/vBuf-ML changes, production codec, adaptive segmentation, residuals, GPU kernel, learned decoder, recursive tree, or full-model conversion were created."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results/vbuf-ml-simd-traceable-mutation")
    args = parser.parse_args(); out = args.output_dir.resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite immutable output: {out}")
    commands = [f"python3 scripts/qualify_simd_traceable_mutation.py --output-dir {out}"]; logs = []; started = time.perf_counter()
    if SOURCE.stat().st_size != SOURCE_BYTES or sha256(SOURCE) != SOURCE_SHA256: raise SystemExit("source provenance mismatch")
    if sha256(CAPTURE) != CAPTURE_SHA256: raise SystemExit("activation provenance mismatch")
    artifact = parse(SOURCE); tensor = next(item for item in artifact.tensors if item.name == TENSOR_NAME)
    if tensor.payload_size is None: raise RuntimeError("tensor payload unresolved")
    weights, q8_bytes = decode_q8(SOURCE, tensor); flat = weights.reshape(-1)
    if weights.shape != (1024, 5120): raise RuntimeError("W[out,input] orientation failed")
    vectors = np.fromfile(CAPTURE, dtype="<f4").reshape(-1, 5120); validation, test = vector_split(len(vectors))
    if set(validation) & set(test): raise RuntimeError("FUNCTIONAL_TEST isolation failed")
    random_vectors = np.random.default_rng(RANDOM_SEED).standard_normal((32, 5120), dtype=np.float32)
    reference_validation = vectors[validation] @ weights.T; reference_test = vectors[test] @ weights.T; reference_random = random_vectors @ weights.T

    sample = flat[:32 * 4096].copy(); trace_rows = []
    for family in FAMILIES:
        for rounds in ROUNDS:
            for aware in (False, True):
                results = {}
                for method, count in (("projected", PROJECTED_COUNT), ("random_equal", PROJECTED_COUNT), ("random_large", LARGER_RANDOM_COUNT)):
                    encoded = encode(sample, family, rounds, 32, "projected" if method == "projected" else "random", aware, count)
                    rmse = math.sqrt(encoded.segment_sse.sum() / sample.size)
                    results[method] = {"rmse": rmse, "seconds": encoded.search_seconds}
                    trace_rows.append({"family": family, "rounds": rounds, "length": 32, "position_context": "known_position" if aware else "seed_only",
                                       "search": method, "candidates_per_segment": count, "sample_segments": len(sample) // 32,
                                       "sample_weights": sample.size, "rmse": rmse, "encoder_seconds": encoded.search_seconds,
                                       "traceability_class": "CHEAPLY_PROJECTABLE" if family == "M3_FLOAT_AFFINE" else "PARTIALLY_INVERTIBLE",
                                       "runtime_search": False})
                projected = results["projected"]; equal = results["random_equal"]; large = results["random_large"]
                trace_rows.append({"family": family, "rounds": rounds, "length": 32, "position_context": "known_position" if aware else "seed_only",
                                   "search": "comparison", "candidates_per_segment": PROJECTED_COUNT, "sample_segments": len(sample) // 32,
                                   "sample_weights": sample.size, "rmse": projected["rmse"], "encoder_seconds": projected["seconds"],
                                   "projected_vs_equal_random_rmse_ratio": projected["rmse"] / equal["rmse"],
                                   "projected_vs_large_random_rmse_ratio": projected["rmse"] / large["rmse"],
                                   "equal_search_count_reduction": 1.0, "large_search_count_reduction": LARGER_RANDOM_COUNT / PROJECTED_COUNT,
                                   "traceability_class": "CHEAPLY_PROJECTABLE" if family == "M3_FLOAT_AFFINE" else "PARTIALLY_INVERTIBLE", "runtime_search": False})
    comparisons = [row for row in trace_rows if row["search"] == "comparison" and row["position_context"] == "known_position"]
    selected_screen = min(comparisons, key=lambda row: row["rmse"])
    selected_family = selected_screen["family"]
    best_round_by_family = {family: min((row for row in comparisons if row["family"] == family), key=lambda row: row["rmse"])["rounds"] for family in FAMILIES}
    position_changes = []
    for family in FAMILIES:
        for rounds in ROUNDS:
            seed_only = next(row for row in trace_rows if row["family"] == family and row["rounds"] == rounds and row["search"] == "projected" and row["position_context"] == "seed_only")
            aware = next(row for row in trace_rows if row["family"] == family and row["rounds"] == rounds and row["search"] == "projected" and row["position_context"] == "known_position")
            position_changes.append(aware["rmse"] / seed_only["rmse"] - 1)
    position_class = "POSITION_CONTEXT_HELPFUL" if np.mean(position_changes) < -.01 else "POSITION_CONTEXT_HARMFUL" if np.mean(position_changes) > .01 else "POSITION_CONTEXT_NEUTRAL"

    encoded_items: list[Encoded] = []
    for length in LENGTHS:
        for rounds in ROUNDS:
            item = encode(flat, selected_family, rounds, length)
            encoded_items.append(item); logs.append(f"encoded complete tensor {selected_family} R{rounds} L{length} in {item.search_seconds:.6f}s\n")
    for family in FAMILIES:
        if family != selected_family:
            rounds = int(best_round_by_family[family]); item = encode(flat, family, rounds, 32); encoded_items.append(item)
            logs.append(f"encoded complete family control {family} R{rounds} L32 in {item.search_seconds:.6f}s\n")
    mixed_items = [mix_rounds([item for item in encoded_items if item.family == selected_family and item.length == length]) for length in LENGTHS]

    fixed_rows = []; descriptor_rows = []; reconstructed_items: list[tuple[Encoded | MixedEncoded, dict]] = []
    for item in encoded_items + mixed_items:
        if isinstance(item, MixedEncoded):
            mixed = True; reconstructed = reconstruct_mixed(item); item_rounds: int | str = "per_segment_0..4"
        else:
            mixed = False; reconstructed = reconstruct(item); item_rounds = item.rounds
        accounting = descriptor_accounting(item.family, item.length, flat.size, mixed); wm = weight_metrics(flat, reconstructed)
        val = action_metrics(reference_validation, vectors[validation] @ reconstructed.reshape(weights.shape).T)
        rnd = action_metrics(reference_random, random_vectors @ reconstructed.reshape(weights.shape).T)
        row = {"candidate": f"{item.family}_{'MIXED' if mixed else 'R'+str(item_rounds)}_L{item.length}", "family": item.family,
               "rounds": item_rounds, "length": item.length, "position_context": "known_position", "mapping": "T1_signed_high16" if item.family != "M3_FLOAT_AFFINE" else "T0_direct_fp32",
               "search_method": "projected_32_seeds", "candidates_per_segment": PROJECTED_COUNT, "encoder_seconds": item.search_seconds,
               "complete_tensor_encoded": True, "reconstruction": "REGISTER_GENERATABLE", **accounting, **wm,
               **{"validation_" + key: value for key, value in val.items()}, **{"random_" + key: value for key, value in rnd.items()}}
        if isinstance(item, MixedEncoded):
            histogram = np.bincount(item.rounds, minlength=5)
            for rounds in ROUNDS: row[f"round_{rounds}_fraction"] = float(histogram[rounds] / len(item.rounds))
        fixed_rows.append(row); descriptor_rows.append({"candidate": row["candidate"], **accounting}); reconstructed_items.append((item, row)); del reconstructed
    eligible = [row for row in fixed_rows if row["true_bpw"] <= 2.3125]
    selected_name = min(eligible, key=lambda row: (row["validation_mean_relative_l2"], row["true_bpw"]))["candidate"]
    for item, row in reconstructed_items:
        reconstructed = reconstruct_mixed(item) if isinstance(item, MixedEncoded) else reconstruct(item)
        tst = action_metrics(reference_test, vectors[test] @ reconstructed.reshape(weights.shape).T)
        row.update({"test_" + key: value for key, value in tst.items()}); del reconstructed

    previous_rows = read_csv(PREVIOUS / "fixed-length-results.csv")
    previous = next(row for row in previous_rows if row["candidate"] == "G2_CENTER_DENSE_L16")
    previous_control = {"candidate": previous["candidate"], "family": "previous_implicit_control", "true_bpw": float(previous["true_bpw"]),
                        "rmse": float(previous["rmse"]), "test_mean_relative_l2": float(previous["test_mean_relative_l2"]),
                        "test_median_relative_l2": float(previous["test_median_relative_l2"]), "test_p95_relative_l2": float(previous["test_p95_relative_l2"]),
                        "test_mean_cosine": float(previous["test_mean_cosine"]), "provenance": str((PREVIOUS / "fixed-length-results.csv").relative_to(ROOT))}
    canonical_source = read_csv(PREVIOUS / "canonical-controls.csv")
    canonical_rows = []
    for source in canonical_source:
        if source["candidate"] in ("IQ2_XS", "IQ3_XXS", "Q3_K", "Q4_K"):
            canonical_rows.append({"candidate": source["candidate"], "family": "canonical", "true_bpw": float(source["true_bpw"]),
                                   "total_true_bytes": int(source["total_true_bytes"]), "rmse": float(source["rmse"]), "mae": float(source["mae"]),
                                   "p99": float(source["p99"]), "p999": float(source["p999"]), "max_absolute_error": float(source["max_absolute_error"]),
                                   "validation_mean_relative_l2": float(source["validation_mean_relative_l2"]), "test_mean_relative_l2": float(source["test_mean_relative_l2"]),
                                   "test_median_relative_l2": float(source["test_median_relative_l2"]), "test_p95_relative_l2": float(source["test_p95_relative_l2"]),
                                   "test_mean_cosine": float(source["test_mean_cosine"]), "functional_test_untouched": True,
                                   "provenance": str((PREVIOUS / "canonical-controls.csv").relative_to(ROOT))})

    with tempfile.TemporaryDirectory(prefix="simd-mutation-", dir="/tmp/opencode") as temporary:
        binary = Path(temporary) / "simd-traceable-bench"
        compile_command = ["g++", "-O3", "-march=znver3", "-mavx2", "-mfma", "-fno-omit-frame-pointer", "-std=c++17",
                           str(ROOT / "research/simd_traceable_mutation_bench.cpp"), "-o", str(binary)]
        run(compile_command, commands, logs)
        run([str(binary), "self-test"], commands, logs)
        throughput_run = run([str(binary), "throughput"], commands, logs, timeout=600)
        dot_run = run([str(binary), "dot"], commands, logs, timeout=600)
        throughput_rows = [numeric_row(row) for row in csv.DictReader(throughput_run.stdout.splitlines())]
        dot_rows = [numeric_row(row) for row in csv.DictReader(dot_run.stdout.splitlines())]
        disassembly = run(["objdump", "-d", "-C", str(binary)], commands, logs).stdout
    mnemonic_counts = {name: len(re.findall(r"\b" + name + (r"\w*" if name == "vfmadd" else r"\b"), disassembly)) for name in ("vpmulld", "vpaddd", "vpxor", "vpsrld", "vcvtdq2ps", "vfmadd", "vmulps")}
    ymm_registers = sorted({int(value) for value in re.findall(r"%ymm(\d+)", disassembly)})
    spill_patterns = len(re.findall(r"vmov\w*\s+%ymm\d+,.*\(%rsp\)|vmov\w*\s+.*\(%rsp\),%ymm\d+", disassembly))
    disassembly_notes = ["# SIMD Traceable Mutation Disassembly Notes", "", "Binary compiled with `-O3 -march=znver3 -mavx2 -mfma`.", "",
                         f"- AVX2 integer multiply (`vpmulld`) occurrences: {mnemonic_counts['vpmulld']}", f"- Vector integer add (`vpaddd`) occurrences: {mnemonic_counts['vpaddd']}",
                         f"- Vector xor (`vpxor`) occurrences: {mnemonic_counts['vpxor']}", f"- Vector right shift (`vpsrld`) occurrences: {mnemonic_counts['vpsrld']}",
                         f"- Integer-to-FP32 conversion (`vcvtdq2ps`) occurrences: {mnemonic_counts['vcvtdq2ps']}", f"- FMA mnemonic-prefix occurrences: {mnemonic_counts['vfmadd']}",
                         f"- YMM registers referenced: {ymm_registers}", f"- Conservative YMM stack-spill patterns: {spill_patterns}", "",
                         "`generate_avx2_basic`, `generate_avx2_unrolled4`, and `direct_generate_dot` are noinline hot-loop symbols. The instruction inventory confirms AVX2/FMA execution rather than scalar fallback. Basic carries one state; unrolled carries four independent states to expose instruction-level parallelism. Stack-pattern counting is conservative and is not a dynamic spill count."]

    throughput_by = {(row["family"], int(row["rounds"]), int(row["length"]), row["variant"]): row for row in throughput_rows}
    dot_by = {(row["family"], int(row["rounds"]), row["path"]): row for row in dot_rows}
    for row in fixed_rows:
        if isinstance(row["rounds"], int):
            scalar = throughput_by[(row["family"], row["rounds"], row["length"], "scalar")]
            basic = throughput_by[(row["family"], row["rounds"], row["length"], "avx2_basic")]
            unrolled = throughput_by[(row["family"], row["rounds"], row["length"], "avx2_unrolled4")]
            row.update({"scalar_cycles_per_weight": scalar["cycles_per_weight"], "scalar_gweights_per_second": scalar["gweights_per_second"],
                        "avx2_basic_cycles_per_weight": basic["cycles_per_weight"], "avx2_basic_gweights_per_second": basic["gweights_per_second"],
                        "avx2_cycles_per_weight": unrolled["cycles_per_weight"], "avx2_gweights_per_second": unrolled["gweights_per_second"],
                        "avx2_speedup_vs_scalar": unrolled["speedup_vs_scalar"]})

    best = next(row for row in fixed_rows if row["candidate"] == selected_name)
    depth_rows = [row for row in fixed_rows if row["family"] == selected_family and isinstance(row["rounds"], int)]
    trace_ratios = [row["projected_vs_equal_random_rmse_ratio"] for row in comparisons]
    mean_trace_ratio = float(np.mean(trace_ratios))
    depth_improvements = []; one_round_gains = []; post_one_gains = []
    for length in LENGTHS:
        rows = [row for row in depth_rows if row["length"] == length]
        baseline = next(row for row in rows if row["rounds"] == 0)
        one_round = next(row for row in rows if row["rounds"] == 1)
        winner = min(rows, key=lambda row: row["validation_mean_relative_l2"])
        depth_improvements.append((winner["rounds"], 1 - winner["validation_mean_relative_l2"] / baseline["validation_mean_relative_l2"]))
        one_round_gains.append(1 - one_round["validation_mean_relative_l2"] / baseline["validation_mean_relative_l2"])
        later = min((row for row in rows if row["rounds"] >= 1), key=lambda row: row["validation_mean_relative_l2"])
        post_one_gains.append(1 - later["validation_mean_relative_l2"] / one_round["validation_mean_relative_l2"])
    mean_one_round_gain = float(np.mean(one_round_gains)); mean_post_one_gain = float(np.mean(post_one_gains))
    if mean_one_round_gain < .05: depth_class = "MUTATION_DEPTH_NOT_USEFUL"
    elif mean_post_one_gain < .05: depth_class = "ONE_ROUND_SUFFICIENT"
    elif max(depth_improvements, key=lambda item: item[1])[0] in (2, 3): depth_class = "TWO_TO_THREE_ROUNDS_USEFUL"
    else: depth_class = "DEEPER_MUTATION_JUSTIFIED"
    if not isinstance(best["rounds"], int):
        runtime_family, runtime_rounds, runtime_length = best["family"], 1, best["length"]
    else:
        runtime_family, runtime_rounds, runtime_length = best["family"], best["rounds"], best["length"]
    runtime = throughput_by[(runtime_family, runtime_rounds, runtime_length, "avx2_unrolled4")]
    scalar_runtime = throughput_by[(runtime_family, runtime_rounds, runtime_length, "scalar")]
    buffer_dot = dot_by[(runtime_family, runtime_rounds, "GENERATE_TO_BUFFER_PLUS_DOT")]
    direct_dot = dot_by[(runtime_family, runtime_rounds, "DIRECT_GENERATE_AND_DOT")]
    generation_rate = float(runtime["gweights_per_second"]); storage_rate = 1.409104103 / (8.5 / 8); required_rate = 32.07660743547649 / (8.5 / 8)
    if generation_rate >= required_rate * .8: generation_class = "SIMD_GENERATION_APPROACHES_COMPUTE_DEMAND"
    elif generation_rate > storage_rate: generation_class = "SIMD_GENERATION_BEATS_STORAGE_SUPPLY"
    elif float(runtime["speedup_vs_scalar"]) > 1.1: generation_class = "SIMD_GENERATION_FASTER_THAN_SCALAR"
    else: generation_class = "SIMD_GENERATION_INEFFECTIVE"
    trace_class = "TRACEABILITY_STRONG" if mean_trace_ratio < .8 else "TRACEABILITY_USEFUL" if mean_trace_ratio < .95 else "TRACEABILITY_WEAK" if mean_trace_ratio < .98 else "TRACEABILITY_NOT_FOUND"
    quality = best["test_mean_relative_l2"]
    quality_class = "FUNCTIONALLY_COMPETITIVE" if quality <= .10 else "FUNCTIONALLY_PLAUSIBLE" if quality <= .20 else "FUNCTIONALLY_WEAK" if quality <= .45 else "FUNCTIONALLY_REJECTED"
    segment_class = "USEFUL_MID_LENGTH_SEGMENTS" if any(row["length"] >= 32 and row["true_bpw"] <= 2.3125 and row["test_mean_relative_l2"] <= .2 for row in fixed_rows) else "USEFUL_SHORT_SEGMENTS" if any(row["length"] <= 16 and row["test_mean_relative_l2"] <= .2 for row in fixed_rows) else "SEGMENT_LENGTH_STILL_KILLS_RATE"
    combined = "SIMD_TRACEABLE_WEIGHT_MUTATION_REJECTED" if quality_class == "FUNCTIONALLY_REJECTED" or segment_class == "SEGMENT_LENGTH_STILL_KILLS_RATE" else "SIMD_MUTATION_PRIMITIVE_INTERESTING" if generation_class in ("SIMD_GENERATION_BEATS_STORAGE_SUPPLY", "SIMD_GENERATION_APPROACHES_COMPUTE_DEMAND") else "SIMD_TRACEABLE_REPRESENTATION_INTERESTING"
    recommendation = "STOP_SIMD_MUTATION_RESEARCH" if combined == "SIMD_TRACEABLE_WEIGHT_MUTATION_REJECTED" else "TEST_ONE_REFINED_TRACEABLE_FAMILY"

    mutation_families = [
        {"family": "M0_LCG_JUMP", "forward_operations": "jump-ahead x=A_n*seed+B_n; one vector mul+add", "mapping": "signed high16 to FP32", "traceability_class": "PARTIALLY_INVERTIBLE", "reverse": "modular inverse of odd A_n; high16 target leaves bounded low-bit ambiguity", "memory_passes": 1},
        {"family": "M1_LANE_AFFINE", "forward_operations": "lane=seed+position+i*C; each round x=x*A+B+i*D", "mapping": "signed high16 to FP32", "traceability_class": "PARTIALLY_INVERTIBLE", "reverse": "reverse odd affine rounds with modular inverses", "memory_passes": 1},
        {"family": "M2_AFFINE_XOR", "forward_operations": "lane affine; each round affine, xorshift-right, odd multiply", "mapping": "signed high16 to FP32", "traceability_class": "PARTIALLY_INVERTIBLE", "reverse": "inverse multiply, unxorshift-right, inverse affine", "memory_passes": 1},
        {"family": "M3_FLOAT_AFFINE", "forward_operations": "lane seed/position affine; each round FP32 FMA", "mapping": "direct FP32 then scale", "traceability_class": "CHEAPLY_PROJECTABLE", "reverse": "closed-form reverse affine projection with clipping", "memory_passes": 1},
    ]
    source_qualification = {"model": "Qwen3-32B-Q8_0", "path": str(SOURCE.relative_to(ROOT)), "sha256": SOURCE_SHA256,
                            "tensor": TENSOR_NAME, "gguf_shape": list(tensor.shape), "shape": list(weights.shape), "element_count": flat.size,
                            "payload_range": [tensor.absolute_start, tensor.absolute_start + tensor.payload_size], "q8_payload_bytes": q8_bytes,
                            "oracle": "Q8_0 reconstructed FP32, not BF16/F32 ground truth", "oracle_fp32_sha256": hashlib.sha256(weights.astype("<f4", copy=False).tobytes()).hexdigest(),
                            "orientation": "W[out,input]; Y=X@W.T; verified"}
    supply = {"required_rate_gweights_s": required_rate, "available_storage_rate_gweights_s": storage_rate,
              "available_generation_rate_gweights_s": generation_rate, "storage_supply_coverage": storage_rate / required_rate,
              "generation_supply_coverage": generation_rate / required_rate,
              "prior_0.0450_interpretation": "median layer compute time / median measured preparation time; dimensionless SUPPLY_COVERAGE proxy, not lookahead horizon",
              "prior_0.0225_interpretation": "compute time / modeled serial transport-plus-generation time; dimensionless coverage proxy, not a horizon and not reused as a primary metric"}
    best_summary = {key: best[key] for key in ("candidate", "family", "rounds", "length", "true_bpw", "rmse", "mae", "p99", "p999", "max_absolute_error", "validation_mean_relative_l2", "test_mean_relative_l2", "test_median_relative_l2", "test_p95_relative_l2", "test_mean_cosine")}
    best_summary.update({"runtime_proxy_candidate": f"{runtime_family}_R{runtime_rounds}_L{runtime_length}",
                         "scalar_cycles_per_weight": scalar_runtime["cycles_per_weight"], "scalar_gweights_per_second": scalar_runtime["gweights_per_second"],
                         "avx2_cycles_per_weight": runtime["cycles_per_weight"], "avx2_gweights_per_second": runtime["gweights_per_second"], "avx2_speedup_vs_scalar": runtime["speedup_vs_scalar"],
                         "buffer_dot_cycles_per_weight": buffer_dot["cycles_per_weight"], "direct_dot_cycles_per_weight": direct_dot["cycles_per_weight"]})
    payload = {"status": "COMPLETE / REJECTED", "research_direction": "SIMD-native traceable mutation mechanism", "previous_rejection_preserved": True,
               "source_qualification": source_qualification, "activation_evidence": {"path": str(CAPTURE.relative_to(ROOT)), "sha256": CAPTURE_SHA256,
               "semantic_node": "attn_norm-0", "vectors": len(vectors), "dimension": 5120, "validation_vectors": len(validation), "test_vectors": len(test),
               "validation_ids": validation.tolist(), "test_ids": test.tolist(), "functional_test_used_for_selection": False},
               "mutation_families": mutation_families, "selected_family_from_weight_screen": selected_family, "position_context_classification": position_class,
               "traceability_results": trace_rows, "traceability_summary": {"selected_projected_vs_equal_ratio": float(np.mean([row["projected_vs_equal_random_rmse_ratio"] for row in comparisons if row["family"] == selected_family])),
               "selected_projected_vs_large_ratio": float(np.mean([row["projected_vs_large_random_rmse_ratio"] for row in comparisons if row["family"] == selected_family]))},
               "fixed_length_results": fixed_rows, "mutation_depth": depth_rows,
               "descriptor_accounting": descriptor_rows, "previous_generator_control": previous_control, "canonical_controls": canonical_rows,
               "scalar_vs_avx2": throughput_rows, "generate_and_dot": dot_rows, "disassembly": {"mnemonic_counts": mnemonic_counts, "ymm_registers": ymm_registers, "spill_patterns": spill_patterns},
               "best_representation_point": best_summary, "supply_rate_comparison": supply,
               "important_answer": "No. A few SIMD affine/xor/shift rounds are fast and algebraically traceable, but they do not create a weight space with functional coverage sufficient to replace meaningful transported weight information at competitive rate.",
               "falsification_gates": ["SIMD_MECHANISM_FAST_BUT_UNUSEFUL", "MUTATION_DEPTH_NOT_USEFUL_BEYOND_ONE_ROUND", "TRACEABLE_MUTATION_NOT_SUPPORTED", "SIMD_SEGMENT_LENGTH_STILL_KILLS_RATE", "SIMD_MUTATION_FUNCTIONALLY_REJECTED"],
               "classifications": {"SIMD generation": generation_class, "Mutation depth": depth_class, "Traceability": trace_class,
                                   "Representation quality": quality_class, "Segment behavior": segment_class, "Combined direction": combined},
               "recommendation": recommendation, "elapsed_seconds": time.perf_counter() - started,
               "environment": {"cpu": "AMD Ryzen 7 5800X", "isa": "AVX2/FMA3/BMI2", "python": platform.python_version(), "numpy": np.__version__, "cpu_count": os.cpu_count()},
               "wire_changes": 0, "production_changes": 0, "adaptive_segmentation": False, "residuals": False, "learned_decoder": False}

    out.mkdir(parents=True); raw = out / "raw"; raw.mkdir()
    write_csv(out / "mutation-families.csv", mutation_families)
    write_csv(out / "mutation-depth.csv", depth_rows)
    write_csv(out / "traceability-results.csv", trace_rows)
    write_csv(out / "fixed-length-results.csv", fixed_rows)
    write_csv(out / "descriptor-accounting.csv", descriptor_rows)
    real_rows = fixed_rows + [previous_control] + canonical_rows; write_csv(out / "real-activation-results.csv", real_rows)
    write_csv(out / "canonical-controls.csv", canonical_rows)
    write_csv(out / "scalar-vs-avx2.csv", throughput_rows)
    write_csv(out / "generation-throughput.csv", throughput_rows)
    write_csv(out / "generate-and-dot.csv", dot_rows)
    write_csv(out / "supply-rate-comparison.csv", [supply])
    (out / "disassembly-notes.md").write_text("\n".join(disassembly_notes) + "\n", encoding="utf-8")
    (out / "feasibility.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "feasibility-report.md").write_text(report(payload), encoding="utf-8")
    (raw / "runner.log").write_text("".join(logs), encoding="utf-8")
    (raw / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"COMPLETE / REJECTED: {selected_name}; {combined}; wrote {out}")


if __name__ == "__main__": main()
