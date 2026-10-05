#!/usr/bin/env python3
"""Independent Q4_K decode and CPU/CUDA effective-activation Q-projection oracles."""
from pathlib import Path
import json, sys
import numpy as np

root = Path(sys.argv[1])
fix = root / "fixture"
width, positions = 5120, 25
packed = np.fromfile(fix / "blk.0.attn_q.weight.q4k.bin", dtype=np.uint8)
assert packed.size == 5120 * 20 * 144
blocks = packed.reshape(5120, 20, 144)
dh = np.frombuffer(blocks[:, :, 0:2].copy().tobytes(), dtype="<f2").astype(np.float32).reshape(5120, 20)
mh = np.frombuffer(blocks[:, :, 2:4].copy().tobytes(), dtype="<f2").astype(np.float32).reshape(5120, 20)
sb = blocks[:, :, 4:16]
qs = blocks[:, :, 16:144]
scales = np.empty((5120, 20, 8), dtype=np.uint8)
mins = np.empty_like(scales)
for g in range(8):
    if g < 4:
        scales[:, :, g] = sb[:, :, g] & 63
        mins[:, :, g] = sb[:, :, g + 4] & 63
    else:
        scales[:, :, g] = (sb[:, :, g + 4] & 15) | ((sb[:, :, g - 4] >> 6) << 4)
        mins[:, :, g] = (sb[:, :, g + 4] >> 4) | ((sb[:, :, g] >> 6) << 4)

codes = np.empty((5120, 5120), dtype=np.uint8)
for b in range(20):
    for half_group in range(4):
        q = qs[:, b, half_group * 32:(half_group + 1) * 32]
        base = b * 256 + half_group * 64
        codes[:, base:base + 32] = q & 15
        codes[:, base + 32:base + 64] = q >> 4

# Independent Q4_K decoder: standard 8x32 scale/min groups, with explicit
# IEEE-754 half reads and packed-nibble expansion.
manual_f32 = np.empty((5120, 5120), dtype=np.float32)
manual_f64 = np.empty((5120, 5120), dtype=np.float64)
for b in range(20):
    for g in range(8):
        sl = slice(b * 256 + g * 32, b * 256 + (g + 1) * 32)
        c = codes[:, sl]
        ws32 = dh[:, b] * scales[:, b, g].astype(np.float32)
        wm32 = mh[:, b] * mins[:, b, g].astype(np.float32)
        manual_f32[:, sl] = c.astype(np.float32) * ws32[:, None] - wm32[:, None]
        ws64 = dh[:, b].astype(np.float64) * scales[:, b, g].astype(np.float64)
        wm64 = mh[:, b].astype(np.float64) * mins[:, b, g].astype(np.float64)
        manual_f64[:, sl] = c.astype(np.float64) * ws64[:, None] - wm64[:, None]

pinned_f32 = np.fromfile(fix / "q4k-weight-ggml-dequant.f32", dtype="<f4").reshape(5120, 5120)
decoder_diff = manual_f32.astype(np.float64) - pinned_f32.astype(np.float64)
decoder = {
    "manual_vs_pinned_ggml_bitwise_equal": bool(np.array_equal(manual_f32, pinned_f32)),
    "manual_vs_pinned_max_abs": float(np.max(np.abs(decoder_diff))),
    "manual_vs_pinned_mismatch_elements": int(np.count_nonzero(manual_f32.view(np.uint32) != pinned_f32.view(np.uint32))),
    "decoded_elements": int(manual_f32.size),
}

x = np.fromfile(fix / "prefix25-q-input.f32.bin", dtype="<f4").reshape(positions, width)
cpu_eff = np.fromfile(fix / "cpu-q8k.effective.f32", dtype="<f4").reshape(positions, width)

# Reconstruct the pinned CUDA MMQ Q8_1 DS4 quantizer. The kernel processes
# 32-value groups as eight consecutive float4 lanes, uses roundf, stores the
# scale and pre-quantization subgroup sum as FP16, and packs four groups per
# 128-value block. The XOR reduction order for the sum is 4, 2, 1.
cuda_q = np.empty((positions, width), dtype=np.int8)
cuda_d = np.empty((positions, width // 32), dtype=np.float16)
cuda_s = np.empty_like(cuda_d)
for p in range(positions):
    for g in range(width // 32):
        v = x[p, g * 32:(g + 1) * 32]
        amax = np.max(np.abs(v)).astype(np.float32)
        d_inv = np.float32(np.float32(127.0) / amax)
        scaled = (v * d_inv).astype(np.float32)
        rounded = np.where(scaled >= 0, np.floor(scaled + np.float32(0.5)),
                           np.ceil(scaled - np.float32(0.5))).astype(np.int32)
        cuda_q[p, g * 32:(g + 1) * 32] = np.clip(rounded, -128, 127).astype(np.int8)
        cuda_d[p, g] = np.float16(np.float32(np.float32(1.0) / d_inv))
        lane_sums = np.empty(8, dtype=np.float32)
        for lane in range(8):
            z = v[lane * 4:(lane + 1) * 4]
            lane_sums[lane] = np.float32(np.float32(np.float32(z[0] + z[1]) + z[2]) + z[3])
        for offset in (4, 2, 1):
            old = lane_sums.copy()
            lane_sums = (old + old[np.arange(8) ^ offset]).astype(np.float32)
        cuda_s[p, g] = np.float16(lane_sums[0])
cuda_eff = cuda_q.astype(np.float64) * cuda_d.astype(np.float64).repeat(32, axis=1)
np.savez_compressed(fix / "cuda-q8_1-ds4-reconstruction.npz", q=cuda_q, d=cuda_d, s=cuda_s)
cuda_eff.astype("<f4").tofile(fix / "cuda-q8_1-effective.f32")

# Oracle A: high-precision dot of independently decoded Q4_K weights and the
# original common F32 activation. Oracle B: same Q4_K decoder with each
# backend's effective quantized activation. These are ordinary FP64 GEMMs.
orig = manual_f64 @ x.astype(np.float64).T
cpu_b = manual_f64 @ cpu_eff.astype(np.float64).T
cuda_b = manual_f64 @ cuda_eff.T

# Oracle C follows the pinned CUDA Q4_K x Q8_1 MMQ algebra. The primary term
# uses int8 q values and the FP16 activation scale; the affine minimum term
# uses the separately stored FP16 sum of original F32 activations.
cuda_c = np.zeros((5120, positions), dtype=np.float64)
for g in range(width // 32):
    b, sg = divmod(g, 8)
    c = codes[:, g * 32:(g + 1) * 32].astype(np.int32)
    q = cuda_q[:, g * 32:(g + 1) * 32].astype(np.int32)
    dot = c @ q.T
    ws = dh[:, b].astype(np.float64) * scales[:, b, sg].astype(np.float64)
    wm = mh[:, b].astype(np.float64) * mins[:, b, sg].astype(np.float64)
    ad = cuda_d[:, g].astype(np.float64)
    asum = cuda_s[:, g].astype(np.float64)
    cuda_c += ws[:, None] * (dot.astype(np.float64) * ad[None, :]) - wm[:, None] * asum[None, :]

cpu_actual = np.fromfile(fix / "cpu-q-output.f32.bin", dtype="<f4").astype(np.float64).reshape(positions, width).T
cuda_actual = np.fromfile(fix / "cuda-q-output.f32.bin", dtype="<f4").astype(np.float64).reshape(positions, width).T

def metrics(reference, actual):
    d = reference - actual
    rms = float(np.sqrt(np.mean(d * d)))
    ar = float(np.sqrt(np.mean(actual * actual)))
    rr = float(np.sqrt(np.mean(reference * reference)))
    return {"rms": rms, "relative_rms_actual_denominator": rms / ar,
            "relative_rms_reference_denominator": rms / rr,
            "max_abs": float(np.max(np.abs(d))),
            "cosine": float(np.sum(reference * actual) / (np.linalg.norm(reference) * np.linalg.norm(actual))),
            "norm_ratio_reference_over_actual": float(np.linalg.norm(reference) / np.linalg.norm(actual))}

for name, out in (("oracle_a_original_f32", orig), ("oracle_b_cpu_q8k", cpu_b),
                   ("oracle_b_cuda_q8_1_vector", cuda_b), ("oracle_c_cuda_ds4_mmq", cuda_c)):
    out.T.astype("<f8").tofile(fix / (name + ".f64"))

input_metrics = {
    "cpu_q8k_effective_rel_rms_vs_f32": float(np.sqrt(np.mean((cpu_eff.astype(np.float64)-x.astype(np.float64))**2)) / np.sqrt(np.mean(x.astype(np.float64)**2))),
    "cuda_q8_1_effective_rel_rms_vs_f32": float(np.sqrt(np.mean((cuda_eff-x.astype(np.float64))**2)) / np.sqrt(np.mean(x.astype(np.float64)**2))),
    "cpu_q8k_vs_cuda_q8_1_effective_rel_rms_cpu_denominator": float(np.sqrt(np.mean((cpu_eff.astype(np.float64)-cuda_eff)**2)) / np.sqrt(np.mean(cpu_eff.astype(np.float64)**2))),
    "cuda_q8_1_scale_groups": int(cuda_d.size),
    "cuda_q8_1_ds4_partial_sums_fp16": True,
}
results = {
    "weight_decoder": decoder,
    "activation_reconstruction": input_metrics,
    "oracles_vs_actual": {
        "oracle_a_vs_cpu": metrics(orig, cpu_actual),
        "oracle_a_vs_cuda": metrics(orig, cuda_actual),
        "oracle_b_cpu_q8k_vs_cpu": metrics(cpu_b, cpu_actual),
        "oracle_b_cuda_q8_1_vector_vs_cuda": metrics(cuda_b, cuda_actual),
        "oracle_c_cuda_ds4_mmq_vs_cuda": metrics(cuda_c, cuda_actual),
    },
    "cpu_actual_vs_cuda_actual": metrics(cpu_actual, cuda_actual),
    "interpretation_boundary": "Independent Q4_K decoder was compared against pinned GGML dequantization; numerical oracle results do not by themselves classify backend kernel dispatch unless the corresponding selected path is established. CUDA Oracle C models the source-defined MMQ Q8_1 DS4 affine-sum algebra.",
}
if not decoder["manual_vs_pinned_ggml_bitwise_equal"]:
    raise SystemExit("independent Q4_K decoder does not match pinned GGML")
if results["oracles_vs_actual"]["oracle_b_cpu_q8k_vs_cpu"]["relative_rms_actual_denominator"] > 1e-6:
    raise SystemExit("CPU Q8_K activation oracle no longer matches the captured CPU projection")
if results["oracles_vs_actual"]["oracle_c_cuda_ds4_mmq_vs_cuda"]["relative_rms_actual_denominator"] > 1e-3:
    raise SystemExit("CUDA DS4 MMQ oracle no longer matches the captured CUDA projection")
if results["oracles_vs_actual"]["oracle_b_cuda_q8_1_vector_vs_cuda"]["relative_rms_actual_denominator"] < 1e-2:
    raise SystemExit("CUDA Q8_1 vector control unexpectedly ceased to distinguish DS4 affine sums")
(fix / "q4k-activation-oracle-results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
