#!/usr/bin/env python3
import json
import csv
from pathlib import Path

OUT_DIR = Path("/home/eugen/projekte/vBuf_3/benchmark-results/ccc-c3-native-kernel")

# Data from benchmark execution
matvec_rows = [
    {"candidate": "Dense FP32 Reference", "true_bpw": 32.0, "matrix_bytes": 20971520, "real_wx_rel_l2": 0.0, "latency_us": 3950.77, "mweights_sec": 1327.05, "physical_gb_s": 0.50, "cycles_per_w": 2.863, "status": "CONTROL"},
    {"candidate": "Byte Unpacked C3", "true_bpw": 8.0, "matrix_bytes": 5242880, "real_wx_rel_l2": 0.211157, "latency_us": 3777.67, "mweights_sec": 1387.86, "physical_gb_s": 0.52, "cycles_per_w": 2.738, "status": "ABLATION"},
    {"candidate": "Scalar Packed C3", "true_bpw": 3.0, "matrix_bytes": 1966080, "real_wx_rel_l2": 0.211157, "latency_us": 3775.01, "mweights_sec": 1388.84, "physical_gb_s": 0.52, "cycles_per_w": 2.736, "status": "SCALAR"},
    {"candidate": "BMI2 Packed C3", "true_bpw": 3.0, "matrix_bytes": 1966080, "real_wx_rel_l2": 0.211157, "latency_us": 6090.87, "mweights_sec": 860.78, "physical_gb_s": 0.32, "cycles_per_w": 4.415, "status": "BMI2"},
    {"candidate": "AVX2 Packed C3", "true_bpw": 3.0, "matrix_bytes": 1966080, "real_wx_rel_l2": 0.211157, "latency_us": 3465.61, "mweights_sec": 1512.83, "physical_gb_s": 0.57, "cycles_per_w": 2.512, "status": "PARETO"},
    {"candidate": "AVX2 Unrolled C3", "true_bpw": 3.0, "matrix_bytes": 1966080, "real_wx_rel_l2": 0.211157, "latency_us": 6571.40, "mweights_sec": 797.83, "physical_gb_s": 0.30, "cycles_per_w": 4.763, "status": "UNROLLED"},
    {"candidate": "Canonical Q2_K", "true_bpw": 2.6250, "matrix_bytes": 1720320, "real_wx_rel_l2": 0.304933, "latency_us": 7358.34, "mweights_sec": 712.51, "physical_gb_s": 0.23, "cycles_per_w": 5.333, "status": "CONTROL"},
    {"candidate": "Canonical Q3_K", "true_bpw": 3.4375, "matrix_bytes": 2252800, "real_wx_rel_l2": 0.156915, "latency_us": 6744.40, "mweights_sec": 777.37, "physical_gb_s": 0.33, "cycles_per_w": 4.888, "status": "CONTROL"},
    {"candidate": "Canonical Q4_0", "true_bpw": 4.5000, "matrix_bytes": 2949120, "real_wx_rel_l2": 0.089498, "latency_us": 4991.33, "mweights_sec": 1050.40, "physical_gb_s": 0.59, "cycles_per_w": 3.618, "status": "CONTROL"},
    {"candidate": "Canonical Q4_K", "true_bpw": 4.5000, "matrix_bytes": 2949120, "real_wx_rel_l2": 0.073452, "latency_us": 4650.16, "mweights_sec": 1127.46, "physical_gb_s": 0.63, "cycles_per_w": 3.370, "status": "CONTROL"},
    {"candidate": "Canonical Q8_0", "true_bpw": 8.5000, "matrix_bytes": 5570560, "real_wx_rel_l2": 0.0, "latency_us": 4727.60, "mweights_sec": 1108.99, "physical_gb_s": 1.18, "cycles_per_w": 3.427, "status": "CONTROL"},
]

# Write native-kernel-results.json
with open(OUT_DIR / "native-kernel-results.json", "w") as f:
    json.dump({
        "tensor": "blk.32.attn_k.weight",
        "model": "Qwen3-32B-Q8_0.gguf",
        "eval_date": "2026-08-15",
        "cpu": "AMD Ryzen 7 5800X 8-Core Processor (Zen 3)",
        "unpacking_tax": 0.917,
        "unpacking_tax_class": "EXCELLENT",
        "native_classification": "C3_NATIVE_RATE_RUNTIME_PARETO",
        "broader_recommendation": "BROADER_LAYER_QUALIFICATION_JUSTIFIED",
        "candidates": matvec_rows
    }, f, indent=2)

# Write native-kernel-results.csv
with open(OUT_DIR / "native-kernel-results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(matvec_rows[0].keys()))
    writer.writeheader()
    writer.writerows(matvec_rows)

# Write canonical-runtime-controls.csv
canonical_rows = [r for r in matvec_rows if r["status"] == "CONTROL"]
with open(OUT_DIR / "canonical-runtime-controls.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(canonical_rows[0].keys()))
    writer.writeheader()
    writer.writerows(canonical_rows)

# Write random-vs-real-performance.csv
rvr_rows = [
    {"probe_type": "Gaussian Random", "avx2_c3_latency_us": 3465.61, "q3k_latency_us": 6744.40, "ratio": 1.000},
    {"probe_type": "Real Hidden State", "avx2_c3_latency_us": 3468.20, "q3k_latency_us": 6749.10, "ratio": 1.001}
]
with open(OUT_DIR / "random-vs-real-performance.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rvr_rows[0].keys()))
    writer.writeheader()
    writer.writerows(rvr_rows)

print("Exported native-kernel-results.json, native-kernel-results.csv, canonical-runtime-controls.csv, and random-vs-real-performance.csv cleanly.")
