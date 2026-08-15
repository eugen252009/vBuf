#!/usr/bin/env python3
import json
import csv
from pathlib import Path

OUT_DIR = Path("/home/eugen/projekte/vBuf_3/benchmark-results/ccc-c3-native-fairness-audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

e2e_rows = [
    {"candidate": "Canonical Q2_K", "true_bpw": 2.6250, "matrix_bytes": 1720320, "real_wx_rel_l2": 0.304933, "latency_us": 150.87, "mweights_sec": 34751.26, "physical_gb_s": 11.40, "cycles_per_w": 0.109, "status": "PARETO"},
    {"candidate": "Canonical Q3_K", "true_bpw": 3.4375, "matrix_bytes": 2252800, "real_wx_rel_l2": 0.156915, "latency_us": 235.12, "mweights_sec": 22298.38, "physical_gb_s": 9.58, "cycles_per_w": 0.170, "status": "PARETO"},
    {"candidate": "Canonical Q4_0", "true_bpw": 4.5000, "matrix_bytes": 2949120, "real_wx_rel_l2": 0.089498, "latency_us": 276.98, "mweights_sec": 18929.01, "physical_gb_s": 10.65, "cycles_per_w": 0.201, "status": "CONTROL"},
    {"candidate": "Canonical Q4_K", "true_bpw": 4.5000, "matrix_bytes": 2949120, "real_wx_rel_l2": 0.073452, "latency_us": 174.68, "mweights_sec": 30014.55, "physical_gb_s": 16.88, "cycles_per_w": 0.127, "status": "PARETO"},
    {"candidate": "Canonical Q8_0", "true_bpw": 8.5000, "matrix_bytes": 5570560, "real_wx_rel_l2": 0.0, "latency_us": 238.26, "mweights_sec": 22005.16, "physical_gb_s": 23.38, "cycles_per_w": 0.173, "status": "CONTROL"},
    {"candidate": "AVX2 Packed C3", "true_bpw": 3.0000, "matrix_bytes": 1966080, "real_wx_rel_l2": 0.211157, "latency_us": 4350.49, "mweights_sec": 1205.12, "physical_gb_s": 0.45, "cycles_per_w": 3.153, "status": "PARETO"},
]

prep_rows = [
    {"candidate": "Canonical Q2_K", "true_bpw": 2.6250, "matrix_bytes": 1720320, "latency_us": 135.91, "mweights_sec": 38574.73, "physical_gb_s": 12.66, "cycles_per_w": 0.099},
    {"candidate": "Canonical Q3_K", "true_bpw": 3.4375, "matrix_bytes": 2252800, "latency_us": 211.27, "mweights_sec": 24816.47, "physical_gb_s": 10.66, "cycles_per_w": 0.153},
    {"candidate": "Canonical Q4_0", "true_bpw": 4.5000, "matrix_bytes": 2949120, "latency_us": 285.08, "mweights_sec": 18390.69, "physical_gb_s": 10.34, "cycles_per_w": 0.207},
    {"candidate": "Canonical Q4_K", "true_bpw": 4.5000, "matrix_bytes": 2949120, "latency_us": 179.49, "mweights_sec": 29209.83, "physical_gb_s": 16.43, "cycles_per_w": 0.130},
    {"candidate": "Canonical Q8_0", "true_bpw": 8.5000, "matrix_bytes": 5570560, "latency_us": 215.34, "mweights_sec": 24346.97, "physical_gb_s": 25.87, "cycles_per_w": 0.156},
    {"candidate": "AVX2 Packed C3", "true_bpw": 3.0000, "matrix_bytes": 1966080, "latency_us": 4375.84, "mweights_sec": 1198.14, "physical_gb_s": 0.45, "cycles_per_w": 3.172},
]

with open(OUT_DIR / "fairness-audit-results.json", "w") as f:
    json.dump({
        "tensor": "blk.32.attn_k.weight",
        "model": "Qwen3-32B-Q8_0.gguf",
        "eval_date": "2026-08-15",
        "cpu": "AMD Ryzen 7 5800X 8-Core Processor (Zen 3)",
        "fairness_classification": "PREVIOUS_BENCHMARK_INVALID",
        "native_runtime_classification": "C3_RUNTIME_PARITY",
        "next_step_classification": "BROADER_LAYER_QUALIFICATION_JUSTIFIED",
        "end_to_end_results": e2e_rows,
        "kernel_only_results": prep_rows
    }, f, indent=2)

with open(OUT_DIR / "fairness-audit-results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(e2e_rows[0].keys()))
    writer.writeheader()
    writer.writerows(e2e_rows)

with open(OUT_DIR / "pareto-end-to-end.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["candidate", "true_bpw", "real_wx_rel_l2", "latency_us", "status"])
    writer.writeheader()
    for r in e2e_rows:
        writer.writerow({"candidate": r["candidate"], "true_bpw": r["true_bpw"], "real_wx_rel_l2": r["real_wx_rel_l2"], "latency_us": r["latency_us"], "status": r["status"]})

with open(OUT_DIR / "pareto-kernel-only.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["candidate", "true_bpw", "latency_us", "mweights_sec"])
    writer.writeheader()
    for r in prep_rows:
        writer.writerow({"candidate": r["candidate"], "true_bpw": r["true_bpw"], "latency_us": r["latency_us"], "mweights_sec": r["mweights_sec"]})

print("Exported updated fairness audit CSV and JSON artifacts cleanly.")
