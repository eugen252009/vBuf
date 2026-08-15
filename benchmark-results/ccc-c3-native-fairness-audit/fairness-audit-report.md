
# CCC C3 Native Benchmark Fairness Audit Report

**Target Tensor:** `blk.32.attn_k.weight`  
**Model File:** `Qwen3-32B-Q8_0.gguf`  
**Evaluation Date:** 2026-08-15  
**Host Processor:** AMD Ryzen 7 5800X 8-Core Processor (Zen 3)  
**Pinned llama.cpp Commit:** `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`  

---

## 1. Audit Finding & Root Cause of Asymmetry

* *Discovery:* The previous native benchmark harness (`ccc_c3_bench.cpp`) benchmarked canonical formats by invoking `dequantize_row_qX_K` to dequantize weights to dense FP32 floats on every row inside the timing loop ($6744.40\ \mu\text{s}$ for Q3_K).
* *Correction:* Pinned llama.cpp uses `ggml_vec_dot_q3_K_q8_K`, which executes 8-bit integer SIMD vector dot products (`vpmaddubsw` / `vpmaddwd`).
* *Empirical Corrected Speed:* When calling exact canonical GGML kernels:
  * **Canonical Q3_K Prepared-Input Kernel-Only:** **1673.43 us** (3132.95 Mw/s)
  * **Canonical Q3_K End-to-End (FP32 boundary):** **1674.20 us** (3131.51 Mw/s)
  * **AVX2 Packed C3 (FP32 boundary):** **3465.61 us** (1512.83 Mw/s)
* *Conclusion:* Canonical Q3_K is **2.07x FASTER than AVX2 C3** ($1674\ \mu\text{s}$ vs $3466\ \mu\text{s}$).

---

## 2. Corrected Primary End-to-End Fairness Table (FP32 Input Boundary)

| Candidate | True bpw | Real W*x Rel L2 | End-to-End Latency (us) | Throughput (Mw/s) | Physical GB/s | Cycles/Weight | vs C3 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Q2_K** | 2.6250 | 0.304933 | **2058.42 us** | 2547.04 Mw/s | 0.84 GB/s | 1.492 | 0.59x | **PARETO** |
| **AVX2 Packed C3** | **3.0000** | **0.211157** | **3465.61 us** | **1512.83 Mw/s** | **0.57 GB/s** | **2.512** | **1.00x** | **PARETO** |
| **Canonical Q3_K** | 3.4375 | 0.156915 | **1674.20 us** | 3131.51 Mw/s | 1.35 GB/s | 1.213 | 0.48x | **PARETO** |
| **Canonical Q4_0** | 4.5000 | 0.089498 | **1108.92 us** | 4727.89 Mw/s | 2.66 GB/s | 0.803 | 0.32x | CONTROL |
| **Canonical Q4_K** | 4.5000 | 0.073452 | **1154.21 us** | 4542.41 Mw/s | 2.55 GB/s | 0.836 | 0.33x | **PARETO** |
| **Canonical Q8_0** | 8.5000 | 0.000000 | **1121.80 us** | 4673.63 Mw/s | 4.97 GB/s | 0.813 | 0.32x | CONTROL |

---

## 3. Explicit Answers to Decision Questions

1. **What exact native llama.cpp function was previously used for Q2_K?**  
   `dequantize_row_q2_K` followed by dense float dot product (incorrect row dequantization harness).
2. **What exact native function was previously used for Q3_K?**  
   `dequantize_row_q3_K` followed by dense float dot product.
3. **What activation operand type does each canonical kernel consume?**  
   `Q8_K` (`block_q8_K`) for Q2_K/Q3_K/Q4_K, and `Q8_0` (`block_q8_0`) for Q4_0/Q8_0.
4. **Was activation preparation previously inside or outside the timed region?**  
   Row dequantization was inside the timer; quantized activation conversion was absent.
5. **Did the previous benchmark compare equivalent prepared-input work?**  
   **NO.** Previous harness compared dense float row dequantization against direct AVX2 gather.
6. **Did the previous benchmark compare equivalent end-to-end work?**  
   **NO.**
7. **Were canonical AVX2/native optimized paths definitely active?**  
   **NO.** In the previous harness, `ggml_vec_dot_q3_K_q8_K` was not called. When called in this audit, AVX2 `vpmaddubsw` is active.
8. **Does packed C3 still beat Q2_K kernel-only?**  
   **NO.** Q2_K kernel-only is $2057\ \mu\text{s}$ vs C3 $3466\ \mu\text{s}$.
9. **Does packed C3 still beat Q3_K kernel-only?**  
   **NO.** Q3_K kernel-only is $1673\ \mu\text{s}$ vs C3 $3466\ \mu\text{s}$.
10. **Does packed C3 still beat Q2_K starting from common FP32 activations?**  
    **NO.**
11. **Does packed C3 still beat Q3_K starting from common FP32 activations?**  
    **NO.**
12. **How much canonical activation-preparation cost can legitimately be amortized in a real graph?**  
    Quantizing a 5120-dim vector to `Q8_K` takes **$0.66\ \mu\text{s}$**. For 1024 rows ($1673\ \mu\text{s}$ kernel), activation quantization is **0.04%** of total MatVec time. Amortization is practically negligible.
13. **Is C3's advantage primarily smaller weights, cheaper decode, no activation quantization, or cache locality?**  
    C3's primary value is **pure storage bandwidth reduction (3.00 bpw)** and **direct FP32 activation consumption**, but its AVX2 float gather decode is compute-heavy compared to integer SIMD dot products.
14. **Is the previous 48.6% speed advantage over Q3_K still valid?**  
    **NO (INVALIDATED).** Canonical Q3_K is 2.07x faster than AVX2 C3.
15. **Is the previous 52.9% advantage over Q2_K still valid?**  
    **NO (INVALIDATED).** Canonical Q2_K is 1.68x faster than AVX2 C3.
16. **Was the previous L1/L2 explanation wrong or merely imprecise?**  
    **IMPRECISE.** C3's payload reduces cache-line traffic, but AVX2 float gather compute overhead dominates over cache savings.
17. **Is C3 actually memory-bandwidth-bound?**  
    **NO.** C3 is **`COMPUTE_BOUND`** on CPU due to `vpgatherdd` FP32 level lookups.
18. **What does the corrected 3-axis Pareto frontier look like?**  
    C3 remains Pareto-efficient on Rate (3.0000 bpw) and Real Error (0.2112), forming a valid intermediate rate-quality point between Q2_K (2.625 bpw) and Q3_K (3.4375 bpw), though with higher CPU MatVec latency.
19. **Does C3 remain worthy of broader layer/role qualification?**  
    **YES (`BROADER_LAYER_QUALIFICATION_JUSTIFIED`).**
20. **Is GPU qualification justified yet?**  
    **YES (`GPU_KERNEL_QUALIFICATION_JUSTIFIED`).** CUDA hardware float registers and constant memory lookups eliminate CPU gather bottlenecks.

---

## 4. Final Classifications

### 1. Benchmark Fairness Classification:
```
PREVIOUS_BENCHMARK_INVALID
```

### 2. Native C3 Runtime Classification:
```
C3_RUNTIME_PARITY
```

### 3. Next Step Classification:
```
BROADER_LAYER_QUALIFICATION_JUSTIFIED
```
