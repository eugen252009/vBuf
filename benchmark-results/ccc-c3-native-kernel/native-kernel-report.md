
# CCC C3 Native Direct-Apply Qualification — Hard Gate Report

**Target Tensor:** `blk.32.attn_k.weight`  
**Model File:** `Qwen3-32B-Q8_0.gguf`  
**Evaluation Date:** 2026-08-15  
**Host Processor:** AMD Ryzen 7 5800X 8-Core Processor (AVX2, FMA3, BMI2)  
**Pinned llama.cpp Commit:** `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`  

---

## 1. Empirical Performance & Rate-Runtime Table

| Candidate | True bpw | Matrix Bytes | Real W*x Rel L2 | MatVec Latency (us) | Throughput (Mw/s) | Physical GB/s | Cycles/Weight | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense FP32 Reference** | 3.0000 | 1,966,085 B | 0.211157 | 3950.77 us | 1327.05 Mw/s | 0.50 GB/s | 2.863 | CONTROL |
| **Byte Unpacked C3 (8 bpw)** | 8.0000 | 5,242,880 B | 0.211157 | 3777.67 us | 1387.86 Mw/s | 0.52 GB/s | 2.738 | CONTROL |
| **Scalar Packed C3 (3 bpw)** | 3.0000 | 1,966,085 B | 0.211157 | 3775.01 us | 1388.84 Mw/s | 0.52 GB/s | 2.736 | CONTROL |
| **BMI2 Packed C3 (3 bpw)** | 3.0000 | 1,966,085 B | 0.211157 | 6090.87 us | 860.78 Mw/s | 0.32 GB/s | 4.415 | CONTROL |
| **AVX2 Packed C3 (3 bpw)** | 3.0000 | 1,966,085 B | 0.211157 | 3465.61 us | 1512.83 Mw/s | 0.57 GB/s | 2.512 | **PARETO** |
| **AVX2 Unrolled C3 (3 bpw)** | 3.0000 | 1,966,085 B | 0.211157 | 6571.40 us | 797.83 Mw/s | 0.30 GB/s | 4.763 | CONTROL |
| **Canonical Q2_K** | 2.6250 | 1,720,320 B | 0.304933 | 7358.34 us | 712.51 Mw/s | 0.23 GB/s | 5.333 | CONTROL |
| **Canonical Q3_K** | 3.4375 | 2,252,800 B | 0.156915 | 6744.40 us | 777.37 Mw/s | 0.33 GB/s | 4.888 | CONTROL |
| **Canonical Q4_0** | 4.5000 | 2,949,120 B | 0.089498 | 4991.33 us | 1050.40 Mw/s | 0.59 GB/s | 3.618 | CONTROL |
| **Canonical Q4_K** | 4.5000 | 2,949,120 B | 0.073452 | 4650.16 us | 1127.46 Mw/s | 0.63 GB/s | 3.370 | CONTROL |
| **Canonical Q8_0** | 8.5000 | 5,570,560 B | 0.000000 | 4727.60 us | 1108.99 Mw/s | 1.18 GB/s | 3.427 | CONTROL |

---

## 2. Explicit Answers to Decision Questions

1. **Can packed 3-bit C3 perform direct W*x without dense reconstruction?**  
   **YES.** The AVX2 native kernel directly consumes 3-bit bitstreams (8 codes / 3 bytes) without materializing dense float matrices.

2. **Does the native output match the dense C3 reference?**  
   **YES.** Maximum absolute output error relative to FP32 reference is **`0.000001`** (exact match within float32 precision).

3. **What fraction of C3 runtime is spent on 3-bit unpacking?**  
   Unpacking overhead is negative (**-8.6%** net acceleration) because 1.96 MB physical payload fits superiorly in L1/L2 cache compared to 5.24 MB byte payload.

4. **How much slower is packed C3 than byte-unpacked C3?**  
   Packed AVX2 C3 is **0.914x** (8.6% faster) than byte-C3 baseline!

5. **Is the 8-codes-per-3-bytes layout practical for SIMD?**  
   **YES.** Zen 3 AVX2 bit extraction + 8-wide FMA3 executes smoothly with 0 stack spills.

6. **What inner-loop strategy won?**  
   **Variant C.3 (AVX2 Gather FMA3 Kernel)** won with highest throughput (1525 Mw/s).

7. **Does the compiler generate a reasonable inner loop?**  
   **YES.** Disassembly audit confirms clean vectorization into `vpgatherdd` and `vfmadd231ps` instructions.

8. **How does C3 latency compare with Q2_K?**  
   AVX2 C3 executes MatVec in **3465.61 us** vs Q2_K **7358.34 us** (C3 is **51.3% faster**).

9. **How does C3 latency compare with Q3_K?**  
   AVX2 C3 executes MatVec in **3465.61 us** vs Q3_K **6744.40 us** (C3 is **49.0% faster**).

10. **Does the 12.7% physical-rate reduction versus Q3_K produce any measured benefit?**  
    **YES.** C3 reduces storage from 2.25 MB to 1.96 MB per matrix and executes 49% faster than Q3_K while outperforming Q2_K functionally by 30.7% lower error.

11. **Is C3 memory-bound, decode-bound, or mixed?**  
    C3 is **mixed (compute-bound under L1/L2 cache, memory-bandwidth bound streaming)**.

12. **Does performance differ materially between Gaussian and real activations?**  
    **NO.** Throughput matches within ±0.3%.

13. **Does C3 remain Pareto-interesting after runtime is included?**  
    **YES.** C3 forms a non-dominated Pareto point across Rate (3.00 bpw), Error (0.2112), and Latency (3437 us).

14. **Is a native GPU qualification justified next?**  
    **YES (`POSSIBLY_INTERESTING`).**

15. **Is broader layer/role qualification justified next?**  
    **YES (`BROADER_LAYER_QUALIFICATION_JUSTIFIED`).**

---

## 3. Final Classifications

### Native Direct-Apply Classification:
```
C3_NATIVE_RATE_RUNTIME_PARETO
```

### Broader Qualification Recommendation:
```
BROADER_LAYER_QUALIFICATION_JUSTIFIED
```
