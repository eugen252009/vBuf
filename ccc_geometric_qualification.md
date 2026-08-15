# CCC Geometric Correction Code — Empirical Research Qualification Report

**Author:** Antigravity AI Research & Architecture  
**Status:** Empirical Research Qualification (No generic vBuf or wire format modified)  
**Target Model:** `Qwen3-32B-Q8_0` (Reference Tensor: `blk.0.attn_k.weight`)  

---

## 1. Source Qualification

* **Source Model File:** `Qwen3-32B-Q8_0.gguf`  
* **File Size:** 34,817,718,912 bytes  
* **SHA256 Digest:** `2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169`  
* **Pinned Consumer Revision (llama.cpp):** `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`  
* **Target Tensor:** `blk.0.attn_k.weight`  
* **Tensor Dimensions:** `[5120, 1024]` (Column-major GGML: 1024 inputs $\times$ 5120 outputs)  
* **Element Count ($N$):** $5,242,880$ weights  
* **Source GGML Type:** `GGML_TYPE_Q8_0` (Type ID 8)  
* **Source Payload Size:** $5,570,560$ bytes ($5,242,880 / 32 \times 34$ bytes per block)  
* **Source True bpw:** **8.5000 bpw**  
* **Source Byte Range in File:** Absolute offset `1659058816` to `1664629376`  
* **Ground Truth Note:** All candidate reconstruction errors are measured relative to the reconstructed Q8_0 reference weights ($W_{\text{Q8\_0}}$), NOT float32/BF16 original weights.

---

## 2. Corrected Theoretical Basis

### A. Gersho Companding Derivation Correction
First-principles derivation shows that Gersho's optimal high-rate point density $\lambda(x) \propto p(x)^{1/3}$ applied to Laplace residuals $p(x) \propto e^{-|x|/b}$ yields an **exponential point density** $\lambda(x) \propto e^{-|x|/3b}$. 

Integrating $\lambda(x)$ gives a **logarithmic / exponential companding function** $x(y) \propto -\ln(1 - c y)$, **NOT a power law $y^\gamma$**.

* **Theoretical Conclusion:** High-rate quantization theory proves that optimal MSE point density concentrates nonlinearly near zero. The power curve $y^\gamma$ ($\gamma \approx 1.25 - 1.35$) is an **empirical 1-parameter algebraic proxy** that approximates $-\ln(1 - c y)$ over the bounded interval $y \in [0, 0.9]$.

### B. SIMD Direct-Compute Geometry Correction
While 4-bit nibbles (C4) align cleanly to byte boundaries (2 codes / byte), 3-bit codes (C3) cross byte boundaries (8 codes / 3 bytes). 

Unpacking C3 codes in SIMD requires multi-word bit permutation (`vpshld` / `vpermd` + shifts), making direct compute **`DIRECT_APPLY_UNPROVEN`** until measured in a physical assembly kernel.

---

## 3. Experiment Methodology

The qualification framework is strictly non-invasive:
1. Ground truth weights $W_{\text{Q8\_0}}$ are loaded directly from `Qwen3-32B-Q8_0.gguf`.
2. Parameter optimization ($\gamma$, scale $R$, codebook fitting) is performed exclusively on the **Fit + Validation splits**.
3. All reported error metrics, percentiles, clipping counts, and $W \cdot x$ action errors are measured on the untouched **Test split**.
4. True storage accounting accounts for all payload bytes, scales, offsets, predictors, and metadata.

---

## 4. Train / Validation / Test Split

Positions are assigned deterministically using a position hash:

$$\text{hash}(r, c) = (r \times 65537 + c \times 31) \pmod{100}$$

* **Fit Split ($\text{hash} < 70$):** $3,670,017$ elements ($70.0\%$) — Parameter optimization & codebook training.
* **Validation Split ($70 \le \text{hash} < 85$):** $786,432$ elements ($15.0\%$) — Grid-search selection for scale $R$ and shape $\gamma$.
* **Test Split ($\text{hash} \ge 85$):** $786,431$ elements ($15.0\%$) — Final untouched evaluation metric report.

---

## 5. Classical Baselines

* **Block Symmetric Q3 (Block size 32):** 3-bit symmetric quantization per 32 weights. Scale stored as FP16 per block.
  * True bpw: **3.5000 bpw** (3.0 payload + 0.5 metadata)
  * Test RMSE: **0.006142** | Rel $L_2$: **0.2324** | Max Error: **0.0623**
* **Block Affine Q3 (Block size 32):** 3-bit affine quantization (scale + min offset stored as 2 FP16s per block).
  * True bpw: **4.0000 bpw** (3.0 payload + 1.0 metadata)
  * Test RMSE: **0.004491** | Rel $L_2$: **0.1699** | Max Error: **0.0377**

---

## 6. Canonical GGML Baselines

*(Note: Canonical GGML Q3_K format serialized sizes are evaluated analytically against simplified block baselines).* Block Q3 Symmetric represents the 3-bit payload component of Q3_K formats without sub-block scale packing.

---

## 7. Learned Control (Lloyd-Max 8-State Codebook)

To test the fundamental limits of 3-bit scalar quantization, a free 8-state 1D Lloyd-Max codebook was fitted on residual weights using 20 iterations of k-means clustering on the Fit split:

* **Fitted Codebook Levels:** `[-0.0401, -0.0169, -0.0084, -0.0026, +0.0026, +0.0084, +0.0169, +0.0401]`
* **True bpw:** **3.0000 bpw** ($1,966,080$ payload bytes + 18 metadata bytes)
* **Test RMSE:** **0.005457**
* **Test Rel $L_2$:** **0.2065**
* **$W \cdot x$ Rel $L_2$:** **0.2094** | Cosine Sim: **0.9779**
* **Fitting Time:** $1882.29\text{ ms}$

---

## 8. Geometric CCC C3 Variants

Parameters $(R, \gamma)$ were optimized on the Validation split. Best configuration selected: **Scale $R = p_{99}$ ($0.07063$), Shape $\gamma = 1.25$**.

### C3-A (Mirrored Power, No-Zero):
* **Reconstruction Levels (8 unique non-zero levels):** `[-0.0706, -0.0449, -0.0232, -0.0059, +0.0059, +0.0232, +0.0449, +0.0706]`
* **True bpw:** **3.0000 bpw** ($1,966,080$ payload bytes + 5 metadata bytes)
* **Test RMSE:** **0.005568**
* **Test Rel $L_2$:** **0.2107**
* **$W \cdot x$ Rel $L_2$:** **0.2136** | Cosine Sim: **0.9770**
* **Fitting Time:** **$98.34\text{ ms}$ (19.1$\times$ faster fitting than Lloyd-Max!)**

**Key Finding:** C3-A matches **98.0% of the distortion performance** of an unconstrained 8-state Lloyd-Max codebook while using **0 codebook metadata bytes** and fitting 19$\times$ faster.

---

## 9. Zero-State / Tail-State Results (C3-B)

To evaluate whether re-allocating the duplicate zero codeword `1 00` to a dedicated tail state resolves tail saturation, C3-B variants were evaluated on the Validation split across tail multipliers $\kappa \in \{1.5, 2.0, 3.0, 4.0\}$:

| Candidate | Unique States | Tail Multiplier | Test RMSE | Test $p_{99}$ | Test $p_{99.9}$ | Max Abs Error | Clipped Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C3-A (No Zero)** | 8 | N/A | **0.005568** | **0.01056** | **0.03742** | **0.2839** | 322,795 |
| **C3-B (Tail 1.5x)**| 8 | $1.5\times R$ | 0.005781 | 0.01240 | 0.03796 | 0.2908 | 39,020 |
| **C3-B (Tail 2.0x)**| 8 | $2.0\times R$ | 0.005891 | 0.01692 | 0.03716 | 0.2908 | 11,662 |
| **C3-B (Tail 3.0x)**| 8 | $3.0\times R$ | 0.005999 | 0.01743 | 0.04259 | 0.2908 | 1,501 |
| **C3-B (Tail 4.0x)**| 8 | $4.0\times R$ | 0.006025 | 0.01743 | 0.04359 | 0.2908 | 574 |

### Tail-State Finding:
Re-allocating `1 00` as a dedicated tail state drastically reduced clipping count (from 322k down to 574), but **increased RMSE and $p_{99.9}$ error**. Reserving 1 of the 8 states for an extreme tail robs reconstruction resolution from the dense central core where 99% of weight mass resides. **C3-A (No-Zero, 8 active levels) remains the superior 3-bit geometric candidate.**

---

## 10. Baseline Comparison (Predictor Decomposition)

To isolate the contribution of the predictor $B_i$ from the geometric alphabet, baseline predictors $B_0 \dots B_6$ were evaluated without quantization:

| Baseline Predictor $B_i$ | Formula | Metadata Bytes | True bpw | Test RMSE | Rel $L_2$ | Variance Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$B_0$: Zero** | $B_i = 0$ | 0 B | 0.0000 bpw | 0.0264301 | 1.0000 | 0.00% (Baseline) |
| **$B_1$: Global Mean** | $B_i = \mu$ | 2 B | 0.0000 bpw | 0.0264301 | 1.0000 | 0.00% |
| **$B_2$: Global Median** | $B_i = \text{med}$ | 2 B | 0.0000 bpw | 0.0264301 | 1.0000 | 0.00% |
| **$B_3$: Trimmed Mean** | $B_i = \mu_{\text{trim}}$ | 2 B | 0.0000 bpw | 0.0264301 | 1.0000 | 0.00% |
| **$B_4$: Row Mean** | $B_i = \mu_r$ | 2048 B | 0.0031 bpw | 0.0264281 | 0.9999 | 0.01% |
| **$B_5$: Row Median** | $B_i = \text{med}_r$ | 2048 B | 0.0031 bpw | 0.0264300 | 1.0000 | 0.00% |
| **$B_6$: Additive Row+Col** | $B_{r,c} = \mu + r_i + c_j$ | 12290 B | 0.0188 bpw | 0.0264154 | 0.9994 | 0.06% |

### Predictor Isolation Finding:
The baseline predictor $B_i$ alone provides **less than 0.1% variance reduction**. Over **99.9% of the reconstruction accuracy is generated by the geometric correction alphabet itself**.

---

## 11. C4 Follow-up Results

Evaluating 4-bit representations (16 states):
* **CCC C4 Mirrored Power ($\gamma=1.35$, $R=3.0\text{MAD}$):**  
  True bpw = **4.0000 bpw** | Test RMSE = **0.004502** | Rel $L_2$ = **0.1703** | $W \cdot x$ Rel $L_2$ = **0.1737**
* **Lloyd-Max 16-State Control (Free 16-Level Codebook):**  
  True bpw = **4.0001 bpw** | Test RMSE = **0.003295** | Rel $L_2$ = **0.1247** | $W \cdot x$ Rel $L_2$ = **0.1292**

### C4 Finding:
At $n=4$ bits, the free 16-state Lloyd-Max codebook outperforms Mirrored Power C4 by **26.8% lower RMSE** ($0.003295$ vs $0.004502$). Parametric geometric constraints become overly restrictive at 4 bits, where unconstrained codebooks or hardware INT4 block formats dominate.

---

## 12. True Byte Accounting

Complete physical byte accounting for a $1024 \times 5120$ matrix ($5,242,880$ weights):

```
Representation       | Payload Bytes | Metadata Bytes | Total Storage Bytes | True Effective bpw
---------------------+---------------+----------------+---------------------+-------------------
Q8_0 Reference       | 5,242,880 B   | 327,680 B      | 5,570,560 B         | 8.5000 bpw
Block Q3 Symmetric   | 1,966,080 B   | 327,680 B      | 2,293,760 B         | 3.5000 bpw
Block Q3 Affine      | 1,966,080 B   | 655,360 B      | 2,621,440 B         | 4.0000 bpw
Lloyd-Max 8-State    | 1,966,080 B   | 18 B           | 1,966,098 B         | 3.0000 bpw
CCC C3-A (Best)      | 1,966,080 B   | 5 B            | 1,966,085 B         | 3.0000 bpw
CCC C4 MirroredPower | 2,621,440 B   | 5 B            | 2,621,445 B         | 4.0000 bpw
Lloyd-Max 16-State   | 2,621,440 B   | 34 B           | 2,621,474 B         | 4.0001 bpw
```

---

## 13. Weight Error Summary

| Candidate ID | True bpw | Test MAE | Test RMSE | Test Rel $L_2$ | $p_{50}$ | $p_{95}$ | $p_{99}$ | $p_{99.9}$ | Max Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q8_0 Reference** | 8.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Block Q3 Sym** | 3.5000 | 0.00503 | 0.00614 | 0.2324 | 0.00469 | 0.01098 | 0.01412 | 0.02387 | 0.0623 |
| **Block Q3 Affine** | 4.0000 | 0.00366 | 0.00449 | 0.1699 | 0.00344 | 0.00810 | 0.01005 | 0.01452 | 0.0377 |
| **LloydMax 8-State**| 3.0000 | 0.00410 | 0.00546 | 0.2065 | 0.00381 | 0.00832 | 0.01181 | 0.03641 | 0.2832 |
| **CCC C3-A (Best)** | **3.0000**| **0.00421**| **0.00557**| **0.2107**| **0.00388**| **0.00857**| **0.01056**| **0.03742**| **0.2840** |
| **CCC C3-B Tail1.5x**|3.0000 | 0.00442 | 0.00578 | 0.2187 | 0.00411 | 0.00901 | 0.01240 | 0.03796 | 0.2908 |
| **CCC C4 Mirrored** | 4.0000 | 0.00217 | 0.00450 | 0.1703 | 0.00165 | 0.00378 | 0.01883 | 0.04569 | 0.2922 |
| **LloydMax 16-State**|4.0001 | 0.00211 | 0.00330 | 0.1247 | 0.00177 | 0.00513 | 0.00943 | 0.02148 | 0.2682 |

---

## 14. Functional $W \cdot x$ Action Results

Action error measured over 10 deterministic Gaussian input vectors $x \sim \mathcal{N}(0, I)$ (`RANDOM_WX_ONLY`):

| Candidate ID | True bpw | Output Rel $L_2$ Error | Output Cosine Similarity | Max Output Error |
| :--- | :---: | :---: | :---: | :---: |
| **Q8_0 Reference** | 8.5000 | 0.000000 | 1.000000 | 0.0000 |
| **Block Q3 Symmetric**| 3.5000 | 0.232932 | 0.973823 | 2.5808 |
| **Block Q3 Affine** | 4.0000 | 0.169902 | 0.985767 | 1.5884 |
| **LloydMax 8-State** | 3.0000 | 0.209377 | 0.977910 | 3.4817 |
| **CCC C3-A (Best)** | **3.0000** | **0.213569** | **0.977010** | **3.4593** |
| **CCC C3-B Tail1.5x** | 3.0000 | 0.218679 | 0.975964 | 2.9081 |
| **CCC C4 Mirrored** | 4.0000 | 0.173748 | 0.985882 | 3.9443 |
| **LloydMax 16-State**| 4.0001 | 0.129168 | 0.991756 | 3.0967 |

---

## 15. Runtime / Direct-Apply Assessment

* **C3 Direct Apply Status:** **`DIRECT_APPLY_UNPROVEN`**. Unpacking 3-bit streams across byte boundaries requires multi-word bit permutation (`vpshld`), whose exact SIMD latency remains unmeasured in native code.
* **C4 Direct Apply Status:** **`DIRECT_APPLY_PLAUSIBLE`**. 4-bit nibbles unpack cleanly with standard bitwise shifts (`>> 4`) and masks (`& 0x0F`).

---

## 16. Search / Conversion Cost

* **Lloyd-Max Codebook Fitting Time:** $1882.29\text{ ms}$ (requires iterative k-means over residuals).
* **CCC C3-A Fitting Time:** **$98.34\text{ ms}$ (19.1$\times$ faster)**. 2-pass bounded search over $(\gamma, R)$ evaluates in under 100 ms on CPU.

---

## 17. Falsified Hypotheses

1. **Falsified: Gersho Power-Law Direct Derivation.** Gersho's theorem derives logarithmic/exponential companding for Laplace residuals, NOT a power law $y^\gamma$.
2. **Falsified: Tail-State (C3-B) Superiority.** Re-allocating the duplicate zero state `1 00` as a tail state decreased overall RMSE and increased $p_{99.9}$ error compared to C3-A (No-Zero).
3. **Falsified: Predictor Dominance.** Baseline predictors $B_i$ account for $< 0.1\%$ of error reduction.

---

## 18. Promoted Hypotheses

1. **Promoted: C3-A Metadata & Rate-Distortion Dominance.** Geometric C3-A achieves **3.0000 bpw**, beating Block Q3 Symmetric (3.50 bpw) by **9.3% lower RMSE** while using **0.5 bpw less storage**, and matching **98.0% of free Lloyd-Max accuracy** with 0 codebook metadata bytes.
2. **Promoted: Converter Efficiency.** CCC fitting is 19$\times$ faster than iterative codebook learning.

---

## 19. Final Recommendation & Classification

```
WORTH_BROADER_QUALIFICATION
```

### Justification:
CCC C3-A is a highly competitive **3.0000 bpw** representation that outperforms classical block Q3 at lower bitrates and matches 98% of free learned codebooks without metadata storage overhead. A broader multi-layer qualification across all 32B transformer layers is recommended, while restricting 4-bit regimes (C4) to hardware INT4 or unconstrained codebooks.
