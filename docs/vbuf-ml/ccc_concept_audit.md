# CCC concept audit — Correction Code (structured baseline + residual alphabet + packed code)

Status: **REJECTED after empirical qualification**. This document supersedes the earlier
CCC concept audit. It records the corrections required by the qualification protocol and
the empirical result that motivated the final classification.

## 1. Concept (refined)

CCC (Correction Code) = a three-part weight representation:

1. **Structured baseline predictor** — a low-cost per-row/per-column/global value `B_ij`
   subtracted before quantization (candidate family B0–B5).
2. **Small residual alphabet** — a fixed 8-state scalar codebook `L` applied to the
   residual `W_ij − B_ij` (candidate family G0–G4).
3. **Packed correction code** — the 3-bit codes stored as the tensor payload; metadata
   (baseline values, alphabet parameters) stored once per tensor/block.

## 2. Corrections applied (audit errata)

| # | Previous claim | Correction |
|---|---|---|
| 1 | Column baseline "free" | `W = C + c` ⇒ `y = C@x + dot(c,x)`: one inner product per token. Cheap (O(cols)), not free. |
| 2 | Row+column baseline direct-apply unclear | `B = g + r + c` ⇒ `y = C@x + (g + r_i)·Σx + dot(c,x)`; verified numerically to 1.4e-15 relative. |
| 3 | Metadata bpw per-sample | Must be accounted over the full tensor: N = 5,242,880 for `blk.0.attn_k.weight` (5120×1024). B3 = 32·1024/N ≈ 0.00625 bpw; B4 = 32·5120/N ≈ 0.03125 bpw; B5 = 32·6145/N ≈ 0.0375 bpw. |
| 4 | Q2_K described as "scales[16] + qs[64], d/dmin first" | Pinned block_q2_K: `scales[16]`, `qs[64]`, then `d`+`dmin` fp16 **at block end** (union layout); value = `d·(sc&0xF)·q − dmin·(sc>>4)`. |
| 5 | Q3_K described as "hmask[32], qs[64], scales[12], d first" | Pinned block_q3_K: `hmask[32]`, `qs[64]`, `scales[12]`, `d` fp16 **at block end**; 16 six-bit scales merged from 12 bytes; hmask bit = element_index/32. |
| 6 | Runtime instruction counts | No instruction counts without measurement. Only classifications: `DIRECT_APPLY_PLAUSIBLE` (algebra verified), `DIRECT_APPLY_UNPROVEN`, `MEASURED_BY_GGML`. |
| 7 | "The low-bit RMSE effect is real" | Replaced: the preliminary low-bit effect is consistent with known companding behavior but remained unverified against canonical controls — now verified, and it does not survive (see §6). |

## 3. Provenance and pins

- Artifact: `research-models/Qwen3-32B-Q8_0.gguf`, SHA256
  `2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169` (verified).
- Tensor: `blk.0.attn_k.weight`, GGUF shape `(5120, 1024)`, Q8_0, 5,242,880 elements,
  payload 5,570,560 B. Rows = ne1 = 1024 (output dim); columns = ne0 = 5120 (input dim).
- llama.cpp pinned at `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` (worktree
  `/tmp/ccc-llama-pinned`) + vBuf patches
  `0001-user-metadata-tensor-source.patch`, `0002-source-neutral-model-source.patch`;
  `llama-quantize` built from that tree.
- Ground truth: Q8_0-dequantized reconstruction (Q8-relative qualification, not BF16/F32).

## 4. Qualification seam

- Script: `scripts/qualify_ccc_structured_baseline.py` (research code only; no vBuf /
  vBuf-ML / canonical-artifact modifications; no production implementation).
- Controls: canonical Q2_K / Q3_K / Q4_0 / Q8_0 produced by pinned llama-quantize and
  dequantized with numpy ports verified against the pinned C `dequantize_row_*`
  (max diff 5e-10). Simplified uniform Q3 marked SIMPLIFIED CONTROL.
- Split: representation-fit split via deterministic splitmix64(row,col), 50/25/25
  fit/val/test; every row contributes FIT samples. Per-block R_b fit on FIT, selected on VAL.
- Invariant: dense vs separated B3/B4/B5 evaluation, max relative diff 1.4e-15.

## 5. First-stage matrix result (111 candidates)

Full table: `benchmark-results/ccc-structured-baseline-qualification/ccc_structured_baseline_qualification.csv`;
report: `ccc_structured_baseline_qualification.md` in the same directory.

Key numbers (TEST split, Q8-relative):

| candidate | true bpw | weight RMSE | p99.9 | max |
|---|---|---|---|---|
| B0_G0_free8 (upper bound) | 3.000 | 0.005496 | 0.0365 | 0.371 |
| B0_G3_tail_g1.25 (best geometry) | 3.000 | 0.005809 | 0.0380 | 0.379 |
| B0_G4_nozero (best family) | 3.000 | 0.005225 | 0.0266 | 0.361 |
| B1_G2_power_g1.25_perblock256 | 3.063 | 0.006216 | 0.0416 | 0.325 |
| canonical Q2_K | 2.625 | 0.007911 | 0.0284 | 0.078 |
| canonical Q3_K | 3.438 | 0.004048 | 0.0151 | 0.052 |
| canonical Q4_0 | 4.500 | 0.002332 | 0.0093 | 0.026 |

Findings:

1. **Baselines contribute zero.** B1–B5 residual RMS ≥ 0.999× raw RMS; row, column and
   row+column predictors are vacuous on this tensor. Context contribution: none.
2. **Geometry does not beat a free codebook.** All G1/G2/G3 are 5.7–15.6% worse in RMSE
   than G0 at identical storage; G4 (no zero state) is the best family member and is a
   plain 3-bit non-uniform scalar quantizer, not a correction code.
3. **Per-block scale is negative.** Block sizes 32–256 all worse than global scale.
4. **Canonical controls dominate.** No family candidate is competitive with Q3_K at
   comparable bpw; the best candidate (3.0 bpw) beats only Q2_K (2.625 bpw) on RMSE at
   14% more bits and with a 4.6x worse maximum error. W*x rel-L2: best family 0.212 vs
   Q3_K 0.155.

## 6. Final classification

**`CCC_REJECTED`** (scope: single tensor, Q8-relative, first-stage matrix).

- Falsified: exploitable row/column structure on `blk.0.attn_k.weight`; geometric
  alphabet advantage; per-block scale gain; low-bit RMSE effect surviving canonical
  controls.
- Surviving but immaterial: the separation algebra is exact (1.4e-15) and the special
  tail state gives a modest p99.9 improvement (0.0444 → 0.0380 at 0.75% occupancy).
- The standing best practice remains canonical GGML K-quants (block-local scales).

## 7. Scope caveats

- Single tensor, single artifact; other tensor classes (FFN gates, embeddings) were not
  in scope and could in principle carry more row/column structure. No follow-up is
  recommended from this qualification, since the family's residual part is dominated by
  canonical quantizers even where baselines might help.
- C4 was not promoted and not evaluated. Direct-apply remains `UNPROVEN`; no production
  kernel was written.