# CCC Structured-Baseline Qualification — blk.0.attn_k.weight

**Scope:** First-stage qualification matrix for the CCC (Correction Code) weight-compression concept, executed against the pinned Qwen3-32B Q8_0 artifact with canonical controls produced by pinned llama.cpp. Q8-relative ground truth only; single tensor; no production implementation.

**Result: `CCC_REJECTED`** — no candidate in the first-stage matrix establishes a useful rate-distortion point versus canonical controls; the structured-baseline contribution measures zero on the chosen tensor.

---

## 1. Executive Result

- The row/column/row+column baselines (B1–B5) reduce residual RMS by at most **0.12%** versus no baseline. The structured-baseline component of CCC contributes nothing on `blk.0.attn_k.weight`.
- The best CCC-family candidate, `B0_G4_nozero` (3.000 bpw, RMSE 0.005225), beats canonical Q2_K (2.625 bpw, RMSE 0.007911) on RMSE but at 14.3% more bits and with a **4.6x worse maximum error** (0.361 vs 0.078). It is worse than canonical Q3_K (3.438 bpw, RMSE 0.004048) in every metric.
- The geometric alphabets (G1/G2/G3) are 5.6–15.6% worse in RMSE than the free 8-level Lloyd-Max codebook at identical storage; the special tail state (G3) buys a modest p99.9 improvement (0.0380 vs 0.0444) at 0.75% occupancy but does not change the classification.
- Per-block scale (R_b, blocks 32/64/128/256) provides **no gain** over a global scale (best per-block RMSE 0.006216 ≈ global 0.006212).
- The algebraic row+column separation invariant holds to **1.4e-15** relative — the concept's algebra is correct; it simply does not pay on this tensor.
- W*x probe error: best CCC candidate rel-L2 0.2125, vs canonical Q3_K 0.1545, vs Q4_0 0.0899.

## 2. Audit Corrections Applied

The accompanying `ccc_concept_audit.md` supersedes the earlier concept audit. Applied corrections:

1. Column baseline `W = C + c` is algebraically cheap: `y = C@x + dot(c,x)` (one inner product per token), not "free" but O(cols) vs O(rows·cols). Row+column baseline `B = g + r + c` yields `y = C@x + (g + sum(r))·Σx + dot(c,x)`; verified numerically (see §4).
2. Metadata bpw is accounted over the **full tensor** (N = 5,242,880), not per-sample: B3 = 32·1024/N ≈ 0.00625 bpw, B4 = 32·5120/N ≈ 0.03125 bpw, B5 = 32·(1024+5120+1)/N ≈ 0.0375 bpw.
3. Canonical GGML block structures stated exactly (see §11); Q2_K is *not* "scales[16] + qs[64] + d/dmin at start" in the pinned source — `d`/`dmin` live at the end of the block (union layout); Q3_K carries `hmask[32]` first and `d` last.
4. Runtime claims are classifications only: `UNPROVEN` / `MEASURED_BY_GGML`. No instruction counts.
5. The earlier "low-bit RMSE effect is real" claim is replaced by: the preliminary low-bit effect is consistent with known companding behavior but remained unverified against canonical controls — now it is verified against them, and it does not survive (see §12).

## 3. Exact Source Tensor

From the artifact, not from memory:

| field | value |
|---|---|
| artifact | `research-models/Qwen3-32B-Q8_0.gguf` (34.8 GB, GGUF v3) |
| artifact SHA256 | `2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169` (verified) |
| tensor | `blk.0.attn_k.weight`, GGUF shape `(5120, 1024)` |
| orientation | rows = ne1 = 1024 (output dim), columns = ne0 = 5120 (input dim) |
| type | Q8_0; elements 5,242,880; payload 5,570,560 B; absolute_start 1,659,058,816 |
| ground truth | Q8_0-dequantized reconstruction (Q8-relative qualification) |

llama.cpp pinned at `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` (worktree `/tmp/ccc-llama-pinned`) with vBuf loader patches `0001-user-metadata-tensor-source.patch` + `0002-source-neutral-model-source.patch` applied; `llama-quantize` built from that tree.

## 4. Baseline Algebra

Definitions (fit on FIT split): `g` = global mean; `r_i` = row_mean_i − g; `c_j` = col_mean_j − g.

- B3: `B_ij = r_i` ⇒ `y_i = C@x + r_i·Σx` — one scalar per row.
- B4: `B_ij = c_j` ⇒ `y = C@x + dot(c,x)` — one inner product per token.
- B5: `B_ij = g + r_i + c_j` ⇒ `y = C@x + (g + r_i)·Σx + dot(c,x)` — one scalar per row plus one inner product per token.

All three are O(cols)+O(rows) per token after the quantized matmul, asymptotically negligible. The separation invariant (dense `(B+C)@x` vs the separated form) was verified on 3 probes: **max relative diff 1.4e-15**. The algebra is correct; the cost model is not why CCC fails here.

## 5. Metadata Accounting

Full-tensor bpw (N = 5,242,880): B0 = 0, B1/B2 = 32/N ≈ 6.1e-6, B3 = 32·1024/N ≈ 0.00625, B4 = 32·5120/N ≈ 0.03125, B5 = 32·6145/N ≈ 0.0375. Alphabet metadata: G0 free codebook = 8·32 bits = 256/N ≈ 4.9e-5; G1/G2 gamma = 4 bits (grid code); G3 = 12 bits (gamma + T grid + side); per-block R_b = 16·rows·nb bits (fp16 per block). All candidates' `true_bpw = 3 + metadata_bpw`.

## 6. Predictor Quality (B0–B5)

Test-split residuals vs raw:

| baseline | meta bpw | residual RMS | RMS/raw | p99.9 | max |
|---|---|---|---|---|---|
| raw | 0 | 0.026448 | 1.0000 | 0.0972 | 0.4317 |
| B0 | 0 | 0.026448 | 1.0000 | 0.0972 | 0.4317 |
| B1 (mean) | 6e-6 | 0.026448 | 1.0000 | 0.0972 | 0.4317 |
| B2 (median) | 6e-6 | 0.026448 | 1.0000 | 0.0972 | 0.4317 |
| B3 (row) | 0.00625 | 0.026453 | 1.0002 | 0.0972 | 0.4333 |
| B4 (col) | 0.03125 | 0.026474 | 1.0010 | 0.0973 | 0.4314 |
| B5 (row+col) | 0.0375 | 0.026479 | 1.0012 | 0.0973 | 0.4330 |

Residual entropy estimate is unchanged (~21.76 bits). **No baseline reduces residual energy by more than 0.12%.** The chosen tensor carries no exploitable row/column structure (consistent with RMSNorm-scaled attention projections). Context contribution: zero.

## 7. Free Residual Codebook

G0 = free 8-level Lloyd-Max fit on FIT residuals, evaluated on TEST: RMSE 0.005496 (B0; identical within 1e-6 across all baselines), p99.9 0.0365, max 0.371, clipping 0.2–4.7% depending on alphabet. This is the upper bound of what any 3-bit scalar codebook can do on the residual — and it already loses to canonical Q3_K.

## 8. Geometric C3

| alphabet | RMSE | vs G0 | p99.9 | max | clip% |
|---|---|---|---|---|---|
| G0 free8 | 0.005496 | — | 0.0365 | 0.371 | ~0.2–4.7 |
| G1 linear | 0.006353 | +15.6% | 0.0444 | 0.379 | 4.71 |
| G2 power γ=1.25 (best γ) | 0.006212 | +13.0% | 0.0444 | 0.379 | 4.71 |
| G3 tail γ=1.25 (best) | 0.005809 | +5.7% | 0.0380 | 0.379 | 0.45 |
| G4 no-zero γ=1.35 | 0.005225 | −4.9% | 0.0266 | 0.361 | 0.0 |

The γ grid (1.0–2.0) produces a smooth optimum at γ≈1.25 for G2/G3. All geometric alphabets are within the 25%-of-free-Lloyd-Max survival band, but none improves on it; the best residual quantizer in the CCC family is G4 (no zero state), which is a plain 3-bit non-uniform scalar quantizer — not a "correction code".

## 9. Special Tail State

G3 adds a special state at ±T·R (T ∈ {1.25, 1.5, 2, 3, 4}), chosen from fit residuals; selected T for γ=1.25 on this tensor. Occupancy 0.75%; effect: p99.9 0.0380 vs 0.0444 (G2), clipping 0.45% vs 4.71%. Tail behavior improves but remains worse than canonical Q3_K (p99.9 0.0151). The tail state is a genuine, if minor, improvement — it does not rescue the family.

## 10. Per-Block Scale

Global curve (best G2, γ=1.25) + per-block fp16 R_b (fit on FIT, selected on VAL, q75/q90/q95/q99/max candidates):

| block size | bpw | RMSE | p99.9 | max |
|---|---|---|---|---|
| global | 3.000 | 0.006212 | 0.0444 | 0.379 |
| 256 | 3.063 | 0.006216 | 0.0416 | 0.325 |
| 128 | 3.125 | 0.006371 | 0.0446 | 0.324 |
| 64 | 3.250 | 0.006757 | 0.0496 | 0.336 |
| 32 | 3.500 | 0.007419 | 0.0552 | 0.333 |

Per-block scale adds metadata and adds error — the residual distribution is block-stationary on this tensor.

## 11. Canonical Controls

Produced by pinned llama-quantize from a minimal synthetic GGUF carrying the identical tensor (dequantized to F32), then dequantized with numpy ports verified bit-equivalent to the pinned C `dequantize_row_*` (max diff 5e-10, fp32 rounding). Block layouts (pinned source): Q8_0 = 32 w/34 B (`d` fp16 + int8 qs); Q4_0 = 32 w/18 B (nibbles, ±8); Q2_K = 256 w/84 B (`scales[16]`, `qs[64]`, `d`/`dmin` fp16 at block end; value = d·(sc&0xF)·q − dmin·(sc>>4)); Q3_K = 256 w/110 B (`hmask[32]`, `qs[64]`, `scales[12]`, `d` fp16 at block end; 6-bit scales merged from 12 bytes; hmask bit = index/32). Q8_0 re-quantization roundtrip: 100% of values bit-identical (self-consistency).

## 12. True Rate-Distortion Comparison

| candidate | true bpw | RMSE | rel-L2 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| canonical Q8_0 | 8.500 | 0 | 0 | 0 | 0 | 0 |
| canonical Q4_0 | 4.500 | 0.002332 | 0.0882 | 0.0055 | 0.0093 | 0.0263 |
| canonical Q3_K | 3.438 | 0.004048 | 0.1531 | 0.0098 | 0.0151 | 0.0516 |
| simplified uniform Q3 | 3.500 | 0.005136 | 0.1942 | 0.0152 | 0.0231 | 0.0859 |
| B0_G4_nozero (best CCC) | 3.000 | 0.005225 | 0.1976 | 0.0101 | 0.0266 | 0.3610 |
| B0_G3_tail_g1.25 | 3.000 | 0.005809 | 0.2196 | 0.0123 | 0.0380 | 0.3788 |
| B0_G0_free8 | 3.000 | 0.005496 | 0.2078 | 0.0118 | 0.0365 | 0.3710 |
| B0_G2_power_g1.25 | 3.000 | 0.006212 | 0.2349 | 0.0175 | 0.0444 | 0.3788 |
| B1_G2_power_g1.25_perblock256 | 3.063 | 0.006216 | 0.2350 | 0.0184 | 0.0416 | 0.3246 |
| canonical Q2_K | 2.625 | 0.007911 | 0.2991 | 0.0200 | 0.0284 | 0.0784 |

At 3.0 bpw the best CCC-family candidate beats Q2_K on RMSE (0.0052 vs 0.0079) but loses to Q3_K at 3.44 bpw in every metric and carries a catastrophic max-error tail (0.36 = the codebook edge) that no K-quant exhibits. There is no bpw niche where the CCC family wins.

## 13. W*x Probes

3 deterministic N(0,1) activation vectors (seed 12345), y = W@x; error on y in Q8-relative frame:

| candidate | W*x rel-L2 | cosine | max abs err |
|---|---|---|---|
| canonical Q8_0 | 0 | 1.000000 | 0 |
| canonical Q4_0 | 0.0899 | 0.995956 | 0.677 |
| canonical Q3_K | 0.1545 | 0.988289 | 1.135 |
| simplified uniform Q3 | 0.1978 | 0.980363 | 1.308 |
| B5_G3_tail_g1.25 | 0.2125 | 0.977140 | 1.635 |
| B1_G2_power_g1.25_perblock256 | 0.2315 | 0.972681 | 2.148 |

The best CCC candidate underperforms canonical Q3_K on actual W@x error (0.212 vs 0.155).

## 14. Direct-Apply Assessment

`DIRECT_APPLY_UNPROVEN` for all CCC-family candidates. The row+column separated form is algebraically verified (1.4e-15, §4), so a kernel would be *implementable*, but there is no numerical incentive: even the best candidate would ship a worse quantizer than canonical Q3_K. No production kernel is proposed.

## 15. Ablation

- Baselines off/on (B0 vs B5): ΔRMSE = 0.000007 (0.13%) — null.
- Geometry (G1/G2/G3 vs G0): −5.7%…+15.6% RMSE — null to negative; G3 tail: +0.5% p99.9-class gain only.
- Zero state (G4 vs G2/G3): −16% RMSE — the zero state is expensive; largest single factor within the family, still not competitive.
- Per-block scale vs global: negative.
- Metadata (full-tensor): ≤0.04 bpw for baselines; per-block adds 0.06–0.5 bpw for nothing.

## 16. Falsified Hypotheses

1. **"blk.0.attn_k.weight has exploitable row/column structure"** — falsified: B1–B5 residual RMS ≥ 0.999× raw.
2. **"The geometric alphabet recovers most of a free codebook's performance with less metadata"** — falsified as an advantage: geometry is 5.7–15.6% *worse* at identical storage; the free codebook itself already loses to canonical Q3_K.
3. **"Per-block scale (R_b) yields a rate-distortion gain"** — falsified on this tensor: strictly worse than global scale.
4. **"The low-bit RMSE effect survives canonical controls"** — falsified: at comparable bpw the family is dominated by canonical Q3_K, and its max-error tail (0.36) is unique to the family.

## 17. Surviving Hypotheses

1. The special tail state (G3) genuinely improves p99.9 (0.0444 → 0.0380) at 0.75% occupancy — a real but minor quantizer property, not a CCC-level effect.
2. The row+column separation algebra is exact (1.4e-15) — the concept is *implementable*; the empirical premise fails, not the math.
3. Structure may exist on *other* tensors (e.g., FFN gates, embeddings), which this single-tensor qualification cannot see — however, per protocol the first-stage matrix on the chosen tensor is decisive for this tensor.

## 18. Recommendation

Do not pursue CCC on attention projection tensors. If tensor-level structure is to be exploited, the evidence points to *block-local* structure (which GGML already exploits in Q4_0/Q3_K via per-block scales) rather than row/column structure, and to *canonical K-quants* as the standing best practice. Any follow-up would require first demonstrating residual-energy reduction >5% from a structured baseline on a tensor class where row/column statistics actually vary (none found in this matrix); no such follow-up is recommended from this qualification.

---

## Comparison Table (§26)

| Candidate | baseline | alphabet | scale scope | true bpw | residual RMS | weight RMSE | p99 | p99.9 | max | W*x rel L2 | cosine | metadata bpw | direct apply | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0_G0_free8 | none | Lloyd-Max 8 | global | 3.000 | 0.0265 | 0.005496 | 0.0118 | 0.0365 | 0.371 | — | — | 0.00005 | UNPROVEN | control |
| B0_G1_linear | none | linear C3 | global | 3.000 | 0.0265 | 0.006353 | 0.0175 | 0.0444 | 0.379 | — | — | 0 | UNPROVEN | control |
| B0_G2_power_g1.25 | none | power C3 | global | 3.000 | 0.0265 | 0.006212 | 0.0175 | 0.0444 | 0.379 | — | — | 0 | UNPROVEN | control |
| B0_G3_tail_g1.25 | none | power + tail | global | 3.000 | 0.0265 | 0.005809 | 0.0123 | 0.0380 | 0.379 | — | — | 0 | UNPROVEN | control |
| B5_G3_tail_g1.25 | row+col | power + tail | global | 3.038 | 0.0265 | 0.005816 | 0.0124 | 0.0379 | 0.354 | 0.2125 | 0.9771 | 0.0375 | UNPROVEN | best CCC W*x |
| B0_G4_nozero | none | power, no zero | global | 3.000 | 0.0265 | 0.005225 | 0.0101 | 0.0266 | 0.361 | — | — | 0 | UNPROVEN | best CCC |
| B5_G4_nozero | row+col | power, no zero | global | 3.038 | 0.0265 | 0.005225 | 0.0101 | 0.0268 | 0.363 | — | — | 0.0375 | UNPROVEN | best CCC w/ baseline |
| B1_G2_power_g1.25_perblock256 | mean | power C3 | per-block 256 | 3.063 | 0.0265 | 0.006216 | 0.0184 | 0.0416 | 0.325 | 0.2315 | 0.9727 | 0.0625 | UNPROVEN | best per-block |
| canonical_Q8_0 | — | ggml | block 32 | 8.500 | — | 0 | 0 | 0 | 0 | 0 | 1.000000 | 0.5 | MEASURED_BY_GGML | reference |
| canonical_Q4_0 | — | ggml | block 32 | 4.500 | — | 0.002332 | 0.0055 | 0.0093 | 0.0263 | 0.0899 | 0.995956 | 0.5 | MEASURED_BY_GGML | control |
| canonical_Q3_K | — | ggml | block 256 | 3.438 | — | 0.004048 | 0.0098 | 0.0151 | 0.0516 | 0.1545 | 0.988289 | 0.4375 | MEASURED_BY_GGML | **best <3.7 bpw** |
| canonical_Q2_K | — | ggml | block 256 | 2.625 | — | 0.007911 | 0.0200 | 0.0284 | 0.0784 | — | — | 0.625 | MEASURED_BY_GGML | low-bit control |
| simplified_uniform_Q3 | — | uniform 3 | block 32 | 3.500 | — | 0.005136 | 0.0152 | 0.0231 | 0.0859 | 0.1978 | 0.980363 | 0.5 | UNPROVEN | SIMPLIFIED CONTROL |

## Final Classification

**`CCC_REJECTED`** — the first-stage matrix on `blk.0.attn_k.weight` (Q8-relative, pinned artifact `2c50eb8a…`, pinned llama.cpp `4c1a0af4…`) shows: zero structured-baseline contribution (≤0.12% residual RMS), geometric alphabets no better than a free codebook, per-block scale negative, and no candidate establishing a useful rate-distortion point versus canonical controls. The concept's algebra is verified correct (separation invariant 1.4e-15); its empirical premise is not supported by this qualification.

*Scope caveat: single tensor, single artifact, Q8-relative ground truth, first-stage matrix only. C4 promotion was not applicable. No production code was written.*