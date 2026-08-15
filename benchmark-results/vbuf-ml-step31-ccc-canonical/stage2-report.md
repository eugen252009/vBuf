# CCC Stage-2 — canonical quantizers and real hidden states

## 1. Executive Result

Correction alphabet: **LEARNED_ALPHABET_LOW_BIT_NICHE**
Geometry: **GEOMETRY_APPROXIMATES_LEARNED_LEVELS**
Overall: **CCC_LOW_BIT_NICHE_INTERESTING**

## 2. Stage-1 Carry-Forward

Stage-1 Q8-relative learned-alphabet signal is preserved without stronger reinterpretation. This stage freezes C2/C3/C4, optional lane continuity, C4 0.1% tail, and bounded C3/C4 power geometry.

## 3. Exact Source Qualification

`blk.32.attn_k.weight`: [1024, 5120], 5242880 weights, Q8_0, 5570560 bytes, byte range [18238388864, 18243959424), 8.500 bpw.

## 4. Hidden-State Capture Qualification

Pinned llama.cpp `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` captured 128 F32 vectors at `attn_norm-32`, the exact `cur` passed to `build_qkv` and `layer.wk`. Native `Kcur-32` correlation: 0.999990843; relative difference 0.004279 (native quantized dot path versus FP32 dequantized oracle).

## 5. Canonical Quantizer Qualification

Canonical controls use `ggml_quantize_chunk` and `ggml_type_traits.to_float`. Exact block geometry and rates are in `canonical-controls.csv`; IQ2_XXS/XS use only functional-validation activation importance.

## 6. CCC Candidate Freeze

C2 learned/no-position and lane-mod32; C3 learned/no-position and optional lane; C4 learned/no-position; C4+0.1% FP16 tail; bounded C3/C4 power alphabets.

## 7. Geometry vs Learned Levels

Actual levels are in `level-sets.csv`.
- C3: gamma 1.25, level RMS distance 0.032742, RMSE penalty 1.018x, real W*x penalty 1.062x.
- C4: gamma 1.25, level RMS distance 0.133950, RMSE penalty 1.008x, real W*x penalty 1.109x.

## 8. Weight-Domain Results

See the central matrix below and `stage2-results.csv`; all metrics use the untouched position-hash weight test split.

## 9. Random W*x Results

64 deterministic Gaussian vectors are secondary controls.

## 10. Real Hidden-State W*x Results

64 untouched vectors from the final two prompts are the primary metric; per-vector rows are in `hidden-state-wx.csv`.

## 11. Random vs Real Activation Comparison

See `random-vs-real-wx.csv`; no assumption that real activations help CCC is made.

## 12. Canonical Rate/Distortion Comparison

Each CCC row names nearest lower/higher canonical rates. Activation-aware IQ controls are explicitly labeled.

## 13. Pareto Frontier

`pareto.csv` uses true bpw and real hidden-state mean relative L2.

## 14. Tail Behavior

Weight p99.9/max and real-output worst-vector/max-output errors are reported.

## 15. Direct-Apply Status

Canonical GGML formats have native CPU kernels. CCC remains `DIRECT_APPLY_UNPROVEN`; all CCC scoring reconstructs dense matrices offline.

## 16. Falsified Hypotheses

Gate A: learned C3/C4 are dominated by lower-rate canonical controls; C2 has no equal-or-lower canonical control but is far worse than IQ2_XXS at +0.0624 bpw. Gate B passes numerically for power C3/C4, but both geometric operating points are dominated. Gate C rejects broad transfer of Stage-1 weight gains: real W*x favors canonical controls. Gate E leaves C4+0.1% tail as a Pareto point but with unproven direct apply.

## 17. Surviving Hypotheses

Outcome 6 is the closest classification: C2 survives only as an extreme-rate niche below the lowest tested canonical rate. Separately, fixed C4+0.1% tail remains Pareto. Power geometry approximates learned levels but does not create a competitive C3/C4 point. None is production-ready.

## 18. Recommendation

STOP after evidence. Do not implement a native CCC kernel unless the overall classification explicitly justifies that separate gate.

## Required summary matrix

Candidate | true bpw | weight RMSE | p99.9 | random W*x rel L2 | real W*x rel L2 | real cosine | geometry/free | canonical/experimental | Pareto | direct apply | verdict
---|---:|---:|---:|---:|---:|---:|---|---|---|---|---
CCC C2 learned | 2.0001 | 0.00841038 | 0.0507099 | 0.361577 | 0.33865 | 0.960325 | FREE | CCC | PARETO | DIRECT_APPLY_UNPROVEN | SURVIVES
CCC C2 learned lane-mod32 | 2.0008 | 0.00847885 | 0.0514738 | 0.36348 | 0.344102 | 0.959141 | FREE+POSITION | CCC | DOMINATED | DIRECT_APPLY_UNPROVEN | DOMINATED
IQ2_XXS ACTIVATION_AWARE | 2.0625 | 0.00852156 | 0.0348726 | 0.36681 | 0.130084 | 0.993033 | canonical | canonical | PARETO | YES — native GGML CPU kernel exists | SURVIVES
IQ2_XS ACTIVATION_AWARE | 2.3125 | 0.00742592 | 0.0309126 | 0.320479 | 0.114825 | 0.995003 | canonical | canonical | PARETO | YES — native GGML CPU kernel exists | SURVIVES
IQ2_S | 2.5625 | 0.00628373 | 0.0259545 | 0.268831 | 0.127439 | 0.993032 | canonical | canonical | DOMINATED | YES — native GGML CPU kernel exists | DOMINATED
Q2_K | 2.6250 | 0.00698122 | 0.0236606 | 0.299545 | 0.133884 | 0.990989 | canonical | canonical | DOMINATED | YES — native GGML CPU kernel exists | DOMINATED
CCC C3 learned | 3.0001 | 0.00475977 | 0.0311749 | 0.20326 | 0.201591 | 0.98434 | FREE | CCC | DOMINATED | DIRECT_APPLY_UNPROVEN | DOMINATED
CCC C3 power gamma=1.25 | 3.0001 | 0.00484771 | 0.0341245 | 0.207766 | 0.214043 | 0.982797 | POWER | CCC | DOMINATED | DIRECT_APPLY_UNPROVEN | DOMINATED
CCC C3 learned lane-mod32 | 3.0016 | 0.00479475 | 0.0334734 | 0.203835 | 0.210978 | 0.983387 | FREE+POSITION | CCC | DOMINATED | DIRECT_APPLY_UNPROVEN | DOMINATED
IQ3_XXS | 3.0625 | 0.00503327 | 0.0248508 | 0.216183 | 0.100572 | 0.995366 | canonical | canonical | PARETO | YES — native GGML CPU kernel exists | SURVIVES
Q3_K | 3.4375 | 0.00357228 | 0.0120472 | 0.153849 | 0.0682399 | 0.997665 | canonical | canonical | PARETO | YES — native GGML CPU kernel exists | SURVIVES
IQ3_S | 3.4375 | 0.00392403 | 0.0190679 | 0.168358 | 0.0814913 | 0.997328 | canonical | canonical | DOMINATED | YES — native GGML CPU kernel exists | DOMINATED
CCC C4 learned | 4.0001 | 0.00284862 | 0.0194152 | 0.120838 | 0.147348 | 0.991475 | FREE | CCC | DOMINATED | DIRECT_APPLY_UNPROVEN | DOMINATED
CCC C4 power gamma=1.25 | 4.0001 | 0.00287258 | 0.0250713 | 0.121683 | 0.16344 | 0.989764 | POWER | CCC | DOMINATED | DIRECT_APPLY_UNPROVEN | DOMINATED
CCC C4 learned + 0.1% FP16 tail | 4.0481 | 0.00246678 | 0.0113382 | 0.105778 | 0.0556431 | 0.998552 | FREE+SPARSE | CCC | PARETO | DIRECT_APPLY_UNPROVEN | SURVIVES
IQ4_XS | 4.2500 | 0.00181772 | 0.00568448 | 0.0781974 | 0.0356543 | 0.999364 | canonical | canonical | PARETO | YES — native GGML CPU kernel exists | SURVIVES
Q4_0 | 4.5000 | 0.00206119 | 0.00684357 | 0.088405 | 0.039559 | 0.999216 | canonical | canonical | DOMINATED | YES — native GGML CPU kernel exists | DOMINATED
Q4_K | 4.5000 | 0.00168868 | 0.00512042 | 0.0725485 | 0.033321 | 0.999449 | canonical | canonical | PARETO | YES — native GGML CPU kernel exists | SURVIVES
IQ4_NL | 4.5000 | 0.00180297 | 0.0055714 | 0.0772458 | 0.0355166 | 0.99937 | canonical | canonical | DOMINATED | YES — native GGML CPU kernel exists | DOMINATED
Q8_0 oracle | 8.5000 | 0 | 0 | 0 | 0 | 1 | canonical | canonical | PARETO | YES — native GGML kernel exists | SURVIVES

## Final classifications

Correction alphabet: `LEARNED_ALPHABET_LOW_BIT_NICHE`
Geometry: `GEOMETRY_APPROXIMATES_LEARNED_LEVELS`
Overall CCC direction: `CCC_LOW_BIT_NICHE_INTERESTING`
