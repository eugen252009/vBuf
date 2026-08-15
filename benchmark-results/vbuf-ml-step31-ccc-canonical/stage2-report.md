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

---

## Post-merge addendum — later independent C3 evidence

This addendum was appended only after independent parallel research commit
`889bd58c674195b9e40f0156e56ca18c1fd00e78` was preserved and merged by
`d11d16791eb63d5cb6d9ff78c424a4f2a87eb8b0`. The original Stage-2 text,
measurements, and classifications above remain historical and unchanged.

### Original Stage-2 classification

Stage-2 commit `b9316d9b83f4d193aec10d4b5cbead53d0963e79` classified C3 from numerical
evidence available at that time. It found C3 dominated in rate × quality,
while C4 plus a fixed 0.1% tail remained numerically Pareto.

### Later parallel evidence

The parallel branch independently found learned/geometric C3 near 0.206/0.211
real W*x error, consistent with Stage-2's approximate 0.202/0.214 operating
point. It also demonstrated correct direct W*x from packed three-bit codes
without a dense reconstructed matrix. C3 direct-apply feasibility is therefore
confirmed by later evidence.

The parallel hard gate reported much larger canonical errors. Reconciliation
found a tensor-orientation bug in that evaluation: flat GGML data was reshaped
as `(5120, 1024)` and then transposed instead of being reshaped directly to
`W[out,in] == (1024, 5120)`. Canonical serialized bytes and dequantized tensors
are identical between branches. Correcting only the reshape reproduces the
Stage-2 canonical errors on eight shared activations.

The parallel branch's initial claim that packed C3 was faster than canonical
kernels was later invalidated by its own fairness audit, which used native GGML
dot kernels. The audit consistently finds canonical kernels faster, although
its report and raw/structured files disagree on exact latency magnitudes.
Comparative direction is retained; exact speed ratios are not promoted.

### Combined reconciliation classification

```text
GEOMETRY_APPROXIMATES_LEARNED
C3_NUMERICALLY_DOMINATED
C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED
CANONICAL_RUNTIME_COMPARISON = PARTIALLY_VALID
STOP_C3
```

Full provenance and localization evidence:
`../vbuf-ml-step32-ccc-reconciliation/reconciliation-report.md`.

---

## Post-merge addendum — independent structured-baseline branch

A second independent research lineage became available at preservation commit
`28588a048a02ee3f4df33530dff8ee789e290114` and was merged without squashing by
`53afb10b26af6a57a8a4ff6bf7c80728c850f39e`. It branched from `33a4d03`, before
Stage-2 and before either reconciliation. Its historical artifacts remain
unchanged.

This branch tested 111 structured-baseline and residual-alphabet candidates on
`blk.0.attn_k.weight`. Unlike the defective parallel hard-gate evaluator, it
uses the correct GGML mapping `flat.reshape(1024, 5120)`.

It independently confirms that contextual prediction is not the source of the
low-bit signal:

- global/tensor anchors are effectively tied with zero;
- row, column, and row+column predictors do not reduce residual RMS;
- predictor metadata generally worsens the result;
- exact separated row+column algebra is implementable but empirically vacuous;
- no structured-baseline kernel was qualified;
- zero-state power alphabets trail free Lloyd-Max, while a no-zero mid-riser
  control is the stronger three-bit scalar form.

The branch uses three Gaussian W*x probes on layer 0, not real hidden states,
and therefore does not supersede Stage-2 layer-32 functional evidence. Its
findings strengthen the combined rejection of the structured/contextual CCC
premise without changing the original Stage-2 classification retroactively.

Combined additional classification:

```text
STRUCTURED_BASELINE_REJECTED
GEOMETRY_APPROXIMATES_LEARNED
C3_NUMERICALLY_DOMINATED
C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED
STOP_C3
```

Full second-lineage provenance:
`../vbuf-ml-step33-ccc-structured-reconciliation/reconciliation-report.md`.

---

## Post-merge addendum — independent geometric/C4 branch

A third independent lineage was preserved at
`da93e439c7836837f13c11341767142d892fad68` and merged without squashing by
`51311440ee955acdb452202f17ab5d01536767e9`. It also branched from `33a4d03`,
prior to Stage-2 and the other branch disclosures.

Its final hard gate uses the correct `W[out,in] = (1024,5120)` orientation,
pinned canonical Q3/Q4/IQ controls, and 69 real F32 inputs captured from the
exact layer-0 `attn_norm-0` key-projection input. Thirty-four vectors were used
for functional validation and 35 remained untouched for functional test.

The layer-0 results independently agree with Stage-2 layer 32:

- free learned C4 real W*x is approximately 0.1530 versus Stage-2 0.1473;
- geometric C4 is worse, approximately 0.1997 versus Stage-2 0.1634;
- Q4_K/IQ4_XS are dramatically better than plain free/geometric C4;
- C3 remains inferior to canonical IQ/K controls.

The branch strengthens the free C4 control and concludes
`C4_FREE_DOMINATES` and `C4_CANONICALLY_DOMINATED`. It does not test the
Stage-2 fixed 0.1% sparse FP16 residual tail: its "tail" experiments are
alphabet-state variants, not sparse indexed exceptions. The Stage-2 sparse
C4-tail point therefore remains visible but independently unreplicated and
`DIRECT_APPLY_UNPROVEN`.

Additional combined classification:

```text
STOP_C3
C4_FREE_DOMINATES
C4_CANONICALLY_DOMINATED
C4_SPARSE_TAIL_SURVIVES_UNREPLICATED
C4_DIRECT_APPLY_UNPROVEN
KEEP_C4_SPARSE_TAIL_AS_EVIDENCE_ONLY
```

Full third-lineage provenance:
`../vbuf-ml-step34-ccc-c4-reconciliation/reconciliation-report.md`.

**FINAL CCC STATUS:** see
[`../../docs/vbuf-ml/ccc_research_conclusion.md`](../../docs/vbuf-ml/ccc_research_conclusion.md).
