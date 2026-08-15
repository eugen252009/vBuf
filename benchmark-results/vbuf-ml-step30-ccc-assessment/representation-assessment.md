# Qwen3-32B CCC / model representation assessment

Status: **PARTIAL — bounded 32B Stage-1 CCC/baseline assessment**

## 1. Environment and source artifacts

- Host: `debian`
- Source SHA-256: `2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169`
- Oracle: dequantized Q8_0 source, not BF16 ground truth.
- Source geometry: 32 weights / 34 bytes = 8.5 bits/weight.

## 2. Exact tensor inventory tested

49 tensors: layers [0, 1, 16, 32, 48, 62, 63] × roles ['Q', 'K', 'V', 'O', 'Gate', 'Up', 'Down']. See `tensor-inventory.csv`.

## 3. Source quantization qualification

Every selected matrix is Q8_0. No BF16 evidence is mixed into this assessment.

## 4. Methodology

Deterministic coordinate sampling and position-hash split: 70% fit, 15% validation, 15% untouched test. Fixed candidates are reused across roles/layers; no report-on-training metrics.

## 5. True byte accounting

Every row records packed codes, metadata/tables, sparse indexes/residuals, total bytes, and true bpw. Derived lane/coordinate context costs zero bytes but its lookup/modulo compute is named.

## 6. Baseline results

Uniform symmetric and affine Q2–Q8 plus canonical Q8_0 control were run across all 49 tensors. Canonical GGML Q4/Q5/K/IQ comparators remain blocked, not approximated.

## 7. CCC results

Additive C2–C8 was tested with zero, tensor mean/median, row mean/median, column mean/median, and row+column anchors. See aggregate and per-tensor CSV files.

## 8. Position/context contribution

- C2: lane-mod32 RMSE change -0.00033662; true bpw 2.0003 vs 2.0000.
- C3: lane-mod32 RMSE change -7.33715e-06; true bpw 3.0005 vs 3.0000.
- C4: lane-mod32 RMSE change +9.11878e-05; true bpw 4.0011 vs 4.0000.
- C5: lane-mod32 RMSE change +0.000121353; true bpw 5.0021 vs 5.0001.
- C6: lane-mod32 RMSE change +0.000144296; true bpw 6.0042 vs 6.0001.
- C7: lane-mod32 RMSE change +0.000210774; true bpw 7.0085 vs 7.0003.
- C8: lane-mod32 RMSE change +0.000269703; true bpw 8.0169 vs 8.0005.

## 9. Statistical-scale results

Mean/median anchors were executed. The full requested dispersion/range/nonlinear Cartesian family was intentionally not run and remains `NOT_TESTED`.

## 10. Codebook results

Lloyd-Max correction tables were fitted from fit positions only. Row/column and lane-conditioned tables include every persisted float.

## 11. Outlier/hybrid results

C2/C3/C4 with FP16 sparse outliers and 32-bit indexes were measured on layer-32 K for all requested fractions.

## 12. Structured representation results

Prior 0.6B structured evidence is not relabeled as 32B evidence. 32B structured families remain `NOT_TESTED` in this bounded Stage 1.

## 13. Cross-tensor results

Not run; promotion gate was not reached in this pass.

## 14. Functional W*x results

Layer-32 K C2/C3/C4/C6/C8 random-Gaussian W*x is recorded. The path reconstructs dense W and is marked `DIRECT_APPLY = NO`. Real hidden states are unavailable.

## 15. Pareto fronts

See `pareto.csv`; the frontier uses true bpw and untouched-test weight RMSE. Runtime is not ranked because no native direct kernel exists.

## 16. Promoted candidates

Only candidates selected by validation and represented in functional W*x evidence are provisional; none is promoted to production.

## 17. Rejected hypotheses

Position context is rejected where its matched RMSE change is non-negative after table overhead. Ultra-low-bit candidates remain rejected if W*x error is high.

## 18. Remaining uncertainty

Canonical GGML comparators, full statistical curves, vectors, structured matrices, cross-tensor sharing, native direct apply, and real hidden states remain unresolved.

## 19. Recommended next experiment

Integrate canonical llama.cpp quantization controls and capture real hidden states for layer-32 K before expanding Stage 2.

## Explicit answers

- Does CCC provide a measurable advantage? **Yes versus the simplified uniform/affine controls at 2–6 bpw; not established versus canonical GGML families.**
- At which budgets? **Largest weight-RMSE advantage at C2–C4, smaller at C6, and a loss to affine at C8.**
- Anchor or structural context? **Zero and tensor-mean anchors are effectively tied; the gain is primarily learned nonlinear correction levels, not the anchor.**
- Does position help? **Lane-mod32 modestly helps C2, is negligible at C3, and hurts C4–C8 after table accounting.**
- Row/column statistics? **They generally worsen test RMSE despite their metadata cost.**
- Is CCC mainly useful at C2/C3? **Most promising there relative to simple controls, but representative W*x error remains 0.358/0.203 and canonical Q2/Q3 controls are missing.**
- Sparse outliers? **Yes: they reduce C2/C3/C4 W*x error monotonically, but 32-bit indexes quickly consume the storage gain.**
- Best near 2 bpw: **CCC additive / tensor_mean**.
- Best near 3 bpw: **CCC additive / tensor_mean**.
- Best near 4 bpw: **CCC additive / tensor_median**.
- Best near 6 bpw: **CCC additive / zero**.
- Best near 8 bpw: **Q8_0 source control / **.
- Hidden-state qualification: **none until a corpus is captured.**

## Compact table

Candidate | True bpw | Weight RMSE | p99 | MaxErr | W*x error | Direct apply | Status
---|---:|---:|---:|---:|---:|---|---
uniform_symmetric   | 2.0000 | 0.021874 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive zero none | 2.0000 | 0.00859942 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive tensor_median none | 2.0000 | 0.00859941 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive tensor_mean column lane mod 32 | 2.0003 | 0.0082628 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive zero none | 3.0000 | 0.00468419 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive tensor_mean column lane mod 32 | 3.0005 | 0.00467686 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive zero none | 4.0000 | 0.00290174 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive tensor_median none | 4.0000 | 0.00290173 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive zero none | 5.0001 | 0.00196119 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
uniform_symmetric   | 6.0000 | 0.00171236 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
affine_asymmetric   | 6.0000 | 0.00150527 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
CCC additive zero none | 6.0001 | 0.00139402 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
uniform_symmetric   | 7.0000 | 0.000862093 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
affine_asymmetric   | 7.0000 | 0.000772374 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
uniform_symmetric   | 8.0000 | 0.000454527 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
affine_asymmetric   | 8.0000 | 0.000419541 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
Q8_0 source control   | 8.5000 | 0 | see CSV | see CSV | representative only | NO | TESTED_INTERESTING
