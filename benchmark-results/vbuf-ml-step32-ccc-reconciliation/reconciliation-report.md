# CCC evidence reconciliation

## Git provenance

- Stage-2 commit: `b9316d9b83f4d193aec10d4b5cbead53d0963e79`
- Parallel research commit: `889bd58c674195b9e40f0156e56ca18c1fd00e78`
- Merge base: `33a4d037015b09394b762d5df159dad6d42f8643`
- No-ff merge commit: `d11d16791eb63d5cb6d9ff78c424a4f2a87eb8b0`
- The merge has Stage-2 as first parent and the independently preserved parallel branch as second parent. No rebase or squash was used.

## Agreements

- Learned C3 remains near 0.20–0.21 real W*x error in both independent branches.
- Bounded power geometry remains close to learned C3.
- Packed 3-bit direct W*x without dense reconstruction is feasible.

## Disagreements

- Parallel canonical errors were about 2x larger. This is resolved below as a tensor-orientation bug.
- The original parallel native benchmark claimed C3 was faster; its later fairness audit invalidates that claim.

## Result

**Root cause:** `DISCREPANCY_TENSOR_ORIENTATION`

Parallel hard-gate reshaped flat GGML data as (5120,1024) and transposed it. Correct W[out,in] is flat.reshape(1024,5120). Canonical bytes/dequantization and metric aggregation are identical. The wrong reshape alone reproduces the ~2x canonical errors on the same eight activations.

The parallel branch used:
```python
W_mat = flat.reshape(5120, 1024)
W_eval = W_mat.T
y = X @ W_eval.T
```
The correct GGML-to-NumPy mapping is:
```python
W = flat.reshape(1024, 5120)  # W[out, in]
y = X @ W.T
```

Both produce an apparent `[1024,5120]` matrix, but only the second preserves contiguous 5120-weight GGML rows.

For both branches the intended metric is:
```python
y_ref       = X @ W_ref.T
y_candidate = X @ W_candidate.T
error_i     = y_candidate[i] - y_ref[i]
rel_l2_i    = norm(error_i, 2) / norm(y_ref[i], 2)
reported    = mean(rel_l2_i for i in vectors)
```
The parallel bug changes both `W_ref` and `W_candidate` before these otherwise identical formulas.

## Common eight-vector localization

Candidate | parallel as written | Stage-2 | parallel corrected
---|---:|---:|---:
CCC C3 learned | 0.209488 | 0.210560 | 0.210560
Q2_K | 0.302677 | 0.137272 | 0.137272
Q3_K | 0.154279 | 0.069992 | 0.069992
Q4_K | 0.074032 | 0.034580 | 0.034580

Correcting only the reshape makes the parallel pipeline bit/formula equivalent to Stage-2. Prompt population and split are therefore not the cause.

## Equivalence chain

- Same model and tensor byte range.
- Same flat Q8 FP32 hash.
- Q2_K/Q3_K/Q4_K serialized bytes are byte-identical across `ggml_quantize_chunk` and the parallel direct entry points.
- Canonical dequantized FP32 tensors are bit-identical.
- Both branches average per-vector relative L2.
- The discrepancy appears only when flat tensors are mapped to matrix coordinates.

## Orientation invariant

- GGML metadata: `ne[0]=5120`, `ne[1]=1024`.
- Logical input: 5120.
- Logical output: 1024.
- Flat storage: 1024 contiguous rows, each containing 5120 input weights.
- NumPy: `W.shape == (1024,5120)`.
- Apply: `X.shape == (n,5120)` and `Y = X @ W.T`.

## Hidden-state seam

Both captures are semantically pre-key-projection inputs. The parallel callback identifies the F32 operand paired with `blk.32.attn_k.weight`; Stage-2 names `attn_norm-32` and additionally validates against `Kcur-32` at correlation 0.999990843. The common-vector test uses the known-good Stage-2 seam and removes capture variation.

## IQ controls

Only IQ2_XXS and IQ2_XS in Stage-2 are activation-aware. Their imatrix is mean-square feature importance from the 64 functional-validation vectors. Functional test vectors remain untouched. IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, and IQ4_XS use no imatrix in the pinned API and must not be labeled activation-aware.

## Combined classifications

Geometry: `GEOMETRY_APPROXIMATES_LEARNED`
Numerical rate/quality: `C3_NUMERICALLY_DOMINATED`
Direct compute: `C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED`
Canonical runtime comparison: `PARTIALLY_VALID`
Combined direction: `STOP_C3`

## Corrected numerical interpretation

`C3_NUMERICALLY_DOMINATED`

Independent Stage-2 canonical values are confirmed. IQ2_XS at 2.3125 bpw / 0.1148 dominates C3 at 3.0001 bpw / 0.2016. Runtime is excluded.

C4+0.1% tail remains visible and unchanged: 4.0481 bpw, 0.0556 real W*x, Stage-2 numerical Pareto, direct apply unproven.

## Runtime interpretation

`C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED`

Canonical comparison: `PARTIALLY_VALID`

Packed 3-bit direct W*x feasibility is accepted. The finished fairness audit invalidates the original C3 speed advantage and finds native canonical kernels faster, but its report and raw/structured latency magnitudes disagree; comparative direction is usable while exact ratios are not.

Numerical rate/quality Pareto and future rate/quality/runtime Pareto are separate claims.

## Recommendation

`STOP_C3`

The bounded recommendation is to stop C3 promotion: corrected rate/quality is dominated and the finished CPU fairness audit removes the claimed runtime advantage. Preserve direct-apply feasibility as a technical result. Do not execute more implementation or broad search in this reconciliation step.

## Stop

No vBuf, vBuf-ML, representation, native-kernel, C4-tail, cross-layer, or production-runtime changes were made.
