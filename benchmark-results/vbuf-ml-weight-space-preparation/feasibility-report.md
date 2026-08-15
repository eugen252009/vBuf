# Weight-Space Preparation / Functional Reparameterization Feasibility Gate

Status: **COMPLETE / REJECTED**

## Do Not Rediscover

The prior conclusions remain unchanged:

- Arbitrary procedural seed spaces did not cover raw trained weight segments well enough.
- Algebraic invertibility did not provide useful projection.
- Additional mutation rounds did not solve coverage.
- AVX2 materially accelerated arithmetic generation.
- Arithmetic generation can exceed measured Q8 storage supply.

This gate changes only the equivalent model coordinate system around the frozen codecs.

## Baseline

Identity reproduced frozen G2 and mixed M2 within the configured tolerance: `BASELINE_REPRODUCED`.
Source `Qwen3-32B-Q8_0` / `blk.0.attn_k.weight`, W[out,input] `[1024, 5120]`, oracle hash `3a91bb0ff6879abcea4fa12cc6f5acfa85b4fe55a93a5dd7fed8127aaa2611f1`.

## Transformation Families

- `identity` (identity): T=I; model metadata 0 bytes; placement `LAYOUT_ONLY`.
- `statistic_sort` (permutation): x'=P x; W'=W P^T; model metadata 8320 bytes; placement `FOLDABLE_INTO_MODEL_CONVERSION`.
- `shape_cluster` (permutation): x'=P x; W'=W P^T; model metadata 8320 bytes; placement `FOLDABLE_INTO_MODEL_CONVERSION`.
- `codec_aware_order` (permutation): x'=P x; W'=W P^T; model metadata 8320 bytes; placement `FOLDABLE_INTO_MODEL_CONVERSION`.
- `random_permutation_control` (permutation_control): deterministic random P control; model metadata 8320 bytes; placement `FOLDABLE_INTO_MODEL_CONVERSION`.
- `signed_shape_cluster` (signed_permutation): x'=S P x; W'=W P^T S; model metadata 8960 bytes; placement `FOLDABLE_INTO_MODEL_CONVERSION`.
- `power2_column_equalization` (diagonal_scaling): x'=D x; W'=W D^-1; model metadata 2560 bytes; placement `FOLDABLE_INTO_ADJACENT_WEIGHTS`.
- `fp16_column_equalization` (diagonal_scaling): x'=D x; W'=W D^-1; model metadata 10240 bytes; placement `FOLDABLE_INTO_ADJACENT_WEIGHTS`.
- `hadamard8` (fixed_orthogonal): blockwise normalized H; x'=H x; W'=W H^T; model metadata 0 bytes; placement `EXPENSIVE_RUNTIME_ACTIVATION_TRANSFORM`.
- `hadamard16` (fixed_orthogonal): blockwise normalized H; x'=H x; W'=W H^T; model metadata 0 bytes; placement `EXPENSIVE_RUNTIME_ACTIVATION_TRANSFORM`.
- `hadamard32` (fixed_orthogonal): blockwise normalized H; x'=H x; W'=W H^T; model metadata 0 bytes; placement `EXPENSIVE_RUNTIME_ACTIVATION_TRANSFORM`.

## Equivalence

Every candidate was checked as `W' x'` versus `W x` before encoding. Permutation/sign candidates differ only by FP32 reduction order; scaling and Hadamard use strict FP32 numerical tolerances. See `transformation-equivalence.csv`.

## Weight-Space Effect

Distribution and local segment diagnostics are in `weight-statistics.csv` and `segment-statistics.csv`. The decision metric is frozen-codec distance, not visual smoothness or marginal statistics.

## Frozen-Codec Effect

| Transform | Codec | L | Total bpw | RMSE | Validation W*x | Gain |
|---|---|---:|---:|---:|---:|---:|
| identity | G2_CENTER_DENSE | 16 | 1.5001 | 0.019255 | 0.5505 | 1.000x |
| identity | G2_CENTER_DENSE | 32 | 0.7501 | 0.022699 | 0.7489 | 1.000x |
| identity | G2_CENTER_DENSE | 64 | 0.3751 | 0.024526 | 0.8644 | 1.000x |
| identity | M2_AFFINE_XOR_MIXED | 16 | 3.1878 | 0.019743 | 0.6024 | 1.000x |
| identity | M2_AFFINE_XOR_MIXED | 32 | 1.5940 | 0.023047 | 0.7825 | 1.000x |
| identity | M2_AFFINE_XOR_MIXED | 64 | 0.7972 | 0.024734 | 0.8844 | 1.000x |
| signed_shape_cluster | G2_CENTER_DENSE | 16 | 1.5138 | 0.019240 | 0.5383 | 1.023x |
| signed_shape_cluster | G2_CENTER_DENSE | 32 | 0.7638 | 0.022695 | 0.7510 | 0.997x |
| signed_shape_cluster | G2_CENTER_DENSE | 64 | 0.3888 | 0.024518 | 0.8589 | 1.006x |
| signed_shape_cluster | M2_AFFINE_XOR_MIXED | 16 | 3.2015 | 0.019713 | 0.5671 | 1.062x |
| signed_shape_cluster | M2_AFFINE_XOR_MIXED | 32 | 1.6077 | 0.023037 | 0.7700 | 1.016x |
| signed_shape_cluster | M2_AFFINE_XOR_MIXED | 64 | 0.8108 | 0.024729 | 0.8825 | 1.002x |

## Segment-Length Behavior

L16/L32/L64 were evaluated without adaptive fallback. L128 was not reached because L64 did not cross the material-signal gate.

## Real Activation Result

Best Stage-1 transform `signed_shape_cluster` with `M2_AFFINE_XOR_MIXED` L16 changes validation error from 0.6024 to 0.5671 (5.9% reduction).
FUNCTIONAL_TEST status: `NOT_REACHED`. No Stage-1 candidate met >=30% validation reduction; FUNCTIONAL_TEST was not opened.

## Canonical Quantizers

Raw and prepared IQ2_XS/IQ3_XXS/Q3_K/Q4_K controls use the same 17 FIT / 17 VALIDATION split; historical untouched-test values are carried only as provenance and were not reopened. Prepared controls were run for the best permutation, scaling, and orthogonal candidates; see both canonical CSV files.

## Metadata Cost

Permutation IDs use 13 bits/channel (8,320 bytes); signs use 1 bit/channel; power-of-two exponents use 4 bits/channel; FP16 scaling uses 16 bits/channel. Universal Hadamard carries no model metadata. All blobs receive explicit alignment accounting.

## Runtime Cost

Activation-transform cycles, read/write traffic, temporary bytes, and folding classifications are in `runtime-transform-cost.csv`. The bounded Hadamard reference costs 7.4-9.0 cycles/element plus a 40,960-byte activation traversal and is classified expensive. Layout/folding scenarios are separated from explicit runtime fallback; no storage credit assumes a free transform without graph support.

## Graph Audit

`attn_norm-0` feeds Q, K, and V. K-only preparation is local algebra, not a complete graph rewrite. Diagonal scales/signs can plausibly fold into the local norm/QKV fan-out; permutations and Hadamards require coordinated residual-basis propagation or explicit activation work. See `graph-compatibility-audit.md`.

## Central Answer

No. Within permutation, signed permutation, diagonal scaling, and fixed block-Hadamard families, equivalent preparation did not move the frozen procedural codecs materially closer to a useful low-rate representation.

## Falsification Gates

- `WEIGHT_PREPARATION_NO_SIGNAL`
- `PROCEDURAL_REPRESENTATION_STILL_FUNCTIONALLY_REJECTED`

## Final Classifications

- Baseline: `BASELINE_REPRODUCED`
- Preparation signal: `WEIGHT_PREPARATION_NO_SIGNAL`
- Best transformation class: `SIGNED_PERMUTATION_BEST`
- Codec interaction: `NO_CODEC_BENEFIT`
- Runtime placement: `TRANSFORM_FOLDABLE`
- Graph viability: `GRAPH_PROPAGATION_CONSTRAINED`
- Final direction: `WEIGHT_SPACE_PREPARATION_REJECTED`

## Recommendation

`STOP_WEIGHT_PREPARATION_RESEARCH`

## Stop

No frozen codec tuning, residuals, adaptive segmentation, retraining, full-model conversion, production format, vBuf/vBuf-ML change, GPU kernel, or graph propagation implementation was performed.
