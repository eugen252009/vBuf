# Layer-Conditioned LUT / Bucket Prediction Feasibility Gate

Status: **COMPLETE / REJECTED**

## Prior Research Boundary

This gate is not procedural seed reconstruction, numerical residual correction, CCC, a new quantizer, or weight-space reparameterization. The canonical quantized payload is the source of truth; the LUT predicts only code fields and the exact correction carries every mistake.

Prior conclusions preserved unchanged: `CCC_DIRECTION_REJECTED`, `STRUCTURED_BASELINE_REJECTED`, `IMPLICIT_WEIGHT_REPRESENTATION_REJECTED`, `SIMD_TRACEABLE_WEIGHT_MUTATION_REJECTED`, `TRACEABILITY_NOT_FOUND`, `SEGMENT_LENGTH_STILL_KILLS_RATE`, and `WEIGHT_SPACE_PREPARATION_REJECTED`. `SIMD_GENERATION_BEATS_STORAGE_SUPPLY` remains positive.

## Canonical Target

Qwen3-32B-Q8_0.gguf `blk.0.attn_k.weight`, W[out,input] `[1024, 5120]`. Q2_K and IQ2_XS are primary; Q3_K and IQ3_XXS are optional controls. No activation vectors were opened.

## Field Policy

Only the canonical `qs`/code-bearing fields are predicted. Q2_K scales plus d/dmin, IQ2_XS d plus scales, Q3_K scales plus d, and IQ3_XXS d are copied unchanged. The exact serialized payload is restored before any canonical consumer.

## Prediction and Correction

L0 global mode, L1 column-block mode, L2 row mode, and L3 row-class x column-class mode were evaluated. C0 is a deterministic packed match bitmap plus fixed-width exact exceptions. Entropy values are lower bounds; realized rates include bitmap, exception, headers, LUT, class assignments, and untouched metadata.

## Physical Ordering

Bucket ordering is storage-only. It uses stable original-index order inside each column bucket and records the inverse mapping. No logical channel identity or graph semantics change.

## Direct Entropy and Controls

Direct static Huffman accounting, frequency-only prediction, random class assignments, and deterministic random physical permutation are included. The direct entropy control is the critical comparison.

## Exactness

Every selected candidate uses the exact correction decoder and reconstructs the original canonical payload byte-for-byte. No model-quality gate or FUNCTIONAL_TEST data is used.

## Metadata and True Rate

| Format | Canonical bpw | Best LUT bpw | Reduction | LUT bytes | Class bytes | Exceptions | Match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2_K | 2.6250 | 2.8632 | -9.07% | 512 | 3200 | 1289009 | 1.6564% |
| IQ2_XS | 2.3125 | 2.4460 | -5.77% | 8 | 768 | 655181 | 0.0273% |
| Q3_K | 3.4375 | 3.8172 | -11.05% | 1 | 768 | 1953019 | 0.6643% |
| IQ3_XXS | 3.0625 | 3.4332 | -12.11% | 512 | 3200 | 1944210 | 1.1124% |

## Correction Entropy

C0 uses a real packed match bitmap, per-block exception-count header, and fixed-width exact replacements. Bitplane and block-size measurements are in `correction-bitplanes.csv` and `correction-block-results.csv`.

## Direct Entropy Control

Direct static Huffman controls, including untouched canonical metadata, are in `direct-entropy-controls.csv`. The conditional entropy comparison is recorded in `direct_entropy_gap_bits_per_weight`.

## Random Controls

R0 random classes, R1 random physical permutation, and R2 frequency-only controls are in `random-controls.csv`. Random class assignments do not improve the learned grouping.

## LUT-Size Curve

The explicit budget-ceiling rows cover 0 B, 64 B, 256 B, 1 KiB, 4 KiB, and 16 KiB. No early table-size regime produced a material exact true-rate gain; see `lut-size-sweep.csv`.

## Runtime

`RUNTIME_NOT_REACHED`: the exact true-rate promotion threshold was not reached, so no native LUT reconstruction or supply model was authorized.

## Classifications

- Exactness: `CANONICAL_ROUNDTRIP_EXACT`
- Structural signal: `LUT_STRUCTURE_NOT_FOUND`
- Best mechanism: `IDENTITY_BEST`
- Predictor geometry: `GLOBAL_FREQUENCY_ONLY`
- Entropy interaction: `LUT_REDUCES_CONDITIONAL_ENTROPY`
- Real storage: `NO_TRUE_RATE_GAIN`
- LUT state: `LUT_STATE_ACCEPTABLE`
- Runtime: `RUNTIME_NOT_REACHED`
- Final direction: `LUT_PREDICTIVE_REPRESENTATION_REJECTED`

## Central Answer

No. The bounded row/column LUT plus exact correction did not produce a promotion-worthy exact true-rate reduction over canonical storage and direct entropy controls.

## Recommendation

`STOP_LUT_PREDICTION_RESEARCH`

## Stop

No canonical quantizer, vBuf/vBuf-ML format, inference runtime, GPU kernel, or production integration was changed. The LUT direction is closed at this gate.
