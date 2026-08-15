# SIMD-Native Traceable Weight Mutation Feasibility Gate

Status: **COMPLETE / REJECTED**

Post-run audit corrected the mechanism interpretation: round 1 provides the material coverage gain, while rounds 2–4 add negligible quality for extra cost. Runtime and supply comparisons therefore use the fixed one-round proxy for the mixed-round representation. The raw runner output remains preserved.

## Mutation Design

- `M0_LCG_JUMP`: jump-ahead x=A_n*seed+B_n; one vector mul+add Mapping: signed high16 to FP32 Traceability: `PARTIALLY_INVERTIBLE`.
- `M1_LANE_AFFINE`: lane=seed+position+i*C; each round x=x*A+B+i*D Mapping: signed high16 to FP32 Traceability: `PARTIALLY_INVERTIBLE`.
- `M2_AFFINE_XOR`: lane affine; each round affine, xorshift-right, odd multiply Mapping: signed high16 to FP32 Traceability: `PARTIALLY_INVERTIBLE`.
- `M3_FLOAT_AFFINE`: lane seed/position affine; each round FP32 FMA Mapping: direct FP32 then scale Traceability: `CHEAPLY_PROJECTABLE`.

All integer operations are modulo 2^32. Lane state stays in AVX2 registers through each fixed mutation round; standalone generation writes one final output only. No runtime convergence loop is used.

## Reverse / Trace Mechanism

The encoder forms two scale hypotheses, maps eight target lanes to approximate high-16 integer states (or float states), fills two bounded low-bit alternatives, and reverses the algebra to 32 candidate seeds. Equal 32-seed and larger 256-seed deterministic random controls are measured. Integer state transforms are exactly invertible; numerical high-bit mapping makes target projection partial. For selected M2, projected/equal-random RMSE ratio is 0.9996; the 256-seed random control is 5.1% better. Projection does not locate a better region.

## Descriptor

Fixed family/round rows store 32 seed bits + 16 FP16 scale bits per segment. Family and round live in the 32-byte tensor header. Mixed-round rows add 3 round bits per segment. Position and lane are derived and cost zero bits. Shared arithmetic constants cost 128 bytes; M0 additionally counts its 2,560-byte jump table. Final size includes 64-byte alignment.

## Source And Activations

- Source: `Qwen3-32B-Q8_0`, SHA-256 `2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169`.
- Tensor: `blk.0.attn_k.weight`, W[out,input] `[1024, 5120]`, 5,242,880 weights.
- Payload byte range: `[1659058816, 1664629376]`; Q8-reconstructed FP32 oracle hash `3a91bb0ff6879abcea4fa12cc6f5acfa85b4fe55a93a5dd7fed8127aaa2611f1`.
- Real seam: `attn_norm-0`; 34 FUNCTIONAL_VALIDATION and 35 untouched FUNCTIONAL_TEST vectors.

## Mutation Depth Curve

| Family | Rounds | L | bpw | RMSE | Real W*x | AVX2 cycles/w |
|---|---:|---:|---:|---:|---:|---:|
| M2_AFFINE_XOR | 0 | 8 | 6.0003 | 0.020868 | 0.6373 | 0.861 |
| M2_AFFINE_XOR | 1 | 8 | 6.0003 | 0.016783 | 0.4552 | 1.760 |
| M2_AFFINE_XOR | 2 | 8 | 6.0003 | 0.016690 | 0.4529 | 2.394 |
| M2_AFFINE_XOR | 3 | 8 | 6.0003 | 0.016707 | 0.4497 | 3.171 |
| M2_AFFINE_XOR | 4 | 8 | 6.0003 | 0.016713 | 0.4473 | 3.821 |
| M2_AFFINE_XOR | 0 | 16 | 3.0003 | 0.023699 | 0.8201 | 0.768 |
| M2_AFFINE_XOR | 1 | 16 | 3.0003 | 0.021625 | 0.7020 | 1.612 |
| M2_AFFINE_XOR | 2 | 16 | 3.0003 | 0.021614 | 0.7019 | 2.222 |
| M2_AFFINE_XOR | 3 | 16 | 3.0003 | 0.021608 | 0.7023 | 2.974 |
| M2_AFFINE_XOR | 4 | 16 | 3.0003 | 0.021611 | 0.7008 | 3.714 |
| M2_AFFINE_XOR | 0 | 32 | 1.5003 | 0.025022 | 0.9045 | 0.774 |
| M2_AFFINE_XOR | 1 | 32 | 1.5003 | 0.024049 | 0.8442 | 1.563 |
| M2_AFFINE_XOR | 2 | 32 | 1.5003 | 0.024048 | 0.8434 | 2.209 |
| M2_AFFINE_XOR | 3 | 32 | 1.5003 | 0.024049 | 0.8443 | 2.981 |
| M2_AFFINE_XOR | 4 | 32 | 1.5003 | 0.024052 | 0.8422 | 3.696 |
| M2_AFFINE_XOR | 0 | 64 | 0.7503 | 0.025706 | 0.9507 | 0.782 |
| M2_AFFINE_XOR | 1 | 64 | 0.7503 | 0.025250 | 0.9194 | 1.529 |
| M2_AFFINE_XOR | 2 | 64 | 0.7503 | 0.025251 | 0.9183 | 2.239 |
| M2_AFFINE_XOR | 3 | 64 | 0.7503 | 0.025253 | 0.9187 | 2.928 |
| M2_AFFINE_XOR | 4 | 64 | 0.7503 | 0.025251 | 0.9193 | 3.723 |

## Segment-Length Curve

The complete tensor was encoded for L=8/16/32/64 at every round for the validation-selected family. L=128 was not reached because the L<=64 evidence crossed the quality/rate falsification gate. See `fixed-length-results.csv`.

## Runtime

One-round fixed runtime proxy: scalar 7.241 cycles/weight, 0.580 Gweights/s.
One-round fixed runtime proxy: AVX2 unrolled 1.563 cycles/weight, 2.687 Gweights/s, 4.63x scalar.
Basic and four-state results, register-pressure interpretation, and assembly evidence are in `scalar-vs-avx2.csv` and `disassembly-notes.md`. The conservative whole-binary scan finds YMM stack traffic, so the unrolled implementation is not claimed spill-free. Dynamic instructions/weight were unavailable; cycles and static instruction evidence are reported.

## Generate And Dot

Direct generate-and-dot was reached. For the one-round proxy, buffer+dot is 2.880 cycles/weight and direct generate-and-dot is 2.506 cycles/weight with zero generated-weight memory writes. This is a bounded microbenchmark, not inference.

## Functional Quality

The best validation-selected point at or below IQ2_XS rate is `M2_AFFINE_XOR_MIXED_L32`: 1.5940 bpw, RMSE 0.023047, untouched real W*x 0.7887. Previous G2 is 1.5001 bpw / 0.5518 real W*x. The one-round SIMD proxy cuts standalone generation from the previous 5.63 to 1.56 cycles/weight, but its fixed-round real W*x is 0.8442 and the mixed representation is still worse than previous G2. This is a speed-only win.

## Canonical Comparison

| Candidate | bpw | RMSE | Real W*x |
|---|---:|---:|---:|
| IQ2_XS | 2.3125 | 0.008174 | 0.0855 |
| IQ3_XXS | 3.0625 | 0.005708 | 0.0620 |
| Q3_K | 3.4375 | 0.004051 | 0.0412 |
| Q4_K | 4.5000 | 0.001914 | 0.0196 |

## Supply-Rate Comparison

- `REQUIRED_RATE_GWEIGHTS_S`: 30.190
- `AVAILABLE_STORAGE_RATE_GWEIGHTS_S`: 1.326
- `AVAILABLE_GENERATION_RATE_GWEIGHTS_S`: 2.687
- Storage `SUPPLY_COVERAGE`: 0.0439
- Generation `SUPPLY_COVERAGE`: 0.0890

The prior values 0.0450 and 0.0225 were coverage ratios, not lookahead horizons. This report retires the overloaded name and uses explicit rates and coverage.

## Falsification Gates

- `SIMD_MECHANISM_FAST_BUT_UNUSEFUL`
- `MUTATION_DEPTH_NOT_USEFUL_BEYOND_ONE_ROUND`
- `TRACEABLE_MUTATION_NOT_SUPPORTED`
- `SIMD_SEGMENT_LENGTH_STILL_KILLS_RATE`
- `SIMD_MUTATION_FUNCTIONALLY_REJECTED`

## Important Answer

No. A few SIMD affine/xor/shift rounds are fast and algebraically traceable, but they do not create a weight space with functional coverage sufficient to replace meaningful transported weight information at competitive rate.

## Final Classifications

- SIMD generation: `SIMD_GENERATION_BEATS_STORAGE_SUPPLY`
- Mutation depth: `ONE_ROUND_SUFFICIENT`
- Traceability: `TRACEABILITY_NOT_FOUND`
- Representation quality: `FUNCTIONALLY_REJECTED`
- Segment behavior: `SEGMENT_LENGTH_STILL_KILLS_RATE`
- Combined direction: `SIMD_TRACEABLE_WEIGHT_MUTATION_REJECTED`

## Recommendation

`STOP_SIMD_MUTATION_RESEARCH`

## Stop

No vBuf/vBuf-ML changes, production codec, adaptive segmentation, residuals, GPU kernel, learned decoder, recursive tree, or full-model conversion were created.
