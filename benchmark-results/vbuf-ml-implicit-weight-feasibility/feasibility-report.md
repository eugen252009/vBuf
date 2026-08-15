# vBuf-ML Implicit Weight Representation Feasibility Gate

Status: **COMPLETE / REJECTED**

Post-run audit: the runtime classification was conservatively corrected from a raw-byte-rate comparison to a weight-stream comparison. Raw runner output remains preserved; G4 LUT accounting was also corrected to zero because the measured implementation regenerates its basis procedurally.

## Source Qualification

- Model: `Qwen3-32B-Q8_0`
- SHA-256: `2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169`
- Tensor: `blk.0.attn_k.weight`
- W[out,input]: `[1024, 5120]` (5,242,880 values)
- Exact source payload: bytes `1659058816` through `1664629376` (end exclusive)
- Oracle: `Q8_0 reconstructed FP32, not BF16/F32 ground truth`; FP32 hash `3a91bb0ff6879abcea4fa12cc6f5acfa85b4fe55a93a5dd7fed8127aaa2611f1`

## Generator Families

- `G0_HASH_SIGN`: counter-hash Rademacher values plus one FP16 scale.
- `G1_HASH_AFFINE`: counter-hash uniform values plus FP16 scale and bias.
- `G2_CENTER_DENSE`: hash-byte binomial/popcount mapping plus FP16 scale.
- `G3_RECURRENCE`: xorshift32 recurrence plus FP16 scale.
- `G4_TINY_BASIS`: eight shared sign bases, seed-selected signs, plus FP16 scale.

All fixed points use an exhaustive 256-entry seed codebook per segment. Encoding minimizes segment SSE; decoding performs no search.

## Descriptor Formats

G0/G2/G3/G4 use 8 seed bits + 16 FP16 scale bits per segment. G1 adds a 16-bit FP16 bias. Fixed length, tensor identity, segment ordinal, and offset are implicit. Every row includes a 16-byte tensor header, 64-byte final alignment, and shared constants/bases.

## Seed Width And Position

At L=32, widening the exhaustive search from 8 to 16 seed bits changes sampled RMSE from 0.021237 at 0.75 descriptor bpw to 0.018058 at 1.00 bpw. Bounded 24/32-bit searches evaluate 65,536 deterministic seeds and provide no further material gain. Address context is `POSITION_CONTEXT_NEUTRAL` across the six sampled lengths. Encoding ran on an AMD Ryzen 7 5800X CPU; exact counts, times, and stopping conditions are in `seed-width-results.csv` and `raw/search-summary.json`.

## Real Activation Evidence

- Seam: `attn_norm-0` feeding `blk.0.attn_k.weight`.
- Corpus: 69 F32 vectors x 5120; four recorded prompts.
- Split: 34 validation, 35 untouched test.

## Length Curve

| Generator | L | Descriptor bytes | True bpw | RMSE | Real test W*x | Cycles/w |
|---|---:|---:|---:|---:|---:|---:|
| G0_HASH_SIGN | 8 | 1966080 | 3.0001 | 0.015222 | 0.4851 | 1.59 |
| G0_HASH_SIGN | 16 | 983040 | 1.5001 | 0.019377 | 0.6388 | 1.12 |
| G0_HASH_SIGN | 32 | 491520 | 0.7501 | 0.022727 | 0.7863 | 0.86 |
| G0_HASH_SIGN | 64 | 245760 | 0.3751 | 0.024539 | 0.8814 | 0.94 |
| G0_HASH_SIGN | 128 | 122880 | 0.1876 | 0.025479 | 0.9342 | 0.90 |
| G0_HASH_SIGN | 256 | 61440 | 0.0939 | 0.025955 | 0.9666 | 1.03 |
| G1_HASH_AFFINE | 8 | 3276800 | 5.0001 | 0.011159 | 0.2157 | 5.41 |
| G1_HASH_AFFINE | 16 | 1638400 | 2.5001 | 0.018267 | 0.5360 | 5.06 |
| G1_HASH_AFFINE | 32 | 819200 | 1.2501 | 0.022240 | 0.7362 | 4.95 |
| G1_HASH_AFFINE | 64 | 409600 | 0.6251 | 0.024315 | 0.8585 | 4.91 |
| G1_HASH_AFFINE | 128 | 204800 | 0.3126 | 0.025371 | 0.9226 | 5.09 |
| G1_HASH_AFFINE | 256 | 102400 | 0.1564 | 0.025901 | 0.9602 | 5.03 |
| G2_CENTER_DENSE | 8 | 1966080 | 3.0001 | 0.013319 | 0.2790 | 5.89 |
| G2_CENTER_DENSE | 16 | 983040 | 1.5001 | 0.019255 | 0.5518 | 5.63 |
| G2_CENTER_DENSE | 32 | 491520 | 0.7501 | 0.022699 | 0.7486 | 5.52 |
| G2_CENTER_DENSE | 64 | 245760 | 0.3751 | 0.024526 | 0.8641 | 5.74 |
| G2_CENTER_DENSE | 128 | 122880 | 0.1876 | 0.025472 | 0.9299 | 5.57 |
| G2_CENTER_DENSE | 256 | 61440 | 0.0939 | 0.025951 | 0.9639 | 5.48 |
| G3_RECURRENCE | 8 | 1966080 | 3.0001 | 0.013504 | 0.3345 | 5.14 |
| G3_RECURRENCE | 16 | 983040 | 1.5001 | 0.019284 | 0.5940 | 5.08 |
| G3_RECURRENCE | 32 | 491520 | 0.7501 | 0.022708 | 0.7710 | 5.32 |
| G3_RECURRENCE | 64 | 245760 | 0.3751 | 0.024530 | 0.8739 | 5.67 |
| G3_RECURRENCE | 128 | 122880 | 0.1876 | 0.025475 | 0.9350 | 5.96 |
| G3_RECURRENCE | 256 | 61440 | 0.0939 | 0.025953 | 0.9658 | 6.01 |
| G4_TINY_BASIS | 8 | 1966080 | 3.0032 | 0.016348 | 0.3779 | 9.98 |
| G4_TINY_BASIS | 16 | 983040 | 1.5032 | 0.021540 | 0.6610 | 7.91 |
| G4_TINY_BASIS | 32 | 491520 | 0.7532 | 0.024097 | 0.8358 | 6.59 |
| G4_TINY_BASIS | 64 | 245760 | 0.3782 | 0.025283 | 0.9130 | 6.11 |
| G4_TINY_BASIS | 128 | 122880 | 0.1907 | 0.025867 | 0.9603 | 5.80 |
| G4_TINY_BASIS | 256 | 61440 | 0.0970 | 0.026152 | 0.9808 | 5.72 |

## Best Fixed-Length Point

Validation-selected candidate under 2 bpw: `G2_CENTER_DENSE_L16` at 1.5001 bpw. Weight RMSE 0.019255; untouched real W*x mean relative L2 0.5518; generation 2.982 GB/s float-equivalent.

The free-information control permutes the same reconstructed values within each segment. RMSE rises from 0.019255 to 0.031996, and real test W*x rises from 0.5518 to 1.0072. The gap is coordinate-specific information carried by seed selection, but it is not enough for viability.

## Adaptive Result

`NOT_TESTED`. Fixed-length evidence crossed the dimensional-collapse and functional rejection gates, so adaptive segmentation was not authorized.

## Canonical Comparison

| Format | True bpw | RMSE | Real test W*x |
|---|---:|---:|---:|
| IQ2_XS | 2.3125 | 0.008174 | 0.0855 |
| IQ3_XXS | 3.0625 | 0.005708 | 0.0620 |
| Q3_K | 3.4375 | 0.004051 | 0.0412 |
| IQ4_XS | 4.2500 | 0.002063 | 0.0210 |
| Q4_K | 4.5000 | 0.001914 | 0.0196 |

## Compute As Bandwidth

Measured storage supply is 1.409 GB/s, or 1.326 Gweights/s for packed Q8_0. The selected generator supplies 0.745 Gweights/s versus 30.190 Gweights/s required by the measured compute window. Its 2.982 GB/s FP32-equivalent rate is not directly comparable to packed Q8 transport. The memcpy control is 17.593 GB/s; no fused inference speedup is claimed.

## Supply-Horizon Implication

Current median host-local supply horizon (compute/preparation) is 0.0450; the selected candidate's modeled serial horizon is 0.0225. The candidate model includes descriptor transport and measured standalone generation; it does not claim end-to-end overlap or speedup. See `supply-horizon-model.csv`.

## Failure Modes

- Functional quality degrades sharply as descriptor amortization enters the strategically interesting rate region.
- The validation-selected sub-2-bpw point remains catastrophically worse than the canonical low-bit controls on real hidden states.
- Wider bounded seed searches improve sampled distortion too slowly to overcome high-dimensional segment collapse.
- Position-aware address context is distributionally neutral on the bounded screen and does not provide a stable model-specific gain.
- The selected generator supplies fewer weights per second than measured Q8_0 storage and far fewer than the measured compute window consumes.

## Final Classifications

- A. Implicit signal: `IMPLICIT_WEIGHT_SIGNAL_WEAK`
- B. Segment-length behavior: `SEGMENT_LENGTH_KILLS_RATE`
- C. Adaptive segmentation: `NOT_TESTED`
- D. Functional viability: `FUNCTIONALLY_REJECTED`
- E. Runtime reconstruction: `GENERATION_COST_DOMINATES`
- F. Overall direction: `IMPLICIT_WEIGHT_REPRESENTATION_REJECTED`

## Recommendation

`STOP_IMPLICIT_WEIGHT_RESEARCH`

## Nano-Index Analogy

Nano-index length amortizes index metadata against padding/waste. Implicit-weight length amortizes descriptor metadata against nonlinear reconstruction distortion. Here the penalty dominates before useful quality survives, so adaptive length was not used to hide the failed fixed-length curve.

## Stop

No production format, vBuf/vBuf-ML representation, runtime, GPU kernel, residual stream, neural decoder, or second tensor was implemented or evaluated.
