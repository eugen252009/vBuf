

## Qualification Update — Qwen3 First-Block Boundary (2026-09-26)

The prior update's statement that no Qwen3 block had been executed is superseded
only for the isolated, non-production qualification path. See
[`qwen3-first-block-qualification.md`](qwen3-first-block-qualification.md) for
source commands, numerical data, and limitations.

- The standalone vBuf-range-backed GGML CPU qualifier passes every captured
  model/reference checkpoint at one position. Follow-up capture of raw attention scores
  shows a strict `1e-5` checkpoint failure at two and three positions even
  though attention context and final block outputs pass. Four, five, and eight
  positions fail both raw attention-score and FFN-down/final-output checks.
- At four positions, FFN down and block output differ by about `8.76e-5` max
  absolute. The investigation classifies this as upstream F16 attention-value
  arithmetic drift amplified by Q8_K activation quantization—not a down-weight,
  layout, or Q6_K×Q8_K kernel defect. The fixed threshold is unchanged; see
  [`qwen3-first-block-qualification.md`](qwen3-first-block-qualification.md)
  for controls, oracle comparisons, and exact checkpoint metrics.
- Block 1, autoregressive KV persistence, ordinary Qwen text generation,
  model-native tool behavior, and Pi remain **NOT TESTED**. The production
  `VbufGenerationSession` still explicitly rejects Qwen3 before inference.
- This isolated block path does not change the architecture ownership boundary
  or qualify the full-model runtime.
