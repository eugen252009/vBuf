# Qwen3 Multi-Block / Full-Forward Qualification

## Decision and scope

Qualification advanced through block 7 (eight blocks total) for 4- and
8-position CPU prefill cases, and through block 1 for 1, 2, 4, and 8 positions.
All observed block-boundary tensors were finite and the isolated blocks execute
with the expected Qwen3 graph/tensor shapes. Numerical error, however, grows
substantially across blocks. The 8-block run reaches max absolute hidden-state
error `0.365293503` (RMS `0.0118544` at 4 positions; `0.0147593` at 8) while
cosine remains near 1. This is a qualification stop, not a pass: the requested
16-, 20-, and 40-block checkpoints and all logits/generation gates remain
untested. Production Qwen3 remains disabled.

No tolerance was changed. The fixed max-absolute `1e-5` field is retained as a
historical comparison in checkpoint logs, not used as the sole structural or
numerical qualification criterion. Numerical differences are measured and
reported rather than clamped, skipped, or treated as a payload/layout failure.

## Harness and reference

- Isolated runner: `integrations/ggml/tools/qwen3_block_qualification.cpp`.
  It obtains validated tensor ranges through the vBuf-ML materializer, then
  executes the selected dense blocks in GGML CPU. It is not routed through
  `VbufGenerationSession`.
- Reference: `integrations/ggml/qualification/qwen3_llama_reference_dump.cpp`,
  llama.cpp/GGML revision `a97123e497968f3440264c0464a7adc7c999c027`.
- vBuf GGML revision: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, with the
  qualification build's LLAMAFILE path disabled unless explicitly noted.
- Artifact: Qwen3-14B Q4_K_M GGUF SHA-256
  `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6`; matching
  vBuf payload SHA-256
  `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- CPU thread count: 8. Reference checkpoint directories and logs are under
  `/tmp/qwen3-multiblock-*`, `/tmp/qwen3-fullref-{4,8}-t8`, and
  `/tmp/qwen3-{layer1,layer6,layer7}-refinput-*`.

The 4- and 8-position reference dumps contain all 40 block checkpoints, but the
vBuf execution was intentionally stopped at eight blocks after the observed
error growth. Capturing all reference checkpoints is not a full-forward vBuf
qualification.

## Phase 1 — block 1

Positions 1 and 2 have bit-identical block-0 and block-1 boundary outputs in
the normal vBuf CPU configuration. At 4 positions, normal sequential execution
has block-0 max error `8.75555e-5` and block-1 max error `9.46999e-3` (RMS
`5.16313e-4`). At 8 positions, block 1 has max error `9.46999e-3` (RMS
`6.72621e-4`). The 4-position block-1 output using the *reference* block-0
output as its input differs by at most `1.90735e-6`; this separates block-1's
large sequential error from an intrinsic block-1 structure/payload defect.

An 8-position reference-input control exposed a separate build-path effect:
with LLAMAFILE disabled, layer 1 raw attention scores differ by up to
`6.10352e-5`, and the block output by `1.60353e-3`. With an isolated pinned
vBuf GGML LLAMAFILE-enabled build, the same reference-input layer-1 control
matches the reference exactly at raw scores, probabilities, attention context,
SwiGLU, down projection, and block output. This localizes that control's
numerical difference to the CPU attention reduction path; the LLAMAFILE result
is diagnostic only and does not change the default qualification build or
production behavior.

## Phase 2 — sequential boundary ladder, 4 and 8 positions

Normal sequential vBuf execution, max absolute / RMS hidden-state error at each
completed block boundary:

| Last block | 4 positions max abs | 4 positions RMS | 8 positions max abs | 8 positions RMS |
|---:|---:|---:|---:|---:|
| 0 | `8.75555e-5` | `1.15195e-5` | `8.75555e-5` | `8.14567e-6` |
| 1 | `9.46999e-3` | `5.16313e-4` | `9.46999e-3` | `6.72621e-4` |
| 3 | `2.47765e-2` | `1.48341e-3` | `4.17004e-2` | `2.52595e-3` |
| 5 | `4.12312e-2` | `5.33354e-3` | `8.46786e-2` | `7.75405e-3` |
| 6 | `1.70601e-1` | `7.59711e-3` | `1.70601e-1` | `9.99122e-3` |
| 7 | `3.65294e-1` | `1.18544e-2` | `3.65294e-1` | `1.47593e-2` |

Every reported boundary was finite. Cosine at block 7 remained `0.999999992`
(4 positions) and `0.999999977` (8 positions), but that does not override the
absolute/RMS error growth or qualify logits/token parity. The first block-0
perturbation is the already classified one-F32-ULP attention value-projection
difference crossing Q8_K activation quantization thresholds; see
[`qwen3-first-block-qualification.md`](qwen3-first-block-qualification.md).
The sequential errors spread to more positions in later blocks and grow through
attention/FFN computations. They are not evidence of incorrect tensor shapes or
non-finite execution, but they are too large to pass off as numerical parity.

## Phase 3 — isolate the later-block growth

Reference-input controls replace a block's incoming state with the independent
llama.cpp output for the preceding block, then execute just that vBuf block:

- Layer 6, at both 4 and 8 positions, finishes within `1.90735e-6` max error;
  its FFN-down same-input control is exact. Thus the `0.1706` sequential error
  at that boundary is inherited/amplified state error, not a layer-6 payload or
  structural mismatch.
- Layer 7, at both position counts, has max boundary error `3.24965e-4` in
  this control; Q/K/V projections and attention context are within F32-level
  differences, and its FFN-down same-input control is exact. This is much
  smaller than the `0.3653` sequential block-7 error. The layer's accumulated
  incoming-state difference is the dominant distinction; local FFN arithmetic
  still does not satisfy the fixed `1e-5` comparison at every checkpoint.
- In the eight-block sequential trace, layer 7's FFN-down checkpoint reaches
  max error `0.39618`; the corresponding same-input GGML control is exact.
  This excludes that layer's down-weight bytes/layout/backend row-dot behavior
  as the source of the sequential difference. The discrepancy is upstream in
  the perturbed activation path and its quantization-sensitive projections.

These controls do not establish full-model parity. They demonstrate that later
blocks execute on reference inputs with the expected tensors and that the large
sequential error is dominated by propagated numerical drift, including
activation-quantization sensitivity. No tolerance adjustment or input
substitution was applied to the sequential runs.

## Gate status

| Gate | Status |
|---|---|
| Block 1, positions 1/2 | **PASS** for exact boundary output |
| Block 1, positions 4/8 | **EXECUTED; numerical parity not passed**; reference-input controls isolate build-path / propagated drift |
| Sequential vBuf through 4 blocks, positions 4/8 | **EXECUTED; numerical drift grows** |
| Sequential vBuf through 8 blocks, positions 4/8 | **STOPPED HERE; numerical drift too large to continue without investigation** |
| Blocks 16, 20, and 40; full vBuf forward | **NOT TESTED** |
| Final normalization and logits parity | **NOT TESTED** |
| Deterministic greedy-token parity / ordinary generation | **NOT TESTED** |
| Persistent autoregressive KV | **NOT IMPLEMENTED / NOT TESTED** |
| Qwen3 tools, streaming parity, and Pi | **NOT TESTED; remain disabled** |

There was no shared runtime/source change in this qualification turn, so the
DeepSeek HTTP regression was not rerun. The current pinned-GGML CTest run passed
30/30, including the DeepSeek semantics contract; prior bounded DeepSeek HTTP
qualification remains separate evidence and was not reclassified here.
