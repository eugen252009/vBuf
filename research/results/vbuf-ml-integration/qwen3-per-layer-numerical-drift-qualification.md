# Qwen3 Per-Layer Numerical Drift and Full-Stack Qualification

## Status

The original layers 0–7 local/sequential analysis remains below. Qualification
was subsequently extended through all 40 transformer blocks, final RMSNorm,
output projection, logits, greedy next-token selection, and an 8-token
full-recompute generation trajectory. All tested states were finite. Local
reference-reset runs at block 39 remain close to the fixed `1e-5` checkpoint
criterion except isolated F32-rounding maxima; the large sequential block-39
and head errors are propagated drift, not evidence of a local layout/tensor
defect. The fixed tolerance was not widened.

The full 4- and 8-position prompt runs produce the same greedy next token as the
pinned reference. Eight consecutive full-recompute generated tokens also match
exactly at every step. This is bounded execution/token-parity evidence for the
tested synthetic prompt(s), not a production-model enablement or a general
quality claim. Persistent KV, ordinary user prompts, chat templates, streaming,
tools, and Pi remain unqualified.

## Artifact and method

- Artifact: `bartowski/Qwen_Qwen3-14B-GGUF`, revision
  `bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2`,
  `Qwen_Qwen3-14B-Q4_K_M.gguf`.
- GGUF SHA-256: `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6`.
- vBuf payload SHA-256:
  `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Sequences: 4 and 8 positions, 8 CPU threads; token IDs are `0..N-1`.
- Reference: llama.cpp/GGML `a97123e497968f3440264c0464a7adc7c999c027`.
- vBuf GGML: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- Local path: `reference_input:N` injects reference embedding/layer output before
  exactly layer N, then compares vBuf output to reference `l_out-N-0`.
- Sequential path: executes layers 0..39, feeding each vBuf output forward and
  comparing each boundary to the corresponding reference output. The final
  llama.cpp block selects the requested last prompt row before its FFN; the
  isolated runner mirrors that selection for block 39.
- Both paths use the same strict finite/shape checks and report the previous
  max-absolute `1e-5` comparison. No tolerance was widened and numerical
  differences do not stop a finite execution.

## Per-layer tables

Each cell is `max_abs / RMS / relative_RMS / cosine`; relative RMS is divided
by reference hidden-state RMS. Local is reference-reset; sequential is the
vBuf-fed trajectory. All listed outputs are finite.

### 4 positions

| Layer | Local error | Sequential error |
|---:|---:|---:|
| 0 | `8.75555e-5 / 1.15195e-5 / 3.39571e-5 / .999999999424` | same |
| 1 | `1.90735e-6 / 3.78739e-8 / 7.22179e-8 / 1` | `.00946999 / .000516313 / .000984503 / .999999515665` |
| 2 | `2.38419e-7 / 2.04298e-8 / 2.95908e-8 / 1` | `.0355682 / .000969062 / .00140360 / .999999018405` |
| 3 | `.000827670 / .000108023 / .000139033 / .999999990335` | `.0247765 / .00148341 / .00190926 / .999998180815` |
| 4 | `1.90735e-6 / 5.59256e-8 / 4.87117e-8 / 1` | `.0245727 / .00323071 / .00281398 / .999996060558` |
| 5 | `1.90735e-6 / 8.66743e-8 / 6.78338e-8 / 1` | `.0412312 / .00533354 / .00417418 / .999991319242` |
| 6 | `1.90735e-6 / 9.23412e-8 / 9.68715e-10 / 1` | `.170601 / .00759711 / 7.96982e-5 / .999999996824` |
| 7 | `.000324965 / 4.55627e-5 / 4.71480e-7 / 1` | `.365294 / .0118544 / .000122669 / .999999992476` |

### 8 positions

| Layer | Local error | Sequential error |
|---:|---:|---:|
| 0 | `8.75555e-5 / 8.14567e-6 / 2.66928e-5 / .999999999644` | same |
| 1 | `.00160353 / .000140830 / .000314337 / .999999950597` | `.00946999 / .000672621 / .00150131 / .999998873267` |
| 2 | `1.90735e-6 / 4.25630e-8 / 7.44526e-8 / 1` | `.0355682 / .00167081 / .00292263 / .999995740234` |
| 3 | `.000827670 / 7.63835e-5 / .000118464 / .999999992983` | `.0417004 / .00252595 / .00391752 / .999992327726` |
| 4 | `1.90735e-6 / 5.75139e-8 / 6.38053e-8 / 1` | `.107292 / .00433102 / .00480478 / .999988457552` |
| 5 | `1.90735e-6 / 1.13262e-7 / 1.10093e-7 / 1` | `.0846786 / .00775405 / .00753713 / .999971671421` |
| 6 | `1.90735e-6 / 8.75821e-8 / 1.29932e-9 / 1` | `.170601 / .00999122 / .000148225 / .999999989015` |
| 7 | `.000324965 / 3.22179e-5 / 4.71462e-7 / 1` | `.365294 / .0147593 / .000215981 / .999999976676` |

The original detailed comparison logs were under `/tmp` and were cleared by
the system restart; the aggregate metrics below are preserved here. They
included mean absolute error, reference RMS/L2, vBuf L2, norm ratio, max-error
index and values, ULP distance, and per-token max errors. After restart, new
reference dumps with operator-source metadata were regenerated at
`/tmp/qwen3-resetref-{4,8}-t8`; the original sanitizer control log was
`/tmp/qwen3-perlayer-asan.log`. The later full-depth references were regenerated
under `/tmp/qwen3-finalref-{4,8}-t8` and `/tmp/qwen3-full-recompute-reference`;
full-run logs are in `/tmp/qwen3-finalhead-vbuf-*.log` and
`/tmp/qwen3-full-recompute-vbuf-step-*.log`.

## Sequential amplification

RMS and max amplification are output sequential error divided by input
sequential error (the preceding boundary). These are trajectory-specific
ratios, not Lipschitz constants.

| Layer | 4-pos max × | 4-pos RMS × | 8-pos max × | 8-pos RMS × |
|---:|---:|---:|---:|---:|
| 0 | — | — | — | — |
| 1 | 108.2 | 44.82 | 108.2 | 82.57 |
| 2 | 3.756 | 1.877 | 3.756 | 2.484 |
| 3 | .697 | 1.531 | 1.172 | 1.512 |
| 4 | .992 | 2.178 | 2.573 | 1.715 |
| 5 | 1.678 | 1.651 | .789 | 1.790 |
| 6 | 4.138 | 1.424 | 2.015 | 1.289 |
| 7 | 2.141 | 1.560 | 2.141 | 1.477 |

Sequential RMS error increases at every boundary for both lengths, while
max-absolute error is non-monotonic at some layers. The first large jump is
block 0→1, consistent with the known Q8_K sensitivity. At block 7, max errors
are `0.365294`; relative RMS is `1.22669e-4` (4 positions) and
`2.15981e-4` (8 positions). Reference/vBuf hidden L2 norms at that boundary
are respectively `13829.6454/13829.6460` and `13830.2491/13830.2525`.

## Local checkpoint findings and classification

- **Layer 0 — LOCAL NUMERICAL DRIFT.** The first meaningful local difference
  includes attention-value aggregation: with identical F16 V and reference
  probabilities, vBuf's compact-context result differs from llama.cpp by one
  ULP at the max-absolute location (`1.49012e-8` at 4 positions). The vBuf
  result matches the FP64 sum; llama.cpp is one F32 ULP away. The FFN SwiGLU
  pre-Q8 max difference is `1.19209e-7` at 4 positions, while the Q6_K×Q8_K
  down output reaches `8.75592e-5`.
- **Layer 1 — LOCALLY STABLE at 4; LOCAL NUMERICAL DRIFT at 8.** At 8 positions,
  Q/K/V projections from reference input match exactly, but raw QK scores differ
  by `6.10352e-5`; context differs by `2.27690e-5`, and the local boundary by
  `1.60353e-3` (relative RMS `3.14337e-4`, cosine `.9999999506`). An earlier
  LLAMAFILE-enabled reference-input control matched layer 1 exactly throughout;
  it is diagnostic evidence for CPU attention reduction-path sensitivity, not
  a production change.
- **Layer 2 — LOCALLY STABLE.** Boundary max error is at most `1.90735e-6`.
- **Layer 3 — LOCAL NUMERICAL DRIFT.** Q/K/V projections match exactly. The
  4-position pre-Q8 SwiGLU max difference is `4.76837e-7`; one Q8_K quantized
  value changes, with 106 blocks/scales changed, 111 serialized bytes changed,
  and post-dequant max difference `9.06640e-3`. FFN-down same-input GGML output
  is exact, while local down output differs by `8.27670e-4`. At 8 positions,
  263 blocks/scales, one quantized value, and 271 bytes differ; the same max
  local output error results. This is quantizer sensitivity, not a down-weight
  defect.
- **Layers 4 and 5 — LOCALLY STABLE.** Boundary max error at most `1.90735e-6`.
- **Layer 6 — LOCALLY STABLE.** Boundary max error `1.90735e-6` at both lengths;
  exact FFN-down same-input control. The large sequential boundary error is
  inherited/amplified input error.
- **Layer 7 — LOCAL NUMERICAL DRIFT.** Boundary max error `3.24965e-4`, but
  relative RMS is only `4.7147e-7` and cosine is 1. The pre-Q8 SwiGLU max
  difference is `1.43051e-6` (4 positions); one Q8 value changes, with 77
  blocks/scales and 80 bytes changed; post-dequant max is `4.12330e-3`. The
  same-input down control is exact. At 8 positions 220 blocks/scales, one
  quantized value, and 227 bytes change. The sequential layer-7 down error
  reaches `0.396181`, which is not reproduced by the same-input down control.

No layer meets the evidence bar for **LOCAL STRUCTURAL SUSPECT**. Catalog,
materialization, finite-state, tensor geometry, attention/GQA, RoPE, residual,
and FFN graph checks remain active. The same-input down controls are exact for
the examined layers; there is no evidence of a wrong down tensor, layout, or
backend row-dot result.

## Controlled attention projection and CPU reduction order

The qualification runner now prints matmul source types/shapes/strides. The
reference checkpoint dumper also records op and source metadata. For the
representative 4-position `kqv` operation, llama.cpp uses `MUL_MAT`, F16 V
source `[256,128,8]`, F32 probabilities `[256,4,40]`, output F32
`[128,4,40]`: the KV-cache key extent is 256, with masked positions carrying
zero probabilities. The vBuf compact operation uses F16 `[4,128,8]` and F32
`[4,4,40]`. Both use F16 operands for the dot after CPU dispatch; output is F32.

A vBuf reference-extent control padded the same live F16 V values and supplied
reference F32 probabilities over K=256. It still differed from llama.cpp by
one ULP at the max-error location. Compact versus padded vBuf itself also
changed by one ULP. The FP64 oracle over the live F16 operands matches compact
vBuf exactly; llama.cpp is one ULP away. Thus different reduction extent
changes rounding, but does not alone explain the residual llama/vBuf ULP.

Source/build inspection identifies a concrete dispatch difference. The pinned
llama.cpp build enables `GGML_USE_LLAMAFILE`, `GGML_USE_CPU_REPACK`, and OpenMP;
the isolated vBuf build has these disabled (`-march=native`, AVX2/FMA/F16C on
this host). For F16×F16→F32 matmul, llama's eligible LLAMAFILE path dispatches
to `llamafile_sgemm`/tinyBLAS. vBuf uses generic `ggml_compute_forward_mul_mat`
and `ggml_vec_dot_f16`. The generic AVX2 path uses F32-converted F16 operands,
FMA vector accumulators/reduction for K≥32, and its scalar double accumulator
for K<32; the tinyBLAS kernel has its own tiled reduction order. This is
consistent with mathematically equivalent, non-bit-identical results. The
precise individual instruction/accumulator sequence responsible for the
remaining one-ULP difference is not yet proven; no attempt was made to force
bit identity.

The padded-control invocation is available through the generic local-reset
mode for layer 0; its extra result is named
`v_context_reference_kv_extent_vbuf`. The geometry and output were captured in
`/tmp/qwen3-padded-control-{4,8}.log` and
`/tmp/qwen3-padded-control-4-final.log`.

## Q8_K amplification sample

| Path/layer/positions | Pre-Q8 max / RMS | Changed blocks | Changed int8 values | Changed bytes | Scale changes | Post-dequant max / RMS | Down max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local 0 / 4 | `1.19209e-7 / 1.87100e-9` | 40 | 1 | 41 | 39 | `.00101620 / 3.85100e-6` | `8.75592e-5` |
| Local 3 / 4 | `4.76837e-7 / 6.83873e-9` | 106 | 1 | 111 | 106 | `.00906640 / 3.43582e-5` | `.000827670` |
| Local 3 / 8 | `9.53674e-7 / 7.34040e-9` | 263 | 1 | 271 | 263 | `.00906640 / 2.42949e-5` | `.000827670` |
| Local 7 / 4 | `1.43051e-6 / 1.51817e-8` | 77 | 1 | 80 | 77 | `.00412330 / 1.56257e-5` | `.000324965` |
| Seq 0 / 4 | `1.19209e-7 / 1.87100e-9` | 40 | 1 | 41 | 39 | `.00101620 / 3.85100e-6` | `8.75592e-5` |
| Seq 3 / 4 | `.0355988 / .000315011` | 130 | 2,126 | 3,103 | 130 | `.0332957 / .000369953` | `.0121074` |
| Seq 7 / 4 | `.281512 / .00285866` | 136 | 12,045 | 14,343 | 136 | `.281513 / .00329954` | `.396181` |

This shows discrete activation representation changes after tiny local
pre-quantization differences. It does not justify changing the quantizer.

## Full-stack continuation: layers 8–39, final head, and generation

### Full 40-block sequential runs

Pinned reference checkpoints were captured for 40 layers at prompt lengths 4
and 8. Sequential vBuf runs completed all 40 blocks. Selected boundary metrics
(max absolute / relative RMS / cosine):

| Boundary | 4 positions | 8 positions |
|---|---:|---:|
| block 31 | `1.97943 / .001558 / .99999879` | `1.97943 / .002470 / .99999695` |
| block 32 | `2.56073 / .001776 / .99999842` | `3.13116 / .002802 / .99999607` |
| block 38 | `5.63245 / .004330 / .99999063` | `5.68909 / .006956 / .99997582` |
| block 39 (selected last row) | `21.4110 / .023274 / .99978141` | `4.39141 / .025147 / .99968633` |

All boundaries were finite. Error growth through blocks 31–38 was gradual,
though max error is non-monotonic. Block 39 is a clear amplification in the
4-position run. Its reference-reset control, using the exact reference input
before block 39, ends at max error `3.05176e-5`, relative RMS `1.51e-7`
(4-position) and `2.03e-7` (8-position). Q/K/V and causal attention also match
closely under reset. Conversely, with the sequentially drifted input, block-39
attention/FFN intermediates diverge materially (e.g. final-token attention
residual relative RMS about `.0226` and SwiGLU relative RMS `.0271` at 4
positions; `.0226` and `.0465` at 8 positions). This identifies propagated
input drift sensitivity at the final block, not a locally incorrect block.

The reset output's max absolute error slightly exceeds `1e-5`, but its RMS and
relative error are tiny; this remains reported rather than hidden. The
reference-derived attention-projection diagnostic subtracts large F32 residuals
and is cancellation-sensitive; residual and direct block output are the
reliable comparisons there.

### Final RMSNorm, output projection, and logits

Final-head outputs were compared against the independent pinned llama.cpp
`result_norm` and `result_output` checkpoints:

| Prompt positions | Final norm relative RMS / cosine | Logits relative RMS / cosine | Greedy next token (reference = vBuf) | Top-10 overlap |
|---:|---:|---:|---:|---:|
| 4 | `.031834 / .99949424` | `.027983 / .99961871` | `4 = 4` | `10/10` |
| 8 | `.039178 / .99924250` | `.030804 / .99952757` | `89401 = 89401` | `10/10` |

These logit differences are not within the fixed absolute checkpoint bound;
matching argmax here is reported as token agreement, not numerical parity.

### Full-recompute generation (seed token 0)

The pinned reference harness rebuilt a fresh llama.cpp context for every prefix.
For each step, the vBuf qualification process independently recomputed token
embeddings, all 40 blocks, final norm, and logits from the current complete
prefix; no KV state was reused. Every vBuf greedy token matched the independent
reference token, so the tested autoregressive paths stayed identical.

| Generated tokens so far | Prefix length recomputed | Next token, reference = vBuf |
|---:|---:|---:|
| 1 | 1 | `25` |
| 2 | 2 | `220` |
| 4 | 4 | `13` |
| 8 | 8 | `198` |

Full generated sequence: `25, 220, 16, 13, 15, 13, 15, 198`.
All eight stepwise greedy decisions matched. At prefix lengths 1 and 2, final
logits matched bitwise. By prefix length 8, final-logit relative RMS was
`.04897`, cosine `.99892335`, while the argmax still matched and top-10
overlap was 10/10. This is a narrowly scoped parity result, not evidence that
all prompts or longer generation lengths will agree. No KV persistence was
implemented or exercised.

### Classification and decision

**Structural status:** no local tensor/layout/graph defect was found in the
40-block execution, final RMSNorm, or output projection controls.
**Numerical status:** propagated drift grows with depth and is amplified by
block 39 and the final head; the local reset is near-reference. **Token status:**
greedy agreement passed for the tested 4/8-position synthetic prompts and all
8 recomputed generation steps. This completes the requested isolated
qualification scope only; production Qwen3 remains disabled.

## Tests and save-point boundaries

- Pinned GGML target build after the harness changes: **PASS**.
- Qwen local-reset runs: layers 0–7 (historical table) and block 39 at 4/8
  positions; block-39 reset completed and finite.
- Qwen sequential runs: all 40 blocks at 4/8 positions, final norm/output,
  and 8 full-recompute token steps; all completed and finite.
- Padded reference-extent value-projection control: **EXECUTED** at 4/8 before
  restart; one-ULP difference remains. The 4-position case was rerun under
  sanitizers after restart.
- Full pinned-GGML CTest after rebuild: **30/30 PASS**.
- ASan/UBSan: **PASS** for the 4-position layer-0 padded-projection control
  and the 4-position block-39 reference-reset run; qualification TU and C++
  GGML objects instrumented, C objects and Rust library were not.
- `git diff --check`: **PASS**.
- DeepSeek HTTP suite: **NOT RERUN**; this turn's changes are isolated to the
  Qwen qualification harness/reference dumper and report, with no shared runtime
  changes. The DeepSeek semantics contract passed within the 30/30 CTest run;
  prior bounded DeepSeek HTTP evidence remains separate.
- Production `VbufGenerationSession` Qwen3 rejection remains unchanged.
- Persistent KV, tools, and Pi were not implemented or qualified.
- Artifacts/logs are under `/tmp/qwen3-finalref-{4,8}-t8`,
  `/tmp/qwen3-full-recompute-reference`, and `/tmp/qwen3-full-recompute-vbuf-step-*.log`.
- The qualification harness/reference dumper changes remain isolated; no
  production inference path was enabled.
