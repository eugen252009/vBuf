# Independent high-precision tensor-operation reference qualification

**Disposition:** bounded reference foundation implemented and exercised; Qwen3 native AV remains experimental and canonical packed-V AV remains authoritative. This work adds no production dispatch, changes no numerical threshold, and promotes no candidate. The existing numerical policy remains v2; QK, softmax, RMSNorm, general matmul, and actual-model AV observations remain `NEEDS_CALIBRATION` and are recorded as `NOT_TESTED`/replay-only diagnostics.

## Reference contract

The new host-only module is `integrations/ggml/include/vbuf_high_precision_reference.h` and `integrations/ggml/src/vbuf_high_precision_reference.cpp`. Its operations accept explicit logical tensor shapes and signed byte strides, so transposed, sliced, broadcast, and reversed views are handled independently of the production GGML execution path. Each operation has an explicit output dtype; allocation is capped at 16M elements by default. FP64 products and additions are separately rounded to prevent compiler contraction from changing the declared left-to-right oracle. Inputs must be finite for QK, AV, RMSNorm, and matmul. Softmax requires finite unmasked scores, treats a nonzero mask byte as excluded, supports one logical extent per row, writes exact zero outside the logical/active positions, and rejects empty or fully masked rows.

Implemented operations and semantics:

- **QK:** Q `[rows, query_heads, head_dim]` times K `[positions, kv_heads, head_dim]`; raw, unscaled scores; GQA mapping `floor(query_head / (query_heads / kv_heads))`.
- **Softmax:** stable binary64 max-subtracted exponentials/normalization; scale is explicit. Qwen3 applies `1/sqrt(128)` here, not in the raw QK reference.
- **AV:** P `[rows, query_heads, positions]` times V `[positions, kv_heads, head_dim]`, ascending-position accumulation, explicit per-query visible extents.
- **RMSNorm:** binary64 mean-square, adds the explicitly supplied positive epsilon before reciprocal square root, then multiplies by the supplied scale vector.
- **Matmul:** conventional `[M,K] × [K,N]`, with explicit F32 rounded multiply/add or F64 left-to-right accumulation.
- **GGML decoding:** row-size checked F32/F16 and supported quantized representations. Q4_K uses the pinned GGML `to_float` trait. Q8_K uses pinned `dequantize_row_q8_K` because this GGML revision has no public Q8_K `to_float` trait. Decoded operands are F32; the reference matmul accumulates their represented values in F64.

The versioned Numerical Contract System remains the source of reference kind, normative contract identity, policy/contract version, metrics, and status. Each evaluation also records `vbuf-high-precision-reference-cpp-v1`, accumulation precision, input/output representations, and operation parameters in its integrity snapshot and JSON. Reference metadata must be complete; the implementation revision is checked against the module's fixed revision string. Captured historical tensors are marked `replayed_metrics_only`, use diagnostic-only candidate identities, and cannot authorize admission. The provenance layer is not cryptographic attestation and does not change the documented same-process trust boundary.

## Synthetic tests

`vbuf_high_precision_reference_contract` passes tests for:

- strided/transposed matmul and explicit F32-vs-F64 cancellation behavior;
- QK GQA grouping and unsupported head ratios;
- scaled stable softmax, variable causal extents, explicit masks, exact zeros, NaN/Inf and fully masked-row behavior;
- AV over equivalent packed/native logical views and clipped visible extents;
- RMSNorm stride, reversed scale, epsilon, and nonfinite-input behavior;
- Q4_K encode/decode round-trip against the pinned upstream `to_float` trait, explicit F32-source versus Q4_K reconstruction metrics under the existing `NEEDS_CALIBRATION` contract, exact payload bounds, and matmul over dequantized weights;
- existing active synthetic AV criteria and provenance serialization; mutation of reference metadata invalidates the integrity snapshot, while replay-only results are rejected for admission.

No new numeric tolerances were introduced. The active synthetic AV test uses the existing policy-v2 thresholds and is scoped to synthetic fixture geometry only.

## Qwen3 captured attention results

The capture qualification consumes the existing layer-0 boundary files in
`qwen3-native-layout-av/raw/av-boundary-qk/` for the admitted Qwen3-14B Q4_K_M
artifact (`sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`),
pinned GGML `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, and the documented
26/14 placement. It recomputes raw QK from captured F32 post-RoPE Q and F16 K,
softmax from captured F32 scores with the causal extents and `1/sqrt(128)` scale,
and AV from captured F32 probabilities and F16 V. The recomputed AV output is
bitwise identical to the saved independent ascending-position FP64 oracle in
both phases. The QK and softmax input/output boundaries are kept separate;
softmax agreement does not hide upstream QK drift.

| Layer-0 capture | QK candidate vs FP64 relative RMS | Softmax candidate vs FP64 relative RMS | Native AV vs FP64 relative RMS | Canonical packed AV vs FP64 relative RMS |
|---|---:|---:|---:|---:|
| 32-row prefill, visible context 32 | `5.1696854e-4` | `1.1153038e-7` | `4.2341173e-8` | `2.6373705e-4` |
| one-row decode, visible context 33 | `1.3394157e-4` | `1.2774701e-7` | `5.0257471e-8` | `2.4018408e-4` |

The softmax candidate satisfies the captured row-normalization, unit-interval,
and excluded-position-zero invariants. These are measured fixture diagnostics,
not approved universal thresholds. The two AV captures independently reproduce
the existing reduction-order observation: native AV is closer to this FP64
reference on these same inputs, while canonical remains the authoritative
runtime path because full-model compatibility and phase/topology qualification
still fail elsewhere.

## Actual Q4_K Qwen3 projection fixture

An additional optional run uses the previously verified block-0
`blk.0.attn_q.weight` Q4_K payload and its captured Q8_K activation/output
fixture. The model artifact is the same admitted Qwen3 SHA above. The fixture's
manifest records the Q4_K payload hash
`3c4dd39531acff6eb56af4dfb0e39fed24ec3a71e6efe958a96c944596845bf9`, shape
`[5120,5120]`, and exact match to the vBuf tensor slice. The activation capture
is the 25-row `prefix25-q-input` quantized to Q8_K; the CPU backend output is the
existing pinned-GGML Q4_K projection capture. The reference decodes the exact
Q4_K and Q8_K representations and computes the first 64 output channels with
binary64 accumulation, then compares to those same 64 GGML outputs:

- relative RMS: `1.1473033e-7`;
- max absolute error: `2.2351742e-8`;
- output shape: `[25,64]`;
- contract: `vbuf.general_matmul.fp64.operation_accuracy` v1,
  `NEEDS_CALIBRATION`, replay-only.

This bounded slice verifies the independent matmul integration on actual
represented Qwen3 weights/activations. It is not a full projection qualification
and does not estimate Q4 quantization error against unquantized weights. The
reference and candidate both use Q4_K/Q8_K represented values; this comparison
isolates the represented-operand arithmetic path for the measured channels.

Machine-readable policy-v2 records are in
[`raw/evaluations.jsonl`](qwen3-independent-tensor-reference/raw/evaluations.jsonl);
phase summaries are in [`raw/summary.txt`](qwen3-independent-tensor-reference/raw/summary.txt).
All seven records explicitly carry the reference implementation revision,
precision, input/output representation, operation parameters, candidate/run
identity, and replay-only state. They are diagnostics, not candidate-admission
evidence.

## Reproduction

Synthetic and checked-in capture tests run through CTest:

```bash
cmake --build /tmp/vbuf-qwen-native-matrix-build \
  --target vbuf_high_precision_reference_contract vbuf_qwen3_reference_capture_qualification -j8
ctest --test-dir /tmp/vbuf-qwen-native-matrix-build \
  -R 'vbuf_(high_precision_reference_contract|qwen3_reference_capture_qualification)' --output-on-failure
```

The optional actual-Q4_K projection uses existing local qualification fixtures
(not copied into this repository):

```bash
/tmp/vbuf-qwen-native-matrix-build/vbuf_qwen3_reference_capture_qualification \
  research/results/vbuf-ml-integration/qwen3-native-layout-av/raw/av-boundary-qk \
  --q4k-qproj \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/fixture/blk.0.attn_q.weight.q4k.bin \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/fixture/cpu-q8k.q8_k.bin \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/fixture/cpu-q-output.f32.bin
```

The Q4_K tensor and its external fixture provenance are documented in
`qwen3-cuda-fixtures/prefix25-q4k-qproj-manifest.json`. No model payload or
external GGML worktree file was modified by this qualification.

## Decision

The reference layer supplies a reusable, independent arithmetic path and
machine-readable evidence without granting a new tolerance or authority.
Continue using the active policy-v2 system and keep the QK/softmax/RMSNorm/
matmul/actual-model AV contracts at `NEEDS_CALIBRATION` until operation-specific
scope and thresholds are separately justified. Keep native AV experimental,
canonical AV authoritative, and all admission safeguards unchanged.
