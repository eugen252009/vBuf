# Step 32I: Retained-KV One-Token Decode

Date: 2026-08-27

## Result

`REAL_RETAINED_KV_ONE_TOKEN_DECODE_QUALIFIED`

The portable generic Rust runtime executed the fixed real-text prefill for
`Test`, retained one checked KV state per layer, and then executed exactly one
incremental decode for token `220`. The decode used query length `1`, past
length `4`, position `4`, and capacity `5`; it did not recompute the prefix,
execute token 6, or enter a generation loop.

All 46 layers completed with four past KV positions read and one new KV
position appended. The independent full-prefix NumPy reference reproduced the
token and embedding fixtures, matched routing membership at every MoE layer,
and matched the production decode argmax and top-10 ordering.

## Qualified Input And Geometry

- Text: `Test`
- Prefill token IDs: `[51,68,82,83]`
- Decode token ID: `220` (`STEP32H_ARGMAX`)
- Full reference sequence: `[51,68,82,83,220]`
- Layers: `0..45` (`46`)
- Layer `0`: dense
- Layers `1..45`: MoE
- Hidden size: `4096`
- Query/KV heads: `96/8`
- Head dimension: `128`
- Routed experts: Top-8 from `128`
- Vocabulary: `151552`
- Request state ID: `801`
- KV capacity: `5`

The production path mapped the immutable semantic sidecar and payload
read-only. It used the persisted tokenizer and embedding row `220`; no
Hugging Face, Safetensors, GGML, device, remote, or loader-specific source was
accessed.

## Production Evidence

| Item | Result |
|---|---:|
| Prefill logits hash | `5c0c52e803670402726c729f53e81415efbb80113fc8d7748169c7f0d2d24afc` |
| Prefill state length | `4` at all 46 layers |
| Prefill KV state bytes | `1507328` |
| Prefill source bytes | `25247329280` |
| Decode complete | `YES` |
| Decode past positions read | `184` (`46 x 4`) |
| Decode positions appended | `46` (`46 x 1`) |
| Decode state length | `5` at all 46 layers |
| Decode KV state bytes before/after | `1507328` / `1884160` |
| KV bytes appended | `376832` |
| Decode source bytes | `13463473152` |
| Decode final transformer hash | `9ddbd98b42d6499043e3d180ac46d12f5fcb62381fde13890048ec54bad39652` |
| Decode final norm hash | `3dcc4bb0f5cfbcc4da83100423c8863baa729b3b505c0da4989db66178496542` |
| Decode logits hash | `cee96ed0dacaf569f242ee50887d410b326ae0c8befb9b06a0aa56a6047dfcea` |
| Decode argmax | `16` |
| Decode top-10 | `[16,17,18,19,20,21,22,23,98668,24]` |
| Decode routing decisions | `45` |
| Selected expert occurrences | `360` (`45 x 8`) |
| Unselected expert tensors touched | `0` |
| Expert overfetch bytes | `0` |
| Layer state mismatches | `0` |
| Decode completion | `YES` |
| Cleanup state bytes before/after | `1884160` / `0` |
| Active leases/states after cleanup | `0` / `0` |

The decode completed in `876270.221 ms`; final output projection took
`39049.707 ms`. The peak converted-weight cache was `1061249536` bytes and
the peak KV state was `1884160` bytes. These are runtime measurements, not
transport ceilings.

## Independent Reference

The reference consumed the validated manifest ranges and independently
computed the five-token causal prefix, including attention, router stages,
selected experts, shared experts, final normalization, and logits. It compared
the final reference position with the single-token production decode.

- Token-ID reference: pass
- Prefill embedding max absolute error: `0`
- Decode embedding max absolute error: `0`
- KV key max absolute error: `3.57627869e-05`
- KV value max absolute error: `1.84774399e-05`
- Transformer max absolute error: `2.44140625e-04`
- Transformer max relative error: `1.13032258e+00`
- Final norm max absolute error: `1.02996826e-04`
- Final norm max relative error: `1.94485728e-02`
- Logits max absolute error: `4.76837158e-05`
- Logits max relative error: `3.13069910e-01`
- Routing membership mismatches: `0`
- Production/reference argmax: `16` / `16`
- Production/reference top-5: `[16,17,18,19,20]` / `[16,17,18,19,20]`
- Production/reference top-10: `[16,17,18,19,20,21,22,23,98668,24]` / `[16,17,18,19,20,21,22,23,98668,24]`

The numerical differences are expected from independent F32 execution and
reduction ordering. The qualification claims bounded numerical parity,
routing membership parity, and output ordering parity; it does not claim byte
identity of independently computed activations.

## Implementation Boundary

`GenericExecutionState` now validates payload geometry, exposes checked state
length and KV views for qualification, tracks the last read/append counts, and
supports transactional rollback when graph execution fails. Generic attention
uses the existing state-owned prefix and indexes newly supplied K/V rows from
the old state length. It remains model-agnostic.

The qualification runner builds attention graphs with explicit query length,
current KV length, and position start. It uses the same state object for
prefill and decode, dispatches only the selected experts, records per-layer KV
snapshots, and resets/releases every state at teardown. The independent
reference was extended to five-token causal replay and KV comparison without
consuming production intermediates.

## Verification

```text
cargo test --workspace
python -m py_compile research/results/vbuf-ml-integration/step32g_bounded_reference.py
```

Both passed. The production run used:

```text
rust/target/debug/vbuf-runtime-step32i-retained-kv \
  .step32c/glm-4.5-air-fp8.semantic.vbuf \
  .step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32i-real.checkpoints \
  /tmp/opencode/step32i-real.manifest Test
```

The independent reference used:

```text
python research/results/vbuf-ml-integration/step32g_bounded_reference.py \
  /tmp/opencode/step32i-real.manifest \
  .step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32i-real.checkpoints 46 decode Test
```

## Boundary

This qualifies one retained-KV real-text incremental decode through all 46
portable generic layers, checked state growth, selected-only MoE acquisition,
KV parity, final-position numerical parity, output ordering, and cleanup. It
does not qualify repeated generation, token 6, device execution, GGML parity,
or backend-loader integration. The immutable model artifacts and temporary
checkpoint traces remain outside the repository.
