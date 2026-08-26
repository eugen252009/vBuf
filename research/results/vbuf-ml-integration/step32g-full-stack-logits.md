# Step 32G: Full Stack Execution And Logits

Date: 2026-08-26

## Result

`FULL_PRODUCTION_EXECUTION_PASS_WITH_INDEPENDENT_REFERENCE_NEAR_TIE_NOTE`

The generic portable Rust runner executed the complete persisted base
transformer stack, layers `0..45`, with real activation handoff, final RMSNorm,
and the persisted output head. The additional physical predictor-layer index
space was not executed. The initial hidden state was the only synthetic input.

The output projection uses the same generic MatMul graph in row chunks of
`8192` vocabulary rows. Each BF16 head chunk is converted, consumed, and
released before the next chunk. This avoids converting the complete
`151552 x 4096` head to F32 at once.

Production execution and acquisition invariants pass. The independent NumPy
reference matches the final argmax and top-10 and remains within the observed
full-stack numerical envelope. Its fully propagated replay differs at two
ordered Top-8 slots in one layer-43/token-2 decision: experts `73` and `11`
swap ranks, while the selected expert set remains identical. A targeted
layer-45 replay using the production layer-44 checkpoint as common input has
exact route parity and `1.14440918e-04` maximum output error. The two slots are
retained as an ordered numerical near-tie qualification note, not silently
reported as strict full-chain exact ordered route parity. Detailed evidence is
in `step32g-r-router-near-tie-stability.md`.

## Qualified Artifacts

- Branch: `vbuf-ml`
- Starting commit: `61ef97e`
- Model repository: `zai-org/GLM-4.5-Air-FP8`
- Model revision: `f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc`
- Model artifact: `.step32c/glm-4.5-air-fp8.vbuf`
- Model bytes: `112563538898`
- Model SHA-256: `15b4f0d3b7d72754c0e233bb787e72b1d6cdab655aec9ecab36dac8b654c0f14`
- Semantic sidecar: `.step32c/glm-4.5-air-fp8.semantic.vbuf`
- Sidecar bytes: `14306259`
- Sidecar SHA-256: `817630bd5579ab603c870e6041c75267d89c90e87b45b0f5e5b844d88840bbea`
- Real model redownloaded: `NO`
- Real model payload rewritten: `NO`
- Second model-sized copy created: `NO`
- Payload and sidecar reopened: `PASS`
- Payload size match: `PASS`

The payload was mapped read-only. The source profile resolved materialized
ranges to source ID `1`. No Hugging Face, Safetensors, config, tokenizer,
GGML, or device backend was accessed.

## Model And Scope

- Base transformer layers executed: `0..45` (`46` layers)
- Additional predictor-layer index space: preserved, not executed
- Layer `0`: dense
- Layers `1..45`: MoE
- Experts per MoE layer: `128`
- Routed Top-K: `8`
- Shared expert: `1`
- Batch: `1`
- Sequence length: `4`
- Hidden size: `4096`
- Query/KV heads: `96/8`
- Head dimension: `128`
- Rotary dimension: `64`
- Vocabulary size: `151552`
- Final norm: persisted `512:36322`, BF16 `[4096]`
- Output head: persisted `512:0`, BF16 `[151552,4096]`
- Embedding and output head payload ranges: distinct
- Initial input hash: `5eb4f1bfdd34f9a76ea0189484d387cf1a047c27d2c49d3f2783211bb1ec604b`

The layer catalog is derived from persisted semantic roles and validated tensor
geometry. Runtime execution does not perform source-name lookups.

## Production Gates

| Gate | Layers | Execution | Final output hash |
|---|---:|---|---|
| Smoke | `0..0` | PASS | `72ba3f25716de08ce2c940c079e5b1c284c694bf83bd61d9482cdfccdf4cac11` |
| Progressive | `0..15` | PASS | `ec8b821e24e37341fbcc075a2ed23e245d82e74addc1a0fb1f537a4ceacfbd28` |
| Progressive | `0..31` | PASS | `af0095bbc8b3d222747a276eb3686dbd8748978f83ddae8bf4872eabd1ae4988` |
| Full base stack | `0..45` | PASS | `5386d869591ec7626a1099b861c45af2ed40bae43c23c42498b7fddf5dd1858b` |

The full chunked run reported:

- Cumulative source bytes: `25333942272`
- Unique model bytes: `25333942272`
- Unique model tensors: `7236`
- Peak source range bytes: `612765184`
- Peak converted F32 cache: `2445369856`
- Peak activation bytes: `5849088`
- Peak internal working set: `2451251712`
- Peak KV state bytes: `1507328`
- Peak output-head chunk F32 bytes: `134234112`
- Output-head working set: `136790016`
- Logits shape: `[1,4,151552]`
- Logits bytes: `2424832`
- Final norm hash: `7205408511c1c75de1a45ac5f8af3b2926995cf39cf02a0b616328697ed7aa3c`
- Logits hash: `af73970f2d47e7cd07225b7eb4bcb09c166d9ba5363c1ef8b58b54a8bc6b47db`
- Cross-layer KV reads: `0`
- Unselected routed expert tensors touched: `0` at every layer
- Active transient leases after run: `0`
- Active execution and layer states after run: `0`

The `2445369856` F32 peak is from the largest selected layer-local weight
working set, not from the output head. The output head itself is bounded by the
`8192`-row chunk.

## Independent Reference

The reference decoded the same persisted FP8/BF16 ranges through the manifest
and independently computed attention, routing, experts, shared experts, final
norm, and logits. It did not use production intermediates for the propagated
full replay.

Full replay observations:

- Routing mismatches: `2` ordered Top-8 positions, both in layer `43`, token `2`
- Routing membership mismatches: `0`; only experts `73` and `11` swap ranks
- Transformer maximum absolute error: `2.16674805e-03`
- Final norm maximum absolute error: `5.34057617e-04`
- Logits maximum absolute error: `3.17096710e-04`
- Final argmax: exact, token ID `729`
- Final top-5: exact, `[729,6722,4616,198,8250]`
- Final top-10: exact, `[729,6722,4616,198,8250,1688,32843,829,23216,606]`

The first material numerical difference is the layer-2 shared-expert
projection, where independent BLAS reduction order produces approximately
`1.73950195e-03` absolute difference. The difference remains bounded through
the rest of the stack. A common-input layer-45 replay produced exact routing
IDs and `1.14440918e-04` maximum layer output error, confirming that the
production route and dispatch path are not the source of the two propagated
reference route changes.

The targeted Step 32G-R replay shows that the two positions are an ordered
numerical instability: the production pair gap is approximately one F32 ULP,
while the K/K+1 membership cutoff remains separated by approximately `2717`
ULPs. Common-input cross-feed and float64 controls implicate both propagated
activation drift and reduction/computation order. Strict full-chain ordered
route-ID equality is therefore not claimed until a reference implementation
with a documented compatible reduction order or a documented near-tie policy
is available; production routing semantics are unchanged.

## Verification

- Full base production execution: PASS.
- Final norm and logits execution: PASS.
- Chunked output-head materialization: PASS.
- Selected-only production expert acquisition: PASS.
- Cross-layer KV isolation: PASS, zero prior-state reads.
- Transient lease and state cleanup: PASS.
- Independent full-stack numerical comparison: PASS with near-tie note.
- Step 32G-R ordered near-tie replay: PASS; two slots, zero membership changes.
- Independent final argmax/top-10 comparison: PASS.
- Strict full-chain independent route-ID equality: NOT CLAIMED; two propagated
  near-tie differences remain.

The production command was:

```text
cargo run --manifest-path rust/Cargo.toml -q -p vbuf-runtime --bin vbuf-runtime-step32f-progressive -- \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.semantic.vbuf \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32g-full-chunked.checkpoints \
  /tmp/opencode/step32g-full-chunked.manifest \
  0 46
```

The independent reference control was:

```text
python -u research/results/vbuf-ml-integration/step32g_bounded_reference.py \
  /tmp/opencode/step32g-full.manifest \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32g-full.checkpoints \
  46
```

## Boundary

This qualifies full portable generic F32 execution of the persisted base stack,
selected-only MoE acquisition, bounded layer-local residency, chunked final
projection, final norm, and real logits production. It does not qualify
tokenization, embedding-driven input, decode, generation, device execution,
GGML parity, or strict full-chain exact routing parity for the independent
reference.
