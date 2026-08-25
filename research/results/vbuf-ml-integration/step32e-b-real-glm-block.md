# Step 32E-B: Real GLM Layer-23 Portable Block Qualification

Date: 2026-08-25

## Result

PASS for the bounded qualification scope: one complete real GLM-4.5-Air-FP8
layer-23 MoE transformer block executed through the architecture-neutral Rust
portable graph and generic F32 executor. The run did not execute another
layer, generate tokens, or use GGML, llama.cpp, Safetensors, or model source
names as runtime authority.

## Inputs

- Model: `zai-org/GLM-4.5-Air-FP8`
- Revision: `f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc`
- Payload: `.step32c/glm-4.5-air-fp8.vbuf`
- Payload size: `112563538898` bytes
- Payload SHA-256: `15b4f0d3b7d72754c0e233bb787e72b1d6cdab655aec9ecab36dac8b654c0f14`
- Semantic sidecar: `.step32c/glm-4.5-air-fp8.semantic.vbuf`
- Sidecar size: `14306259` bytes
- Selected physical layer: `23`
- Input shape: `[1, 4, 4096]`
- Input SHA-256: `5eb4f1bfdd34f9a76ea0189484d387cf1a047c27d2c49d3f2783211bb1ec604b`

The payload and sidecar were mapped read-only. The sidecar source profile
resolved all materialized ranges to source ID `1`, the qualified payload
artifact. No artifact bytes were rewritten.

## Graph

The block was lowered from semantic TensorId bindings, not source names:

1. Input RMSNorm.
2. Q/K/V FP8 projection and BF16 bias addition.
3. Q/K/V head reshape and partial RoPE: rotary dimension `64` of head
   dimension `128`, theta `1,000,000`.
4. Causal GQA attention: 96 query heads, 8 KV heads, head dimension 128,
   request KV state ID 7.
5. Output projection and attention residual.
6. Post-attention RMSNorm.
7. Sigmoid router, persisted correction bias, stable Top-8 selection, and
   normalized original sigmoid weights.
8. Selected routed expert gate/up/down dispatch in rank order.
9. Shared expert gate/up/down path.
10. MoE combine and final block residual.

The four input rows selected these unique routed experts:

`[1, 6, 11, 19, 27, 52, 59, 62, 89]`

The complete per-row selection IDs and weights were emitted by the runner and
included in the independent checkpoint comparison.

## Materialization

- Source bytes requested: `283511296` bytes.
- Tensor identities touched: `75`.
- Peak cached F32 tensor bytes: `1130455552` bytes.
- RSS before execution: `43788 KiB`.
- RSS after execution: `1462892 KiB`.
- KV state length after the prefill block: `4`.
- Final output elements: `16384`.

The large F32 residency is bounded to the selected layer-23 projections,
norms, selected expert tensors, shared expert tensors, and their validated
scale ranges. No model-sized F32 copy was created.

## Independent Reference

The production result was compared against an independent NumPy F32 control
at `/tmp/opencode/step32e_block23_reference.py`. The control independently
decoded the persisted FP8 E4M3 values and scales, reconstructed the same
semantic ranges from the generated manifest, and implemented the graph math
separately. It did not consume production intermediate values.

Maximum observed differences across recorded checkpoints:

- Maximum absolute difference: `1.04904175e-04`.
- Maximum relative difference: `5.42973430e-06`.
- Final block output maximum absolute difference: `8.01086426e-05`.
- Selection IDs: exact match.
- Selection weights maximum absolute difference: `8.19563866e-08`.

Production command:

```text
cargo run -q -p vbuf-runtime --bin vbuf-runtime-step32e-block -- \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.semantic.vbuf \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32e-block23.checkpoints \
  /tmp/opencode/step32e-block23.manifest
```

Reference command:

```text
python /tmp/opencode/step32e_block23_reference.py \
  /tmp/opencode/step32e-block23.manifest \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32e-block23.checkpoints
```

## Verification

- `cargo fmt --all -- --check`: passed.
- `cargo test --workspace`: passed.
- `cargo test -p vbuf-runtime`: 11 tests passed.
- Native portable graph adapter contract: built and passed.
- `python scripts/verify_portable_graph_neutrality.py`: passed,
  `FORBIDDEN_LEAKAGE_COUNT=0`.

The native adapter result is retained as a compatibility contract and is not
the source of the real-block qualification. Canonical GGML revision
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` remains unavailable locally.

## Boundary

This qualifies one real portable F32 block, bounded source materialization,
FP8 scale provenance, GLM router semantics, selected expert execution,
attention state, and independent numerical parity. It is not evidence of
full-model logits, generation parity, device execution, or canonical GGML
backend qualification.
