# Step 32D: GLM Semantic Sidecar Qualification

Date: 2026-08-25

## Scope

Step 32D adds persistent tokenizer and GLM MoE semantics without modifying or
rewriting the qualified 112.56 GB payload artifact. The semantic artifact is a
small vBuf sidecar whose tensor directory contains zero-length placeholders
and a persistent source profile binding each placeholder to the existing
artifact.

The implementation does not start backend execution and does not add a
dependency on llama.cpp or GGML.

## Inputs

- Model: `zai-org/GLM-4.5-Air-FP8`
- Revision: `f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc`
- Payload artifact: `.step32c/glm-4.5-air-fp8.vbuf`
- Payload size: `112563538898` bytes
- Payload SHA-256: `15b4f0d3b7d72754c0e233bb787e72b1d6cdab655aec9ecab36dac8b654c0f14`
- Sidecar: `.step32c/glm-4.5-air-fp8.semantic.vbuf`
- Sidecar size: `14306259` bytes

The sidecar builder requires the recorded payload hash. It does not hash or
redownload the large payload during semantic qualification.

## Tokenizer

The sidecar persists the real tokenizer JSON vocabulary, offsets, added-token
flags, special-token IDs, BPE merge IDs, GPT-2 ByteLevel identity, BOS policy,
and `ignore_merges=true` model behavior.

- Token IDs: `151365`
- Added tokens: `36`
- Special added tokens: `22`
- Merge pairs: `318088`
- EOS and PAD: token ID `151329`
- Chat template: absent in the supplied tokenizer metadata

`Gpt2ByteLevelTokenizer` is a runtime-only index and execution view over those
borrowed canonical ranges. The qualification command encoded and decoded
`Hello world` successfully after sidecar reopen.

## MoE Semantics

The generic MoE directory now supports v2 tensor and scale identities while
retaining v1 decoding. Entries refer to tensor-directory key/occurrence
identities, not Safetensors source names. FP8 expert weights carry explicit
scale identities and are validated against canonical FP32 scale descriptors.

The real GLM inventory qualified with:

- Routed expert weight bindings: `17664`
- Shared expert weight bindings: `138`
- Router weight bindings: `46`
- Router score-correction bindings: `46`
- Total MoE entries: `17894`
- Routed experts: `128`
- Active experts per token: `8`
- Shared experts: `1`
- Normalized TopK: `true`
- Routing groups: `1`
- TopK routing groups: `1`
- Routed scaling factor: `1.0`

The physical tensor inventory has layer indices `0..46`. The config reports
`num_hidden_layers=46` and `num_nextn_predict_layers=1`; the sidecar preserves
the physical index space rather than silently collapsing the additional
predictor layer.

The existing rank-3 contiguous expert-bank lowering contract is not used for
this artifact. GLM stores separate rank-2 expert tensors, so the sidecar uses
the generic catalog with explicit member and scale references. No fabricated
fixed-stride bank is claimed and no payload repack was performed.

## Qualification

The real sidecar command completed successfully and verified:

- Canonical v0.6 validation.
- Persistent source-profile parsing.
- `BorrowedModel::open_persistent` close/reopen behavior.
- `36323` tensor descriptors with external source bindings.
- Tokenizer counts, added/special flags, merge storage, and runtime round trip.
- MoE entry counts, routing parameters, tensor identities, and FP8 scale identities.
- Existing Rust vBuf-ML test suite: all tests passed.
- Formatting and diff whitespace checks passed.

## Boundary

This is semantic and source/materialization qualification only. No GGML
execution, logits comparison, generation, or canonical GGML revision
qualification was performed. The sidecar is therefore not evidence of backend
numerical parity or end-to-end generation.
