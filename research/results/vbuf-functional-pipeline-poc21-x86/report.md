# POC21 Functional Pipeline

Artifact: `research-models/DeepSeek-V2-Lite.IQ1_S.vbuf`

## Acceptance Results

- Real token embedding rows for IDs 0 and 1: exact parity after native dequantization.
- Transformer body: `blk.0..blk.26`, both positions, exact reference parity.
- Activation handoff: zero-copy, `activation_boundary_copy_bytes=0`.
- Final RMSNorm and quantized output projection: exact logits parity.
- Vocabulary: `102400` logits.
- Greedy next token: token 0 -> `86711`; token 1 -> `86711`.
- Per-layer runtime state isolation: PASS.
- Unselected expert graph/tensor/source activity: zero.
- Model artifact mutation: NO.
- Execution-prep copy/repack/transcode bytes: zero.

## Milestone Gates

```text
POC21_FUNCTIONAL_INFERENCE_PIPELINE: PASS
INPUT_EMBEDDING: PASS
FULL_TRANSFORMER_BODY: PASS
FINAL_NORMALIZATION: PASS
LM_HEAD: PASS
LOGITS_REFERENCE_PARITY: PASS
NEXT_TOKEN_PARITY: PASS
TOTAL_TRANSFORMER_BLOCKS_EXECUTED: 27
VOCABULARY_SIZE: 102400
REFERENCE_ARGMAX_TOKEN_ID: 86711
VBUF_RUNTIME_ARGMAX_TOKEN_ID: 86711
REAL_ACTIVATION_FORWARDING: PASS
ARCHITECTURE_SPECIFIC_GENERIC_RUNTIME_LOGIC: NO
MODEL_ARTIFACT_MUTATED: NO
VBUF_LAYOUT_CHANGE_REQUIRED: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
GGML_COMPUTE_PATH_CHANGED: NO
VBUF_RUNTIME_OWNS_MODEL_LIFECYCLE: YES
```

## Residency

- Policy: `COST_AWARE`.
- Capacity: `268435456` bytes.
- Peak resident: `268434432` bytes.
- Logical persistent bytes: `4777135104`.
- Peak active persistent bytes: `12607488`.
- Loads: `3061`; hits: `6143`; misses: `5991`.
- Evictions: `2895`; reload events: `2264`; reload bytes: `1835014464`.
- Policy decisions: `2895`; candidate evaluations: `540244`.

The full raw qualification output is captured during execution in `/tmp/poc21-end-to-end-v3.log`.

## Verification

- Affected target build: PASS.
- Full x86 CTest suite: PASS (qualification recorded for the milestone commit).
- `git diff --check`: PASS.
- `ccc index`: PASS, zero indexing errors.
- Cleanup/failure injections: not exercised by this functional run; existing runtime contract tests PASS.

## Scope

This milestone means functional model execution only. It does not claim tokenizer completeness, sampler completeness, chat runtime completeness, production readiness, performance optimization, cache optimality, GPU completion, ARM32/RV2 readiness, or large-model readiness.
