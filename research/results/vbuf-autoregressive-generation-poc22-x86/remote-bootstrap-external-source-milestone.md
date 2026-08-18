# Remote Bootstrap External-Source Milestone

## Status

```text
REMOTE_BOOTSTRAP_MILESTONE: COMPLETE
COMPLETE_VBUF_ML_DISCOVERY_PARITY: PASS
LLAMA_EXTERNAL_MATERIALIZATION_BOUNDARY: COMPLETE
```

This report supersedes the earlier structural-only and pre-integration-stage
status files without deleting them. Those files remain raw historical evidence
for the hypothesis, negative local benchmark, and intermediate blockers.

## Chronology

1. The structural header-only experiment tested whether removing model payloads
   improved local traversal. It produced valid generic vBuf derivatives of
   `24,288` bytes for Qwen3-0.6B and `52,872` bytes for Qwen3-32B.
2. The local benchmark disproved a local SIMD/Nano/layout speed benefit. The
   structural derivative was smaller, but canonical traversal was already
   cheap; SIMD and Nano had no crossover.
3. The HTTP Range experiment proved that a small valid vBuf can identify and
   fetch exact authoritative ranges without downloading or parsing the source
   artifact first.
4. The generic SourceId/SourceDescriptor/TensorRef/SourceSet seam separated
   tensor identity from metadata location, payload source, and residency.
5. The additive vbuf-ML `SourceMetadata` role persisted source descriptors,
   optional binary source hashes, and external TensorRef bindings while keeping
   old TensorDirectory v1 artifacts self-source compatible.
6. Complete semantic bootstrap derivatives were generated and parsed through
   the normal vbuf-ML path with zero tensor payload bytes copied.
7. The pinned llama.cpp/embedded ggml environment was restored, and the
   source-independent pointer/length/lease boundary was qualified with real
   ggml descriptor and bounded compute parity.

## Final Architecture

```text
authoritative model.vbuf
          | exact u64 ranges
          v
Source / RangeSource <---------------- semantic-bootstrap.vbuf
          ^                                      |
          |                                      v
          |                            normal vbuf-ML discovery
          |                                      |
          |                                      v
          |                       TensorRef(SourceId, u64 offset, length)
          |                                      |
          +-------------------------- SourceSet materialization
                                                 |
                                                 v
                                  stable ptr + len + lease
                                                 |
                                                 v
                                  source-agnostic llama / ggml
                                                 |
                                                 v
                                              compute
```

The local path is the same downstream contract:

```text
full local vBuf -> mmap CheckedRange -> borrowed stable span -> llama/ggml
```

Persistent model size, bootstrap transfer size, local storage, process address
space, resident cache, active working set, and backend execution representation
remain distinct quantities.

## Measured Artifacts

| Artifact | Full bytes | Structural derivative | Semantic bootstrap | Ratio |
|---|---:|---:|---:|---:|
| Qwen3-0.6B-Q8_0 | 637,925,504 | 24,288 | 4,438,480 | 0.006957677616225232 |
| Qwen3-32B-Q8_0 | 34,816,197,376 | 52,872 | 4,472,327 | 0.00012845535518140556 |

The structural sizes are not semantic-bootstrap sizes. The semantic artifacts
contain complete bootstrap, model metadata, TensorDirectory, tokenizer, and
SourceMetadata payloads, while copying zero tensor payload bytes.

Scaling from 0.6B to 32B was measured as `54.5772x` for the full artifact,
`2.1769x` for the structural derivative, and `2.1780x` for physical blocks.
The bounded two-model classification is `REMOTE_BOOTSTRAP_SCALING:
STRONGLY_FAVORABLE`; it is not an unsupported universal asymptotic claim.

## Qualification Closure

```text
VBUF_BASE_FORMAT_CHANGED: NO
VBUF_ML_PROFILE_EXTENSION: YES
OLD_VBUFML_ARTIFACT_COMPATIBILITY: PASS
MISSING_SOURCE_BINDING_MEANS_SELF: YES
MODEL_PAYLOAD_BYTES_FETCHED_DURING_DISCOVERY: 0
MODEL_PAYLOAD_BYTES_COPIED_INTO_BOOTSTRAP: 0
COMPLETE_VBUF_ML_DISCOVERY_PARITY: PASS
FILE_SOURCE_TENSOR_BYTE_PARITY: PASS
HTTP_SOURCE_TENSOR_BYTE_PARITY: PASS
REAL_32B_GT4GIB_TENSOR: PASS
REAL_32B_GT4GIB_OFFSET: 6056603320
FULL_SOURCE_MAPPING_REQUIRED: NO
SOURCE_IDENTITY_SEPARATE_FROM_LOCATOR: YES
OPTIONAL_BINARY_SOURCE_HASH_SUPPORTED: YES
HASH_REQUIRED: NO
MULTIPLE_HASHES_ALLOWED: YES
LLAMA_EXTERNAL_MATERIALIZATION_BOUNDARY: COMPLETE
LOCAL_ZERO_COPY_OR_BORROWED_FAST_PATH_PRESERVED: YES
CPP_SOURCE_IO: NO
SOURCESET_VISIBLE_TO_CPP: NO
GGML_DESCRIPTOR_PARITY: PASS
EXTERNAL_WEIGHT_COMPUTE_OUTPUT_PARITY: PASS
COMPUTE_OPERATION: ggml_sum
COMPUTE_LOCAL: 220.409927
COMPUTE_EXTERNAL_FILE: 220.409927
COMPUTE_EXTERNAL_HTTP: 220.409927
COMPUTE_MAX_ABS: 0
COMPUTE_MAX_REL: 0
COMPUTE_MEAN_ABS: 0
COMPUTE_FIRST_DIFF_INDEX: NONE
```

The real ggml descriptor hash was `b0b2fe73f3d84bfa` for both local and
external Qwen3-0.6B tensor descriptors. The real Qwen3-32B tensor at logical
offset `6,056,603,320` crossed the descriptor/materialization boundary without
mapping the 34.8 GB source.

## Negative Local Result

The local-layout benchmark remains negative and is intentionally not rewritten:

```text
LOCAL_LAYOUT_LATENCY_BENEFIT: NO
SIMD_BENEFIT: NO
NANO_BENEFIT: NO
SIMD_CROSSOVER: NONE
NANO_CROSSOVER: NONE
```

The remote/bootstrap result is justified by deployment and source indirection,
not by local parsing speed.

## Provenance

The real llama integration used pinned llama revision
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` and the ggml embedded in that exact
checkout (`ggml commit 4c1a0af40`). The canonical prepared-tree delta hash was
`b365448a51b9e2801d8b86269317e9396a975f19c9562d9534ed34de25e7cb38`.

Detailed artifact, semantic, source, FFI, ggml, and HTTP evidence remains in
the adjacent JSON/Markdown files. Previous model-convergence divergence
evidence is unrelated and remains unchanged.
