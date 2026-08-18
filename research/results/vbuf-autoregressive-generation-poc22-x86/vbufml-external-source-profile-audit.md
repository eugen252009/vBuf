# Persistent vbuf-ML External Source Profile Audit

## Result

`SourceMetadata` is an additive optional vbuf-ML role (`RegionRole::SourceMetadata = 7`).
The canonical v0.6 base format is unchanged.

The profile payload stores:

```text
source descriptors:
    SourceId: u64
    declared size: u64
    locator kind + UTF-8 locator
    zero or more (algorithm: u16, raw binary digest) hashes

tensor bindings:
    existing TensorDirectory KeyId: u16
    existing occurrence: u16
    SourceId: u64
    offset: u64
    length: u64
```

Existing TensorDirectory version 1 is unchanged. Existing artifacts without
the optional role retain `SourceId::SELF` behavior and remain readable.

## Profile Gap Closed

The former structural derivatives had only canonical physical headers. They
did not contain the existing `VBUFML` bootstrap, model metadata, tokenizer,
TensorDirectory, or source binding payloads. They therefore failed normal
semantic discovery with `BootstrapNotFound`.

The new generator,
`rust/vbuf-ml/src/bin/vbuf-ml-semantic-bootstrap.rs`, copies semantic/control
payloads and replaces every tensor payload with a valid zero-length canonical
placeholder. The persistent source profile binds each placeholder to the
authoritative source range. External representation validation uses the
persisted range length and the original tensor representation/shape contract;
it does not weaken overflow or geometry checks.

Generated artifacts:

| Artifact | Full bytes | Semantic bootstrap bytes | Ratio | Reduction |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 637,925,504 | 4,438,480 | 0.00695768 | 143.73x |
| Qwen3-32B Q8_0 | 34,816,197,376 | 4,472,327 | 0.000128455 | 7,784.81x |

HTTP-locator variants are 4,438,486 bytes and 4,472,333 bytes respectively;
the six-byte difference is only the locator string length.

## 0.6B Breakdown

```text
semantic bootstrap bytes:       4,438,480
canonical blocks:                       338
bootstrap payload:                       80
model metadata payload:                 104
TensorDirectory payload:            13,933
tokenizer payload:                     164
SourceMetadata payload:             10,080
source bindings:                    9,920  (310 * 32)
source descriptor/profile overhead:   160
tensor payload bytes copied:             0
authoritative source bytes:   637,925,504
```

The remaining semantic bytes are the existing tokenizer/model control payload
and canonical headers/alignment. No tensor payload bytes are copied.

## 32B Breakdown

```text
semantic bootstrap bytes:       4,472,327
canonical blocks:                       735
bootstrap payload:                       80
model metadata payload:                 104
TensorDirectory payload:            31,900
tokenizer payload:                     164
SourceMetadata payload:             22,783
source bindings:                   22,624  (707 * 32)
source descriptor/profile overhead:   159
tensor payload bytes copied:             0
authoritative source bytes: 34,816,197,376
```

## Gates

```text
CURRENT_VBUFML_PROFILE_AUDITED: YES
VBUF_BASE_FORMAT_CHANGE_REQUIRED: NO
VBUF_ML_PROFILE_EXTENSION_REQUIRED: YES
HEADER_ONLY_IS_VALID_VBUF: YES
GENERIC_VBUF_READER_CAN_OPEN: YES
GENERIC_VBUF_STRUCTURAL_VALIDATION: PASS
SECOND_CONTAINER_FORMAT_INTRODUCED: NO
SPECIAL_NON_VBUF_PARSER_REQUIRED: NO
NANO_SEMANTICS_CHANGED: NO
MISSING_SOURCE_BINDING_MEANS_SELF: YES
SOURCE_DESCRIPTOR_PERSISTED: YES
SOURCE_ID_PERSISTED: YES
EXTERNAL_TENSORREF_PERSISTED: YES
OPTIONAL_BINARY_SOURCE_HASH_SUPPORTED: YES
HASH_REQUIRED: NO
MULTIPLE_HASHES_ALLOWED: YES
PRIMARY_HASH_SCOPE: SOURCE_ARTIFACT
MODEL_PAYLOAD_BYTES_COPIED: 0
PAYLOAD_BYTES_FETCHED_DURING_DISCOVERY: 0
```

The generated semantic artifacts are temporary research outputs under
`/tmp/opencode/header-only-bench/`; the generator and exact measurements are
kept as repository evidence. The prior 24,288-byte and 52,872-byte outputs
remain structural-only derivatives and are not relabeled.
