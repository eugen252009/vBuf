# Header-Only Derived vBuf Audit

## Decision

```text
PRIMARY_MOTIVATION: REMOTE_BOOTSTRAP
RECOMMENDATION: INVESTIGATE_FURTHER
HEADER_ONLY_IS_VALID_VBUF: YES
GENERIC_VBUF_READER_CAN_OPEN_HEADER_ONLY_ARTIFACT: YES
GENERIC_VBUF_STRUCTURAL_VALIDATION: PASS
SECOND_CONTAINER_FORMAT_INTRODUCED: NO
SPECIAL_NON_VBUF_PARSER_REQUIRED: NO
```

The research derivative is a valid v0.6 block stream containing only opaque
metadata-record payloads. It is structurally opened and validated by the
ordinary `parse_v06` reader. The original artifact remains authoritative.
The fixed metadata-record interpretation is benchmark scaffolding, not a
proposed production profile or second container grammar.

The derivative reduced the 337-record Qwen3-0.6B artifact from
`637925504` bytes to `24288` bytes, but this did not produce a readiness or
structural-traversal time improvement. Canonical v0.6 validation checks headers,
ranges, and canonical padding without materializing model payloads; payload
adjacency is therefore not a measured cost in this path.

The remote bootstrap experiment is materially different: the 24288-byte
bootstrap was fetched by HTTP Range, then the first real tensor range was
resolved and fetched without downloading the source artifact. This proves the
remote transport shape, but not complete existing vbuf-ml semantic discovery.

The 32B scaling check is strongly favorable for this deployment axis: the
authoritative artifact grows from `637925504` to `34816197376` bytes (`54.58x`),
while the valid derived bootstrap grows from `24288` to `52872` bytes
(`2.18x`). The measured 32B bootstrap is `0.000151860%` of the source, or
approximately `658500x` smaller. This is a size-scaling result, not a local
latency optimization result.

## Current Path

The current consumer path is:

```text
BorrowedModelView::parse
  -> parse_v06
  -> Bootstrap::discover
  -> ModelMetadata::parse
  -> TensorDirectory::parse
  -> TokenizerMetadata::parse
```

`BorrowedModel::open` is mmap-backed and retains borrowed validated views.
`parse_v06` creates validated block descriptors before payload ranges are
exposed; it does not materialize tensor payloads. The existing 32B warm
`MODEL_READY` median is approximately `261.6 ms`, but that is a loader
readiness boundary, not a metadata-only traversal timer. The existing Step 24
0.6B warm medians were `0.044 ms` canonical validation,
`4.657 ms` borrowed-view establishment, and `5.521 ms` borrowed-model open.

The approximate 0.26 s result must not be attributed solely to header parsing.
It includes the broader model-ready setup and deferred-access policy.

## Remote Bootstrap Result

```text
FULL_MODEL_ARTIFACT_BYTES: 637925504
HEADER_ONLY_VBUF_BYTES: 24288
BOOTSTRAP_SIZE_RATIO: 0.0000380734
BYTES_TRANSFERRED_TO_METADATA_READY_HEADER_ONLY: 24288
BYTES_TRANSFERRED_TO_FIRST_PAYLOAD_REQUEST: 1138400
FIRST_TENSOR_OFFSET: 171965160
FIRST_TENSOR_LENGTH: 1114112
HTTP_RANGE_USED: YES
FULL_SOURCE_DOWNLOADED: NO
FILE_SOURCE_SUPPORTED_BY_SAME_BOOTSTRAP: YES
HTTP_SOURCE_SUPPORTED_BY_SAME_BOOTSTRAP: YES
FIRST_TENSOR_RANGE_RESOLVED_FROM_BOOTSTRAP_ONLY: YES
SOURCE_ARTIFACT_PARSED_BEFORE_RANGE_RESOLUTION: NO
FULL_SOURCE_REQUIRED_BEFORE_RANGE_RESOLUTION: NO
FULL_ARTIFACT_LOCAL_AVAILABILITY_REQUIRED: YES
FULL_ARTIFACT_NETWORK_BYTES_MEASURED: NO
```

The full-artifact local API still requires the authoritative file to be
available; full-artifact network transfer was not measured. The header-only
test used two HTTP requests: the complete 24288-byte bootstrap and one exact
1114112-byte tensor range. The same range matched a local file-source read.

Complete vbuf-ML semantic discovery parity is not yet qualified. Current
`TensorDirectory::parse` resolves tensor entries to `CheckedRange` values in
the same `ValidatedV06` byte mapping, while `RangeSource` is a runtime range
loader rather than a source identity in the semantic model. A small external
TensorRef/source-descriptor seam is still required.

## Source Identity

The smallest justified next design is an optional ordinary vBuf metadata block
for a source descriptor: URI, u64 source size, and optional binary hash headers.
SHA-256 source identity was measured for the test artifact as
`2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998`, but the
POC does not claim that digest is encoded in the generated bootstrap. A future
profile may encode the raw 32-byte digest in a normal opaque vBuf block.

```text
OPTIONAL_SOURCE_HASH_SUPPORTED: YES
HASH_STORAGE: BINARY
HASH_REQUIRED_FOR_VALID_VBUF: NO
HASH_PRIMARY_PURPOSE: SOURCE_ARTIFACT_IDENTITY
MULTIPLE_HASH_HEADERS_ALLOWED: YES
SOURCE_URI_ALONE_CONSIDERED_STABLE_IDENTITY: NO
```

## Nano

Nano is benchmark-only and interpretation-independent. It is not selected by
the v0.6 wire contract. Existing Nano reconstruction marks canonical physical
block starts; it does not encode tensor semantics or ranges. Dense header runs
can be traversed by direct arithmetic without consulting Nano. In this audit,
Nano reconstruction plus enumeration was slower than direct/scalar traversal
for every tested size and BaseStep. No Nano crossover was observed.

## Attribution

The controlled results do not confirm a time benefit from the header-only
layout. The derivative's byte-size reduction is real, but the canonical
metadata parser already avoids payload materialization. The scalar/direct
header-only path was approximately equal to canonical traversal, generic vBuf
validation was slower, SIMD was slower than scalar, and Nano was slower than
both. No hardware-counter attribution was collected; the benchmark is
warm-cache wall-clock evidence.

Remote transfer value is measured for the bounded structural POC, but complete
semantic model discovery and first useful compute remain unmeasured.

## Architecture

```text
DERIVED_FROM_AUTHORITATIVE_VBUF: YES
PAYLOAD_DUPLICATION_REQUIRED: NO
DERIVED_ARTIFACT_AUTHORITATIVE_FOR_PAYLOAD: NO
ORIGINAL_VBUF_REMAINS_AUTHORITATIVE: YES
NANO_SEMANTICS_CHANGED: NO
NANO_PROFILE_SPECIFIC_ENCODING: NO
HEADER_ONLY_NANO_USAGE: OPTIONAL
PERSISTENT_VBUF_FORMAT_DEFECT_FOUND: NO
VBUF_SOURCE_OR_RANGE_DEFECT_FOUND: NO
VBUF_RESIDENCY_DEFECT_FOUND: NO
VBUF_MATERIALIZATION_DEFECT_FOUND: NO
GENERIC_RUNTIME_ARCHITECTURE_CHANGE_REQUIRED: UNKNOWN_PENDING_EXTERNAL_TENSORREF_SEAM
U64_LOGICAL_OFFSETS_PRESERVED: YES
ARM32_FULL_ARTIFACT_MAPPING_REQUIRED: NO
```

The derivative is mechanically derived from canonical block descriptors and
preserves source ranges, type/representation fields, dimensions represented by
the benchmark record, and a source identity/size provenance field. A real
profile would also need a standardized valid-vBuf source descriptor and stale
check, preferably source URI plus artifact size/hash. The POC does not promote
that profile to the format.
