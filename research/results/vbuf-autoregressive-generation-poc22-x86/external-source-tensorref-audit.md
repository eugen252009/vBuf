# External Source TensorRef Seam Audit

## Result

The first generic seam is implemented in `rust/vbuf-ml/src/source.rs`,
`tensor_directory.rs`, `range_loading.rs`, and `consumer.rs`.

The current full-artifact path remains self-source compatible. The new path
accepts declared source descriptors and external tensor references without
borrowing payload bytes from the metadata mapping.

## Coupling Removed

Before this change, `TensorDescriptor` held only a
`CheckedRange<'a>`. That type proves containment by the mapping borrowed from
the current `ValidatedV06`, so tensor identity, payload source, and payload
residency were inseparable. `BorrowedModelView` and `ConsumerModel` then
exposed those bytes directly through `tensor_payload` and the FFI tensor view.

The new normalized representation is:

```text
TensorDescriptor
    -> TensorRef { SourceId, ByteRange(u64 offset, u64 length) }
    -> SourceSet / SourceResolver
    -> RangeSource
```

The local fast path additionally retains `Option<CheckedRange<'a>>`. It is
`Some` only when the payload is in the metadata artifact. External payloads
have `None`, preventing accidental use of the metadata mmap as payload proof.

## Preserved Validation

- Canonical v0.6 validation still owns the metadata artifact.
- Tensor names remain sorted and unique.
- Shape, representation, continuation, geometry, and payload-length checks remain active.
- External ranges require a registered source with a declared size.
- `offset + length` uses checked u64 arithmetic.
- Ranges beyond the declared source size fail closed.
- Unknown source IDs fail before materialization.
- `ReadPlan` never coalesces ranges from different source IDs.
- Discovery constructs references only; it performs zero source reads.

## Source Identity

`SourceId` is independent from `SourceLocator`. Locators currently represent
self, file, and HTTP/Range deployment locations. `SourceDescriptor` accepts
zero or more binary hash descriptors. Hash presence does not trigger a source
download or eager full-artifact verification; verification policy remains
outside this seam.

## Current Qualification Boundary

The existing `/tmp/opencode/header-only-bench/qwen06.headers.vbuf` and
`qwen32.headers.vbuf` artifacts pass generic vBuf parsing, but their current
experimental derivation contains physical metadata records only. It does not
contain the existing `VBUFML` bootstrap, model metadata, tokenizer metadata,
or TensorDirectory payloads required by normal semantic discovery. Running
`vbuf-ml-diagnose` therefore fails closed with `BootstrapNotFound`.

This is an artifact/profile qualification blocker, not a source-resolution
failure. The seam is intentionally exposed through
`BorrowedModelView::parse_with_sources` and
`BorrowedModel::open_with_sources`, but complete bootstrap discovery parity
must not be claimed until a valid vBuf-ML metadata-bearing bootstrap is
generated and its source bindings are encoded in the profile.

## Tests

`rust/vbuf-ml/tests/external_source.rs` covers:

- external file-locator semantics through a source-independent `RangeSource`;
- checked external materialization through `SourceSet`;
- >4 GiB u64 offset preservation without allocation;
- overflow and declared-size rejection;
- unknown-size rejection;
- optional binary multiple hashes;
- locator/identity separation;
- malformed source descriptor rejection.

No HTTP client or transport-specific parser was added. HTTP is represented by
the same `SourceLocator::Http` and `RangeSource` boundary as file sources.
