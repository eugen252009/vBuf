# External Source TensorRef Design

## Contract

```text
SourceId
SourceDescriptor {
    id
    declared_size: Option<u64>
    locator: SelfArtifact | File | Http
    hashes: Vec<binary hash descriptors>
}
TensorRef {
    source_id
    offset: u64
    length: u64
}
```

`SourceDescriptor::checked_range` returns a bounds-validated range only when
the source size is known. `TensorRef` cannot be built from an unknown-size
descriptor. `ByteRange` preserves overflow checks, while `RangeSource` remains
the I/O boundary.

## Local and External Paths

Full local vBuf parsing creates an implicit `SourceId::SELF` descriptor whose
declared size is the validated mapping length. Tensor descriptors retain their
existing checked local range and also expose the normalized `TensorRef`.

External parsing uses `TensorDirectory::parse_with_sources` and
`BorrowedModelView::parse_with_sources`. External descriptors retain all
semantic tensor validation, but their local checked range is absent. Runtime
selection uses the `TensorRef` in either case.

`ReadPlan` carries source identity into `PhysicalRange`; adjacent/coalesced
ranges from different sources cannot merge. `execute_plan_with_sources`
resolves the source at materialization time.

## Deliberate Non-Goals

- no second container format;
- no HTTP-specific tensor type;
- no eager source reads during discovery;
- no cache, retry, prefetch, or request batching policy;
- no Nano or SIMD changes;
- no source hash eager verification;
- no model semantic changes.

## Remaining Profile Work

The current source descriptors and external bindings are runtime contracts,
not yet encoded in the existing vbuf-ML bootstrap payload. The next narrow
step is to add an optional vbuf-ML source-reference region/profile extension
that remains ordinary valid vBuf, then regenerate the header-only artifact
with its actual model metadata, tokenizer metadata, TensorDirectory, and
source bindings. Only that artifact can qualify complete discovery parity.
