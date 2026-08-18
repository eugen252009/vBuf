# Persistent vbuf-ML External Source Profile Design

## Additive Role

The existing bootstrap role table remains version 1 and receives one optional
role:

```text
RegionRole::SourceMetadata = 7
```

Its payload is ordinary opaque vBuf bytes with magic `VBVSRC\0\0` and profile
version 1. The role is optional, so old payload-bearing artifacts are
unchanged. A structurally valid artifact with required TensorDirectory or
ModelMetadata missing still fails normal vbuf-ML discovery.

## Source Payload

The source payload contains bounded descriptor and binding tables. Source
descriptors use binary `u64` IDs and declared sizes. Locators are descriptive
only: `SelfArtifact`, `File`, and `Http`. The parser performs no I/O.

Hashes are raw binary values with an algorithm ID and length. Zero, one, or
multiple hashes are valid. Hashes are identity metadata and do not force an
eager full-source download or verification.

## Tensor Binding

TensorDirectory v1 continues to carry semantic name, shape, representation,
KeyId, and occurrence. The optional SourceMetadata table adds a binding keyed
by that existing `(KeyId, occurrence)` pair. The parser produces the same
normalized `TensorRef` used by the runtime seam.

If no binding exists, the parser constructs the implicit `SourceId::SELF`
reference from the canonical target block. This preserves old full local
artifacts without regeneration.

For external placeholder blocks, the canonical block retains semantic,
physical, and bit-width geometry but has zero payload bytes. The external
validator checks representation geometry and expected payload length against
the persisted TensorRef. This is the only changed ownership assumption.

## Validation

- source IDs must be registered;
- source sizes are mandatory for persistent descriptors;
- offset plus length uses checked u64 arithmetic;
- range end must be within declared source size;
- source bindings must be unique;
- tensor shape, representation, continuation, and existing semantic checks remain active;
- discovery never reads a source;
- materialization still resolves through `SourceSet` and `RangeSource`.

## Runtime Boundary

`BorrowedModelView::parse_with_sources` discovers and validates the persistent
profile using the normal parser. The caller supplies the runtime SourceRegistry
and must provide matching declared sizes. `SourceSet` remains responsible for
mapping SourceId to actual RangeSource implementations. No HTTP/file branch was
added to TensorDirectory.

## Compatibility

`VBUFML` base bootstrap version and TensorDirectory version remain unchanged.
Old artifacts without SourceMetadata use self-source ranges and continue to
pass existing tests. The extension is vbuf-ML profile state, not a second
container format and not a v0.6 base change.
