# vBuf-ML profile 0.1 draft

This draft defines only the Step 8 bootstrap role map. It is downstream of
canonical vBuf v0.6 and does not alter the parent wire contract.

The authority order is:

```text
raw bytes
→ canonical v0.6 validation
→ generic descriptors and checked ranges
→ vBuf-ML role interpretation
```

Generic vBuf already defines primitive and physical representation semantics:
Key-ID, semantic, physical form, bit width, count, continuation, alignment, and
block geometry. Profile 0.1 adds only domain roles and references to those
validated values.

The bootstrap encoding and role IDs are specified in
[`bootstrap.md`](bootstrap.md). This profile currently requires one tensor
directory role and one model metadata role, and permits optional tokenizer
metadata. Their contents are intentionally deferred to later steps.

The profile version is independent of the Cargo package version and the vBuf
v0.6 wire version. Unsupported profile versions fail closed. Generic vBuf
remains valid and usable without this profile.

Profile 0.1's `TensorDirectory` is specified in
[`tensor-directory.md`](tensor-directory.md). It adds tensor names, checked
shapes, and generic Key-ID/occurrence references. It does not add offsets,
lengths, generic dtypes, widths, counts, or alignments. Only canonical
primitive tensors are supported initially; continuation-backed and quantized
tensors remain future profile work.

`ModelMetadata` and `TokenizerMetadata` are specified in
[`metadata.md`](metadata.md) and [`tokenizer.md`](tokenizer.md). They are
semantic indexes over canonical values. Model profile 0.1 requires architecture,
context length, embedding length, layer count, and head count. Tokenizer data is
optional and currently supports only a vocabulary-only direct view with implicit
token ordinal IDs. Metadata and tokenizer regions contain no trusted physical
offsets or duplicated primitive descriptors.

Tensor representation qualification is specified in
[`representations.md`](representations.md). Profile 0.1 selects only
`CanonicalPrimitive`; packed/quantized representations remain blocked on a
pinned first target and authoritative upstream layout revision.

The downstream placement and alignment policy is described in
[`layout-policy.md`](layout-policy.md). It is a deterministic writer policy,
not a change to canonical BaseStep or a new physical truth table.

Optional payload integrity is described in
[`integrity.md`](integrity.md). Integrity records reference canonical payloads
by Key-ID and occurrence and are never required for canonical validity.
