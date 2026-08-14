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
