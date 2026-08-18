# Header-Only Derived vBuf Layout

## Experimental Layout

The POC derivative is `model.headers.vbuf`, generated from an authoritative
v0.6 artifact. It contains:

```text
[normal v0.6 global header]
[ordinary opaque metadata block]
[ordinary opaque metadata block]
...
```

Each metadata block has a normal v0.6 anchor and a 64-byte metadata payload.
The payload records source logical block offset, source payload offset and
length, key/representation fields, count, source identity, source size, and a
profile marker. No authoritative model tensor payload is copied. All offsets
remain u64 logical offsets.

The artifact is valid under normal v0.6 rules for BaseStep values 8, 16, 32,
64, 128, and 256. The generic v0.6 reader validates its global header, block
anchors, alignment, payload ranges, and padding without knowing that it is a
header-only derivative.

## Constraints

The v0.6 anchor alone cannot encode arbitrary external source offsets,
lengths, dimensions, and semantic identity. The POC therefore stores those
fields in ordinary opaque metadata payload bytes. This remains structurally
valid vBuf, but the metadata-record profile is deliberately experimental and
not a new normative wire format.

A production derivative would require a defined vBuf profile/source descriptor
with:

- source URI or source identity;
- source size and artifact hash/version for stale rejection;
- u64 logical payload offset and length;
- representation/type and dimensions;
- profile/schema identity in valid vBuf structure.

The original `model.vbuf` remains authoritative when the derivative is absent,
invalid, or stale. The derivative cannot be required for correctness.

## Remote Source Boundary

The existing `RangeSource` abstraction already supports positioned file reads,
and the C++ research paths have an HTTP range source. However, the current
Rust vbuf-ml semantic view resolves `TensorDirectory` entries to local
`CheckedRange` values inside the validated artifact. It does not carry a
`source_id` plus external u64 range in the semantic tensor descriptor.

Consequently the bootstrap can prove structural records and exact remote
ranges today, but cannot yet be opened as the existing complete ML semantic
view without a small external-TensorRef/source-descriptor seam. This is the
remaining investigation, not a justification for a second container format.

An optional source descriptor should remain ordinary valid vBuf structure and
may include `file:///` or `https://` URI data, u64 source size, and zero or more
optional raw binary hash headers. SHA-256 is 32 raw bytes on disk; hashes are
identity/provenance metadata and are not required for vBuf validity.

## Nano

No profile-specific Nano meaning was introduced. If Nano is present, it keeps
its existing structural meaning. For a dense valid-vBuf run, direct arithmetic
into the known block sequence is cheaper than building and scanning Nano in
the tested cases. Nano remains optional and cannot define artifact validity.
