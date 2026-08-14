# vBuf-ML tensor directory (Step 9)

The tensor directory is the semantic payload of the bootstrap's
`TensorDirectory` role. It is parsed only from that checked semantic region;
it does not rediscover candidate directories.

Profile 0.1 supports the generic canonical primitive plus the qualified
profile-local BF16 and GGML_Q8_0 opaque-byte representations. Each tensor still
uses one physical canonical block. Continuation-backed and other packed layouts
remain unsupported.

## Authority

```text
canonical v0.6 validation
→ Bootstrap::discover
→ checked TensorDirectory region
→ tensor entry parse
→ Key-ID + occurrence resolution
→ canonical block descriptor + CheckedRange
```

The directory stores no offsets, lengths, generic counts, widths, semantic
values, physical forms, or alignments. Those remain canonical vBuf facts.

## Wire layout

The directory is a little-endian payload with magic `VBTDIR\0\0`.

Header: 20 bytes:

| Offset | Width | Meaning |
|---:|---:|---|
| 0 | 8 | magic |
| 8 | 2 | directory version, currently `1` |
| 10 | 2 | flags, currently zero |
| 12 | 4 | tensor count |
| 16 | 4 | reserved, zero |

Each entry is variable-sized:

```text
u16 name length
u8  rank
u8  representation ID
u16 generic Key-ID
u16 generic physical occurrence
u16 reserved, zero
[name bytes, UTF-8]
[rank × u64 little-endian dimensions]
```

The maximum directory payload is 16 MiB, the maximum tensor count is
1,000,000, names are 1..=4096 bytes, and rank is 0..=16. Names are strictly
increasing by UTF-8 byte order. The encoder sorts entries; the reader validates
ordering. Scalar tensors have rank zero. Dimensions must be nonzero.

The exact structural overhead is 20 bytes plus 10 bytes per entry, plus name
bytes and 8 bytes per dimension. For `n` tensors the fixed overhead is
`20 + 10n` bytes.

## Representation and shape

Representation ID `0` means `CanonicalPrimitive`; it is not a duplicate dtype
field. The referenced canonical descriptor supplies semantic, physical form,
bit width, count, and payload range.

For profile 0.1, shape validation computes the checked product of dimensions
(`1` for rank zero), then requires:

```text
canonical count == logical element count
payload length == count × (canonical bit width / 8)
```

The rule applies only to the supported primitive representation. Quantized and
packed representations require separate ML-local layout identities and are not
silently treated as primitive tensors.

## In-memory API

`TensorDescriptor` contains only ML identity and shape plus derived canonical
identity/range:

```text
name
 dimensions
 representation
 Key-ID + occurrence
 canonical block index
 CheckedRange
```

Exact-name lookup uses the validated sorted order and binary search. Iteration
preserves that deterministic order.

No model metadata, tokenizer data, GGUF conversion, runtime placement,
backend, kernel, or device field is part of this contract.
