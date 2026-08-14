# vBuf-ML bootstrap (Step 8)

The bootstrap is an ML-profile control payload carried by one canonical v0.6
opaque byte block with reserved profile-local Key-ID `0xF000`. It is not a new
vBuf physical structure and does not replace canonical parsing.

Discovery is:

```text
canonical v0.6 parse
→ exactly one opaque block with Key-ID 0xF000
→ checked payload range
→ vBuf-ML bootstrap parse
→ role references resolved by generic Key-ID + physical occurrence
→ checked payload ranges
```

The bootstrap never repeats generic semantic, physical, width, count, alignment,
or offset facts. Those remain authoritative in the referenced canonical block
descriptor.

## Payload encoding

All fields are little-endian and all reserved fields must be zero.

Header (16 bytes):

| Offset | Width | Meaning |
|---:|---:|---|
| 0 | 8 | ASCII magic `VBUFML\0\0` |
| 8 | 2 | profile version, currently `1` |
| 10 | 2 | flags, currently `0` |
| 12 | 2 | entry count |
| 14 | 2 | reserved |

Each entry is 16 bytes:

| Offset | Width | Meaning |
|---:|---:|---|
| 0 | 2 | profile-local role ID |
| 2 | 2 | flags; bit 0 means required |
| 4 | 2 | canonical generic Key-ID |
| 6 | 2 | physical occurrence of that Key-ID |
| 8 | 8 | reserved |

The complete payload is bounded by 4096 bytes. The entry count must exactly
match the payload length.

## Initial roles

- `1`: `TensorDirectory`, required
- `2`: `ModelMetadata`, required
- `3`: `TokenizerMetadata`, optional
- `4`: `IntegrityMetadata`, optional

These entries identify regions only. Their later schemas belong to later ML
steps. A referenced block may already carry any valid generic primitive or
physical representation; the bootstrap does not retype it. The current
implementation preserves the referenced canonical block index and checked
payload range.

Unknown optional roles are skipped. Unknown required roles fail. Known roles
may occur once only. Missing required roles, missing references, unsupported
profile versions, reserved-field violations, and malformed lengths fail.

No Nano, checkpoint, region-directory, finalization, tensor, quantization,
backend, or runtime semantics are required.
