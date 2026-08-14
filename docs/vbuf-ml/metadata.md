# vBuf-ML model metadata (Step 10)

Model metadata is a small semantic index over canonical generic values. It is
not a second typed-value/object system.

```text
ModelMetadata role
→ metadata key
→ generic Key-ID + physical occurrence
→ canonical descriptor and CheckedRange
→ domain validation
```

## Profile 0.1 keys

Required:

| ID | Key | Canonical value |
|---:|---|---|
| 1 | Architecture | non-empty UTF-8 byte array |
| 2 | ContextLength | nonzero unsigned scalar |
| 3 | EmbeddingLength | nonzero unsigned scalar |
| 4 | LayerCount | nonzero unsigned scalar |
| 5 | HeadCount | nonzero unsigned scalar |

Optional:

| ID | Key | Canonical value |
|---:|---|---|
| 6 | FeedForwardLength | nonzero unsigned scalar |
| 7 | NormalizationEpsilon | finite positive float scalar |
| 8 | RopeTheta | finite positive float scalar |
| 9 | KVHeadCount | nonzero unsigned scalar; optional architecture semantic |
| 10 | KeyHeadDimension | nonzero unsigned scalar; optional architecture semantic |
| 11 | ValueHeadDimension | nonzero unsigned scalar; optional architecture semantic |

This is a bounded common model configuration, not a universal architecture
schema. Architecture-specific additions are deferred.

## Wire format

Payload magic is `VBMLMD\0\0`, version `1`, little-endian.

Header: 16 bytes:

```text
magic[8], version:u16, flags:u16, entry_count:u16, reserved:u16
```

Each entry is 8 bytes:

```text
metadata key ID:u16
required flag:u16
canonical Key-ID:u16
canonical physical occurrence:u16
```

Entries are sorted by metadata key ID. Duplicate keys and nonzero reserved or
unknown flags fail. Unknown optional keys are skipped; unknown required keys
fail. The region is bounded to 4096 bytes and 256 entries.

The referenced canonical descriptor remains authoritative for semantic,
physical form, width, count, alignment, and payload range. ML parsing validates
compatibility and domain constraints only.
