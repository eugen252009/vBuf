# External Source u64 Offset Qualification

The contract test constructs a source with declared size `2^40` and a tensor
reference at offset `2^32 + 7`, length `9`. The exact u64 value survives
construction and is exposed to range planning without allocating or mapping a
large source.

```text
U64_OFFSET_ABOVE_4GIB: PASS
USIZE_TRUNCATION: NO
FULL_ARTIFACT_MAPPING_REQUIRED: NO
OVERFLOW_REJECTION: PASS
DECLARED_SIZE_BOUNDS_REJECTION: PASS
UNKNOWN_SIZE_REJECTION: PASS
```

Host conversion is deliberately deferred to the concrete `RangeSource` read
boundary, where destination allocation and host indexing are checked. The
logical source offset remains u64 throughout `SourceDescriptor`, `TensorRef`,
`PhysicalRange`, and `RangeSource::read_exact_at`.
