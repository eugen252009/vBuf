# vBuf-ML partial/range loading (Step 15)

Partial loading is a runtime-local operation over canonical validated ranges.

```text
semantic selection
→ canonical TensorDescriptor / CheckedRange
→ physical ReadPlan
→ mmap or positioned reads
→ optional target-scoped integrity verification
```

## Selection

`select_tensor_names` and `select_tensor_ordinals` return semantic selections
in caller order. They contain tensor ordinal, Key-ID, occurrence, and the exact
canonical payload `ByteRange`.

The physical plan is separate and sorts ranges by file offset. Semantic result
ordering is not changed.

## Physical plans

`ReadPlan` contains:

```text
semantic targets
physical read ranges
relative slices for each target
```

Coalescing modes are:

- `None`: no adjacent-range coalescing; overlapping/duplicate ranges are still
  merged safely;
- `ExactAdjacent`: merge only touching ranges;
- `Gap(n)`: experimental local policy allowing gaps up to `n` bytes.

Merged reads never become semantic tensors. Exact target slices remain tied to
canonical payload ranges.

## Sources

- `MmapSource` exposes checked borrowed slices and can execute a plan.
- `PositionedFileSource` uses positioned file reads and allocates only selected
  physical read ranges.
- `RangeSource` is a small runtime-local abstraction.

No whole-file buffering is required by the positioned path.

## Accounting

Plans expose:

```text
semantic_bytes
physical_bytes
number of physical reads
```

Amplification is `physical_bytes / semantic_bytes` when semantic bytes are
nonzero. File headers, padding, and coalescing gaps remain physical I/O bytes,
not semantic payload bytes.

## Integrity

Integrity remains independent:

```text
selected target
→ exact canonical payload range
→ optional IntegrityMetadata::verify_target
```

A coalesced physical read does not change payload-only digest coverage.

## Canonical validation limitation

The current canonical reader accepts a borrowed full mmap and validates the
canonical block sequence without scanning payload contents. Step 15 therefore
proves selective payload reads after canonical validation. A future transport
layer may optimize how the canonical structural bytes are acquired; it must not
weaken canonical validation.

No Nano, layer semantics, persistent PhysicalRangeIndex, GGUF, or backend API is
part of this contract.
