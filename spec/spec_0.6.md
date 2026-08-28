# vBuf v0.6 — normative base specification

- **Status:** Normative generic vBuf base wire contract
- **Version code:** `0x00060000`
- **Byte order:** Little-endian
- **Normative terms:** MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are interpreted as in RFC 2119.

## 1. Scope and authority

This document defines the generic vBuf v0.6 canonical block stream. Earlier specifications are historical design evidence and are not alternate definitions of v0.6. Compatibility behavior is described in [`compatibility.md`](compatibility.md).

Canonical headers, counts, and checked byte ranges are authoritative. Optional finalized-file artifacts such as indexes, directories, checkpoints, integrity records, and finalization footers are not selected by this contract. A conforming v0.6 file is readable without them.

`BaseStep` is the hardware-neutral canonical physical granularity. It supports predictable block placement and efficient direct native consumption, but does not promise optimal behavior for any ISA, SIMD width, cache-line width, page size, or workload.

## 2. Integer and arithmetic rules

All integer fields are unsigned unless stated otherwise. Multi-byte fields are little-endian.

Wire calculations are performed in the mathematical `u64` domain with checked operations. A reader MUST reject before pointer, slice, or typed-view construction if an addition, multiplication, shift, ceiling division, alignment operation, file-offset conversion, or host-size conversion is not representable.

For a positive power-of-two `a`:

```text
align_up_checked(x, a) = checked_add(x, a - 1) & ~(a - 1)
```

The operation fails if `checked_add` overflows. Implementations MUST NOT use wrapping arithmetic to validate wire ranges.

For non-negative integers `n` and positive `d`:

```text
ceil_div_checked(n, d) = n / d + (n % d != 0 ? 1 : 0)
```

## 3. Global header

The minimum global header is 24 bytes.

| Offset | Size | Field | Canonical v0.6 value or meaning |
|---:|---:|---|---|
| 0 | 4 | `Magic` | bytes `56 42 55 46` (`VBUF`) |
| 4 | 4 | `Version` | `0x00060000` |
| 8 | 1 | `BaseShift` | integer in `3..=8` |
| 9 | 1 | `Flags` | bit 0 is `Indefinite`; bits 1–7 are zero |
| 10 | 2 | `HeaderSize` | total global-header bytes, at least 24 and divisible by 8 |
| 12 | 4 | `Reserved` | zero |
| 16 | 8 | `DataRegionSize` | defined below |

A reader MUST validate the complete 24-byte minimum header before deriving `BaseStep` or any offset.

```text
BaseStep = 1u64 << BaseShift
```

Legal values are therefore 8, 16, 32, 64, 128, and 256 bytes. Values outside `3..=8` are invalid. The legal set is an interoperability contract, not a preferred writer default.

`HeaderSize` permits generic header extensions. A canonical writer that emits no extensions MUST write 24. If `HeaderSize > 24`, bytes 24 through `HeaderSize - 1` contain the extension records defined in Section 3.1. The physical file MUST contain all `HeaderSize` bytes.

The canonical data-region start is:

```text
data_region_start = align_up_checked(HeaderSize, BaseStep)
```

Bytes from `HeaderSize` up to `data_region_start` are global alignment padding and MUST be zero.

### 3.1 Generic header-extension framing

Each extension record begins at an offset divisible by 8 relative to file offset zero:

| Relative offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 2 | `ExtensionType` | generic extension identifier |
| 2 | 2 | `ExtensionFlags` | bit 0 is `Required`; bits 1–15 are zero |
| 4 | 4 | `ExtensionSize` | complete record size, including this header |

`ExtensionSize` MUST be at least 8, divisible by 8, and fit entirely within `HeaderSize`. Records are contiguous; zero-length records and implicit gaps are invalid. No extension type is standardized by this base contract.

A reader MUST skip an unknown extension whose `Required` bit is zero, using its checked `ExtensionSize`. A reader MUST reject an unknown required extension. Unknown flag bits are invalid. A canonical v0.6 base writer MUST NOT emit an extension unless another specification defines its type, payload, and validation.

### 3.2 Known-size and indefinite streams

If `Flags.Indefinite == 0`, `DataRegionSize` is the exact number of bytes in the canonical data region. Checked `data_region_start + DataRegionSize` MUST equal the physical file size under this contract. A known-size empty stream has `DataRegionSize == 0`.

If `Flags.Indefinite == 1`, `DataRegionSize` MUST be zero. The physical end of the available stream is the provisional data-region end. During incremental input, an implementation MAY report that more bytes are required. Once end-of-stream is declared, every block MUST be complete and all canonical rules apply; a truncated final block is invalid.

This contract defines no finalized artifacts after the data region. A future specification may add such framing without changing canonical block bytes, but cannot make canonical parsing optional.

## 4. Canonical block anchor

Every non-empty data region begins with a block at `data_region_start`. Every block begins with one 64-bit anchor:

| Bits | Field | Meaning |
|---:|---|---|
| 0–3 | `Semantic` | representation category, Section 4.1 |
| 4–7 | `Physical` | cardinality mode, Section 4.1 |
| 8 | `Continuation` | next-block logical continuation marker, Section 4.2 |
| 9 | `Count64` | a `u64` count follows the anchor |
| 10–15 | `PayloadShift` | payload alignment multiplier exponent |
| 16–31 | `KeyID` | generic numeric block identifier |
| 32–47 | `BitWidth` | bits per element |
| 48–63 | `InlineCount` | inline element count when `Count64 == 0` |

An all-zero anchor is not padding and is invalid at a canonical block start.

### 4.1 Selected semantic and physical codes

`Semantic` codes:

| Code | Meaning | Legal `BitWidth` |
|---:|---|---|
| 0 | unsigned integer | 8, 16, 32, or 64 |
| 1 | IEEE 754 binary floating point | 32 or 64 |
| 2 | signed two's-complement integer | 8, 16, 32, or 64 |
| 3 | opaque bytes | 8 |

`Physical` codes:

| Code | Meaning | Count rule |
|---:|---|---|
| 0 | scalar | count MUST equal 1 |
| 1 | fixed-width array | count MAY be zero |

All other semantic/physical codes are reserved and MUST be rejected by v0.6 readers. Primitive byte encodings are little-endian. Floating-point payload bytes use IEEE 754 binary32/binary64 interchange encodings. Opaque bytes have no base-format meaning.

A zero-length payload is represented by a fixed-width array with count zero. A scalar never has a zero count.

### 4.2 Key IDs, duplicates, and forward continuation

`KeyID` is a generic numeric identifier. Duplicate Key IDs are legal and preserve physical stream order. The base format does not define first-wins, last-wins, or uniqueness lookup behavior.

`Continuation` points forward from the current block. If `Continuation == 1`, another canonical block MUST follow and that next block MUST have the same `KeyID`; the next block logically continues the current value. If `Continuation == 0`, the current logical chain terminates. Therefore A=1, B=1, C=0 represents three physical blocks and one logical value. The final physical block MUST clear `Continuation`. An adjacent duplicate Key-ID following a zero continuation bit is legal but begins independently.

Continuation does not alter physical geometry. Every member remains a complete canonical physical block with its own validated start, header, count, payload, and range. Any future physical-topology index marks every member's physical start independently and does not encode logical continuation. Profiles and applications define higher-level value meaning.

### 4.3 Count encoding

If `Count64 == 0`:

```text
count = InlineCount
block_header_size = 8
```

If `Count64 == 1`, eight bytes immediately after the anchor contain `ExtendedCount: u64`:

```text
count = ExtendedCount
block_header_size = 16
InlineCount MUST be zero
```

Canonical writers MUST use the inline form for counts at most 65535 and the extended form only for larger counts. Readers MUST reject non-canonical count forms.

## 5. Payload and next-block geometry

Payload alignment is orthogonal to block-start granularity:

```text
combined_shift   = BaseShift + PayloadShift
PayloadAlignment = 1u64 << combined_shift
```

`combined_shift` MUST be at most 63. Thus `PayloadAlignment` is always a power-of-two integer multiple of `BaseStep`. It never changes block-start or optional index geometry.

For a block at `block_start`:

```text
header_end    = checked_add(block_start, block_header_size)
payload_start = align_up_checked(header_end, PayloadAlignment)
payload_bits  = checked_mul(count, BitWidth)
payload_bytes = ceil_div_checked(payload_bits, 8)
payload_end   = checked_add(payload_start, payload_bytes)
```

For all selected v0.6 semantic codes, `BitWidth` is byte-sized, but the checked ceiling formula is normative.

Bytes from `header_end` to `payload_start` are payload-alignment padding and MUST be zero. `payload_start` MUST satisfy both:

```text
payload_start % BaseStep == 0
payload_start % PayloadAlignment == 0
```

The next block, when present, starts at:

```text
next_block_start = align_up_checked(payload_end, BaseStep)
```

Bytes from `payload_end` to `next_block_start` are inter-block padding and MUST be zero. A reader does not search padding for an anchor; the formula gives the only canonical next start.

Every block start MUST satisfy:

```text
block_start >= data_region_start
(block_start - data_region_start) % BaseStep == 0
```

The final block ends exactly at `payload_end`. No tail padding is required or permitted after the final payload under this contract. Consequently, a non-empty data region MAY end in a partial final `BaseStep`.

Every header, padding range, and payload range MUST lie within the declared data region for a known-size file or the available stream region at declared end-of-stream. Trailing bytes that cannot form the uniquely expected next block are invalid.

## 6. Canonical writing and reading

A canonical writer MUST:

1. emit the exact global fields and zero reserved/padding bytes;
2. use the smallest canonical count form;
3. emit only selected semantic/physical combinations;
4. compute all positions with checked arithmetic;
5. place each block and payload at the positions defined above;
6. omit final BaseStep tail padding;
7. preserve block order, duplicate Key IDs, and forward-continuation termination;
8. use `BaseStep` for physical granularity without claiming a hardware-specific default.

A conforming reader MUST reject malformed magic/version, invalid shifts, flags, extension framing, reserved fields/codes, non-canonical count forms, arithmetic overflow, non-zero padding, truncated ranges, contradictory known/indefinite length fields, and any range outside the canonical data region before exposing a typed view or pointer.

Readers MAY expose opaque bytes without interpreting profile/application meaning. Typed native views require representation, bounds, host-width, and alignment validation in addition to syntactic conformance.

## 7. Nested canonical vBuf streams

A higher-level contract MAY identify a non-empty checked subrange of a canonical
payload as a nested vBuf stream. The subrange MUST contain a complete canonical
v0.6 stream, including its own global header and data region. It is validated by
running the canonical v0.6 parser on exactly that bounded subrange; bytes outside
the subrange MUST NOT be consulted for child validity.

A nested stream has an independent local geometry, block sequence, duplicate
Key-ID rules, continuation chains, and Key-ID namespace. Its offsets and
alignment are relative to the nested stream root, so moving or extracting the
complete child byte range does not require patching its internal bytes.

The base format does not assign names, roles, tensor meaning, parent-child
relationships, or a discovery directory to nested streams. A profile or
application MUST provide any child discovery information and MUST resolve it to
a checked parent payload subrange before invoking nested validation. Generic
implementations SHOULD expose this operation as a borrowed checked range plus a
validated child view.

An implementation MUST reject empty, overflowing, out-of-parent, truncated, or
non-canonical nested ranges. Nested validation MUST preserve the same checked
arithmetic and host-range safety requirements as top-level validation.

## 8. Optional artifacts and Nano relationship

No Nano-Index, rank/select checkpoint, region directory, checksum, footer, or finalization table is selected by v0.6 base version 0.6. Such structures require separate qualification and specification.

If a future optional Nano-Index is selected:

```text
NanoSlotSize = BaseStep
```

Nano describes the already canonical geometry and cannot select, override, or duplicate `BaseStep`. Canonical headers and ranges remain authoritative.

## 8. Conformance

Normative vectors are in [`vectors/v06-vectors.json`](vectors/v06-vectors.json). The repository verifier checks BaseStep derivation, global/data alignment, block/payload formulas, partial final regions, canonical count selection, invalid shifts/alignment, and arithmetic overflow.

A v0.6 implementation is not conforming merely because it can read historical v0.5-style files. Version dispatch and legacy behavior are separate compatibility modes.
