# vBuf compatibility and specification authority

## Authority

[`spec_0.6.md`](spec_0.6.md) is the normative generic vBuf v0.6 base wire contract. Files bearing v0.6 version code `0x00060000` MUST be interpreted only by that contract.

Earlier documents and current pre-v0.6 implementations are retained as evidence. They do not collectively define an interoperable wire contract and MUST NOT be used to reinterpret malformed or inconvenient v0.6 bytes.

## Four distinct records

### 1. Historical written behavior

- v0.1 through v0.4 describe fixed 16-byte cells/slots and varying checksum/index/header schemes.
- v0.2 describes a trailing one-bit-per-16-byte Nano-Index.
- v0.4 retains Nano language but does not define complete discovery framing.
- v0.5 describes `BaseStep = 16 << AShift`, a 16-byte global header, a 64-bit block anchor, and a 32-bit `DataLen`.
- Historical documents motivate 16-byte geometry with native/SIMD claims, but do not provide portable evidence that one width is universally optimal.

These documents are historical and internally inconsistent across versions.

### 2. Current legacy implementation behavior

The pre-v0.6 Rust and TypeScript implementations derive alignment as `1 << AShift`. Rust, TypeScript, and C traversal can place/scan subsequent anchors on 8-byte boundaries. The Rust writer leaves `DataLen` zero; the TypeScript writer patches a 32-bit total file size. Their malformed/truncated-range behavior also differs.

Immutable examples and observed behavior are recorded in [`../tests/fixtures/v05/manifest.json`](../tests/fixtures/v05/manifest.json). Preserving those observations does not make unsafe behavior conforming.

### 3. Original design intent supplied during v0.6 planning

BaseStep is a generic physical-layout performance knob intended to support direct native CPU consumption, predictable physical addresses, useful alignment, SIMD-friendly layouts, vectorized traversal, and favorable cache behavior. This intent is hardware-neutral and does not select 16, 32, 64, a page size, or any ISA-specific width.

Canonical block geometry derives from BaseStep. Any future Nano geometry derives from canonical BaseStep, not the reverse. Small and composite multi-block representations remain part of the packing/padding trade-off.

### 4. Corrected normative v0.6 behavior

- exact magic bytes `56 42 55 46`;
- version code `0x00060000`;
- minimum 24-byte global header with checked `u64` data-region size;
- `BaseStep = 1 << BaseShift`, where `BaseShift` is in `3..=8`;
- BaseStep-aligned data-region and canonical block starts;
- orthogonal payload alignment as a power-of-two multiple of BaseStep;
- checked payload and next-block formulas;
- no unnecessary final BaseStep padding;
- exact known-size versus indefinite-stream rules;
- no selected finalized-file artifacts.

The legal BaseStep set is an interoperability contract, not a normative writer default or universal performance ranking.

## Version detection

A dispatcher MAY recognize the first four magic bytes and then inspect the little-endian version field:

| Version field | Status | Required handling |
|---:|---|---|
| `0x00060000` | canonical v0.6 | parse only under `spec_0.6.md` |
| `0x00050000` | legacy v0.5-style lineage | use an explicit legacy mode or report unsupported |
| any other value | not defined here | report unsupported; do not guess from layout |

Legacy and v0.6 parsing modes MUST be distinct. A file is never promoted to v0.6 because some offsets happen to satisfy v0.6 alignment.

## Legacy read policy

A conforming project MAY provide read-only legacy support where interpretation is safe and unambiguous. It MUST:

1. label the result as legacy rather than v0.6;
2. use checked arithmetic and physical-file bounds;
3. reject incomplete declared ranges before typed-view or pointer construction;
4. disclose which historical/implementation interpretation it applies;
5. avoid appending v0.6-only artifacts unless the file is fully converted and validated as v0.6.

New canonical writing targets v0.6. Byte-compatible new v0.5 writing is not required.

## Known incompatibilities

| Topic | Historical/current conflict | v0.6 resolution |
|---|---|---|
| magic notation | historical numeric notation conflicts with actual little-endian `VBUF` bytes | magic is specified as exact bytes |
| BaseStep | text says `16 << AShift`; implementations use `1 << AShift` | `1 << BaseShift`, shifts 3–8 |
| block starts | historical 16-byte slots; implementations may use 8-byte stepping | one data-region-relative BaseStep grid |
| data length | 32-bit, zero/indefinite, or total-file interpretations | checked `u64` data-region size plus explicit flag |
| global header | 16-byte historical/current layout | minimum 24-byte v0.6 layout |
| count encoding | implementations routinely force extended count | smallest canonical count form |
| payload bytes | native object memory may be written directly | only selected portable primitive/opaque encodings |
| final padding | current writers add 8-byte tail padding | no final tail padding; partial final BaseStep allowed |
| derived artifacts | incomplete historical Nano discovery/authority | none selected by base v0.6 |

Conversion is a parse-and-rewrite operation, not a version-field patch. Payload bytes may require portable re-encoding; offsets and padding generally change.
