# ADR 0001: vBuf wire authority and compatibility direction

- **Status:** Accepted
- **Scope:** Generic vBuf BASE
- **Related plan decisions:** A, C, G

## Context

The historical specifications and current implementations disagree about magic-number notation, version/header fields, `DataLen`, anchor alignment, and BaseStep derivation. In particular:

- `spec/spec_0.5-alpha.md` specifies `BaseStep = 16 << AShift`.
- Rust and TypeScript derive alignment as `1 << AShift`.
- Current Rust, TypeScript, and C traversal can place/scan anchors at 8-byte boundaries.
- v0.4/v0.5 use a 32-bit `DataLen`, while large generic downstream files require checked 64-bit ranges.

Treating those bytes as an unambiguous parent contract would make independent descendants non-interoperable.

## Decision

1. A corrected generic vBuf v0.6 is the only parent-format path that may unblock new descendant/profile work.
2. Current v0.5-style bytes are legacy evidence and may remain readable only where their interpretation is safe and unambiguous.
3. New canonical writing for this implementation path targets v0.6; byte-compatible new v0.5 writing is not required by vBuf-ML.
4. vBuf-ML remains blocked until v0.6 is specified and validated.
5. v0.6 encodes the canonical geometry as `BaseStep = 1 << BaseShift` with legal `BaseShift` values 3 through 8 inclusive (8–256 bytes). Out-of-range shifts are rejected before offset derivation or view construction.
6. `data_region_start` and every canonical block start are BaseStep-aligned. The next block starts at checked `align_up(previous_block_end, BaseStep)`; no final tail padding is required.
7. Stricter payload alignment is an orthogonal power-of-two multiple of BaseStep and cannot redefine block-start or Nano geometry.

## Invariants

- Legacy parsing and v0.6 canonical parsing are distinct modes.
- Legacy bytes are never relabeled as v0.6 merely because one offset happens to align.
- Sizes, offsets, alignment arithmetic, and platform conversions are checked.
- BaseStep is selected from generic physical-layout requirements, never downstream ML convenience.
- Optional Nano geometry, if later selected, derives from BaseStep and has no independent quantum.
- Optional derived artifacts cannot repair or redefine an incompatible canonical stream.

## Consequences

- Step 2 may preserve exact legacy behavior as fixtures.
- Step 3 is blocked on the BaseStep subdecision.
- A legacy file must satisfy a complete, explicit conversion policy before receiving v0.6-only finalized artifacts.
- Existing generic use cases remain compatibility requirements, not reasons to preserve ambiguous semantics.

## Rejected alternative

Using ambiguous current v0.5 behavior directly as the vBuf-ML parent is rejected because readers can derive different physical offsets from the same header bytes.
