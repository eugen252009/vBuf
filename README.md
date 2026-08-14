# vBuf (Vector-Buffer) ⚡

vBuf is a generic binary block format for checked, mmap-friendly, direct native consumption.

## Architectural intent

`BaseStep` is vBuf's hardware-neutral physical granularity. It provides predictable block starts and can support naturally aligned scalar loads, SIMD-friendly payload positions, simple physical address calculation, vectorized traversal, and favorable cache behavior. The format does not hard-code one contemporary SIMD, cache-line, or page width.

The v0.6 wire contract permits BaseStep values from 8 through 256 bytes. Choosing a writer default is a whole-system trade-off among native access, cache/traversal behavior, padding, and packing density—especially for small or composite multi-block representations. Optional indexes, if qualified later, derive their geometry from BaseStep and do not select it.

### v0.6 base properties

- exact little-endian magic/version and checked 64-bit ranges;
- power-of-two canonical block geometry;
- orthogonal payload alignment as a multiple of BaseStep;
- portable selected primitive encodings and opaque bytes;
- known-size and indefinite canonical streams;
- deterministic next-block calculation with no required final tail padding;
- no mandatory or currently selected index, checksum, directory, or finalization artifact.

The existing Rust, TypeScript, and C implementations still represent legacy v0.5-style behavior until the v0.6 safety/writer steps are implemented. Do not infer implementation conformance from publication of the specification.

---

## 🏗️ Canonical memory layout

```text
minimum global header
alignment padding to BaseStep
canonical block anchor [+ optional extended count]
padding to PayloadAlignment
payload bytes
padding to the next BaseStep block start (only when another block follows)
```

Canonical headers and checked ranges remain authoritative. See the normative specification for exact fields and formulas.

---

## 🛠️ Roadmap

1. **Current:** normative v0.6 base specification and immutable legacy evidence.
2. **Next:** checked v0.6 Rust, TypeScript, and C readers/writers with cross-language conformance.
3. **Qualification:** measure BaseStep and optional generic navigation structures before selecting defaults or artifacts.

## 💻 Legacy TypeScript prototype usage

> This example uses the pre-v0.6 prototype API and does not claim v0.6 wire conformance.

```typescript
import { VBufWriter } from "./src/vbuf";

const writer = new VBufWriter();
writer.add("user_id", "550e8400-e29b-11d4-a716-446655440000"); // UUID
writer.add("balance", 41234); // SMI

const buffer = writer.finish();
// Now ready to be written to disk or sent over the wire.
```

## 📜 Specification

- **Normative generic wire contract:** [`spec/spec_0.6.md`](spec/spec_0.6.md)
- **Compatibility and historical status:** [`spec/compatibility.md`](spec/compatibility.md)
- **Execution and qualification plan:** [`step-by-step.md`](step-by-step.md)

Specifications v0.1 through v0.5 are retained as historical evidence, not alternate definitions of v0.6.

## ⚖️ License

MIT
