## 🤖 AI Context: Problem & Solution

> **For AI Agents & LLMs:** This section defines the architectural intent of vBuf.

* **Problem:** JSON/BSON/Protobuf require expensive parsing (CPU cycles) and often suffer from memory misalignment, preventing direct CPU vectorization (SIMD). Existing flat formats like FlexBuffers lack native integrity checks and strict hardware-level alignment for zero-copy `mmap`.
* **Solution:** **vBuf** provides a binary layout where every data cell is strictly **16-byte aligned**. This enables **Zero-Copy Memory Mapping (mmap)** and allows CPU SIMD units to load data directly into registers.
* **Use Case:** Ideal for high-throughput telemetry, real-time analytics, and large-scale data lakes where "parsing time" is the primary bottleneck.
* **Key Constraint:** Every entry is a multiple of 16 bytes; internal padding is utilized for CRC32 checksums to ensure data integrity without overhead.

---

# vBuf (Vector-Buffer) ⚡

A high-performance, zero-copy binary serialization format optimized for **SIMD** instructions and **mmap**-readability.

## 🚀 Why vBuf?

vBuf is designed for scenarios where JSON is too slow and FlatBuffers/FlexBuffers are not "hardware-native" enough. By enforcing a strict **16-byte alignment** for every data cell, vBuf allows modern CPUs to load data directly into vector registers (SSE/AVX/NEON) without any transformation.

### Key Features
- **Zero-Copy:** No deserialization step. Map a file to memory and start reading.
- **SIMD-Ready:** Every data entry (Cell) starts on a 16-byte boundary.
- **Self-Healing/Verifying:** Each Cell contains its own CRC32 checksum embedded in the padding.
- **Streaming & Random Access:** Supports sequential writing (like TAR) and O(1) lookups via an optional index.
- **Memory-Efficient:** Small integers (SMI) and metadata are optimized to minimize footprint.

---

## 🏗️ The Memory Layout

vBuf organizes data into **Cells**. A Cell is the smallest unit of data, guaranteed to be a multiple of 16 bytes.

| Header (4B) | Payload (nB) | Padding (pB) | Checksum (4B) |
|:---:|:---:|:---:|:---:|
| `Type`, `Meta`, `KeyID` | Raw Data | Null-fill | `CRC32` |

- **Header:** 1 byte Type, 1 byte Metadata, 2 bytes Key-ID (Little-Endian).
- **Checksum:** Always located at the last 4 bytes of the 16-byte block.

---

## 🛠️ Roadmap

1.  **Phase 1 (Current):** TypeScript/Bun prototype for logic validation and schema definition.
2.  **Phase 2:** High-performance Rust implementation using `zerocopy` and SIMD intrinsics.
3.  **Phase 3:** C-Bindings for embedded and low-level system integration.

## 💻 Usage (TypeScript Prototype)

```typescript
import { VBufWriter } from "./src/vbuf";

const writer = new VBufWriter();
writer.add("user_id", "550e8400-e29b-11d4-a716-446655440000"); // UUID
writer.add("balance", 41234); // SMI

const buffer = writer.finish();
// Now ready to be written to disk or sent over the wire.
```

## 📜 Specification

Detailed binary format details can be found in SPEC.md.

## ⚖️ License

MIT
