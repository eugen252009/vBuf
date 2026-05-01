# Specification: vBuf (Vector-Buffer)

**Version:** 0.4-alpha (SoA Transformer Edition)  
**Status:** Experimental / High-Throughput  
**Byte Order:** Little-Endian (LE)  
**Alignment:** Strict 16-Byte (Slot-Aligned)  
**Access Pattern:** Structure of Arrays (SoA)

---

## 1. Overview
vBuf v0.4 is a binary format designed for high-performance data interchange. By utilizing a **Structure of Arrays (SoA)** layout within data blocks, the format is natively optimized for SIMD pipelines (AVX-512, ARM NEON, RISC-V Vector). The intelligence is embedded directly into the memory layout, reducing the parser's workload to simple pointer casts and zero-checks.

---

## 2. Global Header (16 Bytes)
The header is structured to allow version and metadata validation with minimal CPU cycles.

| Offset | Type  | Name       | Description / Layout (LE) |
| :---   | :---  | :---       | :--- |
| 0      | `u32` | `Magic`    | `0x56425546` (ASCII "VBUF") |
| 4      | `u32` | `Version`  | **Layout:** `[Patch][Minor][Major][Major]` |
| 8      | `u8[4]`| `Reserved` | Padding for 16-byte alignment |
| 12     | `u32` | `DataLen`  | Total payload length in bytes |

> **Version Comparison:** Due to the `0xMajorMajorMinorPatch` layout, a `u32` cast results in a value that can be compared directly using standard operators (`>`, `<`, `==`) to determine compatibility.

---

## 3. Block Structure (SoA Container)
Data is organized into homogeneous blocks. Every block must start at a memory address where `offset % 16 == 0`.

### 3.1 Block Header (16 Bytes)
| Field       | Size    | Description |
| :---        | :---    | :--- |
| `Type-ID`   | 1 Byte  | Semantic data type identifier |
| `Meta`      | 1 Byte  | Flags for inlining or sub-type definitions |
| `Key-ID`    | 2 Bytes | Unique field identifier (Symbol-ID) |
| `ElemWidth` | 4 Bytes | Width of a single element in bytes |
| `Count`     | 4 Bytes | Number of elements in this block |
| `Block-CRC` | 4 Bytes | CRC32 covering both Header and Payload |

---

## 4. Slot-Stepping & Zero-Check Logic
The format enforces a "Dumb Parser" architecture through a 16-byte grid.

1. **Scan:** The parser reads a 16-byte slot.
2. **Zero-Check:** A cast to `u64` checks the first 8 bytes.
   - `if (val == 0)`: Slot is padding. Increment offset by 16.
   - `if (val != 0)`: Slot is a Block Header.
3. **Jump:** After processing the block, the parser jumps to the next slot based on `ElemWidth * Count` (rounded up to the next 16-byte boundary).

---

## 5. Nano-Index (Bit-Vector)
A compact index at the end of the data stream enables O(1) random access at the block level.
- **Density:** 1 bit per 16-byte slot.
- **Significance:** `1` marks the start of a Block Header, `0` marks Payload or Padding.
- **Hardware Acceleration:** Utilizes instructions like `POPCNT` (x86/RISC-V) to instantly calculate the physical address of the N-th column/key.

---

## 6. SoA Transformation (.so Interface)
Restructuring unsorted input data (AoS) into the optimized vBuf layout is handled by a shared library. This allows for platform-specific optimizations (e.g., using RISC-V Vector units on the Orange Pi).

```c
// Example: Generates an optimized SoA block
size_t vbuf_pack_block(void* target, uint16_t key_id, uint8_t type, const void* data, uint32_t count);
```

---

## 7. Hardware Advantages
- **Unaligned-Safe:** Strict 16-byte alignment natively prevents "Alignment Faults" on RISC-V hardware.
- **SIMD-Ready:** Data blocks can be processed without transformation using `vle.v` (RISC-V) or `_mm512_load_si512` (x86).
- **Resilience:** Unknown or corrupt headers do not crash the parser; the system "re-syncs" automatically at the next 16-byte boundary thanks to the stepping logic.

---
