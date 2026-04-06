# Specification: vBuf (Vector-Buffer)

**Version:** 0.1-draft  
**Status:** Experimental  
**Byte Order:** Little-Endian (LE)  
**Alignment:** 16-Byte (SIMD-Optimized)

---

## 1. Overview
vBuf is a binary serialization format designed for high-performance data exchange and zero-copy random access. Unlike traditional formats, vBuf enforces a strict **16-byte alignment** for every data element ("Cell"). This layout is specifically optimized to be loaded directly into CPU vector registers (SIMD) without alignment penalties.

## 2. Global Header (16 Bytes)
Every vBuf stream or file begins with a 16-byte header to ensure compatibility and provide structural metadata.

| Offset | Type | Name | Value / Description |
| :--- | :--- | :--- | :--- |
| 0 | `u32` | `Magic` | `0x56425546` (ASCII "VBUF") |
| 4 | `u8` | `Version` | `0x01` |
| 5 | `u8` | `Flags` | Bit 0: HasIndex, Bit 1: IsStreaming |
| 6 | `u16` | `Reserved` | `0x0000` (Reserved for future use) |
| 8 | `u64` | `DataLen` | Total length of the data payload (excluding header) |

---

## 3. The Data Cell (Cell)
The fundamental unit of vBuf is the **Cell**. Every Cell **must** start at a buffer offset divisible by 16 (`offset % 16 == 0`).

### 3.1 Cell Structure (Integrated Checksum)
A Cell is a contiguous block of memory, always a multiple of 16 bytes. 
The checksum is NOT optional and acts as the final tail of every Cell block.

| Component | Position | Size | Description |
| :--- | :--- | :--- | :--- |
| **Header** | Start | 4 Bytes | Type (1B), Meta (1B), Key-ID (2B) |
| **Payload** | Header + 0 | *n* Bytes | Raw data values |
| **Internal Padding** | Variable | *p* Bytes | Null-padding (only if payload + header + checksum < 16B) |
| **Checksum** | End - 4 | 4 Bytes | **CRC32 / XXH3** of [Header + Payload + Internal Padding] |

**Constraint:** Total_Cell_Size = (4 [Header] + n [Payload] + p [Padding] + 4 [Checksum]) 
The value of `p` is chosen such that `Total_Cell_Size % 16 == 0`.

### 3.2 Inline-SMI Optimization
For small integers (SMI) that fit within 16 bits, the value can be stored directly within the `Key-ID` or `Meta` field. In such cases, the `Meta` byte is set to a specific flag (e.g., `0xFF`), and the Cell occupies exactly 16 bytes with no additional payload.

---

## 4. Data Types & Encoding

| ID | Name | Payload Format | SIMD Load Strategy |
| :--- | :--- | :--- | :--- |
| 0x01 | **SMI** | `i32` / `u32` (LE) | `_mm_loadu_si32` |
| 0x02 | **BigInt** | `i64` (LE) or VarInt | `_mm_loadu_si64` |
| 0x03 | **String** | UTF-8 (Length-prefixed) | Vectorized Comparison |
| 0x04 | **UUID** | 16-byte raw data | `_mm_load_si128` |
| 0x05 | **Null** | None | N/A |

---

## 5. Integrity & Bit-Rot Protection
vBuf provides native resilience against memory corruption. Every Cell is self-validating via the checksum embedded in its padding. If a single field's checksum fails, the error is localized, and the rest of the buffer remains addressable and valid.

## 6. SIMD Access Pattern (Implementation Note)
Due to the mandatory 16-byte alignment, a Cell can be loaded with a single instruction in Rust or C:
- **x86_64:** `_mm_load_si128`
- **ARM64 (NEON):** `vld1q_u8`

This allows the parser to validate types, extract keys, and verify checksums for multiple fields in parallel using vector operations.
