# Specification: vBuf (Vector-Buffer)

**Version:** 0.2-draft (High-Throughput Edition)  
**Byte Order:** Little-Endian (LE)  
**Alignment:** Strict 16-Byte (Vector-Aligned)  
**Index-Ratio:** 1 Bit per 16 Bytes (1:128 Storage Overhead)

---

## 1. Overview
vBuf is a binary Intermediate Representation (IR) designed for multi-terabyte data streams and zero-copy execution. It treats memory as a continuous array of 16-byte "Slots". By enforcing strict alignment, vBuf allows for **O(1) random access** via a trailing bit-mask index and **O(N) hardware-accelerated scans** using SIMD (AVX-512/NEON) instructions.

---

## 2. Global File Structure
A vBuf entity (file or stream) consists of three contiguous logical sections:

1.  **Global Header (16 Bytes):** Metadata, versioning, and structural flags.
2.  **Data Payload (N Bytes):** A sequence of 16-byte aligned Cells.
3.  **Nano-Index (M Bytes):** A trailing bit-vector (Bit-Mask) describing the payload structure.

---

## 3. Global Header (16 Bytes)

| Offset | Type | Name | Description |
| :--- | :--- | :--- | :--- |
| 0 | `u32` | `Magic` | `0x56425546` (ASCII "VBUF") |
| 4 | `u8` | `Version` | `0x02` |
| 5 | `u8` | `Flags` | Bit 0: HasIndex, Bit 1: IsStream, Bit 2: StrictChecksum |
| 6 | `u16` | `Reserved` | `0x0000` (Reserved for future extensions) |
| 8 | `u64` | `DataLen` | Total length of the Data Payload section in Bytes |

---

## 4. The Data Cell (Slot Logic)
The fundamental unit is the **Slot** (exactly 16 bytes). A **Cell** consists of one or more Slots.

### 4.1 Cell Layout
- **Start:** Every Cell **must** start at an offset where `offset % 16 == 0`.
- **Header (4 Bytes):** - `Type` (1B): Semantic identifier (String, SMI, UUID, etc.)
    - `Meta` (1B): Flags (Inline-data flag, compression, or custom tags)
    - `Key-ID` (2B): Reference to a Symbol Table or Schema-ID.
- **Payload:** Raw data following the header.
- **Tail:** If the Cell spans multiple slots or `StrictChecksum` is enabled, the last 4 bytes of the final slot contain a CRC32/XXH3 checksum.

### 4.2 Size Calculation
The total size of a Cell is always rounded up to the next 16-byte boundary:
`Total_Slots = ceil((Header + Payload + Checksum) / 16)`

---

## 5. The Nano-Index (Bit-Vector)
The Nano-Index is a compact "map" of the Data Payload. Each bit represents exactly one 16-byte Slot.

- **Bit = 1:** Indicates the **Start** of a new Cell.
- **Bit = 0:** Indicates a **Continuation** of the current Cell or Empty/Padding space.

### 5.1 Storage Logic
The index is stored at the end of the payload. For a payload of `DataLen` bytes, the Nano-Index size is exactly `DataLen / 128` bytes.

**Example Mapping:**
- Payload: `[Cell A (16B)] [Cell B (32B)] [Padding (16B)] [Cell C (16B)]`
- Slots: `[S1] [S2] [S3] [S4] [S5]`
- Bits:  ` 1    1    0    0    1  ` (Binary: `11001...`)

---

## 6. Arithmetical Addressing & SIMD
vBuf is optimized for modern CPU instruction sets.

### 6.1 Random Access (O(1))
To find Item #N without linear parsing:
1. Load the Nano-Index into RAM.
2. Use `POPCNT` or bit-scanning instructions to find the position of the N-th set bit (`1`).
3. `Physical_Offset = Bit_Position * 16`.

### 6.2 SIMD Processing (AVX-512 / NEON)
- **Parallel Loading:** Use 512-bit registers to load 4 Cells simultaneously.
- **Instant Validation:** Mask out the first byte of each slot to check types across 4-16 items in a single clock cycle.
- **Zero-Copy:** Data can be memory-mapped (`mmap`) and cast directly to Rust structs/slices.

---

## 7. Implementation Notes
- **Empty Slots:** An empty slot is represented by a `0` bit in the Nano-Index and (optionally) a `Type = 0x00` in the data.
- **Streaming:** In streaming mode (`Flag Bit 1`), the Nano-Index may be omitted or sent in periodic chunks (Segments).
- **Endianness:** All multi-byte integers (u16, u32, u64) are stored in **Little-Endian**.
