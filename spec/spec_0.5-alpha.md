# Specification: vBuf (Vector-Buffer) v0.5-alpha

**Project Codename:** Kraftpaket  
**Design Goal:** Maximum throughput with deterministic hardware alignment.  
**Byte Order:** Little-Endian (LE)  
**Alignment:** Dynamic Diamond Alignment ($16 \ll x$)

---

## 1. Global Header (16 Bytes)
| Offset | Type  | Name        | Description |
| :---   | :---  | :---        | :--- |
| 0      | `u32` | `Magic`     | `0x56425546` (ASCII "VBUF") |
| 4      | `u32` | `Version`   | `0x00050000` |
| 8      | `u8`  | `AShift`    | $BaseStep = 16 \ll x$. Sets the SIMD/Cache-line grid. |
| 9      | `u8`  | `Flags`     | Reserved. |
| 12     | `u32` | `DataLen`   | File size. If `0`, mode is **Indefinite Stream**. |

---

## 2. The Atomic Block Header (u64)
Every block is initiated by a 64-bit anchor. This is the "SIMD-Brake" where the parser extracts the branch logic before entering high-speed payload processing.

| Bit-Range | Name | Description |
| :--- | :--- | :--- |
| **0-3** | **SEM** | **Semantics:** `0: (u)Int`, `1: Float`, `2: String`, `3: Byte/Raw` |
| **4-7** | **PHYS** | **Physical Mode:** `0: Scalar`, `1: Array/Container` |
| **8** | **Chain** | `1` = Subsequent block belongs to the same Key-ID (Chaining). |
| **9** | **Ovrflw** | `1` = **Overflow Active**: Read 64-bit Count from the padding area. |
| **10-15** | **Res** | Reserved. |
| **16-31** | **Key-ID** | **Symbol-ID**: Mapping to the database field/column. |
| **32-47** | **PLen** | **Bit-Width**: Size of a single element in bits. |
| **48-63** | **Count** | **Quantity**: Standard max ~65K. If Bit 9 is set, ignore this field. |

---

## 3. Storage & Navigation Logic

### 3.1 The Jump Distance
The parser calculates the offset to the next 16-byte slot using the header's metadata:
$$Jump = \text{round\_up}\left(\frac{PLen \cdot Count}{8}, 16\right)$$
If **Overflow (Bit 9)** is set, the parser pulls a `u64` from the 8 bytes immediately following the header within the same slot to determine the `Count`.

### 3.2 Chaining & Resilienz
To handle "Billions" of items while maintaining CRC integrity and cache-friendliness, large datasets are fragmented into "Chained Blocks". 
*   **The Benefit:** If one block fails the CRC check, the reader can recover at the next 16-byte slot and continue with the next valid block in the chain.
*   **SIMD Efficiency:** Once the header is parsed, the payload is processed at full ISA speed (AVX2/AVX-512) until the end of the block.

### 3.3 Diamond Alignment
The Payload is guaranteed to start on the $BaseStep$ boundary.
1. **Slot Start:** Atomic `u64` Header + Optional 8-byte Metadata (Overflow/CRC).
2. **Alignment Gap:** Null-Headers (padding) are inserted by the writer.
3. **Payload Start:** Aligned to $16 \ll AShift$.

---

## 4. Summary: The "Kraftpaket" Trade-off
By accepting a small branch-penalty at the block header (the "Bremse"), vBuf gains:
1.  **Field Context:** The data is self-describing via `Key-ID`.
2.  **Hardware Flexibility:** Bit-level granularity via `PLen`.
3.  **Infinite Scaling:** Via the combination of `Chain-Bit` and `Overflow-Mode`.
