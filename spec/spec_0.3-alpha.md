# vBuf Specification v0.3-alpha

## 1. Physical Structure
The `vBuf` format version 0.3-alpha defines the physical structure as follows:
- **Global Header:** Contains Magic `VBUF`, Version `0.3`, and overall DataLength.
- The header is always aligned to a 16-byte boundary, following vector-alignment rules for efficient zero-copy operations.

## Specification Details
### 1. Global Structure

The global header includes basic metadata:
- Magic: `VBUF`
- Version: `0.3-alpha`
- Total Length of Data (excluding the header)

### 2. Basic Cell Structure and Alignment

Cells within a vBuf structure adhere to strict alignment rules for efficient data access:
- **Slot Size:** Each cell must begin at an offset where `offset % 16 == 0`.
- The structure ensures zero-copy operation by ensuring all cells align with vector instructions.

### 3. Inlining Rules and Padding (Enhanced v0.3-alpha)
In version 0.3-alpha, vBuf optimizes small data types to minimize RAM pressure:
- **Zero-Key Inlining:** If `KeyLen == 0`, the payload starts immediately at **Offset 8** (directly after the 64-bit header).
- **Single-Slot Rule:** If `(8B Header + ValLen + 4B CRC) <= 16B`, the entire cell fits into one 16-byte slot.
    - This is the "Fast-Path" for SMI (i32), Floats, and Booleans.
- **Padding:** Any cell exceeding 16 bytes or not ending on a 16-byte boundary is padded with zeros. The next cell MUST start at `current_offset % 16 == 0`.

### CRC Positioning
For version v0.3-alpha:
- **CRC Location:** The Checksum (CRC) is located at the end of each cell's data block, precisely positioned at the last 4 bytes of every slot or cell boundary.
  
## Notes for Upgrade to Future Versions
The structure defined here in v0.3-alpha lays the foundation for future enhancements and compatibility with subsequent versions like v0.4 where more detailed rules and optimizations are implemented.

### Compatibility
- Ensure that your implementation adheres strictly to these alignment and size rules to facilitate seamless upgrades or integration with later versions of `vBuf`.
