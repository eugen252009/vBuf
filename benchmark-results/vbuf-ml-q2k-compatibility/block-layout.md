# Q2_K Block Layout

Pinned llama.cpp defines `block_q2_K` as:

- `scales[16]`: packed 4-bit sub-block scales and minima
- `qs[64]`: four packed 2-bit values per byte across the 256 weights
- `d`: FP16 super-block scale
- `dmin`: FP16 super-block minimum scale

The size is 16 + 64 + 2 + 2 = 84 bytes. There is no proven 82-byte Q2_K
layout in the source artifact or pinned runtime.
