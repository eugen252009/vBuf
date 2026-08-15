# Q2_K Compatibility Qualification

## Root Cause

The original GGUF does not contain 82-byte Q2_K blocks. `token_embd.weight`
is GGML type 10 (`Q2_K`), with 819,200 blocks of 256 weights and 84 bytes per
block, for 68,812,800 bytes. The previous vBuf manifest and contracts used
82 bytes per block, so the converter copied only a truncated prefix while
claiming tensor parity. The pinned llama.cpp GGUF loader directly consumes the
full 84-byte payload. After correcting the geometry to 84 bytes, the vBuf
adapter exposes the same bytes and the DeepSeek model loads and decodes.

## Proven GGUF Path

`gguf_init_from_file` preserves the GGUF type. `llama-model-loader.cpp`
constructs `GGML_TYPE_Q2_K`, whose pinned traits are 256 elements and 84 bytes
per block, then maps the source bytes directly. No quantization-version branch
or 82-to-84 conversion exists. CPU repacking, when enabled, occurs after load
and is unrelated to GGUF storage.

## 84-Byte Layout

The pinned `block_q2_K` contains 16 scale/minimum bytes, 64 packed 2-bit
quant bytes, and two FP16 values (`d`, `dmin`): 4 + 16 + 64 = 84 bytes.
`general.quantization_version` is 2 and is not causal to a compatibility
conversion.

## Fix

Corrected Q2_K geometry from 82 to 84 bytes in the vBuf representation and
qualification contracts. No vBuf base format, MoE semantics, DeepSeek graph,
tokenizer, or source payload transformation was changed.

## Verification

- Source tensor: 68,812,800 bytes; 819,200 blocks; 84 bytes/block.
- Tensor payload parity: 377/377 hashes match.
- vBuf canonical validation: passed.
- Full DeepSeek vBuf load: passed; 27 layers, 64 experts, 6 active experts.
- CPU inference: passed; `tokens=3 decode=0`.
- Rust `vbuf-ml` tests: passed, including the updated 84-byte Q2_K geometry.

## Classifications

- Conversion: `CONVERSION_PARITY_CONFIRMED`
- Source representation: `CURRENT_Q2_K_WITH_ALTERNATE_STORAGE`
- Successful GGUF behavior: `GGUF_DIRECTLY_CONSUMES_82_BYTE_LAYOUT` is rejected; actual behavior is direct 84-byte Q2_K consumption.
- Correct vBuf fix: `FIX_SOURCE_TYPE_MAPPING`
- Final runtime: `DEEPSEEK_VBUF_INFERENCE_PASS`
