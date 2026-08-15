# Metadata Branches

The source has `general.quantization_version = 2`. The pinned GGUF loader
records this metadata but does not use it to convert Q2_K storage. GGUF tensor
type 10 is retained as `GGML_TYPE_Q2_K`; `ggml_type_size` and `ggml_blck_size`
produce 84 and 256 respectively. Therefore:

`QUANTIZATION_VERSION_NOT_CAUSAL`

The prior failure was caused by the vBuf-side 82-byte geometry contract and
truncated source range, not by GGUF compatibility logic.
