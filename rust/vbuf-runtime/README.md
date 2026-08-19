# vbuf-runtime

This crate is the vBuf-owned execution control plane. It is intentionally
separate from llama.cpp model loading:

```text
vbuf-ml validation and mmap metadata
    -> bounded tensor range acquisition
    -> layer/expert residency and eviction
    -> ggml kernels through the optional FFI boundary
```

`VBufRuntime::open` validates metadata without reading model payloads into an
owned whole-model buffer. `acquire_layer` reads only the selected layer ranges,
and the configured resident-byte budget evicts older ranges before insertion.
`release_layer` provides the explicit lifetime boundary required by a future
architecture executor.

Enable the ggml link boundary for a prepared ggml build with:

```bash
VBUF_GGML_LIB_DIR=/path/to/ggml/lib \
  cargo check --manifest-path rust/Cargo.toml -p vbuf-runtime --features ggml
```

The DeepSeek graph executor is the next layer. The existing POC22 graph is the
reference for that port; this crate does not depend on a llama model object.
