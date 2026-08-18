# Semantic Bootstrap Compute Slice

The semantic discovery, file-range, HTTP-range, and real >4 GiB gates pass.
The existing llama.cpp integration was audited but not crossed for this slice.

`integrations/llama.cpp/llama_vbuf_loader.cpp` currently stores direct FFI
payload pointers in `VbufRuntime::payloads` and attaches those pointers in
`set_tensor_data`. It assumes `vbuf_ml_consumer_open` exposes a local mmap and
does not have a SourceSet/materialization callback. Adapting that boundary
would be the next runtime/FFI task, not part of the persistent profile
extension.

```text
EXTERNAL_SOURCE_TO_COMPUTE_VERTICAL_SLICE: NOT_REACHED
LLAMA_INTEGRATION_SOURCE_AGNOSTIC_AFTER_MATERIALIZATION: NO
EXTERNAL_PAYLOAD_LIFETIME_DEPENDS_ON_METADATA_MMAP: NO at Rust SourceSet boundary
```

No model semantics, SIMD, Nano, cache, prefetch, or model-switching work was
performed. The next narrow step is to adapt the llama integration to consume
already materialized external bytes without making C++ responsible for source
I/O.
