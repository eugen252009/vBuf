# llama.cpp External Materialization Boundary Audit

## Existing Path

```text
vbuf_ml_consumer_open(path)
  -> BorrowedModel::open
  -> BorrowedModelView / local mmap
  -> vbuf_ml_consumer_tensor_info
  -> VbufMlTensorInfo.payload (raw pointer)
  -> VbufRuntime::payloads[name]
  -> ggml_tensor::data
  -> llama_model_init_from_user callback
```

The direct assumptions were:

- `consumer_ffi.rs:66-67`: payload pointer and length are exposed as an FFI span;
- `consumer_ffi.rs:170-175`: tensor info requires a locally mapped payload;
- `vbuf_ml_adapter.cpp:37-39`: C++ stores the pointer without a separate lease;
- `llama_vbuf_loader.cpp:30-31, 56`: runtime stores raw pointer/size pairs;
- `llama_vbuf_loader.cpp:171-175`: callback assigns the pointer directly to `ggml_tensor::data`.

`llama_model_free_vbuf` calls `llama_model_free` before deleting `VbufRuntime`,
so the existing direct path keeps its mmap alive for the complete llama model
lifetime. The callback path does not copy the pointer into a Rust-owned buffer.

## Boundary Added

The additive FFI contract now separates descriptor discovery from payload
binding:

```text
vbuf_ml_consumer_tensor_descriptor
    -> metadata + expected length + null payload

vbuf_ml_consumer_tensor_info_with_bytes
    -> explicit materialized pointer + exact checked length
```

`vbuf_ml_consumer_open_metadata` opens a persisted semantic bootstrap through
normal vbuf-ML parsing and performs no source I/O. C++ can use
`VbufMlAdapter(path, true)` and `tensor_materialized(...)` to bind an already
materialized span. A `shared_ptr<const void>` lease is carried with the C++
descriptor.

The loader's local map now stores a `PayloadBinding` containing pointer, byte
length, SourceId, source offset, and an optional lease. The local path still
uses the mmap pointer without copying.

## Lifetime

The current `llama_model_init_from_user` callback retains `ggml_tensor::data`
through model construction and subsequent model use. For the existing loader,
the required lifetime is therefore model lifetime. A future external loader
owner must retain the materialization lease until `llama_model_free_vbuf`
returns. The new descriptor lease makes that ownership explicit; no automatic
release occurs at descriptor construction.

No C++ file, HTTP, NAS, SourceId resolution, or SourceSet logic was added.

## Qualification Boundary

Rust FFI tests pass for descriptor-only discovery, explicit materialized-span
binding, exact length validation, and local compatibility. The repository does
not contain the external llama.cpp/ggml headers or a build configuration for
the integration directory, so the real ggml callback and compute path could
not be compiled or executed in this workspace.
