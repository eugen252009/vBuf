# Step 21 pinned llama.cpp vBuf consumer

Pinned base:

```text
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

The contained adapter uses the pinned public `llama_model_init_from_user`
source seam. It creates a GGUF-compatible **in-memory model description** from
validated vBuf-ML descriptors, then calls the ordinary pinned model creation,
vocabulary, tensor-registration, graph, GGML, and decode paths.

Files:

```text
llama_vbuf_loader.{h,cpp}       source adapter and ownership wrapper
vbuf_ml_adapter.{h,cpp}         GGML representation/descriptor bridge
```

The only pinned llama.cpp patch is:

```text
patches/llama.cpp/0001-user-metadata-tensor-source.patch
patches/llama.cpp/0002-source-neutral-model-source.patch
patches/llama.cpp/0003-copy-user-tensor-data-to-backend.patch
```

The historical patches above are retained as research evidence. They are not a
reproducible cumulative application sequence: 0002 overlaps the loader changes
in 0001, and 0003 has stale unified-diff counts. The canonical fresh-checkout
preparation is the consolidated delta
[`patches/llama.cpp/0000-pinned-step21-canonical.patch`](../../patches/llama.cpp/0000-pinned-step21-canonical.patch)
applied by:

```bash
python3 scripts/prepare_step21_llama.py \
  --upstream-root /tmp/llama.cpp-step21
```

The checkout must already be at
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c` and clean. The preparation script
fails closed on a wrong pin or dirty tree, applies one exact patch with
`git apply --check`, and verifies the prepared source delta SHA-256 as
`b365448a51b9e2801d8b86269317e9396a975f19c9562d9534ed34de25e7cb38`.

Build the prepared tree and run the existing Qwen regression with:

```bash
cmake -S /tmp/llama.cpp-step21 -B /tmp/llama.cpp-step21-build \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=ON
cmake --build /tmp/llama.cpp-step21-build --target llama -j2
cargo build --manifest-path rust/Cargo.toml -p vbuf-ml
python3 scripts/qualify_step21_runtime.py \
  --upstream-root /tmp/llama.cpp-step21 \
  --build-dir /tmp/llama.cpp-step21-build
```

The qualification script requires a prepared tree, includes the pinned
`src` headers, and links `vbuf_direct_source.cpp` so the adapter target has all
declared loader symbols.

It makes the existing user-metadata path account for tensor inventory/bytes,
reject absent optional tensors correctly, preserve duplicated-token-embedding
fallback, and count user tensors. GGUF loading is unchanged.

## Mapping

```text
CanonicalPrimitive → GGML_TYPE_F32
BF16               → GGML_TYPE_BF16
GGML_Q8_0          → GGML_TYPE_Q8_0
```

Model metadata, vocabulary, token types/scores, numeric merge ranks, special
IDs, `add_bos`, and chat-template storage are projected into the existing GGUF
metadata API. Local tensor payloads are checked against runtime-created GGML
tensor byte sizes and retain the validated vBuf mmap fast path. The additive
materialization boundary also accepts an already-materialized external span
with an exact length and lease; source resolution remains outside C++.

## Ownership

`llama_model_load_vbuf` retains a vBuf consumer handle in an adapter registry
keyed by `llama_model *`. The handle owns the local mmap and must be released
with `llama_model_free_vbuf`; GGML tensor backing pointers remain valid until
then. External materialized descriptors carry an independent lease through the
same source-independent pointer/length boundary. C++ does not resolve
`SourceId`, locators, or ranges.

## Qualification

Pinned CPU build and actual runtime qualification passed for both BF16 and
Q8_0 vBuf artifacts:

```text
BF16 structural load: PASS
Q8_0 structural load: PASS
BF16 tokenizer token IDs: PASS
Q8_0 tokenizer token IDs: PASS
BF16 first-token logits: max abs diff 0
Q8_0 first-token logits: max abs diff 0
BF16 greedy generation: PASS
Q8_0 greedy generation: PASS
```

The Step-23 source-neutral direct path is retained. Step 24 changes only its
vBuf side: validated tokenizer arrays are now borrowed directly from the mmap
instead of Rust-owned token/merge view tables. The source-neutral
`llama_model_source` seam and common runtime remain unchanged. Step-24 evidence
is under `benchmark-results/vbuf-ml-step24/` and the architecture report is
`docs/vbuf-ml/step24-borrowed-runtime-views.md`.

No GPU, cold-cache, or first-touch performance claim is made.
