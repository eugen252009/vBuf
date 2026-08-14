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
```

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
metadata API. Tensor payloads are checked against runtime-created GGML tensor
byte sizes and attached directly from the validated vBuf mmap.

## Ownership

`llama_model_load_vbuf` retains a vBuf consumer handle in an adapter registry
keyed by `llama_model *`. The handle owns the mmap and must be released with
`llama_model_free_vbuf`; GGML tensor backing pointers remain valid until then.

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

No performance claim is made. See Step-21 evidence for exact commands and
parity records.
