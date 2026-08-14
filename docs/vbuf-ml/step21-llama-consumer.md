# Step 21: pinned llama.cpp consumer adapter

Status: **consumer correctness parity passed for BF16 and Q8_0**.

Pinned base:

```text
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

## Integration seam

The adapter uses the pinned public `llama_model_init_from_user` entry point:

```text
vBuf-ML Rust validation/C ABI
→ in-memory GGUF-compatible model description
→ llama_model_init_from_user
→ existing llama_model / GGML model construction
→ existing tokenizer, graph, kernels, decode, and sampling paths
```

The GGUF path remains `llama_model_load_from_file`. No downstream
`is_vbuf` branches were added. The small pinned patch is recorded at
`patches/llama.cpp/0001-user-metadata-tensor-source.patch`; it only makes the
existing user-metadata path account for tensor count/bytes, reject absent
optional tensors, preserve duplicate-tensor fallback, and count user tensors.

## Mapping and ownership

`llama_vbuf_loader.cpp` projects validated vBuf data into the existing GGUF
metadata API. It maps:

```text
CanonicalPrimitive → GGML_TYPE_F32
BF16               → GGML_TYPE_BF16
GGML_Q8_0          → GGML_TYPE_Q8_0
```

It maps Qwen3 metadata, vocabulary/token types/scores, numeric merge ranks,
special IDs, `add_bos`, and tokenizer identities. Merge IDs are reconstructed
into the pinned runtime's string-keyed BPE map without changing portable
storage.

The Rust handle owns the mmap. A registry retains it for each returned
`llama_model *`; callers must use `llama_model_free_vbuf`. Tensor payload
pointers are checked against runtime GGML tensor byte sizes and remain valid
until that release. The CPU prototype still allocates llama's ordinary backend
buffer bookkeeping, but attaches the validated payload pointers from mmap; no
payload copy, dequantization, Q8_0 repack, or byte reorder was observed.

Q8_0 with absent `output.weight` uses the existing duplicated
`token_embd.weight` fallback. BF16's explicit `output.weight` is registered
independently.

## Results

Qualification used CPU-only pinned builds, two threads, the same prompts, and
the same decode settings for GGUF and vBuf:

```text
                         BF16       Q8_0
metadata parity          PASS       PASS
tensor/model load        PASS       PASS
token-ID parity          PASS       PASS
first-token logits        max diff 0  max diff 0
greedy 8-token sequence  PASS       PASS
```

Tokenizer corpus included empty text, ASCII, whitespace/newline, UTF-8, and a
Qwen-style special-token-looking fragment. Chat-template bytes are retrieved
through the validated descriptor; rendering was not reimplemented.

Evidence:

```text
benchmark-results/vbuf-ml-step21/consumer-metadata-runtime-parity.csv
benchmark-results/vbuf-ml-step21/tokenizer-token-parity.csv
benchmark-results/vbuf-ml-step21/logit-parity.csv
benchmark-results/vbuf-ml-step21/generation-parity.json
benchmark-results/vbuf-ml-step21/adapter-provenance.json
```

No performance claim, GPU claim, layer streaming, or optimization claim is
made. The runtime convergence point is after `llama_model_init_from_user`:
subsequent model, GGML, tokenizer execution, graph, kernel, and decode paths
are shared.
