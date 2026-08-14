# Step 21: pinned llama.cpp consumer adapter

Status: **validated descriptor bridge; full llama runtime seam deferred**.

Pinned base:

```text
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

## What was implemented

Added a validated Rust consumer descriptor:

```text
rust/vbuf-ml/src/consumer.rs
```

and a minimal C ABI:

```text
rust/vbuf-ml/src/consumer_ffi.rs
```

The external-facing C++ wrapper is:

```text
integrations/llama.cpp/vbuf_ml_adapter.{h,cpp}
```

It maps:

```text
CanonicalPrimitive → GGML_TYPE_F32
BF16               → GGML_TYPE_BF16
GGML_Q8_0          → GGML_TYPE_Q8_0
```

## Validated path

```text
mmap vBuf
→ canonical v0.6 validation
→ Bootstrap
→ ModelMetadata
→ TensorDirectory
→ TokenizerMetadata
→ checked consumer descriptor
```

The bridge exposes semantic tensor names, shapes, representations, and
payload slices whose lifetime is tied to the owning model handle. It exposes no
raw target offsets.

Real converted artifacts passed descriptor validation:

```text
Q8_0:  310 tensors
BF16:  311 tensors
```

Model metadata and tokenizer descriptor values were checked, including Qwen3
head dimensions, vocabulary count, merge count, `add_bos`, and chat-template
sizes.

## Pinned llama.cpp inspection

Relevant pinned symbols:

| Location | Symbol | Role |
|---|---|---|
| `src/llama.cpp` | `llama_model_load_from_file_impl` | GGUF-oriented model entry |
| `src/llama-model-loader.cpp` | `llama_model_loader` constructor | owns GGUF metadata/file state |
| `src/llama-model.cpp` | `llama_model_create` | creates architecture model |
| `src/llama-model.cpp` | `llama_model_base::load_hparams` | resolves model parameters |
| `src/llama-model.cpp` | `llama_model_base::load_vocab` | invokes GGUF vocabulary loader |
| `src/llama-model.cpp` | `llama_model_base::load_tensors` | creates runtime tensors |
| `src/llama-model-loader.cpp` | `create_tensor` / `load_data_for` | creates and backs GGML tensors |
| `src/llama-vocab.cpp` | `llama_vocab::impl::load` | tokenizer metadata and BPE construction |

The exact checkout and source mapping are recorded in
`benchmark-results/vbuf-ml-step21/adapter-provenance.json`.

## Seam finding

The pinned loader is not organized around a public model-source interface.
`llama_model_loader` directly owns GGUF metadata, `weights_map`, file mappings,
and GGUF-specific loading behavior. `llama_model_create` and the model classes
also receive this concrete loader type.

Therefore the smallest correct next llama-side change is an internal pinned
loader seam, such as a source interface or equivalent second loader
constructor. Adding vBuf branches throughout model, graph, or kernel code would
violate the intended architecture and was not done.

The current C ABI/C++ wrapper is intentionally descriptor-only. It does not
claim to construct a `llama_model`, configure llama's tokenizer runtime, or run
inference.

## Runtime status

```text
pinned llama.cpp checkout: PASS
pinned llama.cpp CPU library build: PASS
Rust descriptor bridge: PASS
C++ representation wrapper compile: PASS
BF16 llama structural load: DEFERRED
BF16 tokenizer parity: DEFERRED
BF16 logits parity: DEFERRED
BF16 generation parity: DEFERRED
Q8_0 runtime parity: DEFERRED
```

No inference or performance claim is made.
