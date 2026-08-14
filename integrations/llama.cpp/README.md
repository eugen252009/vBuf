# Step 21 pinned-consumer adapter seam

Pinned base:

```text
https://github.com/ggml-org/llama.cpp.git
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

The pinned checkout is kept external (`/tmp/llama.cpp-step21` during local
qualification). The C++ wrapper in this directory is intentionally a
**descriptor-only boundary**. It calls the validated Rust `vbuf-ml` C ABI and
maps profile representations to:

```text
CanonicalPrimitive → GGML_TYPE_F32
BF16               → GGML_TYPE_BF16
GGML_Q8_0          → GGML_TYPE_Q8_0
```

It does not modify llama.cpp or construct a `llama_model` yet.

## Current seam finding

The pinned loader is organized around the concrete `llama_model_loader`, which
owns GGUF metadata, `weights_map`, GGUF mappings, and GGUF-specific tensor data
loading. The model path is:

```text
llama_model_load_from_file_impl
  → llama_model_loader
  → llama_model_create
  → load_hparams / load_vocab / load_tensors
  → llama_model_loader::create_tensor / load_data_for
```

The relevant loader internals are not a public source abstraction. A complete
vBuf runtime load therefore requires a small pinned llama.cpp internal seam
(new source interface or equivalent loader constructor), not more vBuf parser
logic. This step deliberately stops before scattering `is_vbuf` branches into
model, graph, or kernel code.

The Rust bridge nevertheless proves the safe first boundary:

```text
mmap vBuf
→ canonical validation
→ Bootstrap
→ ModelMetadata
→ TensorDirectory
→ TokenizerMetadata
→ checked semantic descriptor/payload views
```

Payload pointers are valid only while the owning adapter handle remains alive.
No raw target offsets are exposed.

## Qualification status

```text
pinned llama.cpp checkout: PASS
pinned llama.cpp CPU library build: PASS
Rust validated consumer descriptor bridge: PASS
C++ representation wrapper compile: PASS
llama_model construction from vBuf: DEFERRED — missing pinned internal loader seam
BF16 tokenizer/logit/generation parity: NOT RUN
Q8_0 tokenizer/logit/generation parity: NOT RUN
```

See `benchmark-results/vbuf-ml-step21/` for provenance and bridge evidence.
