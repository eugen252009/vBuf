# How To Start With vbuf-ml

This guide uses the current `vbuf-ml-0.1` model format and the pinned llama.cpp
consumer. The current profile supports F32, BF16, Q8_0, Q4_0, Q2_K, and IQ1_S
payloads. It also supports optional inline nested vBuf child streams.

```text
Qwen3 GGUF -> vbuf-ml manifest -> Qwen3 .vbuf -> vbuf-ml -> llama.cpp
```

## Requirements

- Linux, Python 3, Rust/Cargo, CMake, Git, and a C++17 compiler.
- The pinned llama.cpp revision:

```text
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

- The source GGUF model and enough disk space for the generated `.vbuf` file.

Model files under `research-models/` are local research artifacts and are not
committed to Git.

## Build vbuf-ml

From the repository root:

```bash
cargo build --release --manifest-path rust/Cargo.toml -p vbuf-ml
cargo test --manifest-path rust/Cargo.toml -p vbuf-ml
```

This produces:

```text
rust/target/release/libvbuf_ml.so
```

## Create A Current `.vbuf` Model

Place the Qwen3 GGUF source in `research-models/`. Example:

```text
research-models/Qwen3-0.6B-Q8_0.gguf
```

Build the current conversion manifest:

```bash
python3 scripts/build_step18_manifest.py --root "$PWD"
```

Convert the GGUF file into the current `vbuf-ml-0.1` format:

```bash
python3 scripts/convert_gguf_to_vbuf_ml.py \
  research-models/Qwen3-0.6B-Q8_0.gguf \
  benchmark-results/vbuf-ml-step18/qwen3-0.6b-q8_0-manifest.json \
  research-models/Qwen3-0.6B-Q8_0.vbuf \
  --integrity none \
  --evidence-dir benchmark-results/vbuf-ml-step20
```

The converter validates the source identity, metadata, tokenizer, tensor
descriptors, representations, and ranges before writing the target. It then
reopens and validates the generated file. Do not edit offsets or hashes by
hand.

For the checked-in Qwen3 0.6B Q8_0 fixture, the expected target SHA-256 is:

```text
2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998
```

The conversion preserves tensor payload bytes. It does not change
quantization, reorder tensors, or add compression.

## Nested vBuf Children

An optional nested directory can index child vBuf streams embedded inside
parent opaque blocks. Each child has a name, parent block reference, relative
offset, and length. The child range must be in bounds, non-overlapping, and a
complete canonical vBuf stream. The Rust API is documented in
[`docs/vbuf-ml/nested-vbuf.md`](docs/vbuf-ml/nested-vbuf.md).

Nested children are validated and exposed by vbuf-ml. MoE routing is not
implicitly inferred from child names; expert-count, router, and graph-routing
contracts must be defined by the target architecture before llama.cpp can
execute an MoE model.

## Build The Pinned llama.cpp Consumer

Clone and patch exactly the pinned llama.cpp revision:

```bash
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp-vbuf
git -C /tmp/llama.cpp-vbuf checkout 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
git -C /tmp/llama.cpp-vbuf apply \
  "$PWD/patches/llama.cpp/0001-user-metadata-tensor-source.patch" \
  "$PWD/patches/llama.cpp/0002-source-neutral-model-source.patch"
```

Build the CPU libraries:

```bash
cmake -S /tmp/llama.cpp-vbuf -B /tmp/llama.cpp-vbuf/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=ON
cmake --build /tmp/llama.cpp-vbuf/build -j"$(nproc)"
```

The vBuf adapter sources are:

```text
integrations/llama.cpp/llama_vbuf_loader.cpp
integrations/llama.cpp/vbuf_ml_adapter.cpp
integrations/llama.cpp/vbuf_direct_source.cpp
```

Compile them into your llama.cpp application. Include:

```text
/tmp/llama.cpp-vbuf
/tmp/llama.cpp-vbuf/include
/tmp/llama.cpp-vbuf/src
/tmp/llama.cpp-vbuf/ggml/include
```

Link against the pinned llama.cpp/GGML libraries and `libvbuf_ml.so`:

```bash
-L/tmp/llama.cpp-vbuf/build/bin \
-lllama -lggml -lggml-cpu -lggml-base -lllama-common \
-L"$PWD/rust/target/release" -lvbuf_ml
```

Use rpaths or set:

```bash
export LD_LIBRARY_PATH="$PWD/rust/target/release:/tmp/llama.cpp-vbuf/build/bin:${LD_LIBRARY_PATH:-}"
```

## Load A `.vbuf` Model From C++

```cpp
#include "llama_vbuf_loader.h"

llama_model_params params = llama_model_default_params();
params.n_gpu_layers = 0;

llama_model * model = llama_model_load_vbuf("Qwen3-0.6B-Q8_0.vbuf", params);
if (!model) {
    // The vbuf-ml validation or llama model construction failed.
}

const llama_vocab * vocab = llama_model_get_vocab(model);
llama_context_params context_params = llama_context_default_params();
llama_context * context = llama_init_from_model(model, context_params);

// Tokenize, create a llama_batch, and call llama_decode as usual.

llama_free(context);
llama_model_free_vbuf(model);
```

Use `llama_model_free_vbuf`, not `llama_model_free`, because the vbuf adapter
keeps the validated mmap-backed model handle alive until the vBuf-specific
free call.

After model construction, llama.cpp uses its normal vocabulary, GGML tensor,
context, graph, kernel, and decode paths. The vbuf-ml library supplies the
validated model descriptors and payload ranges; it is not a separate inference
engine.

## Included Parity Check

The repository includes a small GGUF-versus-vBuf qualification program. Build
it with the same adapter sources and libraries, then run:

```bash
/tmp/step21_qualification \
  research-models/Qwen3-0.6B-Q8_0.gguf \
  research-models/Qwen3-0.6B-Q8_0.vbuf \
  generation
```

Other modes are `metadata`, `tokens`, and `logits`. A successful run confirms
that the current vbuf-ml model reaches the same pinned llama.cpp behavior as
the GGUF reference.

## Troubleshooting

If conversion rejects the source, regenerate the manifest and use the exact
GGUF artifact it describes:

```bash
python3 scripts/build_step18_manifest.py --root "$PWD"
```

If `libvbuf_ml.so` is missing at runtime, check `LD_LIBRARY_PATH` or the
executable rpath. If `llama-model-source.h` is missing, the second llama.cpp
patch was not applied or the llama.cpp `src` include directory is missing.

## References

- Conversion: [`docs/vbuf-ml/step20-gguf-conversion.md`](docs/vbuf-ml/step20-gguf-conversion.md)
- llama.cpp consumer: [`docs/vbuf-ml/step21-llama-consumer.md`](docs/vbuf-ml/step21-llama-consumer.md)
- Integration files: [`integrations/llama.cpp/README.md`](integrations/llama.cpp/README.md)
