# Native Dependency Inventory

Historical reference: llama.cpp commit
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`.

Native ggml pin: `ggml-org/ggml` commit
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.

The native build uses CMake `FetchContent` to checkout the direct ggml pin
reproducibly and adds that checkout as a subdirectory. It does not checkout
llama.cpp, add llama runtime targets, invoke Cargo, or compile Rust. The
historical llama pin is not used by this build and remains the
reference/oracle for existing evidence.

| Component | Needed now? | Source | Reason | Long-term status |
|---|---:|---|---|---|
| ggml core / `ggml-base` | YES | native pin `2d191b5...`, direct source | tensor descriptors, operations, allocator, backend API | KEEP |
| ggml backend registry / `ggml` | YES | native pin `2d191b5...`, direct source | backend registration and selection | KEEP |
| ggml allocator | YES | native pin, included in `ggml-base` | graph/context tensor allocation | KEEP |
| ggml CPU backend | YES | native pin, direct source | required CPU execution substrate | KEEP |
| RVV sources | CONDITIONAL | native pin CPU backend | enabled by ggml for riscv64/RVV toolchains | KEEP |
| CUDA backend | OPTIONAL | native pin, direct source | future device execution and materialization | KEEP |
| vBuf-ML shared library | OPTIONAL NOW | explicit `VBUF_ML_LIBRARY` | future native adapter/lifetime integration; already-built Rust ABI | KEEP |
| llama graph helpers | NO | `src/llama-*` | architecture/lowering work is not substrate work | REIMPLEMENT_SMALLER |
| llama model classes | NO | `src/models` | full-model architecture runtime coupling | REMOVE |
| llama context | NO | `src` | KV, decode, scheduler lifecycle coupling | REMOVE |
| GGUF loader | NO | `src/llama-model-loader.cpp` | persistent source format is vBuf-ML | REMOVE |
| tokenizer | NO | `src/llama-vocab.cpp` and related | model/runtime concern outside substrate | REMOVE |
| sampling | NO | `src/sampling.cpp` | application/runtime concern | REMOVE |

## Non-ggml Source Decision

No non-ggml llama source is linked by this build. The historical pinned llama
sources are still a reference implementation and qualification oracle. Any later helper
candidate must be classified before copying:

- `PURE_GENERIC_HELPER`: compare with ggml and prefer a small local component;
- `MODEL_ARCHITECTURE_HELPER`: keep in vBuf lowering, not this substrate;
- `LLAMA_RUNTIME_COUPLED`: do not import without a documented blocking proof.

## CMake Options

The first POC explicitly uses CPU, disables `GGML_CPU_REPACK`, OpenMP, BLAS,
llamafile, dynamic backend loading, tests, and examples. CUDA is opt-in through
`VBUF_ENABLE_CUDA=ON`. The pinned ggml project still owns architecture-specific
CPU/RVV source selection and compiler flags.

## Build Entry Point

After the Rust library exists, configure and build without Cargo invocation from
CMake:

```sh
cmake -S integrations/ggml -B build/ggml-region \
  -DVBUF_ML_LIBRARY=/path/to/libvbuf_ml.so
cmake --build build/ggml-region
ctest --test-dir build/ggml-region --output-on-failure
```

`VBUF_ML_LIBRARY` is optional for the substrate probes. CMake downloads the
native pin on the first configure. For an offline build, set
`-DVBUF_GGML_SOURCE_DIR=/path/to/ggml` and verify that checkout is the recorded
native commit; CMake also verifies that `HEAD` matches the recorded commit.

## Historical API Boundary

The historical llama.cpp reference embeds ggml `0.19.0`; the direct native pin
reports ggml `0.20.0`. The required public target names and CPU backend APIs
remain usable by this substrate, so no llama runtime source was needed. The
adapter concern exposed by the probe is semantic rather than format-related:
`ggml_backend_buffer_init_tensor()` initializes a buffer callback, while
`ggml_backend_tensor_alloc()` binds an externally owned address to a tensor.
The borrowed-storage path must use the latter and retain the source lifetime.
