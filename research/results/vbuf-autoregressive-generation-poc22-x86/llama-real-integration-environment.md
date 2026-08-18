# Real llama.cpp Integration Environment

## Pinned Environment

```text
upstream: https://github.com/ggml-org/llama.cpp.git
checkout: /tmp/llama.cpp-step21
llama revision: 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
canonical patch: patches/llama.cpp/0000-pinned-step21-canonical.patch
prepared tree delta SHA-256: b365448a51b9e2801d8b86269317e9396a975f19c9562d9534ed34de25e7cb38
compiler: GNU C/C++ 14.2.0
architecture: x86_64
```

The checkout was restored with:

```text
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp-step21
python3 scripts/prepare_step21_llama.py --upstream-root /tmp/llama.cpp-step21
```

The build uses the ggml embedded in the pinned llama checkout. CMake reported:

```text
ggml version: 0.19.0
```

No standalone ggml checkout was mixed into this build. The standalone
historical pin `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` remains a separate
research substrate and was not used here.

## Build

```text
cmake -S /tmp/llama.cpp-step21 -B /tmp/llama.cpp-step21-build \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=ON
cmake --build /tmp/llama.cpp-step21-build --target llama -j2
cargo build --manifest-path rust/Cargo.toml -p vbuf-ml
python3 scripts/qualify_step21_runtime.py \
  --upstream-root /tmp/llama.cpp-step21 \
  --build-dir /tmp/llama.cpp-step21-build
```

Results:

```text
PINNED_LLAMA_ENVIRONMENT_READY: YES
LOCAL_LLAMA_VBUF_INTEGRATION_BUILD: PASS
LOCAL_LLAMA_VBUF_INTEGRATION_BASELINE: PASS
```

The integration probe was compiled with the pinned `include`, `src`, and
embedded `ggml/include` directories, the Rust vbuf-ML library, and the pinned
llama/ggml libraries. The probe source is
`integrations/llama.cpp/llama_external_materialization_probe.cpp`.

## Boundary Result

The C++ adapter consumes only explicit pointer/length spans plus an optional
lease. No SourceSet, URI, HTTP, or file-range code was added to C++.
