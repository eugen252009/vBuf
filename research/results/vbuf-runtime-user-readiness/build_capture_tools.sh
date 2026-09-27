#!/usr/bin/env bash
# Rebuild research-only captures against already prepared native libraries.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
BUILD=${1:?usage: build_capture_tools.sh VBUF_NATIVE_BUILD LLAMA_SOURCE OUTPUT_DIR}
LLAMA=${2:?llama.cpp source checkout required}
OUT=${3:?output directory required}
mkdir -p "$OUT"
c++ -O2 -std=c++17 "$ROOT/research/results/vbuf-runtime-user-readiness/oracle_capture.cpp" \
  -I "$LLAMA/include" -I "$LLAMA/ggml/include" -L "$LLAMA/build/bin" \
  -lllama -lggml-base -Wl,-rpath,"$LLAMA/build/bin" -o "$OUT/oracle-capture"
c++ -O2 -std=c++17 "$ROOT/research/results/vbuf-runtime-user-readiness/native_capture.cpp" \
  -I "$ROOT/integrations/ggml/include" -I "$BUILD/_deps/ggml_source-src/include" \
  "$BUILD/libvbuf_region_executor.a" -L "$BUILD/ggml/src" \
  -lggml -lggml-cpu -lggml-base -L "$ROOT/rust/target/release" -lvbuf_ml -pthread \
  -Wl,-rpath,"$BUILD/ggml/src" -Wl,-rpath,"$ROOT/rust/target/release" \
  -o "$OUT/native-capture"
gcc -O2 -shared -fPIC -pthread "$ROOT/research/results/vbuf-runtime-user-readiness/heap_probe.c" \
  -o "$OUT/heap-probe.so"
