#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
CACHE_ROOT="$HOME/.cache/vbuf-agent-qualification"
ROOT="$CACHE_ROOT/qwen3-cuda-q4k"
FIX="$ROOT/fixture"
GGML="$CACHE_ROOT/ggml"
BUILD="$ROOT/build-cuda"

for file in prefix25-q-input.f32.bin blk.0.attn_q.weight.q4k.bin cpu-q-output.f32.bin cuda-q-output.f32.bin; do
    [[ -s "$FIX/$file" ]] || { echo "Missing persisted fixture: $FIX/$file" >&2; exit 1; }
done
[[ -d "$BUILD/ggml/src" && -d "$GGML/include" ]] || {
    echo 'Run reproduce-prefix25-q4k.sh first to establish the pinned persistent build.' >&2; exit 1;
}

COMMON=(-std=c++17 -O2 -I"$GGML/include")
LIBS=(-L"$BUILD/ggml/src" -Wl,-rpath,"$BUILD/ggml/src")
g++ "${COMMON[@]}" "$REPO_ROOT/research/results/vbuf-ml-integration/qwen3-cuda-fixtures/qwen3-q8k-activation-capture.cpp" \
    "${LIBS[@]}" -lggml-cpu -lggml-base -o "$ROOT/q8k-activation-capture"
g++ "${COMMON[@]}" "$REPO_ROOT/research/results/vbuf-ml-integration/qwen3-cuda-fixtures/qwen3-q4k-ggml-decode.cpp" \
    "${LIBS[@]}" -lggml-base -o "$ROOT/q4k-ggml-decode"
"$ROOT/q8k-activation-capture" "$FIX/prefix25-q-input.f32.bin" "$FIX/cpu-q8k"
"$ROOT/q4k-ggml-decode" "$FIX/blk.0.attn_q.weight.q4k.bin" "$FIX/q4k-weight-ggml-dequant.f32"
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} \
    python3 "$REPO_ROOT/research/results/vbuf-ml-integration/qwen3-cuda-fixtures/analyze-q4k-activation-oracles.py" "$ROOT"
