#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
CACHE_ROOT="$HOME/.cache/vbuf-agent-qualification"
FIXTURE_ROOT="$CACHE_ROOT/qwen3-cuda-q4k"
ARTIFACT_ROOT="$CACHE_ROOT/qwen3-14b-full"
MODEL="$ARTIFACT_ROOT/Qwen_Qwen3-14B-Q4_K_M.vbuf"
SEMANTIC="$ARTIFACT_ROOT/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf"
INPUT="$FIXTURE_ROOT/fixture/prefix25-q-input.f32.bin"
EXPECTED_MODEL_SHA=f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31
EXPECTED_INPUT_SHA=d1cc5ba95059d9ea136b6967e1a57df56d3a276f9873f9e463e4bfc71d6883df
EXPECTED_WEIGHT_SHA=3c4dd39531acff6eb56af4dfb0e39fed24ec3a71e6efe958a96c944596845bf9
EXPECTED_CPU_SHA=1e40f747a6fd949cb0f43bdc3ff727f90b1f5619234347330c6baa90b623851d
EXPECTED_CUDA_SHA=f9bacc3242e0b1e600c3d7f1acafb337c5ba7ed5779c3fba87ad9a2120fe7c63
PORT=${VBUF_QWEN_FIXTURE_PORT:-18783}
TOKENS=tokens:0,25,220,16,13,15,13,15,198,262,549,0,220,16,13,15,13,15,198,262,549,0,220,16,13

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
    local file=$1 expected=$2 actual
    actual=$(sha256_file "$file")
    if [[ "$actual" != "$expected" ]]; then
        printf 'SHA-256 mismatch: %s expected=%s actual=%s\n' "$file" "$expected" "$actual" >&2
        exit 1
    fi
}

[[ -f "$MODEL" && -f "$SEMANTIC" && -f "$INPUT" ]] || {
    echo "Missing exact model/semantic/input fixture under $CACHE_ROOT" >&2; exit 1;
}
require_sha "$MODEL" "$EXPECTED_MODEL_SHA"
require_sha "$INPUT" "$EXPECTED_INPUT_SHA"
[[ $(stat -c %s "$INPUT") == 512000 ]] || { echo 'Q input fixture must be 512000 bytes' >&2; exit 1; }

SOURCE="$FIXTURE_ROOT/source/ggml"
PATCH="$REPO_ROOT/research/results/vbuf-ml-integration/qwen3-cuda-fixtures/qwen3-q4k-capture.patch"
if [[ ! -f "$SOURCE/CMakeLists.txt" ]]; then
    mkdir -p "$(dirname "$SOURCE")"
    cp -a "$REPO_ROOT/integrations/ggml" "$SOURCE"
fi
mkdir -p "$SOURCE/tools"
if ! grep -q 'g_qwen_diagnostic_cuda_backend' "$SOURCE/tools/qwen3_block_qualification.cpp" 2>/dev/null; then
    cp "$REPO_ROOT/integrations/ggml/tools/qwen3_block_qualification.cpp" \
       "$SOURCE/tools/qwen3_block_qualification.cpp"
    patch -s -p1 -d "$SOURCE" < "$PATCH"
fi

GGML_SOURCE="$CACHE_ROOT/ggml"
VBUF_ML_LIBRARY="$REPO_ROOT/rust/target/release/libvbuf_ml.so"
BUILD="$FIXTURE_ROOT/build-cuda"
cmake -S "$SOURCE" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DVBUF_GGML_SOURCE_DIR="$GGML_SOURCE" \
    -DVBUF_ML_LIBRARY="$VBUF_ML_LIBRARY" \
    -DVBUF_ML_INCLUDE_DIR="$REPO_ROOT/integrations/ggml/include" \
    -DVBUF_ENABLE_CUDA=ON -DVBUF_BUILD_PROBES=OFF \
    -DVBUF_BUILD_COMPAT_SERVER=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_CUDA_FA=OFF
cmake --build "$BUILD" --target vbuf_qwen3_block_qualification -j4
BIN="$BUILD/vbuf_qwen3_block_qualification"

RANGE_URL="http://127.0.0.1:$PORT/payload"
SERVER_PID=''
probe_range_server() {
    python3 - "$RANGE_URL" "$EXPECTED_MODEL_SHA" <<'PY'
import sys, urllib.request
url, digest = sys.argv[1:]
try:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=2) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-0/9000232144" or response.headers.get("ETag") != f'"{digest}"':
            raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
}
if ! probe_range_server; then
    python3 "$REPO_ROOT/scripts/range_server.py" --file "$MODEL" --port "$PORT" \
        > "$FIXTURE_ROOT/local-range-server.log" 2>&1 &
    SERVER_PID=$!
    trap 'if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi' EXIT
    for _ in $(seq 1 180); do
        if probe_range_server; then break; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo 'Local vBuf range server exited before becoming ready' >&2; exit 1
        fi
        sleep 1
    done
    probe_range_server || { echo 'Local vBuf range server failed its artifact identity check' >&2; exit 1; }
fi

EMPTY_REFERENCE="$FIXTURE_ROOT/empty-reference"
mkdir -p "$EMPTY_REFERENCE"
mkdir -p "$FIXTURE_ROOT/repro-cpu" "$FIXTURE_ROOT/repro-cuda"
export LD_LIBRARY_PATH="$BUILD/ggml/src/ggml-cuda:$BUILD/ggml/src:$REPO_ROOT/rust/target/release:${LD_LIBRARY_PATH:-}"
export VBUF_QWEN_INTERNAL_ONLY=1
export VBUF_QWEN_Q_INPUT_OVERRIDE="$INPUT"
export VBUF_QWEN_BACKEND=CPU
export VBUF_QWEN_Q_CAPTURE_DIR="$FIXTURE_ROOT/repro-cpu"
"$BIN" "$SEMANTIC" "$RANGE_URL" "$EMPTY_REFERENCE" 25 1 actual "$TOKENS" --serial \
    > "$FIXTURE_ROOT/repro-cpu.log" 2>&1

export CUDA_VISIBLE_DEVICES=0
export VBUF_QWEN_BACKEND=CUDA
export VBUF_QWEN_Q_CAPTURE_DIR="$FIXTURE_ROOT/repro-cuda"
"$BIN" "$SEMANTIC" "$RANGE_URL" "$EMPTY_REFERENCE" 25 1 actual "$TOKENS" --serial \
    > "$FIXTURE_ROOT/repro-cuda.log" 2>&1

for backend in cpu cuda; do
    require_sha "$FIXTURE_ROOT/repro-$backend/q-input-attn-norm.f32" "$EXPECTED_INPUT_SHA"
    require_sha "$FIXTURE_ROOT/repro-$backend/q-weight-q4k.bin" "$EXPECTED_WEIGHT_SHA"
    require_sha "$FIXTURE_ROOT/repro-$backend/q-weight-backend-readback.bin" "$EXPECTED_WEIGHT_SHA"
done
require_sha "$FIXTURE_ROOT/repro-cpu/q-output.f32" "$EXPECTED_CPU_SHA"
require_sha "$FIXTURE_ROOT/repro-cuda/q-output.f32" "$EXPECTED_CUDA_SHA"
python3 - "$FIXTURE_ROOT/repro-cpu/q-output.f32" "$FIXTURE_ROOT/repro-cuda/q-output.f32" <<'PY'
import numpy as np, sys
cpu=np.fromfile(sys.argv[1],dtype='<f4').astype(np.float64)
cuda=np.fromfile(sys.argv[2],dtype='<f4').astype(np.float64)
if cpu.shape != (128000,) or cuda.shape != cpu.shape: raise SystemExit('Q output geometry mismatch')
rms=np.sqrt(np.mean((cpu-cuda)**2))
rel_cpu=rms/np.sqrt(np.mean(cpu**2))
rel_cuda=rms/np.sqrt(np.mean(cuda**2))
print(f'Q4_K prefix25 Q projection: max_abs={np.max(np.abs(cpu-cuda)):.9g} rms={rms:.9g} rel_cpu={rel_cpu:.9%} rel_cuda={rel_cuda:.9%}')
if not 0.015 <= rel_cpu <= 0.017: raise SystemExit('historical CPU/CUDA mismatch class not reproduced')
PY
printf 'Fixture replay PASS; model, input, payload, CPU and CUDA hashes match the manifest.\n'
