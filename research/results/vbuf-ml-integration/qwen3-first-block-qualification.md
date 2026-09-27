# Qwen3 First-Block Qualification

## Scope and decision

A separate, opt-in CPU qualification path now executes the real Qwen3-14B
embedding and dense transformer block 0 from vBuf-ML-validated tensor ranges.
It is deliberately not wired into `VbufGenerationSession`: normal Qwen3
inference remains fail-closed. Later isolated work extended this qualification
runner through block 7 for bounded 4/8-position tests; that does not implement
a production/full-model path, final logits, persistent autoregressive KV, or
generation. See [`qwen3-multiblock-full-forward-qualification.md`](qwen3-multiblock-full-forward-qualification.md).
Tool calling and Pi were not run.

**Checkpoint status (fixed max-absolute threshold `1e-5`):** one-position
block 0 passes. The two- and three-position runs pass the attention context,
FFN, and final block output, but fail the newly captured raw attention-score
checkpoint (`1.52587891e-5` at two positions; `1.71661377e-5` at three).
Four, five, and eight positions also fail raw attention scores; at four or more
the FFN-down and final block output additionally fail. The earlier summary that
position two passed every checkpoint is superseded by this internal-score
measurement; its downstream output result remains unchanged.

At four positions, FFN down has max absolute error `8.75592232e-5` (RMS
`1.15196202e-5`) and the final block output has `8.75554979e-5` max absolute
(RMS `1.15195378e-5`). The failure is preserved; tolerance was not widened.
No result qualifies full-model inference or production Qwen3 support.

## Implementation boundary

- `integrations/ggml/tools/qwen3_block_qualification.cpp` is an opt-in
  qualification executable. It opens the semantic bootstrap, validates the
  Qwen3 metadata and strict 443-tensor dense catalog, verifies the declared
  source size/SHA-256 identity, then fetches required tensor ranges through
  `HttpRangeSource` and `LocalVbufRangeMaterializer`.
- The graph uses the pinned repository GGML CPU backend (`2d191b5d…`). It
  implements embedding lookup; attention RMSNorm; Q/K/V projections; per-head
  Q/K RMSNorm; Qwen3 NEOX/split-half RoPE (128 dimensions, theta 1,000,000);
  batched causal GQA (40 query heads / 8 KV heads); output projection and
  attention residual; FFN RMSNorm; parallel gate/up, SiLU-gated product, down
  projection, and final residual. The batched attention form matches the
  reference graph geometry; it did not remove the observed numerical drift.
- For reference parity, the Q/K/V values are cast to F16 at the KV boundary,
  matching the reference context's default KV cache representation. Qualification
  KV is temporary in-batch state only; no persistent decode cache is implemented.
- Production `VbufGenerationSession` still rejects Qwen3 before inference.
  No model-generation, sampler/logits, full-open resource, residency, or
  persistent-KV qualification is implied by this tool.

## Reference and artifacts

- Model: `bartowski/Qwen_Qwen3-14B-GGUF`, revision
  `bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2`, file
  `Qwen_Qwen3-14B-Q4_K_M.gguf`.
- GGUF SHA-256:
  `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6`.
- vBuf payload SHA-256:
  `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Independent checkpoints were emitted by
  `integrations/ggml/qualification/qwen3_llama_reference_dump.cpp` using
  llama.cpp/GGML source revision `a97123e497968f3440264c0464a7adc7c999c027`.
  The reference harness runs one CPU decode to observe block-0 intermediates;
  it does not generate tokens. Tokens
  are IDs `0..N-1`, with N equal to the tested position count.
- Each run asserts exact checkpoint size and compares finite F32 arrays with
  `max_abs <= 1e-5` (no relative allowance). The output-projection reference is
  derived as `ffn_inp - embd`, because the upstream evaluation callback does not
  expose that intermediate directly.

## Reproduction

Build the vBuf qualification target with the pinned GGML and the existing
vBuf-ML shared library, then build the reference dumper against the local
llama.cpp checkout at the revision above. Example vBuf build:

```bash
cmake -S integrations/ggml -B /tmp/vbuf-qwen3-block-build \
  -DVBUF_GGML_SOURCE_DIR=/home/eugen/.cache/vbuf-agent-qualification/ggml \
  -DVBUF_ML_LIBRARY="$PWD/rust/target/release/libvbuf_ml.so" \
  -DVBUF_BUILD_COMPAT_SERVER=OFF -DVBUF_BUILD_PROBES=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vbuf-qwen3-block-build --target vbuf_qwen3_block_qualification -j4
```

The reference dumper is a qualification-only source file. The recorded build
used the local llama.cpp checkout at `a97123e497968f3440264c0464a7adc7c999c027`
and its built `libllama.so.0.1.0` / GGML libraries:

```bash
g++ -O2 -std=c++17 \\
  -I/home/eugen/projekte/llama.cpp/include \\
  -I/home/eugen/projekte/llama.cpp/ggml/include \\
  integrations/ggml/qualification/qwen3_llama_reference_dump.cpp \\
  -L/home/eugen/projekte/llama.cpp/build/bin \\
  -Wl,-rpath,/home/eugen/projekte/llama.cpp/build/bin \\
  -l:libllama.so.0.1.0 -l:libggml-base.so.0 -l:libggml.so.0 \\
  -l:libggml-cpu.so.0 -lpthread -ldl -o /tmp/qwen3_llama_reference_dump
```

Export reference checkpoints for each position count, serve the converted vBuf
payload (not the original GGUF), and invoke the qualifier:

```bash
/tmp/qwen3_llama_reference_dump MODEL.gguf /tmp/qwen3-reference-1 1
/tmp/qwen3_llama_reference_dump MODEL.gguf /tmp/qwen3-reference-2 2
/tmp/qwen3_llama_reference_dump MODEL.gguf /tmp/qwen3-reference-4 4
python3 scripts/range_server.py --file MODEL.vbuf --port 18783
LD_LIBRARY_PATH="$PWD/rust/target/release:/tmp/vbuf-qwen3-block-build/ggml/src" \
  /tmp/vbuf-qwen3-block-build/vbuf_qwen3_block_qualification \
  MODEL.semantic.vbuf http://127.0.0.1:18783/payload \
  /tmp/qwen3-reference-1 1
```

Observed statuses with the expanded checkpoint set: position 1 **PASS**;
positions 2 and 3 **FAIL** the raw-score checkpoint while their context and
final block outputs pass; positions 4, 5, and 8 **FAIL** raw scores and the
FFN-down/final-output checkpoints. The server's SHA-256 ETag and
`Content-Range` identity checks were active during these runs.

## Four-position discrepancy investigation (2026-09-26)

The first checkpoint to exceed the strict absolute threshold is raw QK attention
scores: max cross-reference error `3.05175781e-5` at compact index 364 (query
position 3, head 22, key 0; reference `179.573334`, vBuf `179.573303`). An
independent FP64 dot over the F16-rounded Q/K operands gives `179.573318` at
that index; the two implementations are one F32 ULP on opposite sides. Across
the full score matrix, the FP64-oracle maximum errors are `1.62124634e-5` for
llama.cpp and `3.05175781e-5` for vBuf. Softmax probabilities remain within
`4.77e-7`, and the attention context within `1.49e-8`. The vBuf V projection
fed actual probabilities is bit-identical to the same projection fed the
*reference* probabilities, so the raw-score/softmax difference does not cause
the later FFN error. The first downstream divergence is the F16 attention-value
projection/context: that vBuf result differs from llama.cpp by at most
`1.49e-8`. A direct FP64 oracle over the F16-rounded V and probability operands
matches the vBuf context exactly; llama.cpp differs by one F32 ULP at the
maximum-error location. This isolates the relevant discrepancy upstream of
FFN down.

The FFN-down tensor is not the defect:

- `blk.0.ffn_down.weight` is Q6_K, shape `[17408,5120]`, row size 14,280
  bytes (68 blocks/row), source offset 579,061,448, payload 73,113,600 bytes.
  Materialized bytes and GGML backend readback have the same FNV-1a64
  (`a4c3145d3cafb9c3`); rows 0, 2560, and 5119 decode identically.
- The down input and output layouts are contiguous and non-transposed. The
  operation uses Q6_K weights with Q8_K activation rows and F32 accumulation.
  For four positions the CPU plan is 320 output chunks × 1 token chunk at
  chunk size 16; one and eight vBuf worker threads give the same output.
- Feeding the *reference* SwiGLU values to the vBuf GGML down projection
  reproduces the llama.cpp output exactly. A separate pinned-GGML Q6_K×Q8_K
  row-dot control (independent of matrix-matmul dispatch, using GGML's
  registered vector-dot kernel) also reproduces both outputs exactly for their
  respective inputs. Thus weight bytes, row interpretation, and matrix dispatch
  are excluded; this control is not a second implementation of the Q6_K dot.
- The actual/reference SwiGLU difference is at most `1.1920929e-7`, only at
  position 3. Applying the same Q8_K quantizer changes 41 serialized bytes in
  40 of 68 activation blocks, all in that row. The down result then differs by
  `8.75592232e-5` at output feature 1839 (flat index 17199): reference
  `-0.000656396151`, vBuf `-0.000743955374`. Positions 0–2 are identical.
  The final block output carries the same error at that feature.

The independent llama.cpp reference is stable across one and eight CPU threads
(bit-identical captured attention, FFN, and block outputs); vBuf one/eight
thread runs also reproduce the discrepancy. This is not a thread-count or
FFN-down chunk-dispatch threshold effect. The reference checkout is llama.cpp/GGML
`a97123e…`; the vBuf executable uses standalone GGML `2d191b5d…`. Their build options also differ: reference has LLAMAFILE,
CPU_REPACK, and OpenMP enabled; the vBuf target forces these off. In an isolated
control build with pinned vBuf GGML and LLAMAFILE enabled, raw attention scores
match the reference exactly, but the reference-probability V projection remains
one ULP different and FFN down still fails. This confirms that raw-score drift
is a CPU-kernel/build-path issue but is not by itself the cause of the large
down error. The exact instruction/reduction-order source of the one-ULP V
projection difference remains unresolved.

**Root-cause classification:** an F16 attention-value projection result that
is one F32 ULP different between the reference GGML path and the vBuf GGML path
is the first downstream numerical divergence. That difference propagates into
the position-3 SwiGLU input, where Q8_K activation quantization changes 40
blocks, yielding the `8.76e-5` FFN-down error. A separate raw-QK score drift
fails the internal checkpoint but is removed by softmax/F16 boundaries and is
not causal for the FFN-down mismatch. The exact CPU reduction-order difference
for the reference V projection remains unresolved. This is not a vBuf payload,
tensor layout, FFN-down weight, or Q6_K×Q8_K kernel defect. No tolerance or
production behavior was changed.

An AddressSanitizer/UndefinedBehaviorSanitizer run of the qualification C++
translation unit completed the four-position diagnostic with the expected
qualification failure and no sanitizer report. GGML and the Rust runtime
libraries in that run were prebuilt and not sanitizer-instrumented; this is not
full-stack sanitizer coverage.

## Other regression checks

- Pinned-GGML CMake build, including the new qualification executable: **PASS**.
- Repository CTest suite: **30/30 PASS**.
- DeepSeek bounded HTTP compatibility-server suite, using its converted vBuf
  payload and two-block setup: `VBUF_COMPAT_SERVER_HTTP_TEST=PASS`. Ordinary
  bounded DeepSeek requests completed; no Qwen code was routed into that path.
- `git diff --check`: **PASS**.

## Remaining gates

- Isolated blocks 1–7: **EXECUTED** for bounded qualification cases; sequential numerical error grows substantially by block 7. Full-model parity remains unqualified; see the multi-block report.
- Four-position block-0 parity: **FAIL** for raw attention scores and FFN down;
  the down-projection numerical root cause is classified above. Do not claim
  broader prefill parity from this isolated block-0 diagnostic.
- Persistent KV state and autoregressive Qwen3 generation: **NOT IMPLEMENTED /
  NOT TESTED**; production support remains disabled.
- Ordinary Qwen3 text generation, native tool-call/result/continuation behavior,
  tool error handling, streaming parity, and Pi qualification: **NOT TESTED**.
