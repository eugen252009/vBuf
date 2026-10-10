# Capture commands and build provenance

The commands below document the bounded runs. They use the existing local, identity-checked artifacts and bind the temporary range source only to loopback. The temporary diagnostic executables were built outside the repository; see `execution-manifest.json` for SHA-256 values.

## vBuf

A temporary copy of `integrations/ggml/src/qwen3_generation.cpp` received `qwen3_generation_logit_capture.patch`, then replaced that object in a copy of the existing core archive. The original archive remained unchanged (SHA-256 `343422158bb9f60def3dd3237ca1a4e0773c2efddff39e27ebb160303a0630df`); the temporary archive hash is in the manifest.

```bash
# Compile the temporary instrumented object with the flags from:
ninja -C /var/tmp/vbuf-qwen3-8b-ab-build -t commands vbuf_compat_server

cp /var/tmp/vbuf-qwen3-8b-ab-build/libvbuf_qwen3_cuda_core.a /tmp/libvbuf_qwen3_cuda_core_logitdiag.a
ar d /tmp/libvbuf_qwen3_cuda_core_logitdiag.a qwen3_generation.cpp.o
ar r /tmp/libvbuf_qwen3_cuda_core_logitdiag.a /tmp/qwen3_generation_diag.o
ranlib /tmp/libvbuf_qwen3_cuda_core_logitdiag.a

python3 scripts/range_server.py \
  --file /home/eugen/.cache/vbuf-agent-qualification/qwen3-8b/Qwen3-8B-Q4_K_M.vbuf \
  --host 127.0.0.1 --port 18337 --log

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 \
  /tmp/vbuf_fixed_prefix_logits \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-8b/Qwen3-8B-Q4_K_M.semantic.vbuf \
  http://127.0.0.1:18337/model.vbuf \
  prompt-token-ids.csv generated-token-ids.csv vbuf \
  12288 1350 positions.csv
```

The 63-token boundary replay used the same command with `63 positions-first-divergence.csv`. A repeat using `incremental-only` captured selected logits in the retained-KV generation path. Generated token IDs were compared against the saved sequence; 1,350/1,350 and 63/63 matched.

## Pinned llama.cpp reference

The source checkout was clean at `a97123e497968f3440264c0464a7adc7c999c027`. The reference harness uses raw token IDs, context 12,288, batch 512, ubatch 128, flash attention off, and all model layers on logical CUDA device 0. Prompt tokens are decoded in batches up to 512; known output tokens are then teacher-forced one at a time.

```bash
g++ -O3 -std=c++17 \
  -I/home/eugen/projekte/llama.cpp/include \
  -I/home/eugen/projekte/llama.cpp/ggml/include \
  harness/llama_fixed_prefix_logits.cpp \
  -L/home/eugen/projekte/llama.cpp/build/bin \
  -Wl,-rpath,/home/eugen/projekte/llama.cpp/build/bin \
  -lllama -o /tmp/llama_fixed_prefix_logits

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 \
  LD_LIBRARY_PATH=/home/eugen/projekte/llama.cpp/build/bin:/home/eugen/projekte/llama.cpp/ggml/src:/home/eugen/projekte/llama.cpp/ggml/src/ggml-cuda \
  /tmp/llama_fixed_prefix_logits \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-8b/Qwen3-8B-Q4_K_M.gguf \
  prompt-token-ids.csv generated-token-ids.csv llama positions.csv
```

`llama_prefix_top1.cpp` performs the same teacher-forced sweep and records top-1 at every target through 1,348. The emitted CSV's header separator was accidentally escaped by the temporary harness; the raw output and exact one-line correction are retained. The corrected CSV was byte-reproduced by `fix_llama_top1_csv_header.py`. The malformed header did not affect inference or row values.

## Offline analysis

```bash
python3 harness/fix_llama_top1_csv_header.py \
  llama/greedy-fixed-prefix-top1.raw.csv /tmp/greedy-fixed-prefix-top1.csv
python3 harness/analyze_fixed_prefix_logits.py .
```
