# Qwen3 CUDA Residency / Transfer Audit — Gate 1

Status: **GATE 1 BLOCKED FOR DEPTH PROGRESSION.** The current bounded Qwen3 CUDA run is an isolated diagnostic, not a resident vBuf Qwen runtime. It transfers weights and intermediate activations through host memory at multiple stage boundaries, synchronizing on every tensor transfer. The path is understood and quantitatively measured; scaling this implementation to additional layers would produce a misleading GPU qualification. No production behavior changed.

## Scope and method

- Real artifact: Qwen3-14B Q4_K_M vBuf, SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- One complete block (`blk.0`), Sequence-A prefix 25, CUDA device 0 (RTX 3060, SM 8.6), pinned GGML `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, one thread. The Q projection used the persisted common F32 activation override so the real Q4_K baseline remained reproducible. No second block or depth progression was run.
- This is the diagnostic copy of `qwen3_block_qualification`, with the out-of-tree backend selector selecting GGML CUDA. It is not `VbufGenerationSession`, and is not the production Qwen3 path.
- A process-local `LD_PRELOAD` audit interposed CUDA Runtime `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMalloc`, `cudaFree`, and synchronization calls. H2D/D2H copy events and allocation totals are persisted under `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/`. Pinned GGML source inspection confirms its CUDA tensor set/get handlers issue `cudaMemcpyAsync` followed immediately by `cudaStreamSynchronize` (`ggml-cuda.cu`, `ggml_backend_cuda_buffer_{set,get}_tensor`).
- The probe runs with internal-only diagnostic controls and Q output capture enabled. In particular, the Q weight and FFN-down weight device readbacks and many activation/checkpoint readbacks are diagnostic overhead, not necessary production transfers. Reported totals describe this exact diagnostic invocation, not a production session.

## Device and allocation observations

| Measurement | Result |
|---|---:|
| GPU used for the run | RTX 3060, SM 8.6 |
| Physical VRAM | 12,288 MiB |
| Free / used before run | 12,027 / 12 MiB |
| Free / used after run | 12,027 / 12 MiB |
| Runtime `cudaMalloc`/`cudaMallocAsync` allocations | 7 allocations; all 7 freed before process exit |
| Peak simultaneously live runtime allocations | 597,986,432 bytes (~570.4 MiB) |
| Live runtime allocation bytes at exit | 0 |
| Persistent model-weight residency after the block | 0 bytes |
| Whole-model GPU preload | NO |

The 597,986,432-byte peak is the peak of device allocations observed through the interposed CUDA Runtime allocation APIs; it includes backend buffers/workspace and activations and is **not** a measured weight-only subtotal. It may not include driver-internal/context allocations. The 9.0-GB vBuf artifact size is not treated as GPU residency. Device free memory was sampled before and after, not continuously during the short run.

There is no separate GPU model-initialization phase in this diagnostic. GGML backend buffers are allocated when each temporary graph context is built, used for that stage, and freed with the context. The first and second block contexts overlap in lifetime; Q/K/V and gate/up projection contexts are additional temporary allocations. No weights remain resident across blocks or requests.

## PCIe transfer accounting — one block, prefix 25

| Direction / class | Calls | Bytes |
|---|---:|---:|
| H2D model-weight payloads | 12 | 647,742,464 |
| H2D activations and controls | 42 | 9,625,912 |
| **H2D total** | **54** | **657,368,376** |
| D2H diagnostic weight readbacks | 2 | 87,859,200 |
| D2H activations/checkpoints/output | 28 | 16,895,296 |
| **D2H total** | **30** | **104,754,496** |

The 12 uploaded weight tensors are the complete token-embedding table, three first-stage norm vectors, Q/K/V projection weights, attention-output and FFN-norm weights, FFN-down weight, and FFN gate/up weights. The embedding tensor alone transfers **437,575,680 bytes** to select 25 token rows. The Q/K/V projection weights and gate/up weights are uploaded into separate temporary CUDA contexts; attention output and FFN-down weights are uploaded into the second-stage context. The Q and FFN-down whole-weight D2H reads are identity/diagnostic checks.

The activation/control bytes include three copies of the 512,000-byte Q input (one to each independent Q/K/V projection context), Q/K/V projection outputs downloaded to host and then re-uploaded into the first block context, attention context and hidden state crossing into the second context, FFN norm sent to gate/up contexts, and gate/up outputs returned to the second context. They also include token IDs, position IDs, causal mask, and the internal-only zero reference-probability diagnostic input.

### Explicit and implicit synchronization

- CUDA Runtime observed **131 `cudaStreamSynchronize` calls**, zero `cudaDeviceSynchronize`, and zero `cudaEventSynchronize` calls.
- The 54 H2D plus 30 D2H transfers each cause an immediate stream synchronization in the pinned GGML CUDA tensor buffer handlers: **84 transfer-associated synchronizations**.
- The active Qwen diagnostic call path contains 36 explicit `ggml_backend_synchronize` calls, including a second synchronization after each `get_f32` readback; the CUDA tensor-get handler already synchronizes its transfer. The remaining 11 observed stream synchronizations are backend/graph/runtime synchronization activity not separately attributed by this probe. Thus, the measured low-level count is 131; it is not represented as only the source-level calls.
- Each transfer is therefore blocking at the CUDA stream boundary. H2D/D2H work cannot be usefully overlapped across the host staging points in this path.

## Residency by tensor class

| Tensor class | Observed residency and movement |
|---|---|
| Model weights | Acquired/materialized in host memory by the diagnostic vBuf source/materializer. Twelve weight tensors are copied H2D (647,742,464 bytes total) into temporary GGML CUDA graph contexts. Q/K/V and gate/up projection contexts are separate; there is no cross-layer/request GPU weight cache. The embedding table is uploaded in full for 25 selected rows. |
| Input hidden / token embedding | Token IDs are host-originated and copied as 25 separate 4-byte writes. The embedding table and embedding result are GPU-resident in the first context. Its attention-normalized F32 result is read back to host; the fixed Q/K/V activation is uploaded independently to each projection context. For later layers, the previous block returns a host F32 hidden vector and the next block uploads it again. |
| Q/K/V outputs | Each projection runs in its own temporary CUDA context. Its result is downloaded to a host vector, then uploaded again into the first block context. Additional checkpoint readbacks download Q/K/V intermediates for diagnostics. |
| Attention intermediates | Q/K normalization, RoPE, F16 K/V for the current prefix, scores, probabilities, and attention context execute in the first CUDA context. Diagnostics download several checkpoints. The attention context and hidden state cross to host and are uploaded to the separate FFN context. |
| FFN intermediates | Attention output/residual/FFN norm execute in the second context. FFN norm crosses to host for separate gate/up CUDA contexts; gate/up outputs are downloaded and re-uploaded for SwiGLU/down projection. Down output and final block output are read back for diagnostics and return. |
| KV | This prefix-25 block keeps its temporary F16 K/V tensors on the first context only for the duration of this call; they are freed at block exit. GPU persistent KV was not exercised. The isolated `Qwen3KvCache` stores K/V in host `std::vector<uint8_t>` buffers; when that path is paired with this backend it must upload past K/V and download appended K/V. It is not GPU-resident decode. |
| Logits | Not created in this one-block run. The qualification tool calls its final-head path only after all 40 blocks; CUDA logits residency/transfer is therefore NOT TESTED here. |

## Architecture finding and gate decision

This diagnostic exhibits the exact undesirable pattern the residency gate is intended to detect:

```text
host materialization -> CUDA weight upload
CUDA projection -> host output -> CUDA block-context upload
CUDA attention context -> host -> CUDA FFN-context upload
CUDA gate/up -> host outputs -> CUDA FFN-context upload
CUDA block output -> host
```

It also uploads the full 437,575,680-byte embedding tensor and serializes each set/get with a stream synchronization. The diagnostic readbacks inflate D2H volume, but even excluding the two weight readbacks and checkpoint-only traffic, the model's projection/attention/FFN boundaries still force device-host-device staging. This implementation must not be depth-scaled.

The production `VbufGenerationSession` in `integrations/ggml/tools/autoregressive_poc22.cpp` explicitly rejects Qwen3 before inference because its direct Qwen graph builder is not implemented. The separate Rust generic CUDA graph backend represents device tensors as F32 and has not been connected to a Qwen3 production session or Q4_K model-weight path. The successful Qwen3 CPU qualification and the diagnostic GGML CUDA backend do not fill that production/runtime seam.

**Gate 1 disposition: BLOCKED for Phase 2.** The audit established a concrete transfer/residency design requirement, but the currently runnable Qwen CUDA path is not a suitable base for progressive depth. Before progressing, the canonical vBuf-owned runtime needs a Qwen execution path that retains/materializes weights under vBuf policy, keeps dependent activations/device KV resident across the block/token work where practical, and returns to host only at genuine API/control boundaries. GGML CUDA may remain a numerical primitive/backend; it must not acquire model acquisition, scheduling, or residency ownership. No CUDA depth, full-model, replay, persistent-device-KV, performance, HTTP, tool, or Pi gate was attempted after this blocker.

## Requested readiness snapshot at this stop point

- CUDA residency: **BLOCKED** — current diagnostic path measured; unacceptable repeated host/device staging; no persistent GPU weights/KV.
- GPU depth: **BLOCKED** — one block only; depths 2/4/8/16/40 not tested.
- GPU full model/logits: **NOT TESTED**.
- GPU incremental replay / persistent KV: **NOT TESTED**; current isolated KV storage is host-resident.
- Production `VbufGenerationSession` Qwen3: **BLOCKED** — explicitly rejected by direct graph builder.
- OpenAI text endpoint with real Qwen GPU: **NOT TESTED**.
- Native Qwen tools / Pi text / Pi tool round trip / coding dogfood: **NOT TESTED**.
- Strict external llama.cpp `1e-5` gate: **FAIL; unchanged**. Production Qwen3 remains **DISABLED**.
- CPU Qwen, canonical attention, replay, persistent CPU KV, ExecutionPolicy, and prior CPU qualification evidence: unchanged; no regressions were run in this audit-only step.

## Evidence and worktree boundary

Raw CUDA transfer counters/events and run logs are under `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/`. The CUDA shim and temporary patched diagnostic build are outside the repository. This report is an uncommitted investigation artifact; no production source or runtime behavior was changed. Existing Q4_K fixture/report work remains uncommitted. Both existing stashes remain untouched.
