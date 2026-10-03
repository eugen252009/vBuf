# Qwen3 CUDA Ownership Objects

## Baseline

- Admission commit: `9ec3928` (`refactor(qwen3): share exact model admission`).
- Generic runtime/session commit: `05389c6091e6cbd19577a39414e732c45252d666` (`refactor(runtime): split model and generation-session ownership`).
- Pinned semantic artifact identity: `sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Production Qwen: **UNSUPPORTED / DISABLED**. No production dispatch or qualification-runner migration was made.

## Runtime state

- Type: `QwenCudaRuntimeState` in `integrations/ggml/include/qwen3_cuda_core.h` and `src/qwen3_cuda_core.cpp`.
- Admission input: an already opened `Qwen3Model`; runtime checks the exact qualified source identity and presence of the shared admission/catalog/materializer. Metadata validation is not duplicated.
- Backend: created once by `QwenCudaRuntimeState::create` (GGML GPU backend); owned by the runtime PIMPL and synchronized/freed in its destructor.
- Residency: the runtime allocates model tensors together, materializes/uploads the admitted model tensors once, registers stable device handles in `TensorResidencyStore`, and exposes both the resident embedding and tensor lookup. Residency leases share the model allocation owner, which is released once after residency clear.
- Immutable configuration: 32-token prefill default, reusable prefill/decode scratch sizes, backend name, and admitted model identity/metadata. Exact-artifact runtime/session resource construction was smoke-tested on CUDA0/RTX 3060; no graph inference was run.
- Lifetime: the runtime borrows the admitted `Qwen3Model`; the model must outlive it. Its tensor GGML bindings are cleared during runtime teardown and on failed construction.

## Session state

- Type: `QwenCudaSessionState` in the same files; `QwenCudaRuntimeState::create_session(capacity)` returns shared ownership of it.
- Runtime reference: retained by `shared_ptr`, preventing a dangling backend/runtime while any session exists.
- KV: each session allocates distinct per-layer F16 K and V tensors in its own GGML context/backend buffer; capacity is bounded to `1..4096`.
- Scratch: session-owned packed-V tensor, reusable prefill scratch buffer sized to the qualified 32-token scratch allocation, and a separate decode scratch buffer. Buffers are distinct across sessions. They are ownership reservations for the next migration; they have not yet been bound to the qualification runner's execution graphs.
- Logical length: starts at zero; `commit_tokens` bounds-checks capacity; `reset()` sets logical length to zero and increments a reset generation without clearing physical KV/scratch allocations.
- Destruction synchronizes the shared backend, frees the session scratch and KV allocation/context, and leaves runtime model residency intact.

## Lifecycle contracts

- Runtime create/destroy: tested using the CPU-backed synthetic fixture.
- Runtime outlives session: tested by dropping the external runtime reference while a session retains it.
- Sequential sessions: tested; runtime backend, embedding handle, and model residency remain stable across session A/B.
- Two sessions: tested distinct K/V tensors, packed-V tensors, prefill/decode scratch buffers, and logical lengths.
- Reset: tested logical reset with physical allocation reuse.
- Partial runtime construction: injected failures after backend, model allocation, and residency setup throw; ownership is RAII-managed.
- Partial session construction: injected failures after KV, prefill-scratch, and decode-scratch allocation throw; runtime remains valid.
- Exact-artifact CUDA resource smoke: **PASS**; the full resident setup was created and torn down with 443 tensors / 8,995,793,920 resident payload bytes, embedding resident, and a capacity-40 session with KV and scratch allocated. No inference was run.

## Next qualification-runner migration map

| Runner-owned resource | New owner/API |
|---|---|
| CUDA/GGML backend | `QwenCudaRuntimeState::backend()` |
| Model tensor/device residency | `QwenCudaRuntimeState::residency()` and runtime model allocation |
| Embedding | `QwenCudaRuntimeState::embedding()` |
| Named model tensors | `QwenCudaRuntimeState::tensor(name)` |
| Persistent K/V | `QwenCudaSessionState::key_cache(layer)` / `value_cache(layer)` |
| Logical position | `QwenCudaSessionState::current_length()` / `commit_tokens()` / `reset()` |
| Per-layer prefill scratch | `QwenCudaSessionState::prefill_scratch()` |
| Decode scratch | `QwenCudaSessionState::decode_scratch()` |
| Packed V workspace | `QwenCudaSessionState::packed_value_scratch()` |

The qualification runner still owns and executes its current backend/KV/scratch path. The next task must migrate it to these objects and bind its graph tensors to their allocations; this change intentionally does not claim inference qualification for the new owners.

## Preserved classifications

- Memory scaling: **MEMORY-SCALING QUALIFIED**; historical 409,728 B/capacity-token pathology remains absent in the qualified runner, with the qualified 25,220 B/capacity-token non-KV workspace slope.
- Incremental/full CUDA: **EXPECTED SHAPE-DEPENDENT CUDA NUMERICAL DIVERGENCE**.
- Chunked/full CUDA: **EXPECTED CUDA KERNEL-DISPATCH-DEPENDENT PREFILL DIVERGENCE**.
- CPU/CUDA prefix 25: **UNRESOLVED CONTEXT-SENSITIVE ACCUMULATED DIVERGENCE**.
- External llama.cpp strict `1e-5`: **FAIL**.

## Verification

- CUDA-enabled build: **PASS** for `vbuf_qwen3_cuda_ownership_contract` and `vbuf_qwen3_block_qualification` against pinned GGML `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- Pinned payload SHA: verified by `sha256sum` and shared admission as `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- CTest: **39/39 PASS**.
- Ownership contract: **PASS** on CPU backend synthetic fixture.
- ASan/UBSan ownership contract: **PASS** (`detect_leaks=1`, `halt_on_error=1`).
- Real Qwen CUDA ownership smoke: **PASS** for construction, residency, embedding, capacity-40 KV/scratch allocation, and teardown; inference/transfer counters: **NOT TESTED**.
- Qualification-runner migration: **NOT IMPLEMENTED**.
- Production Qwen: **UNSUPPORTED / DISABLED**.
