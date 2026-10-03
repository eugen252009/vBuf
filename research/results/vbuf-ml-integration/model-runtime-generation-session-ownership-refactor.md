# Canonical Model-Runtime / Generation-Session Ownership Refactor

## Baseline

Before this change, `VbufGenerationSession::Impl` directly owned parsed metadata/plans, lazily initialized source and materializers, residency, executor, request counters, and fault-injection state. `run()` owned the actual/value K/V vectors as locals and ran the whole request in one loop. Existing call sites constructed one `VbufGenerationSession` per model/server and called `run()` repeatedly.

Qwen3 remains **UNSUPPORTED** in canonical generation and production Qwen remains **DISABLED**. The explicit Qwen rejection is unchanged. No Qwen dispatch, qualification code, or qualified CUDA implementation was changed.

## Phase A — ownership map

| Resource | Previous owner/lifetime | Current owner/lifetime | Notes |
|---|---|---|---|
| Parsed model metadata and layer plans | Session `Impl`, effectively model lifetime | `VbufModelRuntime::Impl`, model lifetime | Loaded once when runtime is constructed. |
| Range source / controlled-failure wrapper | Session `Impl`, lazy and reused only by that object | `VbufModelRuntime::Impl`, lazy model lifetime | Still reconfigured if a later request changes endpoint/capacity, preserving existing request-config behavior. Same-config sessions share it. |
| Local and resident materializers | Session `Impl` | `VbufModelRuntime::Impl` | Source/materializer resources live as long as runtime/session references. Per-call leases are released by `run()`. |
| Residency store | Session `Impl` or supplied shared store | `VbufModelRuntime::Impl` | Shared between sessions created from one runtime. |
| Expert executor | Session `Impl`, lazily recreated on worker-setting change | `VbufModelRuntime::Impl` | Execution is serialized for one runtime; concurrent calls are not thread-safe or qualified. |
| Backend handles and per-operation GGML graphs | Constructed/destroyed by existing execution helpers within an operation | Call-local / temporary | No reusable backend handle exists in this CPU production path; not falsely promoted to model lifetime. |
| Request and active-generation counters | Session `Impl` | `VbufModelRuntime::Impl` | Snapshots now aggregate requests from sessions sharing that runtime. |
| Actual K/V `RuntimeStateSlot`s | Locals in `run()` | `VbufGenerationSession::SessionState`, session lifetime | Reset at the start of a whole-request run; updated after each successfully processed position; cleared on failure. |
| Logical context length | Implicit loop position | `SessionState::current_context_length` | Exposed read-only and cleared by `reset()`. |
| Reference/oracle K/V slots | Locals in `run()` | Call-local | These are comparison state, not production session state. |
| Trace ID sets, timing, logits/result vectors, temporary captures | Locals in `run()` | Call-local | Kept local; no unnecessary persistence introduced. |

The Qwen qualification runner's backend/KV/scratch ownership remains a separate qualification-only concern and was not modified.

## Phase B — model runtime

Added `VbufModelRuntime` in `integrations/ggml/include/vbuf_generation.h` and implemented it in `integrations/ggml/tools/autoregressive_poc22.cpp`. It owns one shared `Impl` containing metadata/plans, source/materializer/residency state, executor, and runtime counters. Source setup remains lazy because endpoint/capacity/fault controls arrive through the existing request configuration.

`create_session()` requires the runtime to be `shared_ptr`-owned and returns a session holding a strong runtime reference. The runtime therefore cannot be destroyed while any session created from it remains alive. Destruction follows RAII; a partially failed metadata/planning constructor releases already-created members. A stack-owned runtime is deliberately rejected by `create_session()` rather than yielding a dangling session.

The old `VbufGenerationSession(path, block_count[, shared_residency])` constructors remain as compatibility conveniences; each creates and retains its own runtime. Existing persistent server call sites therefore keep their prior model-scope lifetime. Callers that need multiple sessions can now explicitly share a `VbufModelRuntime`.

## Phase C — generation session and `run()` compatibility

A `VbufGenerationSession` now owns a `SessionState` with actual K/V slots and current logical length, plus a strong reference to `VbufModelRuntime`. `reset()` clears only that session's state. `current_context_length()` exposes the logical count. Destroying the session destroys its K/V slots before releasing its runtime reference.

`VbufGenerationSession::run()` remains the whole-request operation and retains its existing dispatch, generation loop, policies, result/error structure, and materializer lease release. It resets session state at request start, updates current length after each successful `run_sequence`, and resets state on failure. The completed session state remains inspectable until explicit reset, the next `run()`, or destruction. `VbufModelRuntime::run()` and `run_vbuf_generation()` are convenience wrappers that create a temporary session, execute one request, and destroy it; the existing persistent-session call sites remain source-compatible.

No duplicate generation engine was added. The Qwen3 dispatch still returns the same “direct graph builder is not implemented” error. Existing source/materializer/residency behavior is unchanged for same-config repeated requests; the model runtime now supplies their shared lifetime.

## Phase D — lifecycle and existing-model qualification

Added `integrations/ggml/tests/generation_lifecycle_contract.cpp`. It checks session creation from one shared runtime, strong runtime lifetime across external-owner/session teardown, independent session logical lengths/reset, shared request/residency/source counters, stack-owned/null runtime rejection, invalid model construction, failed-request state cleanup, and the `run_vbuf_generation()` empty-prompt error contract.

With the local DeepSeek-V2-Lite IQ2_XXS semantic artifact and local range server, the same contract first injects a range-source failure, confirms the failed session is reset, and then successfully executes two one-block requests through distinct sessions sharing one runtime. Each successful session reaches processed-context length 2; running or resetting one leaves the other's state intact. Source request/byte and residency materialization counters remain monotonic across the successful requests, confirming reuse of the same model-runtime resource set after request failure. This is a focused existing-model regression, not full-model DeepSeek qualification.

| Verification | Result |
|---|---|
| Pinned GGML build, all targets | **BUILT** |
| CTest | **36/36 PASSED** (including the opt-in lifecycle/DeepSeek regression) |
| DeepSeek semantics contract | **PASSED** |
| Lifecycle ownership/failure contract, ASan + UBSan | **PASSED** for metadata/session creation, error, and destruction paths |
| Actual two-session DeepSeek execution | **PASSED**, sequential, one block per session |
| Qwen CPU/CUDA qualification rerun | **NOT RUN**; no Qwen execution code changed |
| Q4_K/Q6_K focused controls | **NOT RUN** |
| DeepSeek full-model regression | **NOT RUN** |
| `git diff --check` | **PASS** |
| `ccc index` | **PASS** |

The contract accepts a semantic-model path and an optional range-source URL. CMake exposes `VBUF_TEST_SEMANTIC_MODEL` and `VBUF_TEST_SOURCE_URL`; without these cache values the binary is built but the artifact-dependent test is not registered. The recorded CTest run used the cached DeepSeek semantic bootstrap and local range endpoint, not the Qwen artifact. The configured integration contract observed three runtime requests (one injected failure and two successful generations).

## Phase H — concurrency and failure contract

Multiple sessions may exist on a shared runtime. Sequential/interleaved session runs were exercised. Concurrent `run()` or `snapshot()` calls on one runtime are **NOT SUPPORTED / NOT THREAD-SAFE**: the shared counters, executor selection, diagnostic clearing, and source reconfiguration remain mutable. The public header documents this. Existing compat-server generation admission serializes work; this change does not expand concurrency guarantees.

Session construction with a null runtime and runtime creation from an invalid model path fail explicitly. Strong shared ownership prevents runtime destruction while sessions remain. Request failures clear only the failing session's K/V/length while leaving the runtime usable; the test then successfully runs the other session. Allocation-failure injection during backend/source creation was not performed.

## Phase J — Qwen integration readiness

- Qwen model-global metadata, source/materializer, residency, and any eventual CUDA backend/execution resources belong in `VbufModelRuntime` (or a Qwen model-runtime component owned by it).
- Qwen persistent GPU K/V, logical position, per-layer prefill scratch, decode scratch, and graph/session state belong in `VbufGenerationSession::SessionState` or a model-specific session-state object owned by the session.
- The qualification runner still needs its resident-CUDA model loading/graph setup and execution code extracted into a shared core in a later task. Do not connect it to production in this refactor.
- The next integration point is runtime/session-specific dispatch behind `VbufModelRuntime::create_session()` while preserving generic `run()` as the compatibility entry. Qwen remains explicitly unsupported until the shared core exists and is qualified.

## Repository and final status

- Production files changed: `integrations/ggml/include/vbuf_generation.h`, `integrations/ggml/tools/autoregressive_poc22.cpp`, `integrations/ggml/CMakeLists.txt`.
- New test: `integrations/ggml/tests/generation_lifecycle_contract.cpp`.
- Existing dirty Qwen qualification and runtime-integration changes were preserved outside this generic ownership refactor.
- Checkpoint commit: `refactor(runtime): split model and generation-session ownership`; Qwen admission remains in parent commit `9ec3928`.
- `stash@{0}` (`android: unverified vbuf_parallel_executor CMake hunk`): untouched.
- `stash@{1}` (`pre-merge local Step-32 reconciliation draft`): untouched.

| Capability | Status |
|---|---|
| Model-runtime / generation-session ownership split | **IMPLEMENTED** |
| Existing whole-request `run()` compatibility | **IMPLEMENTED / TESTED** |
| Sequential model-resource reuse across sessions | **TESTED** with focused DeepSeek run |
| Session logical state/reset/destruction isolation | **TESTED** |
| Same-runtime concurrent execution | **NOT SUPPORTED / NOT TESTED** |
| Qwen3 production session | **UNSUPPORTED** |
| Qwen shared execution core | **NOT IMPLEMENTED** |
| OpenAI HTTP, tools, Pi, dogfood, llama.cpp comparison | **NOT TESTED** |
