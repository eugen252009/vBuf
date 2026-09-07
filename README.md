# vBuf

vBuf is a checked, portable binary substrate for zero-copy native access,
vectorized scanning, and `mmap`-friendly storage. Its canonical structure is
self-navigating: readers can validate and traverse blocks without a mandatory
global index.

**The goal:** separate how data is stored from where its bytes come from, how
much is resident in memory, and how a consumer executes on it. For large models,
this means working toward useful execution without requiring the entire model
in RAM or VRAM.

vBuf itself is generic, not an LLM format. **vBuf-ML** adds optional model
semantics and runtime services on top: metadata, tokenizers, tensor directories,
source resolution, materialization, bounded residency, and backend integration.
Neither layer is defined by a particular model or by llama.cpp/GGML.

## Current Findings

See the **[vBuf-ML current-state summary](research/results/vbuf-ml-current-state-summary.md#latest-findings)**
for the latest two or three major achievements, their important limits, and
links to measured evidence. Larger qualified achievements replace older items
in that snapshot; detailed research records remain available.

The current qualification spans:

| Target | Demonstrated | Boundary |
|---|---|---|
| 112.56 GB GLM-4.5-Air-FP8 on an x86_64 workstation | All 46 layers and retained-KV greedy generation, with independent reference parity | Portable CPU F32 correctness qualification, not a throughput claim |
| GLM on RTX 3060 | Up to eight consecutive real CUDA layers with bounded device residency and selected-expert-only transfers | Not full-stack GPU prefill, decode, or generation |
| 5.64 GB DeepSeek-V2-Lite IQ2_XXS on Pixel 7 Pro | ARM64 direct runtime, prompt prefill, and bounded generation with a 256 MiB residency cap | The cap is vBuf-ML payload residency, not total process RSS |

These are specific qualification cases, not universal model support or minimum
hardware requirements. DeepSeek and GLM are examples, not format definitions.

## How It Works

The architecture keeps semantic identity, metadata, payload source, payload
residency, and backend representation separate:

```text
artifact / semantic bootstrap
    -> vBuf discovery + vBuf-ML semantic bindings
    -> TensorRef(SourceId, checked u64 offset, length)
    -> file / HTTP Range / other RangeSource
    -> materialization + readiness + residency + lease
    -> validated borrowed or materialized tensors
    -> backend compute
```

A **semantic bootstrap** holds discovery metadata and tensor references without
copying the tensor payload. Payload bytes may remain in an external file or
remote range source. A local payload-bearing artifact still uses the `SELF`
source and borrowed `mmap` fast path; remote access does not replace local
zero-copy access.

vBuf-ML acquires required ranges and manages their lifetime. A lease prevents
eviction while a backend uses a tensor. Model size, bootstrap size, bytes
transferred, process RSS, and runtime residency are different measurements.
A small residency cap does not mean the full model fits inside that cap, and
fetching or reloading weights can impose substantial latency.

**Ownership stays with vBuf-ML:** semantic discovery, sources, materialization,
payload ownership, leases, residency, load planning, and request scheduling.
Backends consume ready tensors and own execution mechanics. GGML is an allowed
compute backend; llama.cpp is an oracle and compatibility/qualification target,
not the canonical vBuf-ML loader or owner of acquisition policy.

The portable path lowers imported semantics through `PortableProgram` and
`ExecutionGraph` to validated tensor bindings and backend adapters. Persistent
and execution representations may differ without rewriting the wire contract.

## Format And Design Principles

The normative wire contract is **[vBuf v0.6](spec/spec_0.6.md)**. Earlier
specifications are historical, not alternate current definitions.

- Checked `u64` arithmetic for offsets, lengths, alignment, and host conversions.
- Little-endian fields and payload encodings, with explicit alignment.
- Self-navigation from canonical block geometry; no required index, directory,
  checkpoint, checksum, or finalization artifact.
- Source identity is separate from source location and runtime residency.
- Backend representations and model policy do not redefine persistent layout.
- Optional physical accelerators such as Nano are not guaranteed speedups.

vBuf-ML adds semantic interpretation without creating a second container
format. Its optional source metadata supports source descriptors, locators,
hashes, and checked external tensor bindings. Existing payload-bearing
artifacts without this role continue to resolve through `SELF`.

A structural header-only derivative is **not** a complete semantic bootstrap.
For example, the historical Qwen3-32B proof used about 53 KB of structural
headers but a 4.47 MB semantic bootstrap for a 34.82 GB artifact.

See [vBuf-ML documentation](docs/vbuf-ml/) and the
[Rust profile implementation](rust/vbuf-ml/).

## Runtime And Serving

The direct runtime separates normal inference from qualification:

- **NormalInference:** actual computation only; no reference/oracle work.
- **Qualification:** reference work, parity checks, and fail-closed behavior.

Normal prompt prefill can batch matrix work across prompt rows while keeping
causal attention and KV updates ordered. Routed expert contributions retain
each token's original TopK rank order even when experts are grouped for
execution. Decode remains autoregressive and token-serial.

A recorded Pixel seven-token prefill comparison measured **284.4 s serial vs.
82.2 s batched** (`3.459x`) with the same 256 MiB cap. This is a specific
model/prompt qualification, not a universal speedup; see the
[batching report](research/results/vbuf-android-demo-poc/prompt-prefill-batching.md)
for parity and measurement limits.

The native `integrations/ggml` target `vbuf_compat_server` exposes a thin
OpenAI-compatible subset: `/health`, `/v1/models`, `/v1/chat/completions`, and
`/v1/completions`. It uses vBuf-ML runtime services, not a llama-server proxy.
Production inference remains serial, with isolated request KV state,
cancellation-aware admission, and responsive read-only control endpoints.
Research concurrency seams do not change that policy; the 256 MiB
shared-concurrency case did not qualify.

See the [serving qualification](research/results/vbuf-ml-integration/step31r-vbuf-compat-server-qualification.md),
[persistent-process lifecycle tests](research/results/vbuf-ml-integration/step31s-persistent-server-lifecycle-qualification.md),
and [concurrency limits](research/results/vbuf-ml-integration/step31z-shared-residency-concurrency.md).

## Limits And Historical Evidence

- The portable GLM CPU path converts demanded FP8/BF16 weights into bounded F32
  working sets. CUDA currently uses bounded host F32 staging and F32 uploads;
  Top-K remains host control. Native FP8 device execution and multi-GPU
  execution remain unqualified.
- Android batching retained first-decode parity and router/MoE audit evidence,
  not full hidden-state/KV/logit hashes for every prompt row.
- Request fragmentation, reload amplification, and materialization costs remain
  observable. No universal cache behavior or general prefetch/compute-overlap
  policy is claimed.
- Historical Qwen3-32B tests qualified checked ranges, remote materialization,
  exact payload parity, and FFI on x86_64, ARM32, ARM64, and riscv64—including
  offsets above 4 GiB. They did **not** qualify full 32B inference or model
  compute on every architecture. See the
  [cross-architecture matrix](research/results/vbuf-autoregressive-generation-poc22-x86/cross-architecture-qualification-matrix.md).
- Historical Android llama.cpp loading proved payload parity, EAGER_ALL
  construction, generation, and GGML interoperability. It remains a bounded
  compatibility path, not the canonical direct runtime. Transport controls and
  observer-distorted timings are documented in the
  [Android qualification records](research/results/vbuf-android-arm64-chat-poc/).
- Local-layout tests did not establish universal SIMD or Nano latency benefits.
  Contextual Correction Code (CCC) research is **closed / rejected**; failed
  controls and reopening criteria remain in the
  [CCC research conclusion](docs/vbuf-ml/ccc_research_conclusion.md).

Detailed architecture, measurements, and limitations are in the
[current-state summary](research/results/vbuf-ml-current-state-summary.md).
Raw, failed, and historical evidence remains under [research/results/](research/results/).
The [pre-consolidation README](https://github.com/eugen252009/vBuf/blob/eb63641d723ad9a66299fd822e9325a4a0c5e3c8/README.md)
also preserves the earlier extended tables and transport interpretation.

## Build And Test

From the repository root:

```sh
cargo test --manifest-path rust/Cargo.toml --workspace
python3 scripts/verify_portable_graph_neutrality.py
```

Implementation and integration entry points:

- [Rust workspace](rust/) and [vBuf-ML](rust/vbuf-ml/)
- [GGML adapter](integrations/ggml/ADAPTER.md) and [dependencies](integrations/ggml/DEPENDENCIES.md)
- [Android application and build instructions](integrations/android-vbuf-chat/README.md)
- [Historical llama.cpp compatibility adapter](integrations/llama.cpp/)

Verification is recorded per scope, not implied by this README. The latest
Step 32 records include Rust workspace tests and CUDA device qualification;
the latest CPU/CUDA work did not rerun the APK build. Consult the Android
instructions for the established SDK/JDK/NDK environment and local GGML source
override.

## License

MIT
