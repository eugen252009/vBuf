# vBuf

vBuf is a self-navigating binary substrate for validated persistent structure
whose payload source, residency, and execution representation can remain
independent runtime concerns. It is designed so a reader can validate and
navigate canonical structure without a mandatory global index, while a runtime
can materialize only the ranges currently required by a consumer.

vBuf itself is a generic binary substrate. [`vbuf-ML`](rust/vbuf-ml/) is an
optional semantic/profile layer for model metadata, tokenizers, tensor
directories, and external tensor sources. vBuf is not defined by machine
learning or by any particular backend such as llama.cpp or ggml.

## Current Runtime Architecture

vBuf-ML is the optional ML semantic/runtime layer above the generic vBuf
substrate. Its canonical direct-runtime path is:

```text
artifact / semantic bootstrap
        |
        v
vBuf discovery
        |
        v
vBuf-ML semantic bindings
        |
        v
TensorRef(SourceId, offset, length)
        |
        v
source resolution
        |
        v
materialization / readiness
        |
        v
bounded residency + lease
        |
        v
portable/backend execution
        |
        v
GGML compute
```

vBuf-ML owns semantic tensor discovery, source resolution, physical ranges,
materialization, readiness, payload lifetime, leases, residency, scheduling,
and runtime integration. A backend consumes ready borrowed or materialized
tensors and owns execution mechanics. GGML is an allowed execution backend and
qualification target, not the owner or definition of vBuf-ML acquisition policy.

The portable execution boundary is:

```text
Importer -> PortableProgram -> generic lowering -> ExecutionGraph
    -> TensorBindings -> materialization/readiness/residency
    -> ready PersistentTensorRef + lease -> C ABI
    -> generic backend adapter -> compute
```

## External Payloads And Residency

Semantic metadata and tensor payload need not be colocated:

```text
semantic-bootstrap.vbuf -> TensorRef(SourceId, offset, length)
    -> file / HTTP Range / other RangeSource -> bounded materialization
```

A semantic bootstrap can contain metadata and TensorRefs while payload bytes
remain in an external file or range source. A local payload-bearing artifact
continues to use the `SELF` source and borrowed mmap path. The runtime can keep
only required tensor payloads resident even when the model is much larger than
available memory. Process RSS and vBuf-ML residency are different metrics.

The current Android qualification used a `5,639,819,878`-byte DeepSeek-V2-Lite
IQ2_XXS vBuf payload with a `268,435,456`-byte (`256 MiB`) vBuf-ML residency
cap. The full payload was not loaded into 256 MiB.

## Runtime Modes

The direct runtime has explicit modes:

| Mode | Behavior |
|---|---|
| `NormalInference` | Actual runtime computation only; reference/oracle work is not executed. |
| `Qualification` | Actual plus reference/oracle computation, parity checks, and fail-closed behavior. |

Qualification controls must not be mistaken for production inference cost.
Qualification remains serial. Normal inference does not execute reference work.
The Android direct runtime keeps its unchanged single-position decode path;
the portable GLM runtime separately qualifies retained-KV greedy generation.

## OpenAI-Compatible Serving

The native `integrations/ggml` target `vbuf_compat_server` exposes a thin
llama-server/OpenAI-compatible subset: `/health`, `/v1/models`,
`/v1/chat/completions`, and `/v1/completions`. The HTTP layer translates
protocol DTOs into the existing vBuf-ML tokenizer, TensorRef, RangeSource,
materialization, TensorWave, and GGML execution path. llama-server is not
required as a runtime or proxy.

Step 31R qualified both a bounded two-block request path and a one-token
full-26-layer host request. Normal serving defaults to `NormalInference`, keeps
request KV state isolated, serializes active generation, and preserves external
RangeSource plus bounded residency behavior. Unsupported generation semantics
are rejected rather than silently ignored. Evidence is in
[`step31r-vbuf-compat-server-qualification.md`](research/results/vbuf-ml-integration/step31r-vbuf-compat-server-qualification.md).

Step 31S additionally qualified one persistent server process across a 20-request
baseline and a 40-request follow-up soak with one deterministic source failure
and same-process recovery, repeated SSE streams, real disconnect cancellation,
failure recovery, serialized simultaneous clients, and Python OpenAI client
traffic. Request leases, generation state, streams, and inflight materialization
returned to zero after each logged request. Model payload residency intentionally
persisted within the configured cap; diagnostic histories are trimmed between
requests. Step 31W now separates the read-only control plane from the serialized
inference plane: `/health` and `/v1/models` remain responsive while one
generation is active, without enabling concurrent inference. The lifecycle
evidence remains in [`step31s-persistent-server-lifecycle-qualification.md`](research/results/vbuf-ml-integration/step31s-persistent-server-lifecycle-qualification.md);
the dispatch qualification is in
[`step31w-control-plane-separation.md`](research/results/vbuf-ml-integration/step31w-control-plane-separation.md).
Step 31X adds cancellation-aware serial admission without changing the
one-generation production policy. Steps 31Y and 31Z qualify isolated-session
and shared immutable-residency concurrency on x86_64 as research seams, but
production serving remains serial; the 256 MiB shared-concurrency case did not
qualify. See [`Step 31X`](research/results/vbuf-ml-integration/step31x-cancellable-admission.md),
[`Step 31Y`](research/results/vbuf-ml-integration/step31y-backend-concurrency-feasibility.md),
and [`Step 31Z`](research/results/vbuf-ml-integration/step31z-shared-residency-concurrency.md).

## Portable FP8, Full-Stack, And CUDA Qualification

The latest portable-runtime lineage uses the pinned public
`zai-org/GLM-4.5-Air-FP8` artifact at revision
`f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc`. The direct-range Safetensors
import produced a canonical `112,563,538,898`-byte vBuf artifact without a
complete local Safetensors copy. Its source payload exceeded measured host RAM
plus aggregate GPU VRAM. A separate `14,306,259`-byte semantic sidecar persists
tokenizer, tensor, FP8 scale, MoE, and source bindings without rewriting the
payload.

The generic Rust portable runtime has qualified:

- real persisted text through all 46 base transformer layers to logits;
- retained-KV one-token decode;
- eight consecutive greedy decode steps with the same generated token sequence,
  routing, argmax, and top-10 ordering as an independent persisted-range NumPy
  reference; and
- selected-expert-only acquisition with bounded layer-local converted-weight
  residency and cleanup.

The generated eight-token continuation for the fixed input `Test` was:

```text
Test 1: 1. The sum
```

This is a CPU F32 portable execution qualification, not GGML parity or a
throughput claim. Full evidence is in
[`Step 32C`](research/results/vbuf-ml-integration/step32c-real-fp8-oversubscription.md),
[`Step 32D`](research/results/vbuf-ml-integration/step32d-glm-semantic-sidecar.md),
[`Step 32H`](research/results/vbuf-ml-integration/step32h-real-text-full-stack.md),
and [`Step 32J`](research/results/vbuf-ml-integration/step32j-repeated-autoregressive-generation.md).

The backend-neutral device contract and CUDA adapter have separately qualified
one complete real GLM block and progressive 2/4/8-layer execution on an RTX
3060. Layer outputs remain device-resident between layers, selected-expert
parity passes, unselected expert transfers are zero, and logical device
residency plateaus at approximately 203 MB in the eight-layer gate. Persistent
FP8 weights are currently materialized through bounded host F32 staging and
uploaded as F32; Top-K remains host control. Full-stack CUDA prefill, CUDA
decode/generation, native FP8 device execution, and multi-GPU execution remain
unqualified. See [`Step 32K-A`](research/results/vbuf-ml-integration/step32k-a-cuda-device-block.md)
and [`Step 32K-B`](research/results/vbuf-ml-integration/step32k-b-progressive-cuda-layers.md).

## Android Qualification And Performance

The current Android qualification target is a Pixel 7 Pro running Android 17, arm64-v8a,
with DeepSeek-V2-Lite IQ2_XXS, a seven-token prompt, and a 256 MiB residency
cap. The prompt was:

```text
Explain the purpose of bounded generation
```

The controlled runtime-mode split is:

| Mode | Position time |
|---|---:|
| Qualification | 73,588 ms |
| Normal position 0 | 35,397 ms |
| Normal position 1 | 40,060 ms |

Normal position 0 versus qualification is a derived `2.079x` speedup and
`51.9%` wall-clock reduction. The directly measured removed reference
component subtotal is `23,772 ms`; the observed wall-clock delta is `38,191 ms`,
with `14,419 ms` unattributed or not separately instrumented. The entire delta
is not attributed to reference compute. An earlier `~109.7 s` position is a
historical qualification-heavy diagnostic, not this controlled baseline.

The measured full-prompt normal-inference comparison is:

| Seven-token prompt prefill | Time | Human-readable |
|---|---:|---:|
| Serial | `284,415 ms` | `284.4 s`, about `4 min 44.4 s` |
| Batched | `82,231 ms` | `82.2 s`, about `1 min 22.2 s` |

This is a measured physical Pixel result for the stated model, prompt, source,
runtime, and cap: `3.459x` speedup, `71.1%` wall-clock reduction, and
`202,184 ms` saved, about `3 min 22.2 s`. It is not a universal speedup claim.

The batched path is layer-major. Causal attention and KV updates remain ordered
per prompt position, while FFN, router, shared expert, and grouped routed-expert
work uses larger GGML operations across prompt rows. This is not full
position-level parallelism. Grouping experts for execution must not change the
semantic reduction order: each token's routed contributions are accumulated in
original TopK-rank order because floating-point addition is order-sensitive.

The seven-token batched run completed through decode and matched the serial
control's first continuation:

```text
serial:  -
batched: -
```

Detailed evidence is in [`vBuf-ML current-state summary`](research/results/vbuf-ml-current-state-summary.md)
and [`prompt-prefill-batching.md`](research/results/vbuf-android-demo-poc/prompt-prefill-batching.md).

The original serial prefill was position-major and token-by-token:

```text
token 0 -> all layers
token 1 -> all layers
...
token N -> all layers
```

The current batching result changes backend matrix granularity, not model
semantics. It does not make the whole transformer or decode loop fully
parallel.

## Current Capabilities

| Capability | Status |
|---|---|
| Generic vBuf substrate | Implemented |
| External `RangeSource` payloads | Implemented |
| vBuf-ML semantic bootstrap and TensorRef ranges | Implemented |
| Bounded materialization, readiness, and leases | Implemented |
| Bounded residency | Implemented |
| Portable program/lowering and generic GGML adapter | Implemented and contract-qualified |
| Streaming Safetensors-to-vBuf import | Implemented; real 112.56 GB FP8 artifact qualified |
| Persistent block-scaled F8_E4M3 plus semantic scale bindings | Implemented and qualified |
| Portable CPU full-stack text-to-logits | Qualified on 46-layer GLM-4.5-Air-FP8 |
| Portable retained-KV autoregressive generation | Eight greedy decode steps qualified against an independent reference |
| Generic CUDA device backend | Complete real block and progressive 2/4/8 layers qualified |
| Full-stack CUDA prefill/decode/generation | Not qualified |
| Android arm64 direct runtime | Physically qualified |
| Android application harness | Physically qualified over the direct runtime; see [`Android app baseline`](research/results/vbuf-android-demo-poc/android-app-baseline.md) |
| Normal/Qualification runtime modes | Implemented and qualified |
| Layer-major prompt batching | Implemented and physically qualified in normal mode |
| Production server inference concurrency | Serial; research concurrency seams do not change policy |

DeepSeek-V2-Lite and GLM-4.5-Air-FP8 are qualification examples, not
model-specific format definitions. Backend/model neutrality remains a design
requirement.

The importer plans canonical vBuf layout from bounded Safetensors metadata and
writes exact tensor bytes from bounded HTTP ranges directly into final aligned
destinations without retaining a complete local shard or second model-sized
payload. Step 32A established the generic path; Steps 32B and 32C added
persistent block-scaled F8_E4M3 semantics and physically qualified the real
oversubscribed artifact. See
[`Step 32A`](research/results/vbuf-ml-integration/step32a-hf-safetensors-streaming-vbuf-conversion.md)
and [`Step 32B`](research/results/vbuf-ml-integration/step32b-f8-e4m3-block-scaled.md).

## Closed Research Directions

Contextual Correction Code (CCC) research is **closed / rejected** after four
independent lineages were preserved and reconciled. Contextual baselines and
power geometry did not survive strong controls; C3 and plain C4 are numerically
dominated, and the claimed C3 runtime advantage failed a native-kernel fairness
audit. Packed sub-byte direct compute remains a positive systems result. The
unreplicated sparse C4 residual-tail point is evidence only and does not
authorize implementation. The canonical decision and reopening criteria are in
[`CCC Research Conclusion`](docs/vbuf-ml/ccc_research_conclusion.md).

## Current Limitations

- Causal attention and KV state transitions remain ordered per prompt position.
- Autoregressive decode remains token-serial by definition.
- Full hidden-state, KV, and final-logit hashes for every Android batched prompt row were not retained; first-decode parity and router/MoE audit were recorded.
- The portable GLM CPU path materializes demanded FP8/BF16 weights to bounded F32 working sets; it is correctness qualification, not production performance qualification.
- CUDA is qualified only through eight consecutive real layers. Full-stack CUDA text prefill, retained-KV decode, generation, native FP8 execution, and multi-GPU remain open.
- CUDA Top-K is host control in the qualified path; inter-layer activations otherwise remain device-resident.
- Production OpenAI-compatible serving remains serial despite bounded x86_64 concurrency feasibility evidence.
- Materialization and request/reload amplification remain observable; no general prefetch or compute/materialization overlap policy was selected.
- Backend thread configuration was not separately tuned; the Android configuration keeps OpenMP disabled.
- The APK was not rebuilt during the final historical direct-probe qualification; the ARM64 direct probe was built and physically run. This is not a claim that the documented Android toolchain is currently unavailable.

## Performance Progression

| Stage | Result | Classification |
|---|---:|---|
| Qualification-heavy diagnostic position | `~109.7 s` | Historical diagnostic |
| Controlled Qualification position | `73,588 ms` | Measured |
| Controlled Normal position 0 | `35,397 ms` | Measured |
| Controlled Normal position 1 | `40,060 ms` | Measured |
| Serial seven-token full prefill | `284,415 ms` | Measured |
| Batched seven-token full prefill | `82,231 ms` | Measured |

The mode split isolates qualification/reference overhead. The full-prefill
comparison isolates batching against serial normal inference; the rows answer
different questions and are not cumulative speedups.

The largest measured isolated runtime optimization is batched prompt prefill:
`3.459x` and `71.1%` for this qualification case. Runtime-mode separation
established the normal-inference baseline by removing qualification/reference
execution: `2.079x` at the controlled position boundary. These scopes are not
combined into a cumulative speedup.

## Core Model

The architecture keeps these concepts separate:

```text
semantic tensor identity
        !=
metadata artifact
        !=
payload source
        !=
payload residency
        !=
backend execution representation
```

Likewise, these sizes are different measurements:

```text
persistent model size
        != bootstrap transfer size
        != local storage requirement
        != process virtual address space
        != resident working set
        != active compute memory
```

The common external-source path is:

```text
semantic-bootstrap.vbuf
        |
        v
normal vbuf-ML discovery
        |
        v
TensorRef(SourceId, u64 offset, u64 length)
        |
        v
SourceSet / RangeSource
        |
        v
file or HTTP Range
        |
        v
bounded materialization
        |
        v
pointer + length + provenance + lease
        |
        v
source-agnostic backend or compute
```

The local path remains supported and keeps its borrowed fast path:

```text
full model.vbuf
        |
        v
SELF source -> mmap -> CheckedRange -> borrowed payload
```

Remote/bootstrap loading is therefore not a replacement for local mmap loading.
They are two source and residency strategies under the same semantic model.

## Design Principles

- Canonical structure is self-navigating: the next block follows checked geometry, not a required index.
- Wire offsets and arithmetic stay in the checked `u64` domain until a host pointer or allocation is explicitly created.
- Alignment is explicit and orthogonal to payload representation.
- Persistent identity is independent of whether bytes are mapped, borrowed, cached, or materialized.
- Source identity is separate from source location. A `SourceId` identifies the authoritative byte artifact; a locator describes where bytes may be fetched.
- Materialization is bounded and lazy. A remote logical offset may exceed 4 GiB while the local materialized span remains checked against the host `usize`.
- Backend representations may differ from persistent representations without changing the vBuf artifact.
- Existing payload-bearing artifacts remain usable through the `SELF` source default.

## vBuf v0.6

The stable generic wire contract is [`spec/spec_0.6.md`](spec/spec_0.6.md).
Its high-level rules are:

- little-endian fields and payload encodings;
- version code `0x00060000`;
- minimum global header size of 24 bytes;
- `BaseShift` in `3..=8`, giving `BaseStep` values from 8 through 256 bytes;
- `PayloadAlignment = 1 << (BaseShift + PayloadShift)`;
- `next_block_start = align_up_checked(payload_end, BaseStep)`;
- checked `u64` arithmetic for offsets, sizes, alignment, and conversions;
- no required index, directory, checkpoint, checksum, or finalization artifact.

Optional physical structures may accelerate a particular workload, but they are
not normative v0.6 requirements. In particular, Nano is an optional physical
or topological acceleration structure, not a semantic index and not a
guaranteed speedup.

## vbuf-ML Profile

vbuf-ML adds semantic interpretation without creating a second container
format. Its additive source profile uses:

```text
RegionRole::SourceMetadata = 7
```

The profile can persist:

- `SourceDescriptor` and `SourceId`;
- file, HTTP, or other `SourceLocator` values;
- optional binary source hashes, with zero or multiple hashes allowed;
- external tensor bindings represented as checked `TensorRef` values.

Older payload-bearing artifacts without this optional role remain compatible and
resolve their tensor payloads through `SELF`.

The vbuf-ML implementation and focused profile documentation are in
[`docs/vbuf-ml/`](docs/vbuf-ml/) and [`rust/vbuf-ml/`](rust/vbuf-ml/).

## Semantic Bootstrap

A structural header-only derivative and a semantic bootstrap have different
purposes. The structural derivative measures a lower bound for generic
navigation. A semantic bootstrap contains the metadata needed for complete
vbuf-ML discovery while copying zero tensor payload bytes.

| Artifact | Full artifact | Structural derivative | Semantic bootstrap |
|---|---:|---:|---:|
| Qwen3-0.6B-Q8_0 | 637,925,504 bytes | 24,288 bytes | 4,438,480 bytes |
| Qwen3-32B-Q8_0 | 34,816,197,376 bytes | 52,872 bytes | 4,472,327 bytes |

The approximately 53 KiB 32B result is therefore a structural lower bound, not
a complete semantic model bootstrap. The 32B bootstrap contains complete
bootstrap, model metadata, tensor directory, tokenizer, and source metadata;
its tensor payload bytes copied during discovery are zero.

## Historical Cross-Architecture Storage/Source Qualification

The following table records an earlier shared Qwen3-32B portability
qualification. It primarily proves storage, source, addressing, materialization,
and FFI portability for that test scope; it is not the current Android direct-
runtime capability matrix. The later DeepSeek ARM64 direct-runtime and batched
prefill qualification is documented in the current Android section above.

The same 32B semantic bootstrap was parsed on four real architectures. The
shared normalized discovery digest for that historical test is
`f731868eafcd1830244c4dbb28c198b13a204e468adb1d89859c127eccce1f3f`.

| Capability | x86_64 | ARM32 / ARMv7 | ARM64 / AArch64 | riscv64 |
|---|---:|---:|---:|---:|
| Generic vBuf parse | PASS | PASS | PASS | PASS |
| Semantic bootstrap parse | PASS | PASS | PASS | PASS |
| Discovery digest parity | PASS | PASS | PASS | PASS |
| u64 TensorRef | PASS | PASS | PASS | PASS |
| Real >4 GiB tensor | PASS | PASS | PASS | PASS |
| HTTP Range source | PASS | PASS | PASS | PASS |
| File RangeSource | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| Exact payload hash parity | PASS | PASS | PASS | PASS |
| Bounded materialization | PASS | PASS | PASS | PASS |
| Committed FFI materialization | PASS | PASS | PASS | PASS |
| Real ggml descriptor | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| Bounded ggml compute | PASS | NOT_QUALIFIED | PASS (Android local) | NOT_QUALIFIED |
| Full external token path | NOT_RUN | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |

Hardware identities and detailed raw results for this historical qualification:

- x86_64: development workstation. See [`research/results/vbuf-autoregressive-generation-poc22-x86/`](research/results/vbuf-autoregressive-generation-poc22-x86/).
- ARM32: Cubietech Cubietruck Plus, Allwinner A83T, `armv7l`, 32-bit userspace. See [`ARM32 evidence`](research/results/vbuf-cross-architecture/arm32/).
- ARM64: Google Pixel 7 Pro, GS201, Android 17, Termux, `aarch64`. See [`ARM64 evidence`](research/results/vbuf-cross-architecture/arm64/).
- riscv64: Orange Pi RV2, Ky X1 / `ky,x60`, Ubuntu 24.04.4, `riscv64`. See [`RISC-V evidence`](research/results/vbuf-cross-architecture/riscv64/).

Within this historical matrix, the table proves storage, source, addressing,
materialization, and committed FFI portability. Its `NOT_QUALIFIED` and
`NOT_RUN` cells remain valid for that exact Qwen3-32B test scope. They do not
override the later current DeepSeek ARM64 direct-runtime qualification, and
they do not imply that every architecture has a qualified llama/ggml backend or
token-generation path.

The ARM64 Android local-compute result is an earlier Phase A Qwen3-0.6B proof:
the Pixel 7 Pro opened the local vBuf artifact and completed bounded real
autoregressive generation. That historical Qwen result did not qualify remote
external token generation; the later DeepSeek direct-runtime qualification
documented above does. See [`Android Phase A evidence`](research/results/vbuf-android-arm64-chat-poc/).

The historical Android Phase B proof adds a direct Wi-Fi HTTP Range source: the Pixel 7 Pro
opened the Qwen3-0.6B semantic bootstrap without the full model artifact on the
device, materialized the external tensor payloads, and completed bounded real
autoregressive generation. Tokenizer text/type, merge, and all 310 tensor
payloads matched the direct source. The 53.203 s model open transferred
633,495,552 bytes in 310 HTTP Range requests under the pinned EAGER_ALL backend
construction behavior; remote transfer and residency are not optimized yet.
See [`Android Phase B evidence`](research/results/vbuf-android-arm64-chat-poc/phase-b-remote-chat.json).

### Historical Android Remote Loading (Qwen Phase B/C/D)

The following Phase C/D measurements are historical Qwen remote-loading
evidence, not the current DeepSeek prompt-prefill result. Phase C measured
remote-load transport without changing `EAGER_ALL` residency.
The historical Phase C large-range probe reached `29.763 MB/s`, but used Pixel
toybox `netcat` and a different direct HTTP Range benchmark scope. A separate
ARM64 source-transfer snapshot used ADB reverse HTTP; neither historical result
is directly comparable to the later Android `HttpRangeSource` model-loading
path. The derived `40.0%` ratio is therefore not used as current transport
utilization or a synthetic ceiling. Phase D0.2 also confirmed that
synchronous `smaps_rollup` diagnostics substantially inflated Phase C absolute
model-open timings. Phase D0.3 establishes the first matched current-path
single-span versus exact 310-range transport control.

| Measurement | Phase B baseline | Selected C1 keep-alive |
|---|---:|---:|
| Requests | 310 | 310 |
| Connections | 310 | 1 |
| Transferred bytes | 633,495,552 | 633,495,552 |
| Overfetch | 0 | 0 |
| Median model open | 53.203 s | 44.451 s |
| Effective throughput | 11.907 MB/s | 14.252 MB/s |

C1 is the only retained production optimization: it reuses one persistent
generic HTTP connection, giving a `1.197x` speedup and `16.450%` lower median
model-open time without changing payload semantics or persistent formats.
Payload parity, remote generation, and local/direct-source regression all
passed. Batching/coalescing and bounded concurrency were tested as selection
evidence but regressed and were not retained.

The full artifact is not copied onto the Pixel, but the backend remains
`EAGER_ALL`, so the selected path still transfers and retains almost all tensor
payload bytes during model construction. That historical path remained
dominated by serialized remote payload acquisition; whether request scheduling
had meaningful transport headroom was measured separately in the matched
[`Phase D0.3 transport control`](research/results/vbuf-android-arm64-chat-poc/phase-d0.3-transport-control.md).

## Historical Shared 32B Proof

This section records the earlier shared Qwen3-32B cross-architecture proof. It
is historical storage/source evidence and is not the current DeepSeek ARM64
model-execution matrix. The authoritative source is `Qwen3-32B-Q8_0.vbuf`,
`34,816,197,376` bytes, SHA-256
`84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d`.

The shared semantic bootstrap is `4,472,327` bytes, SHA-256
`dc3b0c755b5bbb4949ca813131c3a74bd33ea21368c5ac42c59feb3adfdeb278`.

All four architectures resolved the same real tensor:

```text
ordinal:          22
SourceId:         1
logical offset:   6,056,603,320
logical length:   5,570,560
end exclusive:    6,062,173,880
representation:   GGML_Q8_0
dimensions:       [5120, 1024]
payload SHA-256:  bfb4d69b99058c659ce075ec794b5d923b8a513ab6b7745a042383aa8dac13bf
```

The same discovery digest, exact range, and payload hash matched across all
four qualified architectures. ARM32, ARM64, and riscv64 completed this proof
without possessing, mapping, or downloading the full model.

## Historical Cross-Architecture Performance Snapshot

These are directly measured results from the earlier cross-architecture storage,
source, and local-layout qualification. They are not the current DeepSeek
ARM64 direct-runtime or prompt-prefill measurements. TensorRef lookup is a
hot-loop microbenchmark, not end-to-end latency. ARM64 source transfer used
ADB reverse HTTP and is not directly comparable to riscv64 direct LAN HTTP.

| Target | Semantic discovery median | TensorRef lookup | Local materialization | Source transfer |
|---|---:|---:|---:|---:|
| ARM64 | 9.344 ms | 0.852213 ns/op | 2.919 ms, 1908.641 MB/s | 0.506011 s, 11.009 MB/s, ADB reverse HTTP |
| riscv64 | 38.978 ms | 3.18285 ns/op | 1.994 ms, 2793.924 MB/s | 49.094 ms, 113.467 MB/s, direct LAN HTTP |

The local-layout benchmark remains a negative result. On the measured 337-record
artifact:

| Traversal | Time |
|---|---:|
| Canonical | 0.016849 ms |
| Generic header-only | 0.019990 ms |
| Scalar/direct | 0.016500 ms |
| SIMD | 0.019355 ms |
| Nano | 0.020385 ms |

The measured conclusion is `LOCAL_LAYOUT_LATENCY_BENEFIT: NO`,
`SIMD_BENEFIT: NO`, and `NANO_BENEFIT: NO`. The value of semantic bootstrap is
remote deployment, source indirection, and bounded residency, not local parser
acceleration.

## Historical Backend Separation Evidence

The llama.cpp integration is one source-agnostic consumer, not a definition of
vBuf. On x86_64, materialized external bytes crossed the committed FFI boundary
into the pinned embedded ggml CPU backend. The bounded operation was:

```text
ggml_sum
local:         220.409927
external file: 220.409927
external HTTP: 220.409927
```

The outputs matched exactly, with zero observed absolute or relative error and
without full-source mapping or download. Within this earlier bounded-operation
qualification, ARM32 and riscv64 model-compute backends remained separately
unqualified. The later DeepSeek ARM64 direct-runtime work separately qualified
real GGML model execution on the Pixel; this does not qualify ARM32, riscv64,
or every backend path.

The ARM32 qualification also found a hard-coded `i8` assumption in the FFI test
harness. It was fixed generically with `core::ffi::c_char`; no production FFI,
storage, or persistent-format semantics changed.

## Build and Test

From the repository root:

```sh
cargo test --manifest-path rust/Cargo.toml --workspace
python3 scripts/verify_portable_graph_neutrality.py
```

The current focused evidence includes the committed vbuf-ML tests for bootstrap,
consumer, consumer FFI, external sources, range loading, source profiles, and
tensor directories. The Rust package is at [`rust/vbuf-ml`](rust/vbuf-ml/), the
llama adapter is at [`integrations/llama.cpp`](integrations/llama.cpp/), and
qualification records are under [`research/results`](research/results/).

Current verification status:

| Check | Status |
|---|---|
| Rust workspace tests | PASS in the latest Step 32K qualification |
| Native CTest | PASS, 22/22 in the Step 32A verification record |
| Portable graph neutrality guard | PASS, `FORBIDDEN_LEAKAGE_COUNT=0` and `CUDA_TYPE_LEAKAGE_COUNT=0` |
| CUDA backend build and device tests | PASS; four CUDA tests on device 0 |
| Real CUDA execution | PASS for one complete block and progressive 2/4/8 layers |
| ARM64 direct probe build | PASS in its recorded qualification |
| APK build | Not rerun in the latest CPU/CUDA work |

The latest Step 32 work did not require an APK rebuild. The established Android
SDK/JDK/NDK/CMake environment remains documented separately; the historical
APK non-rebuild is not a runtime failure.

## Evidence and Scope

The detailed architecture and qualification evidence is preserved in:

- [`vbuf-ML documentation`](docs/vbuf-ml/);
- [`CCC final research conclusion`](docs/vbuf-ml/ccc_research_conclusion.md);
- [`real FP8 repeated generation`](research/results/vbuf-ml-integration/step32j-repeated-autoregressive-generation.md);
- [`progressive CUDA layers`](research/results/vbuf-ml-integration/step32k-b-progressive-cuda-layers.md);
- [`cross-architecture matrix`](research/results/vbuf-autoregressive-generation-poc22-x86/cross-architecture-qualification-matrix.md);
- [`ARM32 records`](research/results/vbuf-cross-architecture/arm32/);
- [`ARM64 records`](research/results/vbuf-cross-architecture/arm64/);
- [`riscv64 records`](research/results/vbuf-cross-architecture/riscv64/);
- [`v0.6 specification`](spec/spec_0.6.md).

Historical experiments remain under `research/` as evidence. They are not
alternate definitions of the current architecture. This README does not claim
full 32B inference, universal cache behavior, universal parser speedups, or
compute support on every qualified architecture.

## License

MIT
