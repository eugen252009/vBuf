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
Autoregressive decode remains the unchanged single-position path.

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
requests. Health remains dispatch-blocked during active serialized generation by
policy. Evidence is in
[`step31s-persistent-server-lifecycle-qualification.md`](research/results/vbuf-ml-integration/step31s-persistent-server-lifecycle-qualification.md).

## Android Qualification And Performance

The current physical target is a Pixel 7 Pro running Android 17, arm64-v8a,
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
| Android arm64 direct runtime | Physically qualified |
| Android application harness | Physically qualified over the direct runtime; see [`Android app baseline`](research/results/vbuf-android-demo-poc/android-app-baseline.md) |
| Normal/Qualification runtime modes | Implemented and qualified |
| Layer-major prompt batching | Implemented and physically qualified in normal mode |
| Autoregressive decode | Implemented; unchanged by batching |

The real DeepSeek-V2-Lite target is a qualification example, not a
DeepSeek-specific format. Backend/model neutrality remains a design requirement.

## Current Limitations

- Causal attention and KV state transitions remain ordered per prompt position.
- Autoregressive decode remains token-serial by definition.
- Full hidden-state, KV, and final-logit hashes for every batched prompt row were not retained; first-decode parity and router/MoE audit were recorded.
- Materialization and request/reload amplification remain observable; no new prefetch or compute/materialization overlap was implemented.
- Backend thread configuration was not separately tuned; the existing Android configuration keeps OpenMP disabled.
- The APK was not rebuilt in the final direct-probe qualification because the current environment lacks a usable `javac`; the changed ARM64 direct probe was built and physically run.

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
| Rust workspace tests | PASS |
| Native CTest | PASS, 21/21 |
| Portable graph neutrality guard | PASS, `FORBIDDEN_LEAKAGE_COUNT=0` |
| ARM64 direct probe build | PASS |
| APK build | Not rebuilt; blocked by unavailable usable `javac` |

The APK status is an environment/build-verification limitation, not a runtime
failure. The ARM64 direct probe was built and physically qualified separately.

## Evidence and Scope

The detailed architecture and qualification evidence is preserved in:

- [`vbuf-ML documentation`](docs/vbuf-ml/);
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
