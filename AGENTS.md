# Agent Guidance

## Project Scope

vBuf is the generic zero-copy binary substrate: a checked, portable wire format
for direct native consumption, vectorized scanning, and `mmap`-friendly access.
vBuf-ML builds model semantics and runtime services on that substrate. Do not
make generic vBuf format or navigation code depend on model-specific policy.

The current normative base wire contract is [`spec/spec_0.6.md`](spec/spec_0.6.md).
Earlier specifications, including `v0.5-alpha`, are historical compatibility
and design evidence, not alternate definitions of the current wire contract.
The TypeScript, Rust, and C implementations and their baseline format tests are
historical completed work; preserve their cross-language and format invariants
when changing them.

## vBuf vs vBuf-ML

Generic vBuf concerns include the binary substrate, checked structure and byte
ranges, navigation, alignment, and persistent-format invariants. vBuf-ML
concerns model metadata and tensor directories, external sources,
materialization, payload lifetime, residency, loading, and backend integration.
Keep those scopes separate. A vBuf-ML feature must not silently redefine the
generic vBuf wire contract.

## Historical Baseline Status

The legacy project agenda recorded the completed prototype baseline: the Bun
TypeScript prototype, Rust library, and embedded C implementation; prototype
validation and cross-language test vectors; checked block headers, overflow and
chaining behavior; Rust zero-copy decoding and SIMD-oriented benchmarks; and C
libraries, bindings, and format tests. Preserve these completed implementation
and qualification invariants when changing the corresponding code.

The agenda's `v0.5-alpha` alignment target and roadmap wording are historical.
Use the normative v0.6 contract above for current format decisions rather than
treating that older target as an active alternate specification.

## Canonical Runtime Architecture

**Runtime invariant:** vBuf-ML owns source resolution, semantic tensor
discovery, materialization, payload ownership and leases, residency policy, load
planning, request scheduling, and runtime integration. Always use the latest
canonical vBuf-ML runtime implementation for those responsibilities. Historical
PoCs are evidence and lineage, not automatically the active runtime.

The latest direct runtime lineage in this repository is PoC22 and its documented
successors. PoC22 was paused while the Android application path was qualified;
it was not deprecated or rejected as an architecture. Do not infer that the
later Android qualification adapter supersedes the direct vBuf-ML runtime.

The backend consumes the validated tensors and runtime services that vBuf-ML
provides. A backend must not dictate how vBuf-ML fetches persistent payloads,
materializes tensors, manages residency, or schedules load work.

The acceptable integration boundary is:

```text
vBuf-ML runtime
    -> materialized or borrowed validated tensors
    -> GGML-compatible execution representation
```

This does not prohibit GGML or other compute backends. It prohibits allowing a
backend loader to become the owner of vBuf-ML acquisition policy.

## llama.cpp Role

llama.cpp and GGML may be used as a correctness oracle, reference
implementation, behavioral comparison target, generation-parity target, and
backend qualification target. The Android llama.cpp integration proved useful
for real model construction, remote payload parity, EAGER_ALL qualification,
generation, and GGML interoperability. Those results remain valid as
qualification evidence.

The llama.cpp loader is not:

- the canonical vBuf-ML model loader;
- the owner of vBuf-ML materialization;
- the owner of vBuf-ML payload ownership or residency;
- the owner of vBuf-ML request scheduling or load planning;
- the architectural source of truth.

Do not extend `llama_model_loader`, `llama_model_load_*`, or loader-specific
source callbacks merely because a historical Android adapter already exists.
Existing implementation availability is not evidence that it is the preferred
runtime architecture. In particular, do not turn a synchronous pattern such as
`source->tensor()` followed by `materialize_tensor()`, `request()`, and `wait()`
into the vBuf-ML runtime contract.

## Runtime Decision Rule

If a feature concerns where bytes come from, when they are fetched, range
counts, request scheduling, caching, tensor lifetime, materialization,
residency, leases, or load planning, implement it in the canonical vBuf-ML
runtime, source, and materialization layers. Do not implement it by extending
`llama_model_loader`.

If a feature concerns reference logits, generation parity, backend correctness,
numerical behavior, or comparison against llama.cpp, llama.cpp may be used as
the oracle or backend qualification target.

## Prompt Prefill Ordering

Normal prompt prefill may batch backend matrix work across prompt rows while
preserving ordered causal attention and KV state transitions. Position-level
parallelism is not implied. Per-token router and TopK decisions remain semantic
row-local decisions. Routed experts may be grouped for execution, but their
contributions MUST be accumulated in the original per-token TopK rank order;
backend scheduling or expert-ID grouping must not change floating-point
reduction order. Qualification mode remains serial with its reference/oracle
and fail-closed parity behavior; normal inference does not execute reference
work. Autoregressive decode remains the existing single-position path.

## Architectural Ownership

```text
vBuf-ML owns:
- semantic tensor discovery
- source resolution
- tensor source offsets and lengths
- materialization
- payload ownership and leases
- residency policy
- load planning
- request scheduling
- runtime integration

Backend / oracle owns:
- execution semantics
- reference behavior
- numerical and correctness comparison
- backend-specific compute representation where required
```

Do not replace or bypass the canonical vBuf-ML runtime with a llama.cpp loader
because it is convenient or already qualified on Android.

## Historical Qualification Boundary

The llama.cpp loader integration is a compatibility and correctness path, not
the canonical vBuf-ML runtime. Preserve its qualification evidence, including
Android generation and parity results, while keeping its architectural role
explicitly bounded.

Phase D measurements also showed that the current llama loader's exact 310-range
plan incurs serialized request-boundary cost versus a matched single-span
transport control. This is evidence for vBuf-ML load-planning decisions, not a
reason to make llama.cpp the owner of those decisions.

The matched control measured approximately `6,146.521 ms` and `103.066 MB/s`
for the current single-span path versus `10,159.064 ms` and `62.358 MB/s` for
the exact 310-range path. The `4,012.543 ms` (`65.282%`) difference is evidence
that backend-driven per-tensor request structure can impose request-fragmentation
cost; it is not permission to move scheduling ownership into the backend.

## Persistent Format Rules

Treat the current vBuf specification and checked range contracts as normative.
Do not change persistent formats, canonical byte layout, alignment semantics,
or compatibility behavior as a side effect of runtime or backend work. Format
changes require explicit scope and qualification; a convenient loader path is
not a format authority.

## Performance and Benchmark Discipline

Keep benchmark scopes explicit and comparable. Separate transport control,
payload materialization and validation, and full model-open timing. Do not
compare a historical tool-specific transport probe with a current loader path as
if it were a transport ceiling. Record request count, connection count, useful
payload bytes, requested bytes, overfetch, cache state, and timing boundaries.

Inclusive timers such as worker wait, body receive, and materialization may
contain nested copy, hashing, or bookkeeping costs. Do not add nested counters
as if they were exclusive critical-path intervals. Preserve raw historical
evidence while correcting interpretations when scope or client paths differ.

## Research and Evidence

Research results are evidence with explicit qualification boundaries, not
automatic architecture decisions. Preserve failed, historical, and compatibility
results unless a task explicitly authorizes their removal. State whether a
result is measured, derived, comparable, observer-distorted, or unresolved.

The Android llama.cpp qualification path established real model construction,
remote payload parity, EAGER_ALL behavior, generation, and GGML interoperability.
Those results remain valid evidence even though that loader is not canonical.

## Testing and Qualification

For changes to generic vBuf behavior, run the relevant cross-language, checked
range, format, and Rust tests. For vBuf-ML runtime changes, qualify source
resolution, materialization, ownership/lifetime, residency, scheduling, and
backend integration at their respective boundaries. For oracle or backend work,
use llama.cpp for parity, logits, generation, numerical, and interoperability
checks without transferring runtime ownership to its loader.

Do not claim payload parity, generation, direct-source regression, or backend
qualification without an actual recorded result. Keep local, remote, and
cross-platform qualification evidence distinct.

## Android Build / Qualification Environment

The Android SDK is installed at `/home/eugen/Android/Sdk`. The established user
zsh configuration and `source.zsh` setup provide the normal Java/Android
environment exports. If an agent starts in a shell where those variables are
not visible, it must first source that established user shell environment.

Known-good environment and build inputs are:

```text
JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64
ANDROID_HOME=/home/eugen/Android/Sdk
ANDROID_SDK_ROOT=/home/eugen/Android/Sdk
ANDROID_NDK_HOME=/home/eugen/Android/Sdk/ndk/27.1.12297006
GGML source=/home/eugen/projekte/llama.cpp/ggml
CMake=3.22.1
```

The Android build currently requires the local GGML source override
`-PvbufGgmlSrc=/home/eugen/projekte/llama.cpp/ggml`. The historical default
`/tmp/llama.cpp-step21/ggml` may not exist and is not the only valid GGML source
location. Qualification APKs using the remote payload source must also preserve
the documented `vbufRemoteUrl` Gradle property from the Android README/build
instructions.

Before reporting the Android SDK, JDK, NDK, CMake, or GGML checkout as
unavailable, first load the established user shell environment and check the
known-good paths documented here. Missing environment-variable discovery is not
evidence that the toolchain is absent.

Canonical build shape using the repository Gradle wrapper:

```bash
source <existing user source.zsh / established shell setup>

./integrations/android-vbuf-chat/gradlew \
  -p integrations/android-vbuf-chat \
  -PvbufGgmlSrc=/home/eugen/projekte/llama.cpp/ggml \
  <existing documented vbufRemoteUrl property when required> \
  assembleDebug
```

## Change Discipline

Before changing runtime behavior, locate the latest canonical implementation
and its tests. Do not resurrect an older adapter merely because it is easier to
discover or already builds. Keep production changes separate from diagnostic
instrumentation and research controls. Preserve unrelated worktree changes,
do not add compatibility shims without a concrete need, and do not commit or
push unless explicitly requested.

## Architecture Guards

The repository has dependency-boundary tests for validated vBuf-ML ranges, but
no existing guard that expresses a llama.cpp-loader dependency prohibition. Do
not add a new architecture-test framework solely for this rule. If a suitable
existing guard is extended later, it should prevent the canonical vBuf-ML
runtime from acquiring dependencies on `llama_model_loader`,
`llama_model_load_*`, or loader-specific source callbacks while permitting
llama.cpp oracle and backend qualification use.

## Agent Decision Rules

For source resolution, remote fetches, range planning, materialization, caching,
payload ownership, tensor lifetime, residency, load planning, or request
scheduling, modify the canonical vBuf-ML runtime, source, or materialization
layer. Do not implement those concerns by extending `llama_model_loader`.

For reference logits, generation parity, numerical behavior, backend correctness,
or compatibility comparison, llama.cpp may be used as an oracle or qualification
target. GGML remains an allowed compute backend when it consumes validated
materialized or borrowed tensors supplied by vBuf-ML.

Always verify the latest runtime lineage before making a change. Existing code
is not proof of preferred architecture, and a historical Android qualification
adapter must not silently become the runtime source of truth.
