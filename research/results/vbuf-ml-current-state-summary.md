# vBuf-ML Current-State Summary

Date: 2026-08-20
Branch: `vbuf-ml`
Baseline: `27c8bd9 Establish Android normal inference baseline`

## 1. Scope

This report consolidates the current generic vBuf substrate, the vBuf-ML
semantic/runtime layer, the portable backend boundary, and the completed
Android direct-runtime qualification and prompt-prefill batching work. Numbers
are labeled as measured, derived, or historical. This is a documentation and
evidence summary; it does not authorize another optimization stage.

## 2. vBuf vs vBuf-ML

`vBuf` is the generic persistent, self-navigating binary substrate. Its v0.6
contract defines checked canonical block geometry, little-endian fields, u64
range arithmetic, alignment, continuation, and direct native navigation. It is
not an LLM format and does not depend on model policy or a backend.

`vBuf-ML` is an optional semantic and runtime layer above vBuf. It adds model
metadata and tensor-directory interpretation, semantic bindings, external
payload sources, materialization, readiness, leases, bounded residency, runtime
state, load planning, and backend adaptation. DeepSeek-V2-Lite and GGML are
qualification targets and consumers, not the definition of either layer.

## 3. Current Architecture

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

vBuf-ML owns source resolution, physical ranges, materialization, readiness,
payload lifetime, leases, residency, scheduling, and runtime integration. The
backend consumes ready borrowed or materialized tensors and owns execution
mechanics; it does not acquire persistent payloads or define runtime policy.

## 4. Portable Execution Path

```text
Importer
  -> PortableProgram
  -> generic lowering
  -> ExecutionGraph
  -> TensorBindings
  -> materialization / readiness / residency
  -> ready PersistentTensorRef + lease
  -> versioned C ABI
  -> generic backend adapter
  -> compute
```

Rust lowering preserves operation attributes such as RMSNorm epsilon, MatMul
orientation, and TopK policy. The generic C++ adapter consumes lowered
descriptors and ready payloads without model-family or source-name authority.
The adapter and neutrality guard remain separately qualified from the direct
DeepSeek runtime path.

## 5. External Payload / TensorRef Model

Semantic metadata and tensor payload need not be colocated:

```text
semantic-bootstrap.vbuf
        |
        v
TensorRef(SourceId, offset, length)
        |
        v
file / HTTP Range / other RangeSource
        |
        v
bounded materialization
```

The semantic bootstrap can contain metadata and TensorRefs while payload bytes
remain in a file or remote range source. A local payload-bearing artifact still
uses the `SELF` source and borrowed mmap path. Source identity is separate from
the source locator.

## 6. Materialization, Readiness, and Leases

```text
NotRequested -> request -> InFlight -> Ready -> resident + active lease
                                      \-> Failed
resident + active lease -> borrowed backend tensor -> release
```

The dependency executor requests or waits at the persistent-input boundary and
passes only ready views/storage to GGML. A lease prevents eviction while the
backend uses a borrowed payload. Source-read or allocation failure propagates
as a deterministic runtime error; no pending pointer is exposed.

## 7. Bounded Residency

The payload can be much larger than resident memory. In the current Android
qualification, the converted DeepSeek-V2-Lite IQ2_XXS vBuf payload is
`5,639,819,878` bytes, while the vBuf-ML residency cap is `268,435,456` bytes
(`256 MiB`). The runtime materializes only required ranges and evicts
unleased entries according to the existing cost-aware policy.

Residency bytes and process RSS are different metrics. The physical batching
runs remained within the cap: serial peak resident bytes were `268,435,456`,
and batched peak resident bytes were `267,424,768`.

## 8. Android Direct Runtime

The canonical direct Android path uses the semantic bootstrap, validated
external physical ranges, `HttpRangeSource`, vBuf-ML materialization and
residency, borrowed GGML tensors, and the PoC22-derived direct execution path.
The qualification target was a physical Pixel 7 Pro running Android 17,
arm64-v8a, with DeepSeek-V2-Lite IQ2_XXS and the 256 MiB cap.

The qualified capability sequence includes semantic discovery, remote source
access, physical TensorRef ranges, materialization/readiness, bounded
residency, leases, embedding, graph construction, GGML execution, first token,
bounded generation, parity controls, and teardown. Historical D2.3 counters
remain evidence of the earlier four-position control and are not mixed with
the newer batching measurements:

```text
semantic bootstrap: 3,206,424 bytes
converted vBuf:     5,639,819,878 bytes
tensor count:       377
tensor payload:     5,636,623,360 bytes

historical D2.3 source bytes: 4,785,358,848
historical D2.3 reload bytes: 3,967,938,880
historical D2.3 hits/misses/evictions: 16,979 / 8,305 / 5,316
historical HTTP 206 responses: 5,398
historical returned bytes: 5,460,460,160
```

## 9. RuntimeMode Separation

The direct runtime has explicit `NormalInference` and `Qualification` modes.

| Mode | Behavior |
|---|---|
| `NormalInference` | Actual runtime computation only; reference/oracle work is not executed. |
| `Qualification` | Actual computation plus reference/oracle computation, parity checks, and fail-closed behavior. |

This separation prevents qualification controls from being mistaken for
production inference cost. Source resolution, materialization, readiness,
leases, residency, tokenization, and actual model computation remain governed
by the same runtime contracts. Qualification remains serial and decode remains
unchanged.

### Controlled mode split

These are measured complete position timings on the same Pixel qualification
configuration:

| Mode | Position time |
|---|---:|
| Qualification | 73,588 ms |
| Normal position 0 | 35,397 ms |
| Normal position 1 | 40,060 ms |

Position 0 normal versus qualification is a derived `2.079x` speedup and
`51.9%` wall-clock reduction. The directly measured removed reference
component subtotal is `23,772 ms`; the observed total wall-clock delta is
`38,191 ms`, leaving `14,419 ms` unattributed or not separately instrumented.
The entire delta is not attributed to reference compute. The earlier `~109.7 s`
position is a historical qualification-heavy diagnostic, not this control.

## 10. Serial Normal Baseline

Before batching, normal prompt prefill was position-major and token-by-token:

```text
token 0 -> all layers
token 1 -> all layers
...
token N -> all layers
```

The measured seven-token serial full-prompt prefill is `284,415 ms` (`284.4 s`,
about `4 min 44.4 s`). The older `~247.8-280.4 s` range was an extrapolation
from individual positions and is superseded for this prompt by the measured
full-prompt control.

## 11. Prompt Batching Architecture

`PromptBatch` contains multiple prompt token rows and their first absolute
position. The normal path is now layer-major:

```text
Layer 0:
    ordered causal attention/KV for prompt rows
    batched dense FFN/router/MoE for prompt rows
Layer 1:
    ordered causal attention/KV
    batched dense FFN/router/MoE
...
```

Attention and KV transitions still execute in prompt-position order. FFN,
router, shared expert, and grouped routed-expert operations receive multiple
rows in larger GGML operations. Rows selecting the same expert may be grouped
for execution and scattered back to their original prompt row.

This is not full prompt-token parallelism and is not a wrapper around complete
independent `run_step()` calls. Position-level parallelism remains `NO`.
Decode remains the unchanged single-position autoregressive path.

## 12. Correctness Invariants

The batching contract preserves absolute positions, RoPE positions, causal
visibility, ordered KV appends, per-token router decisions, deterministic TopK,
per-token selected weights, and row-local residual composition. The first
decode step starts after all prompt rows have appended their K/V state.

An important reusable invariant is that backend expert execution order is not
semantic contribution order. Grouping experts by expert ID is allowed for
compute efficiency, but each token's routed expert contributions MUST be
accumulated in original TopK-rank order. Floating-point addition is
order-sensitive; changing this reduction order can change model results even
when the selected experts and weights are identical.

The physical seven-token run completed through batched prefill and entered
decode. The first continuation matched the serial control:

```text
serial:  -
batched: -
```

This is strong end-to-end evidence together with the causal/state contracts and
router/MoE audit, but it is not a claim of full hidden-state/KV/logit hash
equivalence for every prompt row.

## 13. Physical Pixel Measurements

Configuration:

```text
device: Pixel 7 Pro, Android 17, arm64-v8a
model: DeepSeek-V2-Lite
quantization: IQ2_XXS
prompt: Explain the purpose of bounded generation
prompt tokens: 7
residency cap: 256 MiB
thread configuration: unchanged; OpenMP disabled
```

### Full-prompt prefill

| Path | Measured prefill | Human-readable |
|---|---:|---:|
| Serial normal | 284,415 ms | 284.4 s, about 4:44.4 |
| Batched normal | 82,231 ms | 82.2 s, about 1:22.2 |

The derived batching improvement is `3.459x`, a `71.1%` wall-clock reduction,
and `202,184 ms` saved, about `3 min 22.2 s`. These are measured physical
results for this model, prompt, source/runtime configuration, and cap; they are
not a universal model or prompt speedup.

The two runs were separate single observations against the same local source
endpoint, and host/server cache state was not independently reset. Source and
residency counters are therefore reported as observations, not a controlled
transport ceiling. Both runs remained within the residency cap.

## 14. Performance Progression

| Stage | Result | Classification |
|---|---:|---|
| Qualification-heavy diagnostic position | ~109.7 s | Historical diagnostic |
| Controlled Qualification position | 73,588 ms | Measured |
| Controlled Normal position 0 | 35,397 ms | Measured |
| Controlled Normal position 1 | 40,060 ms | Measured |
| Serial seven-token full prefill | 284,415 ms | Measured |
| Batched seven-token full prefill | 82,231 ms | Measured |

The mode split isolates qualification/reference overhead. The full-prefill
comparison isolates batching against serial normal inference. These rows answer
different questions and must not be divided together to manufacture a
cumulative speedup.

The largest measured isolated runtime optimization is batched prompt prefill:
`3.459x` and `71.1%` for this qualification case. Runtime-mode separation
established the normal-inference baseline by removing qualification/reference
execution: `2.079x` at the controlled position boundary. These scopes are not
combined into a cumulative speedup.

## 15. Historical vs Current Measurements

Historical reports preserve earlier D2.3 four-position qualification, Qwen
remote-loading, transport controls, and diagnostic observations. They remain
valid within their stated scopes. The current canonical performance comparison
for this DeepSeek prompt is the measured `284,415 ms` serial normal prefill
against the measured `82,231 ms` batched normal prefill. The controlled mode
comparison is separately `73,588 ms` qualification versus `35,397 ms` normal
position 0.

## 16. Current Capabilities

| Capability | Status |
|---|---|
| Generic vBuf substrate | Implemented |
| External `RangeSource` payloads | Implemented |
| vBuf-ML semantic bootstrap | Implemented |
| TensorRef source ranges | Implemented |
| Bounded materialization | Implemented |
| Readiness and leases | Implemented |
| Bounded residency | Implemented |
| Portable program/lowering | Implemented |
| Generic GGML adapter | Implemented and contract-qualified |
| Android arm64 direct runtime | Physically qualified |
| `NormalInference` / `Qualification` modes | Implemented and qualified |
| Layer-major prompt batching | Implemented and physically qualified in normal mode |
| Autoregressive decode | Implemented; unchanged by batching |

## 17. Current Limitations

- Causal attention and KV state transitions remain ordered per prompt position.
- Autoregressive decode remains token-serial by definition.
- Full hidden-state, KV, and final-logit hashes for every batched prompt row were not retained; first-decode parity and router/MoE audit were recorded.
- Materialization and request/reload amplification remain observable; no new prefetch or compute/materialization overlap was implemented.
- Backend thread configuration was not separately tuned; the existing Android configuration keeps OpenMP disabled.
- The APK was not rebuilt in the final direct-probe qualification because the current environment lacks a usable `javac`; the changed ARM64 direct probe was built and physically run.
- The generic portable adapter's real DeepSeek router-prefix execution remains a separate qualification boundary; its Rust/ABI/neutrality contracts are not the same evidence as the direct runtime qualification.

## 18. Next Evidence-Driven Optimization Areas

No next stage was started here. Candidate evidence areas, subject to a new
scope and qualification plan, are:

1. Batched Q/K/V projection with an explicit state-complete causal contract.
2. Execution-aware load planning for observed request fragmentation and reload amplification.
3. Prefetch/compute overlap only if measured dependency and residency evidence supports it.

Any future work must preserve vBuf-ML ownership of source resolution,
materialization, residency, and scheduling, and must not transfer those concerns
to a backend loader.

## 19. Evidence Links

- [`README.md`](../../README.md)
- [`vBuf v0.6 specification`](../../spec/spec_0.6.md)
- [`normal-inference-baseline.md`](vbuf-android-demo-poc/normal-inference-baseline.md)
- [`prompt-prefill-batching.md`](vbuf-android-demo-poc/prompt-prefill-batching.md)
- [`phase-d2.3-materialization-readiness.md`](vbuf-android-arm64-chat-poc/phase-d2.3-materialization-readiness.md)
- [`post-merge-android-hardware-qualification.md`](vbuf-ml-integration/post-merge-android-hardware-qualification.md)
- [`p1-deepseek-portable-backend-adapter.md`](vbuf-import-boundary-audit/p1-deepseek-portable-backend-adapter.md)
