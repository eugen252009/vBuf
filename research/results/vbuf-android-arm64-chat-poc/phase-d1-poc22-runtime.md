# Phase D1: PoC22 Direct Runtime Reconstruction

Date: 2026-08-19

## Decision

PoC22 is the latest direct vBuf-ML runtime lineage and is a viable candidate
for the Android runtime seam. It already separates source acquisition,
materialization, residency, and execution. No production integration change is
made in D1. The next implementation should adapt Android input/output and
threading around these existing interfaces rather than extend
`llama_model_loader`.

## Call Graph

The current PoC22 executable follows this path:

```text
main
  -> load_metadata
       -> read local vBuf metadata/artifact
       -> vbuf_ml_consumer_open
       -> vbuf_ml_consumer_tensor_views
       -> vbuf_ml_consumer_tensor_physical_range
  -> model_lease
  -> HttpRangeSource(endpoint)
  -> LocalVbufRangeMaterializer(source)
  -> TensorResidencyStore(capacity, policy)
  -> ResidentTensorMaterializer(backing, residency)
  -> per generated position:
       -> embedding_row / run_embedding
       -> run_sequence
            -> attention and block execution
            -> routed/shared expert execution
            -> explicit KV state append/read
       -> output graph / logits
       -> greedy next-token selection
       -> feed selected token into next position
```

`autoregressive_poc22.cpp` includes the earlier direct runtime layers as a
library-only implementation. `multi_layer_poc16.cpp` builds block plans from
semantic tensor views and source offsets. The execution layers request tensor
payloads through the materializer and do not use llama.cpp loader callbacks.

## Ownership And Lifetime

- The Rust consumer owns the validated metadata handle and exposes borrowed
  semantic tensor views plus physical source ranges.
- `model_lease` keeps the consumer/model metadata valid for the run.
- `HttpRangeSource` owns transport access and returns requested ranges.
- `LocalVbufRangeMaterializer` allocates aligned owned payload storage, starts
  the range worker, and publishes a `MaterializedTensor` whose borrowed view is
  backed by a shared owner.
- `ResidentTensorMaterializer` and `TensorResidencyStore` retain ready payloads
  while residency leases are active and release them after execution.
- GGML-facing tensors consume borrowed validated storage; the backend does not
  own source resolution or residency policy.
- `RuntimeStateSlot` owns copied KV history for the bounded recurrence. State
  width and maximum positions are checked on append/read.
- PoC22 explicitly clears residency and reports zero resident bytes, active
  leases, materialization resources, and runtime-state resources after teardown.

The current materializer is a qualification implementation: one worker per
request, byte-budget admission, optional fallback source, and no source
selection policy. This is sufficient to preserve the runtime boundary but is
not evidence that Android production scheduling is complete.

## Local Remote Qualification

Build configuration used the normative pinned GGML commit
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` and the Rust `vbuf-ml` debug
library. The executable was run against a local HTTP range server serving:

```text
research-models/DeepSeek-V2-Lite.IQ1_S.vbuf
```

Run parameters were full stack, 27 blocks, 256 MiB residency capacity,
cost-aware replacement, seed `0`, and 4 generated positions.

Observed result:

- `POC22_FUNCTIONAL_AUTOREGRESSIVE_GENERATION=PASS`
- Generated and reference sequences both had four tokens: `94761, 94761, 86711, 86711`.
- Generated-token feedback passed; runtime did not use reference future tokens.
- Router selection parity passed.
- Logits maximum absolute error was `0` at every position.
- State alias violations: `0`.
- Embedding and materializer identity collisions: `0`.
- Teardown: resident bytes `0`, execution leases `0`, materialization resources `0`, runtime-state resources `0`.
- Peak resident bytes: `268120064`.
- Peak active persistent bytes: `12607488`.

This is measured x86 local-HTTP evidence. It is not Android ARM64 evidence,
not tokenizer/chat evidence, and not a production performance result.

## Android Seam

The smallest architectural seam is an Android adapter that supplies:

1. A vBuf-ML consumer handle and semantic tensor snapshot.
2. An Android-compatible `RangeSource` implementation for local/remote ranges.
3. A materializer/residency configuration and execution budget.
4. Tokenization, sampling, and chat-stream orchestration above PoC22-style
   execution.

The execution layer should continue to receive validated materialized or
borrowed tensors. Android transport, request scheduling, and residency policy
must remain below that layer. The llama.cpp integration can continue to serve
as a numerical and generation oracle, but it must not become the owner of this
adapter's acquisition path.

## Gaps And Next Qualification

- No Android ARM64 build or device run has been performed for PoC22.
- No tokenizer, sampler, chat-template, or streaming path is connected.
- No large-model or multi-request qualification has been performed.
- The current direct materializer is intentionally simple and does not provide
  the final Android request scheduler.
- No direct PoC22 versus llama.cpp measured comparison was run in D1.
- A fresh ARM64 qualification should cover local source, remote source,
  materialization leases, residency eviction/reload, generated-token feedback,
  teardown, and backend numerical parity.

## Scope Boundary

This report records measured and derived architecture evidence only. It does
not promote PoC22's qualification executable into the final Android runtime,
change the generic vBuf wire contract, or authorize extending llama.cpp loader
interfaces.
