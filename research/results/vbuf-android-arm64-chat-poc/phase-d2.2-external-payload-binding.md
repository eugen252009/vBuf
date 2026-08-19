# Phase D2.2: External Payload Binding

## Scope

This phase addressed the PoC22 external semantic-bootstrap qualification seam
identified in D2.1. The semantic bootstrap contains tensor metadata with null
inline payload pointers; reference execution must therefore use the same
validated materialized ranges as runtime execution.

No persistent format, GGML execution operation, scheduler, residency capacity,
transport policy, or model input was changed.

## Implementation

The PoC22 lineage now uses the existing `TensorMaterializer` contract for
reference paths:

- reference graphs receive the same namespace-scoped materializer as runtime
  graphs;
- reference-only F32 norm and router reads resolve through
  `obtain_ready_tensor()` rather than `view.payload`;
- embedding row views preserve a null payload without performing null-pointer
  arithmetic;
- the normal PoC22 sequence shares the canonical source/materializer/residency
  with reference execution;
- materialized reference payloads are held by scoped leases and released after
  use.

The isolated `reference_only` path retains its existing local qualification
materializer because it is not the canonical external-source PoC22 path.

## Regression Coverage

`vbuf_tensor_adapter_qualification` now includes a null-inline-payload case:
metadata is constructed with `payload == nullptr`, the range is materialized
from a source, and the resulting payload is bound to GGML. The test passed.

## Validation

The following x86 targets built successfully:

- `vbuf_autoregressive_poc22`
- `vbuf_multi_layer_poc16`
- `vbuf_deep_stack_poc19`
- `vbuf_full_moe_layer_poc13`
- `vbuf_tensor_adapter_qualification`

All 17 configured GGML/vBuf CTest tests passed.

## Qualification Status

The unchanged external PoC22 driver was rerun against the available IQ2_XXS
semantic bootstrap and range endpoint. It reached the embedding materializer
but stopped with:

```text
POC22_FAILURE=poc22_position_0_embedding payload not ready
```

This is not a successful D2.2 qualification result. No Android rerun was
claimed, and no generation, parity, or teardown result was recorded for the
external semantic-bootstrap path. The remaining issue is determining why the
available local range fixture does not produce a ready embedding payload; it
must be resolved before the required unchanged x86 and Pixel ARM64 four-token
qualification reruns.

## Changed Files

- `integrations/ggml/tools/full_moe_layer_poc13.cpp`
- `integrations/ggml/tools/multi_expert_moe_poc12.cpp`
- `integrations/ggml/tools/multi_layer_poc16.cpp`
- `integrations/ggml/tools/autoregressive_poc22.cpp`
- `integrations/ggml/tools/router_driven_moe_poc11.cpp`
- `integrations/ggml/tests/tensor_adapter_qualification.cpp`
