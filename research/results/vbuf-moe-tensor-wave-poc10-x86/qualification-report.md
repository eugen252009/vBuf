# MoE-Oriented Tensor Wave POC10

## Artifact And Expert Fixture

- Artifact: `DeepSeek-V2-Lite.IQ1_S.vbuf`
- SHA256: `780a55b77d2730705a93622338d9747149624d72e558e26176868210fafcc`
- Layer: `1`
- Expert A: `0`
- Expert B: `1`
- Router computation: not implemented; selected ID injected by fixture input.

The real expert tensors are 3D artifact tensors. Each selected expert slice is
derived from the real expert dimension and presented as a 2D `PersistentTensorRef`
with the exact contiguous physical slice range.

## Expert A

| Role | Tensor | Artifact ID | Slice offset | Bytes | Representation | Dimensions |
|---|---|---:|---:|---:|---:|---|
| gate | `blk.1.ffn_gate_exps.weight` | 18 | 103764448 | 563200 | 5 | [2048,1408] |
| up | `blk.1.ffn_up_exps.weight` | 22 | 139809264 | 563200 | 5 | [2048,1408] |
| down | `blk.1.ffn_down_exps.weight` | 16 | 175854080 | 1622016 | 7 | [1408,2048] |

## Expert B

| Role | Tensor | Artifact ID | Slice offset | Bytes | Representation | Dimensions |
|---|---|---:|---:|---:|---:|---|
| gate | `blk.1.ffn_gate_exps.weight` | 18 | 104327648 | 563200 | 5 | [2048,1408] |
| up | `blk.1.ffn_up_exps.weight` | 22 | 140372464 | 563200 | 5 | [2048,1408] |
| down | `blk.1.ffn_down_exps.weight` | 16 | 177476096 | 1622016 | 7 | [1408,2048] |

## Dynamic Wave

The fixture selects Expert A or B before graph construction. The selected
mapping becomes generic graph-local TensorRefs:

```text
gate -> TensorRef 0
up   -> TensorRef 1
down -> TensorRef 2
```

No unselected expert tensor enters the graph, planner candidates, materializer,
residency store, or source path. Expert execution uses the unchanged generic
`TensorDependencyExecutor` with gate/up/SwiGLU/down operations.

## Residency And Materialization

Initial state intentionally preloads only the selected gate tensor into the
RAM residency store. Up and down are nonresident and are fetched from HTTP.
The HTTP materializer receives two misses; the resident gate bypasses source
selection and HTTP I/O. Source-selection decisions for nonresident candidates:
`2` cold, `0` warm. The fixture does not instrument policy calls inside the
materializer itself.

The planner exposes only selected-expert tensors. Up and down transition through
`IN_FLIGHT -> READY -> RELEASED`; the selected gate remains resident after its
execution lease is released. Warm execution reuses the materializer-ready
payloads without network requests.

## Correctness

- Expert A mapped-reference parity: PASS, max abs `0`, max rel `0`.
- Expert B mapped-reference parity: PASS, max abs `0`, max rel `0`.
- Selected Expert A output uses unchanged ggml execution.
- Selected Expert B output uses unchanged ggml execution.
- Unselected expert requests: `0`.
- Unselected expert source reads: `0`.

## Failure

With the HTTP endpoint unavailable and JIT fallback disabled, the selected up
tensor reaches `FAILED`, the expert operation does not execute, the incomplete
destination is discarded, and teardown remains clean.

## Results

- x86 CTest: `10/10 PASS`.
- Peak active persistent bytes: `1622016`.
- Peak resident bytes: `2748416`.
- Warm network requests: `0`.
- vBuf format change: none.
- Architecture-specific runtime logic: none.
- RV2 compute: not executed.
