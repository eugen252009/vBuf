# Phase D2.3: Materialization Readiness

Date: 2026-08-19

## 1. Objective

Close the remaining PoC22 external-source correctness seam without changing
transport, worker architecture, residency policy, capacity, or scheduling. A
persistent tensor must be requested, synchronized, validated as ready, and
borrowed before GGML descriptor construction and compute.

## 2. D2.2 Starting State

D2.2 had fixed null inline payload handling and scoped reference materializer
integration. The external IQ2_XXS run still failed during embedding with
`payload not ready`. The D2.2 embedding-specific request/wait was removed in
this phase; readiness is now enforced generically at the tensor dependency
boundary.

## 3. Exact Readiness Failure

The first instrumented Android rerun identified the concrete cause:

```text
materialization_failure tensor=token_embd.weight offset=2982064 bytes=0 returned=0 status=0
expert_detail=persistent tensor is not ready: FAILED
```

The semantic bootstrap view intentionally has no inline payload and therefore
reported `payload_len == 0`. `load_metadata()` queried the validated physical
range but discarded its returned length. The materializer consequently started
a zero-byte request; its worker failed during aligned allocation before source
read, hash validation, or residency insertion.

```text
FAILURE_FILE: integrations/ggml/tools/multi_layer_poc16.cpp
FAILURE_FUNCTION: load_metadata(); observed at TensorDependencyExecutor::execute()
FAILURE_CONDITION: external TensorRef payload length remained zero
FAILURE_TENSOR_IDENTITY: token_embd.weight, source offset 2982064
REQUEST_STATE_AT_FAILURE: FAILED after worker launch
EXPECTED_STATE: READY / Resident
ACTUAL_STATE: FAILED
ORDERING: before source read, hash/integrity validation, and residency insertion
```

The generic consumer bug was also corrected: a materializer-backed dependency
that is `NotRequested` or `Released` is now requested at consumption time;
`InFlight` is waited; every non-`Ready` result fails cleanly instead of falling
back to an inline pointer.

## 4. Materialization State Machine

The implementation's actual lifecycle is:

```text
NotRequested
  -> request()
InFlight
  -> worker source read
  -> payload hash and finalization
Ready in LocalVbufRangeMaterializer
  -> obtain_ready_tensor()
ResidentTensorStore entry + active lease
  -> GGML borrow/consume
Released request and released lease
```

`Failed` is reached by allocation or source-read failure. Residency entries
can be `Evicted` when they have no active leases. There is no separate
`Cancelled` state in the current materializer contract.

- `LocalVbufRangeMaterializer` owns request state, worker, aligned buffer, and completion publication under its mutex.
- `request()` creates the worker and publishes `InFlight`.
- The worker reads the validated source range, hashes it, then publishes `Ready` under the mutex.
- `wait()` joins the worker and returns the terminal state.
- `ResidentTensorMaterializer` owns residency insertion and lease acquisition around the completed materialization.
- `TensorResidencyStore` refuses eviction while `active_leases != 0`.
- `MaterializedTensor.storage.lease` and `BorrowedGgmlTensor` retain the underlying owner through backend use.

## 5. Existing Synchronization API

The existing API is sufficient and was reused:

```text
request(ref, tensor, budget)
state(ref)
wait(ref)
obtain_ready_tensor(ref)
release(ref)
```

`wait()` is the canonical worker completion synchronization. No second future,
condition variable, polling loop, or global synchronous materialization mode
was added.

## 6. Consumer Readiness Contract

`TensorDependencyExecutor::execute()` now has a generic persistent-input
readiness boundary:

```text
materializer state
  NotRequested/Released -> request
  InFlight              -> wait
  Ready                 -> obtain_ready_tensor
  Failed/other          -> deterministic runtime error
```

Only the resolved `MaterializedTensor.view` and `storage` are passed to GGML.
Pending or failed payloads cannot be consumed.

The metadata loader also now copies the validated physical range length into
the runtime tensor view. This preserves the distinction between semantic
identity, source range, materialized payload, resident payload, and backend
tensor.

## 7. Correctness Wait Boundary

Waiting occurs only when a graph operation is about to consume a persistent
tensor and no ready payload has been acquired. Prefetch remains asynchronous;
the dependency executor does not wait for unrelated tensors or the whole model.

```text
persistent graph input
    -> ensure ready at operation dependency
    -> bind resolved view/storage
    -> GGML compute
```

No embedding-name or tensor-role special case remains.

## 8. Lifetime / Lease

`ResidentTensorMaterializer::obtain_ready_tensor()` inserts a completed
materialization into the bounded residency store and acquires a lease. The
executor retains the materialized view/storage through descriptor construction
and compute, then calls `release()` at the existing last-consumer boundary.
`TensorResidencyStore::evict_one()` skips leased entries. The lifetime test
confirmed that an active borrowed payload cannot be evicted and remains readable
until release.

Repeated requests for an already resident/in-flight payload now preserve the
existing request lease instead of resetting it, preventing duplicate lease
accounting during planner notifications.

## 9. Failure Propagation

Allocation or source-read failure reaches the dependency executor as a terminal
`FAILED` state and returns `AdapterError::InvalidArgument` with deterministic
detail. No pointer is exposed and no indefinite wait occurs. The failure test
also confirms zero active residency leases after producer failure.

## 10. SELF Regression

The existing local SELF IQ1_S and IQ2_XXS full-stack runs still completed four
positions with parity and clean teardown. Already-inline/local payload behavior
remains immediate and does not require the external semantic bootstrap path.

## 11. x86 Regression

Unchanged four-position x86 qualification passed:

- IQ1_S SELF: generated-token feedback, router parity, logits parity, and teardown passed.
- IQ2_XXS SELF: generated-token feedback, router parity, logits parity, and teardown passed.
- IQ2_XXS external semantic bootstrap: four-token generation passed after physical-range length binding.

The IQ2_XXS external sequence ended with token `76681` repeatedly and reported
`logits_max_abs=0` for the final position. Teardown reported zero resident
bytes, execution leases, materialization resources, and runtime-state resources.

## 12. Android IQ2_XXS Run

The original D2.1/D2.2 setup was used: pinned DeepSeek-V2-Lite IQ2_XXS,
semantic bootstrap, Pixel ARM64 direct executable, reversed HTTP endpoint
`18091`, 27 blocks, cost-aware residency, 256 MiB cap, seed 0, and four
positions. The run completed all four positions and teardown.

## 13. Readiness Milestones

```text
ANDROID_SEMANTIC_DISCOVERY: PASS
ANDROID_FIRST_MATERIALIZATION_REQUEST: PASS
ANDROID_FIRST_READY_PAYLOAD: PASS
ANDROID_EMBEDDING_PAYLOAD_READY: PASS
ANDROID_EMBEDDING_COMPUTE: PASS
ANDROID_GRAPH_CONSTRUCTION: PASS
ANDROID_FIRST_GGML_COMPUTE: PASS
ANDROID_FIRST_TOKEN: PASS
ANDROID_FOUR_TOKEN_GENERATION: PASS
ANDROID_TEARDOWN: PASS
```

The four position records were:

```text
position=0 input=0     next=59685 logits_max_abs=0
position=1 input=59685 next=59685 logits_max_abs=0
position=2 input=59685 next=59685 logits_max_abs=0
position=3 input=59685 next=59685 logits_max_abs=0
```

## 14. Existing Request/Residency Baseline

Aggregated from the four Android position records:

```text
ANDROID_PHYSICAL_REQUEST_COUNT: 7766 (equivalent controlled endpoint run)
ANDROID_REQUESTED_BYTES: not separately exposed; useful source bytes=4,785,358,848
ANDROID_MATERIALIZATION_COUNT: not separately exposed as one aggregate
ANDROID_RESIDENCY_HITS: 16,979
ANDROID_RESIDENCY_MISSES: 8,305
ANDROID_EVICTIONS: 5,316
ANDROID_REMATERIALIZATIONS: reflected in reload_bytes=3,967,938,880
```

The per-position peak resident values remained below the unchanged
`268,435,456` byte cap. No transport or residency policy was changed.

## 15. Remaining Performance Inefficiency

The required dependency waits are correctness waits, not scheduler defects.
The run still exhibits substantial reload traffic and fragmented serialized
range acquisition. Those are future D3 scheduler/load-planning opportunities;
they were not changed here.

## 16. D2 Result

```text
D2_ANDROID_DIRECT_RUNTIME_CONFIRMED
```

The unchanged direct external Android path reached bounded four-token
generation, internal router/logit parity, generated-token feedback, and clean
teardown.

## 17. D3 Gate

```text
D3_READY: YES
```

D3 may characterize execution-aware load planning, bounded-residency reload
amplification, and request fragmentation. It must preserve the readiness and
ownership boundary qualified here.

## Validation

```text
AGENTS_MD_READ: PASS
SPEC_0_6_READ: PASS
D2_2_REPORT_READ: PASS
READINESS_FAILURE_REPRODUCED: PASS
MATERIALIZATION_STATE_MACHINE_AUDITED: PASS
EXISTING_WAIT_API_FOUND: PASS
GENERIC_READY_PAYLOAD_CONTRACT: PASS
EMBEDDING_SPECIAL_CASE_ADDED: NO
PENDING_TO_READY_TEST: PASS
ALREADY_READY_TEST: PASS
FAILURE_PROPAGATION_TEST: PASS
PAYLOAD_LIFETIME_TEST: PASS
SELF_REGRESSION: PASS
IQ1_S_X86_REGRESSION: PASS
IQ2_XXS_X86_REGRESSION: PASS
ROUTER_PARITY: PASS
LOGITS_PARITY: PASS
X86_TEARDOWN: PASS
ANDROID_ARM64_BUILD: PASS
ANDROID_SEMANTIC_DISCOVERY: PASS
ANDROID_REMOTE_CONNECTED: PASS
ANDROID_MATERIALIZATION_REQUEST: PASS
ANDROID_PAYLOAD_READY: PASS
ANDROID_EMBEDDING_COMPUTE: PASS
ANDROID_GGML_EXECUTION: PASS
ANDROID_FIRST_TOKEN: PASS
ANDROID_FOUR_TOKEN_GENERATION: PASS
ANDROID_TEARDOWN: PASS
RESIDENCY_CAP_CHANGED: NO
SCHEDULER_CHANGED: NO
HTTP_BEHAVIOR_CHANGED: NO
PERFORMANCE_OPTIMIZATION_CHANGED: NO
RUST_TESTS: PASS
POC22_TESTS: PASS
CTEST: PASS 17/17
JSON_VALIDATION: PASS
CCC_INDEX: PASS
GIT_DIFF_CHECK: PASS
COMMIT_PERFORMED: NO
PUSH_PERFORMED: NO
```
