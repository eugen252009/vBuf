# POC20 Cost-Aware Residency / Replacement Qualification

## Outcome

POC20 qualifies a generic online COST_AWARE policy. It materially beats LRU at the fixed 64 MiB and 128 MiB capacities while preserving POC19 semantics and the bounded active execution wave.

## Policy Comparison

| Capacity | Policy | Source bytes | Reload bytes | Reload events | Evictions | Useful retained bytes | Peak resident |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 MiB | LRU | 750,665,728 | 516,276,224 | 645 | 920 | 5,564,416 | 8,363,008 |
| 8 MiB | COST_AWARE | 750,665,728 | 516,276,224 | 645 | 923 | 7,614,464 | 8,388,608 |
| 64 MiB | LRU | 750,665,728 | 516,276,224 | 645 | 849 | 50,104,320 | 67,082,240 |
| 64 MiB | COST_AWARE | 668,123,136 | 433,733,632 | 592 | 828 | 60,059,648 | 67,102,720 |
| 64 MiB | MIN | offline only | 319,285,248 | 388 | 598 | offline only | offline only |
| 128 MiB | LRU | 750,665,728 | 516,276,224 | 645 | 765 | 95,456,256 | 134,154,240 |
| 128 MiB | COST_AWARE | 522,614,784 | 288,225,280 | 457 | 633 | 87,656,448 | 134,215,680 |
| 128 MiB | MIN | offline only | 118,007,808 | 124 | 263 | offline only | offline only |
| 256 MiB | LRU | 234,389,504 | 0 | 0 | 0 | 140,943,360 | 234,389,504 |
| 256 MiB | COST_AWARE | 234,389,504 | 0 | 0 | 0 | 140,943,360 | 234,389,504 |

## Oracle Headroom

| Capacity | LRU reload | MIN reload | COST_AWARE reload | Saved vs LRU | Headroom capture |
|---:|---:|---:|---:|---:|---:|
| 8 MiB | 516,276,224 | 495,354,880 | 516,276,224 | 0 | 0% |
| 64 MiB | 516,276,224 | 319,285,248 | 433,733,632 | 82,542,592 | 41.90% |
| 128 MiB | 516,276,224 | 118,007,808 | 288,225,280 | 228,050,944 | 57.26% |
| 256 MiB | 0 | 0 | 0 | 0 | n/a |

Source-byte savings versus LRU are `82,542,592` at 64 MiB and `228,050,944` at 128 MiB. The 8 MiB case remains capacity-bound as expected.

## Reuse And Eviction Quality

- Total token-to-token reusable bytes: `140,943,360`.
- COST_AWARE useful retained bytes: `60,059,648` at 64 MiB and `87,656,448` at 128 MiB.
- Dead retained bytes are reported in `retention-analysis.json`; runtime policy does not use future labels.
- Evictions classified as never reused, reused, bytes evicted then reused, and median post-eviction reuse distance are in `eviction-quality.json`.
- Representative score decisions, including generic score components and eligible candidates, are in `representative-decisions.json`.

## Runtime Information Boundary

- Used: generic identity, size, active lease state, request ordinal, observed request count, recency, and byte-based reacquisition cost.
- Not used: model semantics, layer/tensor names, expert/router classification, completed-trace future, MIN future, source-specific latency, or speculative scheduling.
- `KNOWN_RUNTIME_FUTURE`: none used.
- `ORACLE_FUTURE`: MIN offline replay only.
- Async-overlap-aware scoring: deferred.

## Overhead And Safety

- 64 MiB COST_AWARE: 828 decisions, 40,899 candidates, 805,441 ns total policy CPU, 10,160 ns maximum decision.
- 128 MiB COST_AWARE: 633 decisions, 77,285 candidates, 1,027,912 ns total policy CPU, 4,600 ns maximum decision.
- Candidate inspection is a deterministic linear scan over current eligible residents.
- Active leased entries are excluded before policy selection; `ACTIVE_LEASE_EVICTED=0`.
- One resident entry exists per identity; `DUPLICATE_RESIDENT_PAYLOAD_BYTES=0`.
- 256 MiB remains reload-free under COST_AWARE.

## Semantic And Failure Controls

- Per-block parity: PASS.
- Token 0 and token 1 reference parity: PASS.
- Zero-copy activation forwarding: PASS; boundary copy bytes `0`.
- Cross-layer state isolation: PASS.
- Unselected expert graphs/acquires/source reads/materializations: `0 / 0 / 0 / 0`.
- Peak active persistent bytes remain `1,892,352`.
- Early, middle, and late 128 MiB failures fail closed with cleanup PASS.
- Resident bytes, execution leases, materialization resources, and runtime-state resources after teardown: `0`.
- Exact source-range identity matching remains PASS; whole-model, whole-block, and whole-packed-expert loading remain NO.
- Execution-prep copied/repacked/transcoded bytes remain `0 / 0 / 0`.

## Guards And Verification

- Architecture-specific policy logic: NO.
- Architecture-specific runtime logic: NO.
- Model artifact mutated: NO.
- vBuf layout/format change: NO / NO.
- ggml compute path changed: NO.
- Full x86 CTest: `17/17 PASS`.
- `git diff --check`: PASS.
- `ccc index`: PASS.
- Commit created: NO.

## Final Result

```text
POC20_COST_AWARE_RESIDENCY: PASS
LRU_BASELINE_REPRODUCED: PASS
MIN_ORACLE_REPRODUCED: PASS
COST_AWARE_POLICY_IMPLEMENTED: YES
COST_AWARE_POLICY_RUNTIME_ENABLED: YES
POLICY_USES_MODEL_SEMANTICS: NO
POLICY_USES_ORACLE_FUTURE: NO
POLICY_USES_RUNTIME_KNOWN_FUTURE: NO
ASYNC_OVERLAP_AWARE_SCORING: DEFERRED
ACTIVE_LEASE_EVICTED: 0
DUPLICATE_RESIDENT_PAYLOAD_BYTES: 0
LRU_RELOAD_BYTES_8M: 516276224
MIN_RELOAD_BYTES_8M: 495354880
COST_AWARE_RELOAD_BYTES_8M: 516276224
HEADROOM_CAPTURE_8M: 0
LRU_RELOAD_BYTES_64M: 516276224
MIN_RELOAD_BYTES_64M: 319285248
COST_AWARE_RELOAD_BYTES_64M: 433733632
RELOAD_BYTES_SAVED_64M: 82542592
HEADROOM_CAPTURE_64M: 41.90%
LRU_RELOAD_BYTES_128M: 516276224
MIN_RELOAD_BYTES_128M: 118007808
COST_AWARE_RELOAD_BYTES_128M: 288225280
RELOAD_BYTES_SAVED_128M: 228050944
HEADROOM_CAPTURE_128M: 57.26%
LRU_RELOAD_BYTES_256M: 0
COST_AWARE_RELOAD_BYTES_256M: 0
SOURCE_BYTES_LRU_64M: 750665728
SOURCE_BYTES_COST_AWARE_64M: 668123136
SOURCE_BYTES_LRU_128M: 750665728
SOURCE_BYTES_COST_AWARE_128M: 522614784
TOKEN_TO_TOKEN_REUSABLE_BYTES: 140943360
TOKEN_TO_TOKEN_RETENTION_LRU_64M: 50104320
TOKEN_TO_TOKEN_RETENTION_COST_AWARE_64M: 60059648
TOKEN_TO_TOKEN_RETENTION_LRU_128M: 95456256
TOKEN_TO_TOKEN_RETENTION_COST_AWARE_128M: 87656448
POLICY_DECISIONS_64M: 828
POLICY_CANDIDATES_EVALUATED_64M: 40899
AVERAGE_POLICY_DECISION_TIME_64M_NS: 972
MAX_POLICY_DECISION_TIME_64M_NS: 10160
POLICY_DECISIONS_128M: 633
POLICY_CANDIDATES_EVALUATED_128M: 77285
AVERAGE_POLICY_DECISION_TIME_128M_NS: 1624
MAX_POLICY_DECISION_TIME_128M_NS: 4600
COST_AWARE_EVICTION_SEQUENCE_DETERMINISTIC: PASS
PEAK_ACTIVE_PERSISTENT_BYTES_LRU: 1892352
PEAK_ACTIVE_PERSISTENT_BYTES_COST_AWARE: 1892352
ACTIVE_MEMORY_REMAINS_BOUNDED: YES
REAL_ACTIVATION_FORWARDING: PASS
ACTIVATION_BOUNDARY_COPY_BYTES: 0
DEEP_STACK_REFERENCE_PARITY_TOKEN_0: PASS
DEEP_STACK_REFERENCE_PARITY_TOKEN_1: PASS
PER_BLOCK_REFERENCE_PARITY: PASS
CROSS_LAYER_STATE_ISOLATION: PASS
UNSELECTED_EXPERT_GRAPHS_CREATED: 0
UNSELECTED_EXPERT_TENSORS_ACQUIRED: 0
UNSELECTED_EXPERT_SOURCE_READS: 0
UNSELECTED_EXPERT_MATERIALIZATIONS: 0
WHOLE_MODEL_LOADING: NO
WHOLE_BLOCK_LOADING: NO
WHOLE_PACKED_EXPERT_LOADING: NO
TOTAL_BYTES_COPIED_FOR_EXECUTION_PREP: 0
TOTAL_BYTES_REPACKED: 0
TOTAL_BYTES_TRANSCODED: 0
EARLY_BLOCK_FAILURE_FAILS_CLOSED: PASS
MIDDLE_BLOCK_FAILURE_FAILS_CLOSED: PASS
LATE_BLOCK_FAILURE_FAILS_CLOSED: PASS
FAILED_REQUEST_WORKER_CLEANUP: PASS
RESIDENT_BYTES_AFTER_TEARDOWN: 0
EXECUTION_LEASES_AFTER_TEARDOWN: 0
MATERIALIZATION_RESOURCES_AFTER_TEARDOWN: 0
RUNTIME_STATE_RESOURCES_AFTER_TEARDOWN: 0
ARCHITECTURE_SPECIFIC_POLICY_LOGIC: NO
ARCHITECTURE_SPECIFIC_RUNTIME_LOGIC: NO
MODEL_ARTIFACT_MUTATED: NO
VBUF_LAYOUT_CHANGE_REQUIRED: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
GGML_COMPUTE_PATH_CHANGED: NO
FULL_X86_CTEST: PASS
GIT_DIFF_CHECK: PASS
CCC_INDEX: PASS
COMMIT_CREATED: NO
OUTCOME_CLASSIFICATION: A
READY_FOR_ASYNC_COST_MODEL_POC: YES
READY_FOR_RAM_VRAM_PLACEMENT_POLICY_POC: YES
READY_FOR_FULL_TRANSFORMER_STACK_POC: YES
READY_FOR_LARGE_MODEL_EXECUTION_POC: NO
```
