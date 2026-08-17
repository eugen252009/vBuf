# POC17 Residency Policy Under Real Multi-Layer Sparse Execution

## Outcome

POC17 closes with **Outcome B**. The exact POC16 request trace was replayed
under the current policy and a lease-constrained MIN diagnostic oracle. MIN
reduces reload bytes by only `21,980,160` bytes, or `11.0554%`, at the same
`8,388,608` byte capacity. The workload is primarily capacity-bound; no
speculative runtime policy was implemented.

## Current Policy Semantics

- Victim ordering is ascending `last_use` timestamp.
- Ties use ascending tensor reference.
- `lookup` updates recency; admission assigns a newest timestamp.
- Active leases are never evicted.
- Released tensors become immediately eligible.
- Admission is unconditional when the tensor fits.
- Prefetch planning does not directly admit payloads; execution requests do.

## Baseline

- Capacity: `8,388,608` bytes.
- Logical requests: `684`.
- Unique persistent identities: `102`.
- Runtime residency hits: `696`.
- Runtime residency misses: `684`.
- Evictions: `340`.
- Reload events: `246`.
- Reload bytes: `198,818,816`.
- Source bytes across cold, warm1, and warm2 workload repetitions: `283,414,528`.
- Peak resident bytes: `8,312,832`.
- Peak active persistent bytes: `1,892,352`.

Top reload offenders are recorded in `top-reload-offenders-before.json`.
The largest is shared-expert down at `5,677,056` reload bytes. The remaining
top offenders are routed-expert down tensors at `4,866,048` reload bytes each.

## Reuse and Causes

- Reuse pairs at `<=2 MiB`: `336`.
- Reload pairs at `>16 MiB`: `246`.
- Reload bytes at `>16 MiB`: `198,818,816`.
- Reloads with reuse distance fitting the 8 MiB capacity: `0`.
- Primary cause: sequential layer traversal over a working set whose useful
  reuse distance exceeds the configured residency budget.
- `EVICTION_POLICY_PRIMARY_PROBLEM`: `NO`.
- `ADMISSION_POLICY_PRIMARY_PROBLEM`: `NO`.
- `PREFETCH_CONTRIBUTES_TO_THRASH`: `NO`.

The current policy audit, reuse-distance rows, cause classification, class
summary, prefetch audit, and capacity sweep are in this directory. The
capacity sweep shows MIN continues to improve with larger budgets, while the
current trace remains capacity-limited at 8 MiB.

## Oracle Gate

- Current LRU reload events: `246`.
- MIN reload events: `213`.
- Current LRU reload bytes: `198,818,816`.
- MIN reload bytes: `176,838,656`.
- Avoidable reload bytes: `21,980,160`.
- Avoidable reload fraction: `0.110554`.
- Policy headroom captured: `0`, because no candidate policy was introduced.

MIN is an offline diagnostic lower bound using the exact request ordering,
tensor sizes, capacity, and observed active-lease schedule. It is not runtime
code.

## Candidate

No candidate policy was implemented or run. This is intentional: the hard
oracle gate showed limited policy headroom, and a generic LRU variant, LFU,
probation admission, or heuristic expert policy would not be evidence-backed.
Candidate capacity, traffic, timing, and peak metrics are therefore
`NOT_APPLICABLE`, not fabricated PASS values.

## Semantics and Guards

- Real workload `blk.1 -> blk.2 -> blk.3`: PASS.
- Final token-0 and token-1 parity: PASS.
- Router selections: unchanged.
- Cross-layer state isolation: PASS.
- Unselected expert source reads/materializations: `0`.
- Activation boundary copy bytes: `0`.
- Execution-preparation copies/repacked/transcoded bytes: `0/0/0`.
- Middle-block attention failure: PASS.
- Middle-block selected-expert failure: PASS.
- Failed-request worker cleanup: PASS.
- Model artifact mutated: NO.
- vBuf layout/format changes required: NO/NO.
- RV2 logical range manifest changed: NO; POC16 transport qualification remains applicable.
- RV2 compute remains NOT_EXECUTED due the existing pinned RVV FP16 blocker.

## Verification

- Full x86 CTest: `15/15 PASS` before the POC17 generic contract; the added
  generic residency-policy contract also passes, making the current suite
  `16/16 PASS`.
- `git diff --check`: PASS.
- `ccc index`: PASS.
- Commit created: NO.

## Files

- `scripts/qualify_residency_policy_poc17.py`
- `integrations/ggml/include/vbuf_residency.h`
- `integrations/ggml/src/vbuf_residency.cpp`
- `integrations/ggml/tests/residency_policy_poc17_contract.cpp`
- `integrations/ggml/CMakeLists.txt`
- `integrations/ggml/tools/multi_layer_poc16.cpp` trace instrumentation

## Final Status

```text
REAL_MULTI_LAYER_WORKLOAD: PASS
BASELINE_RESIDENCY_CAPACITY_BYTES: 8388608
CANDIDATE_RESIDENCY_CAPACITY_BYTES: NOT_APPLICABLE
BASELINE_PEAK_ACTIVE_PERSISTENT_BYTES: 1892352
CANDIDATE_PEAK_ACTIVE_PERSISTENT_BYTES: NOT_APPLICABLE
BASELINE_PEAK_RESIDENT_BYTES: 8312832
CANDIDATE_PEAK_RESIDENT_BYTES: NOT_APPLICABLE
BASELINE_RELOAD_EVENTS: 246
CANDIDATE_RELOAD_EVENTS: NOT_APPLICABLE
BASELINE_RELOAD_BYTES: 198818816
CANDIDATE_RELOAD_BYTES: NOT_APPLICABLE
RELOAD_BYTE_REDUCTION: NOT_APPLICABLE
RELOAD_BYTE_REDUCTION_PERCENT: NOT_APPLICABLE
BASELINE_SOURCE_BYTES: 283414528
CANDIDATE_SOURCE_BYTES: NOT_APPLICABLE
SOURCE_BYTE_REDUCTION_PERCENT: NOT_APPLICABLE
BELADY_MINIMUM_RELOAD_BYTES: 176838656
AVOIDABLE_BASELINE_RELOAD_BYTES: 21980160
POLICY_HEADROOM_CAPTURED: 0
EVICTION_POLICY_PRIMARY_PROBLEM: NO
ADMISSION_POLICY_PRIMARY_PROBLEM: NO
PREFETCH_CONTRIBUTES_TO_THRASH: NO
CACHE_THRASH_PRIMARY_CAUSE: capacity-bound sequential layer traversal
POLICY_HYPOTHESIS: none; oracle gate closed speculative implementation
POLICY_GENERIC: NOT_APPLICABLE
RESIDENCY_BUDGET_INCREASED: NO
ROUTER_SELECTIONS_UNCHANGED: PASS
MULTI_LAYER_REFERENCE_PARITY: PASS
CROSS_LAYER_STATE_ISOLATION: PASS
UNSELECTED_EXPERT_SOURCE_READS: 0
ACTIVATION_BOUNDARY_COPY_BYTES: 0
TOTAL_BYTES_COPIED_FOR_EXECUTION_PREP: 0
TOTAL_BYTES_REPACKED: 0
TOTAL_BYTES_TRANSCODED: 0
DUPLICATE_RESIDENCY_PAYLOAD_BYTES: 0
MIDDLE_BLOCK_ATTENTION_FAILURE: PASS
MIDDLE_BLOCK_SELECTED_EXPERT_FAILURE: PASS
FAILED_REQUEST_WORKER_CLEANUP: PASS
ARCHITECTURE_SPECIFIC_RUNTIME_LOGIC: NO
MODEL_ARTIFACT_MUTATED: NO
VBUF_LAYOUT_CHANGE_REQUIRED: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
RV2_LOGICAL_RANGE_MANIFEST_CHANGED: NO
FULL_X86_CTEST: PASS
GIT_DIFF_CHECK: PASS
CCC_INDEX: PASS
COMMIT_CREATED: NO
READY_FOR_TRANSFORMER_STACK_POC: YES
READY_FOR_FULL_MODEL_EXECUTION_POC: NO
```
