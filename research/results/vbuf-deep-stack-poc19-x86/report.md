# POC19 Deep Multi-Layer Transformer Stack

## Scope

- Real artifact: `DeepSeek-V2-Lite.IQ1_S.vbuf`
- Real stack: `blk.1..blk.8`, 8 consecutive blocks, 14 tensors per block
- Tokens: positions 0 and 1, one-hot fixtures
- Flat residency only; no POC18 HOT/WARM policy
- x86 only; RV2 compute not executed because the pinned ggml RVV FP16 blocker remains

## Semantic Qualification

| Metric | Result |
|---|---|
| Per-block reference parity | PASS |
| Final token 0 parity | PASS |
| Final token 1 parity | PASS |
| Zero-copy activation forwarding | PASS |
| Activation boundary copy bytes | 0 |
| Cross-layer state isolation | PASS |
| Unselected expert graphs/acquires/source reads/materializations | 0 / 0 / 0 / 0 |
| Router selections | Recorded for all 8 blocks and both tokens |

The selected blocks share MLA non-split-KV and sparse top-6-plus-shared geometry, while their physical ranges and routed expert selections are distinct.

## Active Versus Logical Scaling

| Metric | Value |
|---|---:|
| Total logical persistent bytes | 1,462,931,456 |
| Peak active persistent bytes | 1,892,352 |
| Peak active fraction | 0.00129353 |
| Logical/active ratio | 773.076 |

Peak active memory remains bounded at the local execution wave while logical weight demand grows across eight blocks.

## Capacity Sweep

Average resident bytes are event-sampled averages over the runtime residency trace.

| Capacity | Peak resident | Average resident | Utilization | Source reads | Source bytes | Hits | Misses | Evictions | Reloads | Reload bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 MiB | 8,363,008 | 7,832,462 | 99.69% | 928 | 750,665,728 | 1,856 | 1,824 | 920 | 645 | 516,276,224 |
| 32 MiB | 33,554,432 | 32,213,864 | 100.00% | 928 | 750,665,728 | 1,856 | 1,824 | 890 | 645 | 516,276,224 |
| 64 MiB | 67,082,240 | 63,675,701 | 99.96% | 928 | 750,665,728 | 1,856 | 1,824 | 849 | 645 | 516,276,224 |
| 128 MiB | 134,154,240 | 122,353,697 | 99.95% | 928 | 750,665,728 | 1,856 | 1,824 | 765 | 645 | 516,276,224 |
| 256 MiB | 234,389,504 | 187,182,641 | 87.32% | 283 | 234,389,504 | 3,105 | 575 | 0 | 0 | 0 |

Relative to 8 MiB, source-byte reduction is 0% at 32/64/128 MiB and 68.78% at 256 MiB. Reload-byte reduction is 0% at 32/64/128 MiB and 100% at 256 MiB. The capacity knee is `268,435,456` bytes under the defined 25% reduction criterion.

## Token-to-Token Reuse

- Reusable bytes between warm token 0 and warm token 1: `140,943,360`
- Routed expert bytes selected in both tokens: `85,200,896`
- Base/always-required reuse bytes: `55,742,464`
- Token-0-only bytes: `46,723,072`
- Token-1-only bytes: `46,723,072`

| Capacity | Retained reusable bytes | Retention rate |
|---:|---:|---:|
| 8 MiB | 5,564,416 | 3.95% |
| 32 MiB | 26,741,760 | 18.97% |
| 64 MiB | 50,104,320 | 35.55% |
| 128 MiB | 95,456,256 | 67.73% |
| 256 MiB | 140,943,360 | 100.00% |

Reuse-distance evidence is in `reuse-distance.json` and `reuse-distance-summary.json`. The observed pairs are 448 at `<=8MiB` and 181 at `<=256MiB`; no pair exceeded 256 MiB in this trace.

Per-block survival is in `layer-cache-survival-by-capacity.json`. At 8 MiB, blocks 1-7 have zero boundary retention and block 8 retains 5,564,416 bytes. At 256 MiB, all eight blocks retain their reusable bytes.

## Cold/Warm Replay

The harness phases are `cold_token0`, `cold_token1`, `warm1_token0`, and `warm1_token1`; no residency flush occurs between them.

| Capacity | Cold source bytes | Warm source bytes | Cold reload bytes | Warm reload bytes |
|---:|---:|---:|---:|---:|
| 8 MiB | 375,332,864 | 375,332,864 | 187,666,432 | 375,332,864 |
| 64 MiB | 375,332,864 | 375,332,864 | 187,666,432 | 375,332,864 |
| 128 MiB | 375,332,864 | 375,332,864 | 187,666,432 | 375,332,864 |
| 256 MiB | 234,389,504 | 0 | 46,723,072 | 0 |

At 256 MiB, the first cold token loads 232 identities and the second cold token needs only the 51 token-1-only identities; both warm replays are source-free.

## LRU/MIN Oracle

| Capacity | LRU reload bytes | MIN reload bytes | Avoidable bytes | Avoidable fraction |
|---:|---:|---:|---:|---:|
| 8 MiB | 516,276,224 | 495,354,880 | 20,921,344 | 4.05% |
| 64 MiB | 516,276,224 | 319,285,248 | 196,990,976 | 38.16% |
| 128 MiB | 516,276,224 | 118,007,808 | 398,268,416 | 77.14% |

The oracle indicates replacement-policy headroom at 64/128 MiB even though the current flat runtime remains capacity-bound for this exact trace. No new runtime policy was implemented.

## Cleanup And Integrity

- Current resident bytes before teardown, 8 MiB: `8,312,832`
- Resident bytes after teardown: `0`
- Execution leases after teardown: `0`
- Materialization resources after teardown: `0`
- Runtime-state resources after teardown: `0`
- Peak active bytes are independent of cache capacity: `1,892,352` at every capacity
- Execution-prep copied/repacked/transcoded bytes: `0 / 0 / 0`
- Cache flushed to reduce metrics: `NO`
- Residency intentionally used for reuse: `YES`
- Whole-model loading: `NO`
- Whole-block loading: `NO`
- Whole packed expert loading: `NO`
- Source-range audit: `928/928` materialization ranges matched exact identity offset and byte length
- Architecture-specific runtime logic: `NO`
- Model artifact mutated: `NO`
- vBuf layout/format change required: `NO / NO`

## Failure Matrix

| Target | Earlier completed | Later executed | Final output | Worker cleanup |
|---|---:|---:|---|---|
| Early block 1 | 0 | 0 | INVALID | PASS |
| Middle block 4 | 3 | 0 | INVALID | PASS |
| Late block 8 | 7 | 0 | INVALID | PASS |

All failure cases injected 12 controlled failures and returned without `std::terminate`.

## Verification

- Full x86 CTest: `17/17 PASS`
- `git diff --check`: PASS
- `ccc index`: PASS, 1,288 files, 44,244 chunks, 0 errors
- RV2 deep-stack transport: NOT EXECUTED
- RV2 deep-stack compute: NOT EXECUTED

## Files And State

- Changed source: `integrations/ggml/CMakeLists.txt`
- Changed source: `integrations/ggml/tools/multi_layer_poc16.cpp`
- Added analysis: `scripts/qualify_deep_stack_poc19.py`
- Evidence directory: `research/results/vbuf-deep-stack-poc19-x86/`
- `git diff --check`: PASS
- Commit created: NO
- Working tree: source and evidence changes remain uncommitted by request

## Final Qualification

```text
REAL_DEEP_TRANSFORMER_STACK: PASS
BLOCK_RANGE: blk.1..blk.8
BLOCK_COUNT: 8
DEEP_STACK_REFERENCE_PARITY_TOKEN_0: PASS
DEEP_STACK_REFERENCE_PARITY_TOKEN_1: PASS
PER_BLOCK_REFERENCE_PARITY: PASS
REAL_ACTIVATION_FORWARDING: PASS
ACTIVATION_BOUNDARY_COPY_BYTES: 0
CROSS_LAYER_STATE_ISOLATION: PASS
UNSELECTED_EXPERT_GRAPHS_CREATED: 0
UNSELECTED_EXPERT_TENSORS_ACQUIRED: 0
UNSELECTED_EXPERT_SOURCE_READS: 0
UNSELECTED_EXPERT_MATERIALIZATIONS: 0
TOTAL_LOGICAL_PERSISTENT_BYTES: 1462931456
PEAK_ACTIVE_PERSISTENT_BYTES: 1892352
PEAK_ACTIVE_FRACTION: 0.00129353
LOGICAL_TO_ACTIVE_RATIO: 773.076
ACTIVE_MEMORY_REMAINS_BOUNDED: YES
PEAK_ACTIVE_GROWS_WITH_BLOCK_COUNT: NO
SOURCE_BYTES_8M: 750665728
SOURCE_BYTES_32M: 750665728
SOURCE_BYTES_64M: 750665728
SOURCE_BYTES_128M: 750665728
SOURCE_BYTES_256M: 234389504
RELOAD_BYTES_8M: 516276224
RELOAD_BYTES_32M: 516276224
RELOAD_BYTES_64M: 516276224
RELOAD_BYTES_128M: 516276224
RELOAD_BYTES_256M: 0
TOKEN_TO_TOKEN_REUSABLE_BYTES: 140943360
TOKEN_TO_TOKEN_REUSABLE_BYTES_RETAINED_8M: 5564416
TOKEN_TO_TOKEN_REUSABLE_BYTES_RETAINED_32M: 26741760
TOKEN_TO_TOKEN_REUSABLE_BYTES_RETAINED_64M: 50104320
TOKEN_TO_TOKEN_REUSABLE_BYTES_RETAINED_128M: 95456256
TOKEN_TO_TOKEN_REUSABLE_BYTES_RETAINED_256M: 140943360
TOKEN_TO_TOKEN_REUSE_RETENTION_RATE_8M: 0.0394798
TOKEN_TO_TOKEN_REUSE_RETENTION_RATE_256M: 1.0
CAPACITY_KNEE_BYTES: 268435456
BASE_ALWAYS_REQUIRED_REUSE_BYTES: 55742464
ROUTED_EXPERT_REUSE_BYTES: 85200896
PEAK_RESIDENT_BYTES_8M: 8363008
PEAK_RESIDENT_BYTES_256M: 234389504
AVERAGE_RESIDENT_BYTES_8M: 7832462
AVERAGE_RESIDENT_BYTES_256M: 187182641
CACHE_FLUSHED_TO_REDUCE_MEMORY_METRICS: NO
RESIDENCY_INTENTIONALLY_USED_FOR_REUSE: YES
CURRENT_RESIDENT_BYTES_AFTER_RUN: 8312832
RESIDENT_BYTES_AFTER_TEARDOWN: 0
TOTAL_BYTES_COPIED_FOR_EXECUTION_PREP: 0
TOTAL_BYTES_REPACKED: 0
TOTAL_BYTES_TRANSCODED: 0
EARLY_BLOCK_FAILURE_FAILS_CLOSED: PASS
MIDDLE_BLOCK_FAILURE_FAILS_CLOSED: PASS
LATE_BLOCK_FAILURE_FAILS_CLOSED: PASS
FAILED_REQUEST_WORKER_CLEANUP: PASS
WHOLE_MODEL_LOADING: NO
WHOLE_BLOCK_LOADING: NO
WHOLE_PACKED_EXPERT_LOADING: NO
ARCHITECTURE_SPECIFIC_RUNTIME_LOGIC: NO
MODEL_ARTIFACT_MUTATED: NO
VBUF_LAYOUT_CHANGE_REQUIRED: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
RV2_DEEP_STACK_TRANSPORT_RESULT: NOT_EXECUTED
RV2_DEEP_STACK_COMPUTE_RESULT: NOT_EXECUTED
FULL_X86_CTEST: PASS
GIT_DIFF_CHECK: PASS
CCC_INDEX: PASS
COMMIT_CREATED: NO
READY_FOR_FULL_TRANSFORMER_STACK_POC: YES
READY_FOR_RAM_VRAM_CACHE_POC: YES
READY_FOR_LARGE_MODEL_EXECUTION_POC: NO
```
