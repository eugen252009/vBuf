# POC18 Tiered Residency Across Backing, Warm, and Hot Tiers

## Result

POC18 proves the generic tier and generation contract, but closes with
**Outcome B** for the selected policy: `8 MiB HOT + 32 MiB WARM` does not beat
an equal-total-capacity `40 MiB` flat cache on backing traffic and introduces
`259,117,056` bytes of inter-tier movement. No tiered policy is enabled in the
CPU execution runtime.

## Required Configurations

| Configuration | Backing bytes | Hot hits | Warm hits | Promotions | Inter-tier bytes |
|---|---:|---:|---:|---:|---:|
| Flat 8 MiB baseline | 283,414,528 | 336 | 0 | 0 | 0 |
| Flat 40 MiB control | 283,414,528 | 336 | 0 | 0 | 0 |
| Tiered 8+32 MiB | 283,414,528 | 84 | 252 | 84 | 259,117,056 |

The complete `8`, `16`, `24`, `32`, `40`, `56`, and `64 MiB` flat/tiered
capacity sweep is in `tiered-capacity-sweep.csv` and `tiered-results.json`.

## Contract

- Existing payload ownership remains backing materializer plus residency store.
- `TieredResidencyStore` separates `ResidencyTier`, `ResidencyGeneration`,
  and execution leases.
- `PersistentTensorRef` identity is stable across all tier moves.
- Exclusive residency is used; duplicate payload bytes are `0`.
- Active leases prevent promotion, demotion, and drop.
- Immutable weights have dirty writeback bytes `0`.
- Runtime state does not use weight tiers.
- No model, layer, expert, tensor-name, or architecture branch exists in the
  generic tier code.

## Generation and Movement

The deterministic host policy admits first loads to WARM as `NEW`, marks the
second request `YOUNG`, and marks the third request `OLD`. OLD WARM entries are
eligible for HOT promotion. HOT pressure demotes LRU entries to WARM when the
warm tier can admit them; otherwise warm entries drop to backing. Promotion
precision in the bounded replay is `1.0` for the promoted set.

For `8+32 MiB`:

- HOT peak: `8,380,416` bytes.
- WARM peak: `33,543,168` bytes.
- Total peak: `41,923,584` bytes.
- Promotions: `84` / `133,365,760` bytes.
- Demotions: `79` / `125,751,296` bytes.
- Backing misses: `348`.
- Backing source bytes: `283,414,528`.
- Total data movement: `542,531,584` bytes.

## Semantic Controls

- POC16 normal multi-layer semantics: PASS.
- Final token parity: PASS.
- Router selections: unchanged.
- Cross-layer state isolation: PASS.
- Unselected expert source reads: `0`.
- Activation boundary copies: `0`.
- Execution-preparation copies/repack/transcode: `0/0/0`.
- Middle-block attention failure: PASS.
- Middle-block selected-expert failure: PASS.
- Promotion-failure contract: PASS.
- Backing-source failure contract: PASS.
- Failed async worker cleanup: PASS.
- RV2 logical range manifest: unchanged; POC16 transport evidence remains
  applicable. RV2 compute remains NOT_EXECUTED due the pinned RVV FP16 blocker.

## Verification

- POC17 residency contract: PASS.
- POC18 tiered residency contract: PASS.
- Full x86 CTest: `18/18 PASS`.
- `git diff --check`: PASS.
- `ccc index`: PASS.
- Commit created: NO.

## Final Status

```text
REAL_TIERED_RESIDENCY: PASS
MODEL_ARTIFACT_MUTATED: NO
PERSISTENT_IDENTITY_STABLE_ACROSS_TIERS: PASS
HOT_TIER_CAPACITY_BYTES: 8388608
WARM_TIER_CAPACITY_BYTES: 33554432
TOTAL_RESIDENT_CAPACITY_BYTES: 41943040
BASELINE_FLAT_CAPACITY_BYTES: 8388608
BASELINE_BACKING_SOURCE_BYTES: 283414528
EQUAL_CAPACITY_FLAT_SOURCE_BYTES: 283414528
TIERED_BACKING_SOURCE_BYTES: 283414528
SOURCE_BYTE_REDUCTION_VS_8M_BASELINE: 0
SOURCE_BYTE_REDUCTION_PERCENT_VS_8M_BASELINE: 0
SOURCE_BYTE_REDUCTION_VS_EQUAL_CAPACITY_FLAT: 0
HOT_HITS: 84
WARM_HITS: 252
BACKING_MISSES: 348
PROMOTION_EVENTS: 84
PROMOTION_BYTES: 133365760
DEMOTION_EVENTS: 79
DEMOTION_BYTES: 125751296
TOTAL_INTER_TIER_MOVEMENT_BYTES: 259117056
TOTAL_DATA_MOVEMENT_BYTES: 542531584
DUPLICATE_RESIDENT_PAYLOAD_BYTES: 0
DIRTY_WRITEBACK_BYTES: 0
PROMOTION_PRECISION: 1.0
GENERATIONAL_POLICY_GENERIC: YES
ARCHITECTURE_SPECIFIC_TIERING_LOGIC: NO
BASELINE_PEAK_ACTIVE_PERSISTENT_BYTES: 1892352
TIERED_PEAK_ACTIVE_PERSISTENT_BYTES: 1892352
HOT_PEAK_RESIDENT_BYTES: 8380416
WARM_PEAK_RESIDENT_BYTES: 33543168
TOTAL_PEAK_RESIDENT_BYTES: 41923584
MULTI_LAYER_REFERENCE_PARITY: PASS
ROUTER_SELECTIONS_UNCHANGED: PASS
UNSELECTED_EXPERT_SOURCE_READS: 0
ACTIVATION_BOUNDARY_COPY_BYTES: 0
TOTAL_BYTES_COPIED_FOR_EXECUTION_PREP: 0
TOTAL_BYTES_REPACKED: 0
TOTAL_BYTES_TRANSCODED: 0
MIDDLE_BLOCK_ATTENTION_FAILURE: PASS
MIDDLE_BLOCK_SELECTED_EXPERT_FAILURE: PASS
PROMOTION_FAILURE_FAILS_CLOSED: PASS
BACKING_SOURCE_FAILURE_FAILS_CLOSED: PASS
RUNTIME_STATE_USES_WEIGHT_RESIDENCY_TIERS: NO
VBUF_LAYOUT_CHANGE_REQUIRED: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
RV2_LOGICAL_RANGE_MANIFEST_CHANGED: NO
FULL_X86_CTEST: PASS
GIT_DIFF_CHECK: PASS
CCC_INDEX: PASS
COMMIT_CREATED: NO
TIERING_BEATS_EQUAL_CAPACITY_FLAT_CACHE: NO
READY_FOR_REAL_RAM_VRAM_TIER_POC: YES
READY_FOR_LARGE_MODEL_STREAMING_POC: NO
```
