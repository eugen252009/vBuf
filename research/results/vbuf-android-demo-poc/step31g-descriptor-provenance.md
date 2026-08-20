# Step 31G Descriptor Provenance

Date: 2026-08-20
Starting commit: `1d2ccc6`
Status: **FIRST-INVALID BOUNDARY QUALIFIED; SHAPE LIFETIME FAILURE PROVEN**

## Objective

Locate the first boundary at which the malformed descriptor for
`blk.1.ffn_down_shexp.weight` appears, without repairing the descriptor or
changing source, materialization, residency, backend, or inference behavior.

## Physical Evidence

The diagnostic build was run on the Pixel 7 Pro with the canonical 512 MiB
local workload from Step 31F. The relevant final sequence was:

```text
executor_persistent: dims=2816,2048 representation=10 payload=0
resident_ready:      dims=12970367413557265264,0 representation=10 payload=<valid>
executor_ready:      dims=12970367413557265264,0 representation=10 payload=<valid>
pre_ggml:             dims=12970367413557265264,0 representation=10 payload=<valid>
```

The failure immediately followed at `expert_down_matmul`. The source offset,
representation, payload size, and payload owner remained valid; only the
borrowed shape values were malformed.

## Boundary Result

`executor_persistent` is the last valid boundary. `resident_ready` is the
first invalid boundary. The executor requests a valid persistent view, but a
residency hit returns a cached `MaterializedTensor` whose `view.dimensions`
pointer refers to a different shape-storage address. That cached pointer then
propagates unchanged through `executor_ready` and `pre_ggml`.

The trace also captured an earlier valid materialization with the same
`2816,2048` geometry and IQ2_XXS representation. This distinguishes the
failure from malformed source metadata and from payload corruption at the
materializer boundary.

The required classification is:

```text
LAST_VALID_BOUNDARY: executor_persistent immediately before residency lookup
FIRST_INVALID_BOUNDARY: resident_ready returning cached MaterializedTensor
PAYLOAD_LIFETIME: VALID
PAYLOAD_CORRUPTION: NOT OBSERVED
SHAPE_STORAGE: BORROWED
SHAPE_POINTER_LIFETIME: INVALID AFTER CACHED MATERIALIZED TENSOR OUTLIVES ORIGINAL SHAPE OWNER
ROOT_CAUSE_CLASS: SHAPE_OWNER_LIFETIME_MISMATCH / DANGLING_BORROWED_DESCRIPTOR_VIEW
LIFETIME_FAILURE_PROVEN: YES
RESIDENCY_ALGORITHM_BUG_PROVEN: NO
RESIDENCY_CACHE_HIT_EXPOSES_LATENT_BUG: YES
```

## Ownership Interpretation

`TensorResidencyStore::insert()` copies `MaterializedTensor` by value, but the
embedded `VbufTensorView` contains a borrowed `dimensions` pointer. The
residency entry therefore retains payload ownership through its lease while
retaining no ownership of the shape storage. The observed `resident_ready`
shape address differs from the current `executor_persistent` shape address and
contains stack-like garbage values.

This proves a stale or otherwise invalid borrowed-shape view is returned by
the residency hit. It does not yet prove which construction, copy/move,
container, or graph reuse event invalidated the shape storage, and it does not
authorize a runtime fix in this step.

## Qualification Boundary

```text
RESIDENCY_POLICY_CHANGED: NO
RESIDENCY_ALGORITHM_CHANGED: NO
SOURCE_BEHAVIOR_CHANGED: NO
MATERIALIZATION_BEHAVIOR_CHANGED: NO
BACKEND_CHANGED: NO
INFERENCE_SEMANTICS_CHANGED: NO
DIAGNOSTIC_PROVENANCE_ADDED: YES (temporary, removed after capture)
FIRST_INVALID_BOUNDARY: resident_ready
ROOT_CAUSE_FIXED: NO
```

The existing 256 MiB successful qualification remains the comparison control
from Step 31E. A fresh 256 MiB rerun was not performed because the local model
artifact and range server are not present in this workspace; no new control
result is claimed here.

```text
FRESH_256_CONTROL: NOT EXECUTED
REASON: local qualification model/range source currently unavailable
EXISTING_256_CONTROL: already qualified in Step 31E
IMPACT_ON_STEP31G_BOUNDARY_PROOF: NONE
```
