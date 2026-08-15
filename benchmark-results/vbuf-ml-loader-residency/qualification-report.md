# Step 33 Loader/Residency Qualification

Runs: 12
One cold sample per model and strategy; repeat before publication.
Output equivalence: `True`

## Strategy Results

| Model | Strategy | First eval (ms) | End-to-end (ms) | Gate wait (ms) |
|---|---:|---:|---:|---:|
| 0.6B | s0 | 588 | 1106 | 0 |
| 0.6B | s1 | 57 | 964 | 0 |
| 0.6B | s2 | 434 | 875 | 0 |
| 0.6B | s3 | 273 | 878 | 0 |
| 0.6B | s4 | 435 | 865 | 375 |
| 0.6B | s5 | 34 | 1085 | 0 |
| 32B | s0 | 37323 | 52317 | 0 |
| 32B | s1 | 2350 | 37743 | 0 |
| 32B | s2 | 25351 | 35635 | 0 |
| 32B | s3 | 22947 | 34973 | 0 |
| 32B | s4 | 24546 | 34960 | 21651 |
| 32B | s5 | 1068 | 41145 | 0 |

## Classifications
- current_loading_behavior: lazy demand-fault residency; cold 32B first eval is about 37 s
- full_preload: explicit pread removes compute-time page faults but has about 25 s host-residency startup on 32B
- overlap: measured; S3 H4 and S4 H2 reduce 32B end-to-end time to about 35 s
- io_geometry: NOT_REACHED
- physical_bottleneck: storage bandwidth; measured loader is about 1.4 GB/s while compute consumes layer spans faster
- final_recommendation: no production change from this qualification; bounded overlap is the leading policy candidate, pending repeated runs

Primary observations are in `strategy-summary.csv`; raw harness JSON is under `raw/`.
Conditional I/O geometry sweeps are marked `NOT_REACHED` until the primary run set is complete.
