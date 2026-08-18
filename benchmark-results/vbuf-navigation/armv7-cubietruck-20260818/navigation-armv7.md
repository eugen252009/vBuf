# Step 5A raw navigation qualification

This report is generated from the CSV; timings are measurements, not wire decisions.

- CPU: `unreported`
- Python validator: `3.14.7`
- Raw rows: 17280; groups: 864; samples/group: 20; operation variants: 18; BaseSteps: 8, 16, 32, 64, 128, 256 bytes

## Median nanoseconds by operation (all layouts pooled per BaseStep)

| BaseStep | Operation | Variant | Median ns |
|---:|---|---|---:|
| 8 | canonical-block-traversal | canonical | 336,750.0 |
| 8 | canonical-validation | parse-and-describe | 397,187.5 |
| 8 | directory-construction | sorted-validated-ranges | 204,520.5 |
| 8 | key-lookup | canonical-linear-scan | 226,687.5 |
| 8 | key-lookup | directory-binary-search | 22,416.5 |
| 8 | nano-construction | reconstruct-from-validated-canonical | 84,396.0 |
| 8 | nano-deployment | embedded-load | 8,979.0 |
| 8 | nano-deployment | local-reconstruction | 89,687.5 |
| 8 | nano-deployment | reconstructed-cache-load | 8,666.5 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 21,637,333.5 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-512 | 18,518,375.0 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 21,629,979.5 |
| 8 | nth-start-plus-next-boundary | raw-nano-linear-select | 92,408,896.0 |
| 8 | parallel-payload-sum | canonical-rayon | 319,708.0 |
| 8 | parallel-payload-sum | dynamic-queue-control | 283,728.5 |
| 8 | parallel-payload-sum | nano-guided-rayon | 9,107,541.5 |
| 8 | parallel-start-enumeration | nano-rayon | 279,583.5 |
| 8 | physical-start-enumeration | nano | 210,166.5 |
| 16 | canonical-block-traversal | canonical | 431,875.5 |
| 16 | canonical-validation | parse-and-describe | 438,104.0 |
| 16 | directory-construction | sorted-validated-ranges | 210,583.0 |
| 16 | key-lookup | canonical-linear-scan | 205,750.0 |
| 16 | key-lookup | directory-binary-search | 25,896.0 |
| 16 | nano-construction | reconstruct-from-validated-canonical | 200,354.0 |
| 16 | nano-deployment | embedded-load | 39,541.5 |
| 16 | nano-deployment | local-reconstruction | 170,854.5 |
| 16 | nano-deployment | reconstructed-cache-load | 42,812.0 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 12,020,000.0 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-512 | 10,830,604.0 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 13,953,896.0 |
| 16 | nth-start-plus-next-boundary | raw-nano-linear-select | 47,338,666.5 |
| 16 | parallel-payload-sum | canonical-rayon | 394,583.0 |
| 16 | parallel-payload-sum | dynamic-queue-control | 362,542.0 |
| 16 | parallel-payload-sum | nano-guided-rayon | 8,561,458.5 |
| 16 | parallel-start-enumeration | nano-rayon | 274,000.0 |
| 16 | physical-start-enumeration | nano | 167,542.0 |
| 32 | canonical-block-traversal | canonical | 815,520.5 |
| 32 | canonical-validation | parse-and-describe | 345,541.5 |
| 32 | directory-construction | sorted-validated-ranges | 153,249.5 |
| 32 | key-lookup | canonical-linear-scan | 259,271.0 |
| 32 | key-lookup | directory-binary-search | 22,395.5 |
| 32 | nano-construction | reconstruct-from-validated-canonical | 193,958.0 |
| 32 | nano-deployment | embedded-load | 68,958.5 |
| 32 | nano-deployment | local-reconstruction | 188,083.5 |
| 32 | nano-deployment | reconstructed-cache-load | 64,750.0 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 7,166,854.0 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-512 | 5,989,500.0 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 11,105,479.0 |
| 32 | nth-start-plus-next-boundary | raw-nano-linear-select | 24,745,729.5 |
| 32 | parallel-payload-sum | canonical-rayon | 582,770.5 |
| 32 | parallel-payload-sum | dynamic-queue-control | 505,458.5 |
| 32 | parallel-payload-sum | nano-guided-rayon | 7,998,354.0 |
| 32 | parallel-start-enumeration | nano-rayon | 273,229.5 |
| 32 | physical-start-enumeration | nano | 167,541.0 |
| 64 | canonical-block-traversal | canonical | 1,203,250.0 |
| 64 | canonical-validation | parse-and-describe | 559,416.5 |
| 64 | directory-construction | sorted-validated-ranges | 147,354.0 |
| 64 | key-lookup | canonical-linear-scan | 202,854.5 |
| 64 | key-lookup | directory-binary-search | 27,833.0 |
| 64 | nano-construction | reconstruct-from-validated-canonical | 36,041.5 |
| 64 | nano-deployment | embedded-load | 7,458.5 |
| 64 | nano-deployment | local-reconstruction | 100,708.5 |
| 64 | nano-deployment | reconstructed-cache-load | 7,459.0 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 4,762,166.5 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-512 | 3,576,479.5 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 9,684,354.0 |
| 64 | nth-start-plus-next-boundary | raw-nano-linear-select | 13,450,771.0 |
| 64 | parallel-payload-sum | canonical-rayon | 596,333.0 |
| 64 | parallel-payload-sum | dynamic-queue-control | 585,312.5 |
| 64 | parallel-payload-sum | nano-guided-rayon | 7,561,687.0 |
| 64 | parallel-start-enumeration | nano-rayon | 268,917.0 |
| 64 | physical-start-enumeration | nano | 167,542.0 |
| 128 | canonical-block-traversal | canonical | 1,487,125.0 |
| 128 | canonical-validation | parse-and-describe | 735,583.0 |
| 128 | directory-construction | sorted-validated-ranges | 202,792.0 |
| 128 | key-lookup | canonical-linear-scan | 191,917.0 |
| 128 | key-lookup | directory-binary-search | 22,291.5 |
| 128 | nano-construction | reconstruct-from-validated-canonical | 11,208.0 |
| 128 | nano-deployment | embedded-load | 30,208.0 |
| 128 | nano-deployment | local-reconstruction | 58,604.5 |
| 128 | nano-deployment | reconstructed-cache-load | 7,542.0 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 3,546,833.0 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-512 | 2,360,999.5 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 9,020,770.5 |
| 128 | nth-start-plus-next-boundary | raw-nano-linear-select | 7,756,625.5 |
| 128 | parallel-payload-sum | canonical-rayon | 653,729.5 |
| 128 | parallel-payload-sum | dynamic-queue-control | 615,770.5 |
| 128 | parallel-payload-sum | nano-guided-rayon | 7,459,645.5 |
| 128 | parallel-start-enumeration | nano-rayon | 255,083.0 |
| 128 | physical-start-enumeration | nano | 167,541.0 |
| 256 | canonical-block-traversal | canonical | 1,522,271.0 |
| 256 | canonical-validation | parse-and-describe | 1,072,104.0 |
| 256 | directory-construction | sorted-validated-ranges | 146,416.5 |
| 256 | key-lookup | canonical-linear-scan | 236,979.5 |
| 256 | key-lookup | directory-binary-search | 27,833.0 |
| 256 | nano-construction | reconstruct-from-validated-canonical | 9,125.0 |
| 256 | nano-deployment | embedded-load | 17,729.0 |
| 256 | nano-deployment | local-reconstruction | 32,916.0 |
| 256 | nano-deployment | reconstructed-cache-load | 17,021.0 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 3,069,625.0 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-512 | 1,783,250.0 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 5,658,833.0 |
| 256 | nth-start-plus-next-boundary | raw-nano-linear-select | 4,943,333.5 |
| 256 | parallel-payload-sum | canonical-rayon | 638,750.5 |
| 256 | parallel-payload-sum | dynamic-queue-control | 605,750.0 |
| 256 | parallel-payload-sum | nano-guided-rayon | 7,230,479.0 |
| 256 | parallel-start-enumeration | nano-rayon | 265,396.0 |
| 256 | physical-start-enumeration | nano | 167,541.0 |

## Measured facts and conservative decisions

- **Measured fact:** all six frozen BaseSteps and all eight generic layouts completed with stable metadata and 20 samples per operation.
- **Measured fact:** exact header and padding bytes are recorded separately from payload, Nano, checkpoint, and directory bytes in the raw CSV.
- **Inference:** these in-memory timings do not establish a universal BaseStep winner, nor do they include OS-controlled cold-cache state.
- **Decision:** BaseStep remains a legal generic tuning input; no BaseStep is promoted or removed by this run.
- **Decision:** Nano, checkpoints, and region-directory paths remain `POSSIBLE BUT NOT YET JUSTIFIED` pending independent end-to-end qualification including construction, persistence, validation, and equivalent lookup work.
- **Decision:** no experimental artifact becomes normative and no finalization envelope is added in Step 5A.

## Environment limitations

The Rust runner records compiler/build context in the repository; this validation report intentionally does not infer cache, fault, SIMD, or pointer-alignment claims that were not instrumented.
