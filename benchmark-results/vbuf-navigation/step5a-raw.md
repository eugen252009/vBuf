# Step 5A raw navigation qualification

This report is generated from the CSV; timings are measurements, not wire decisions.

- CPU: `unreported`
- Python validator: `3.13.5`
- Raw rows: 17280; groups: 864; samples/group: 20; operation variants: 18; BaseSteps: 8, 16, 32, 64, 128, 256 bytes

## Median nanoseconds by operation (all layouts pooled per BaseStep)

| BaseStep | Operation | Variant | Median ns |
|---:|---|---|---:|
| 8 | canonical-block-traversal | canonical | 8,305.0 |
| 8 | canonical-validation | parse-and-describe | 2,690.0 |
| 8 | directory-construction | sorted-validated-ranges | 3,340.0 |
| 8 | key-lookup | canonical-linear-scan | 25,950.0 |
| 8 | key-lookup | directory-binary-search | 15,310.0 |
| 8 | nano-construction | reconstruct-from-validated-canonical | 1,410.0 |
| 8 | nano-deployment | embedded-load | 520.0 |
| 8 | nano-deployment | local-reconstruction | 4,330.0 |
| 8 | nano-deployment | reconstructed-cache-load | 460.0 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 663,106.0 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-512 | 503,669.0 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 631,590.5 |
| 8 | nth-start-plus-next-boundary | raw-nano-linear-select | 3,339,847.5 |
| 8 | parallel-payload-sum | canonical-rayon | 23,360.5 |
| 8 | parallel-payload-sum | dynamic-queue-control | 22,620.0 |
| 8 | parallel-payload-sum | nano-guided-rayon | 296,902.5 |
| 8 | parallel-start-enumeration | nano-rayon | 25,255.0 |
| 8 | physical-start-enumeration | nano | 4,245.0 |
| 16 | canonical-block-traversal | canonical | 8,345.0 |
| 16 | canonical-validation | parse-and-describe | 3,795.0 |
| 16 | directory-construction | sorted-validated-ranges | 6,560.0 |
| 16 | key-lookup | canonical-linear-scan | 26,140.0 |
| 16 | key-lookup | directory-binary-search | 15,305.0 |
| 16 | nano-construction | reconstruct-from-validated-canonical | 930.0 |
| 16 | nano-deployment | embedded-load | 400.0 |
| 16 | nano-deployment | local-reconstruction | 4,370.0 |
| 16 | nano-deployment | reconstructed-cache-load | 400.0 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 323,353.0 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-512 | 288,632.5 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 381,913.0 |
| 16 | nth-start-plus-next-boundary | raw-nano-linear-select | 1,706,909.0 |
| 16 | parallel-payload-sum | canonical-rayon | 23,190.5 |
| 16 | parallel-payload-sum | dynamic-queue-control | 22,725.0 |
| 16 | parallel-payload-sum | nano-guided-rayon | 271,122.0 |
| 16 | parallel-start-enumeration | nano-rayon | 24,305.0 |
| 16 | physical-start-enumeration | nano | 3,940.0 |
| 32 | canonical-block-traversal | canonical | 8,825.0 |
| 32 | canonical-validation | parse-and-describe | 5,790.0 |
| 32 | directory-construction | sorted-validated-ranges | 3,325.5 |
| 32 | key-lookup | canonical-linear-scan | 25,955.0 |
| 32 | key-lookup | directory-binary-search | 15,315.0 |
| 32 | nano-construction | reconstruct-from-validated-canonical | 620.0 |
| 32 | nano-deployment | embedded-load | 400.0 |
| 32 | nano-deployment | local-reconstruction | 4,320.0 |
| 32 | nano-deployment | reconstructed-cache-load | 400.0 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 222,922.0 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-512 | 165,751.0 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 319,483.0 |
| 32 | nth-start-plus-next-boundary | raw-nano-linear-select | 896,027.5 |
| 32 | parallel-payload-sum | canonical-rayon | 25,020.5 |
| 32 | parallel-payload-sum | dynamic-queue-control | 23,635.5 |
| 32 | parallel-payload-sum | nano-guided-rayon | 258,872.0 |
| 32 | parallel-start-enumeration | nano-rayon | 25,510.0 |
| 32 | physical-start-enumeration | nano | 3,940.0 |
| 64 | canonical-block-traversal | canonical | 10,610.0 |
| 64 | canonical-validation | parse-and-describe | 10,050.0 |
| 64 | directory-construction | sorted-validated-ranges | 3,345.0 |
| 64 | key-lookup | canonical-linear-scan | 25,945.0 |
| 64 | key-lookup | directory-binary-search | 15,215.0 |
| 64 | nano-construction | reconstruct-from-validated-canonical | 475.0 |
| 64 | nano-deployment | embedded-load | 400.0 |
| 64 | nano-deployment | local-reconstruction | 4,100.5 |
| 64 | nano-deployment | reconstructed-cache-load | 400.0 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 128,810.5 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-512 | 103,706.0 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 288,302.0 |
| 64 | nth-start-plus-next-boundary | raw-nano-linear-select | 531,554.5 |
| 64 | parallel-payload-sum | canonical-rayon | 23,170.0 |
| 64 | parallel-payload-sum | dynamic-queue-control | 22,920.0 |
| 64 | parallel-payload-sum | nano-guided-rayon | 255,687.5 |
| 64 | parallel-start-enumeration | nano-rayon | 24,186.0 |
| 64 | physical-start-enumeration | nano | 3,940.0 |
| 128 | canonical-block-traversal | canonical | 16,220.0 |
| 128 | canonical-validation | parse-and-describe | 19,385.0 |
| 128 | directory-construction | sorted-validated-ranges | 3,335.0 |
| 128 | key-lookup | canonical-linear-scan | 25,990.5 |
| 128 | key-lookup | directory-binary-search | 15,305.0 |
| 128 | nano-construction | reconstruct-from-validated-canonical | 390.0 |
| 128 | nano-deployment | embedded-load | 400.0 |
| 128 | nano-deployment | local-reconstruction | 2,360.0 |
| 128 | nano-deployment | reconstructed-cache-load | 400.0 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 106,980.5 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-512 | 66,051.0 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 304,892.0 |
| 128 | nth-start-plus-next-boundary | raw-nano-linear-select | 271,422.5 |
| 128 | parallel-payload-sum | canonical-rayon | 23,180.0 |
| 128 | parallel-payload-sum | dynamic-queue-control | 23,050.0 |
| 128 | parallel-payload-sum | nano-guided-rayon | 220,682.0 |
| 128 | parallel-start-enumeration | nano-rayon | 23,720.0 |
| 128 | physical-start-enumeration | nano | 3,940.0 |
| 256 | canonical-block-traversal | canonical | 17,660.0 |
| 256 | canonical-validation | parse-and-describe | 36,375.5 |
| 256 | directory-construction | sorted-validated-ranges | 3,345.0 |
| 256 | key-lookup | canonical-linear-scan | 25,945.0 |
| 256 | key-lookup | directory-binary-search | 15,305.0 |
| 256 | nano-construction | reconstruct-from-validated-canonical | 345.0 |
| 256 | nano-deployment | embedded-load | 405.0 |
| 256 | nano-deployment | local-reconstruction | 1,370.0 |
| 256 | nano-deployment | reconstructed-cache-load | 400.0 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 85,915.5 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-512 | 56,650.5 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 169,831.0 |
| 256 | nth-start-plus-next-boundary | raw-nano-linear-select | 171,041.0 |
| 256 | parallel-payload-sum | canonical-rayon | 24,305.5 |
| 256 | parallel-payload-sum | dynamic-queue-control | 23,515.5 |
| 256 | parallel-payload-sum | nano-guided-rayon | 214,601.5 |
| 256 | parallel-start-enumeration | nano-rayon | 23,850.0 |
| 256 | physical-start-enumeration | nano | 3,940.0 |

## Measured facts and conservative decisions

- **Measured fact:** all six frozen BaseSteps and all eight generic layouts completed with stable metadata and 20 samples per operation.
- **Measured fact:** exact header and padding bytes are recorded separately from payload, Nano, checkpoint, and directory bytes in the raw CSV.
- **Inference:** these in-memory timings do not establish a universal BaseStep winner, nor do they include OS-controlled cold-cache state.
- **Decision:** BaseStep remains a legal generic tuning input; no BaseStep is promoted or removed by this run.
- **Decision:** Nano, checkpoints, and region-directory paths remain `POSSIBLE BUT NOT YET JUSTIFIED` pending independent end-to-end qualification including construction, persistence, validation, and equivalent lookup work.
- **Decision:** no experimental artifact becomes normative and no finalization envelope is added in Step 5A.

## Environment limitations

The Rust runner records compiler/build context in the repository; this validation report intentionally does not infer cache, fault, SIMD, or pointer-alignment claims that were not instrumented.
